from __future__ import annotations
import math
from dataclasses import asdict
import torch
from torch import nn
from hamiformer.config import ExperimentConfig
from hamiformer.flow.rectified_flow import make_rf_pair, sample_tau
from hamiformer.models import ModelBundle
from hamiformer.types import PhaseBatch
from .ema import ExponentialMovingAverage
from .losses import d_rf_loss, occurrence_clean_loss, occurrence_rf_loss
from .checkpoint import validate_checkpoint_payload

class HamiFormerTrainer:

    def __init__(self, config: ExperimentConfig, models: ModelBundle, *, q_scale: torch.Tensor, p_scale: torch.Tensor, device: torch.device | str, data_provenance: dict[str, str] | None=None, expert_mode: str='joint') -> None:
        if expert_mode not in {'joint', 'd_only', 'h_only'}:
            raise ValueError('expert_mode 只能是 joint、d_only 或 h_only')
        self.config = config
        self.device = torch.device(device)
        self.models = models
        self.expert_mode = expert_mode
        self.train_d = expert_mode in {'joint', 'd_only'}
        self.train_h = expert_mode in {'joint', 'h_only'}
        self.train_s = expert_mode == 'joint' and self.models.s is not None
        self.models.d.to(self.device)
        self.models.h.to(self.device, dtype=torch.float32)
        if self.models.s is not None:
            self.models.s.to(self.device)
        self.q_scale = q_scale.to(self.device, dtype=torch.float32)
        self.p_scale = p_scale.to(self.device, dtype=torch.float32)
        self.state_scale = torch.cat([self.q_scale, self.p_scale])
        self.data_provenance = dict(data_provenance or {})
        parameter_groups = []
        if self.train_d:
            parameter_groups.append({'params': self.models.d.parameters(), 'name': 'D'})
        if self.train_h:
            parameter_groups.append({'params': self.models.h.parameters(), 'name': 'H'})
        if self.train_s:
            parameter_groups.append({'params': self.models.s.parameters(), 'name': 'S'})
        train = config.training
        self.optimizer = torch.optim.AdamW(parameter_groups, lr=train.learning_rate, weight_decay=train.weight_decay)
        self.ema_d = ExponentialMovingAverage(self.models.d, train.ema_decay)
        self.ema_h = ExponentialMovingAverage(self.models.h, train.ema_decay)
        self.ema_s = ExponentialMovingAverage(self.models.s, train.ema_decay) if self.models.s is not None else None
        self.step_index = 0

    def _learning_rate(self, step: int) -> float:
        train = self.config.training
        if step < train.warmup_steps:
            return train.learning_rate * float(step + 1) / max(train.warmup_steps, 1)
        progress = (step - train.warmup_steps) / max(train.max_steps - train.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return train.min_learning_rate + cosine * (train.learning_rate - train.min_learning_rate)

    def _set_learning_rate(self) -> float:
        value = self._learning_rate(self.step_index)
        for group in self.optimizer.param_groups:
            group['lr'] = value
        return value

    def _d_autocast(self):
        enabled = self.device.type == 'cuda' and self.config.training.mixed_precision_d == 'bf16'
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def _sample_training_tau(self, batch_size: int) -> torch.Tensor:
        if self.train_h and self.config.training.h_tau_sampling == 'solver_nodes':
            nodes = torch.linspace(0.0, self.config.rectified_flow.tau_max, self.config.sampler.num_intervals, device=self.device, dtype=torch.float32)
            start = self.step_index * batch_size % nodes.numel()
            indices = (torch.arange(batch_size, device=self.device) + start) % nodes.numel()
            return nodes[indices]
        return sample_tau(batch_size, self.config.rectified_flow, device=self.device)

    def train_step(self, batch: PhaseBatch) -> dict[str, float]:
        batch = batch.to(self.device, dtype=torch.float32)
        self.models.d.train(self.train_d)
        self.models.h.train(self.train_h)
        if self.models.s is not None:
            self.models.s.train(self.train_s)
        tau = self._sample_training_tau(batch.future.shape[0])
        pair = make_rf_pair(batch.future, tau, self.state_scale, noise_scale=self.config.rectified_flow.noise_scale)
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate = self._set_learning_rate()

        def require_finite(name: str, loss: torch.Tensor) -> None:
            if not bool(torch.isfinite(loss).item()):
                ids = [] if batch.sample_id is None else batch.sample_id[:4]
                tau_preview = pair.tau.detach().cpu().tolist()[:4]
                raise FloatingPointError(f'{name} 非有限；拒绝 optimizer/EMA 更新。sample_id={ids}, tau={tau_preview}')
        loss_d = None
        if self.train_d:
            with self._d_autocast():
                d_clean = self.models.d(pair.noisy, pair.tau, x0=batch.x0, attrs=batch.attrs, physical_time=batch.time)
                loss_d = d_rf_loss(d_clean, pair, self.state_scale, t_eps=self.config.rectified_flow.t_eps)
            require_finite('D RF loss', loss_d)
            loss_d.backward()
        loss_h = None
        loss_h_denoise = None
        loss_h_clean_relation = None
        if self.train_h:
            h_output = self.models.h(pair.noisy.float(), pair.tau.float(), x0=batch.x0.float(), attrs=batch.attrs.float(), physical_time=batch.time.float(), q_scale=self.q_scale, p_scale=self.p_scale, create_graph=True)
            if self.config.training.h_denoise_objective == 'velocity':
                loss_h_denoise = d_rf_loss(h_output.clean, pair, self.state_scale, t_eps=self.config.rectified_flow.t_eps)
            else:
                loss_h_denoise = occurrence_clean_loss(h_output.occurrences, pair.clean, q_dim=self.config.data.q_dim, q_scale=self.q_scale, p_scale=self.p_scale)
            require_finite('H denoise loss', loss_h_denoise)
            denoise_weight = self.config.training.h_denoise_weight
            if denoise_weight > 0.0:
                (denoise_weight * loss_h_denoise).backward()
            clean_weight = self.config.training.h_clean_relation_weight
            if clean_weight > 0.0:
                clean_occurrences = self.models.h.local_clean_occurrences(batch.future.float(), x0=batch.x0.float(), attrs=batch.attrs.float(), physical_time=batch.time.float(), create_graph=True)
                loss_h_clean_relation = occurrence_clean_loss(clean_occurrences, batch.future, q_dim=self.config.data.q_dim, q_scale=self.q_scale, p_scale=self.p_scale)
                require_finite('H clean generating-relation loss', loss_h_clean_relation)
                (clean_weight * loss_h_clean_relation).backward()
            loss_h = denoise_weight * loss_h_denoise
            if loss_h_clean_relation is not None:
                loss_h = loss_h + clean_weight * loss_h_clean_relation
        loss_s = None
        if self.train_s:
            assert self.models.s is not None
            s_output = self.models.s(pair.noisy, pair.tau, x0=batch.x0, attrs=batch.attrs, physical_time=batch.time, q_scale=self.q_scale, p_scale=self.p_scale)
            loss_s = occurrence_rf_loss(s_output.occurrences, pair, q_dim=self.config.data.q_dim, q_scale=self.q_scale, p_scale=self.p_scale, t_eps=self.config.rectified_flow.t_eps)
            require_finite('S occurrence RF loss', loss_s)
            loss_s.backward()
        clip = self.config.training.grad_clip
        grad_d = torch.tensor(0.0, device=self.device)
        grad_h = torch.tensor(0.0, device=self.device)
        if self.train_d:
            grad_d = nn.utils.clip_grad_norm_(self.models.d.parameters(), clip, error_if_nonfinite=True)
        if self.train_h:
            grad_h = nn.utils.clip_grad_norm_(self.models.h.parameters(), clip, error_if_nonfinite=True)
        grad_s = torch.tensor(0.0, device=self.device)
        if self.train_s:
            assert self.models.s is not None
            grad_s = nn.utils.clip_grad_norm_(self.models.s.parameters(), clip, error_if_nonfinite=True)
        self.optimizer.step()
        if self.train_d:
            self.ema_d.update(self.models.d)
        if self.train_h:
            self.ema_h.update(self.models.h)
        if self.train_s and self.models.s is not None and (self.ema_s is not None):
            self.ema_s.update(self.models.s)
        self.step_index += 1
        metrics = {'learning_rate': float(learning_rate)}
        if loss_d is not None:
            metrics['loss/d_rf'] = float(loss_d.detach().cpu())
            metrics['grad/d'] = float(torch.as_tensor(grad_d).detach().cpu())
        if loss_h is not None:
            metrics['loss/h_total'] = float(loss_h.detach().cpu())
            assert loss_h_denoise is not None
            metrics['loss/h_denoise'] = float(loss_h_denoise.detach().cpu())
            if self.config.training.h_denoise_objective == 'velocity':
                metrics['loss/h_occurrence_rf'] = float(loss_h_denoise.detach().cpu())
            if loss_h_clean_relation is not None:
                metrics['loss/h_clean_relation'] = float(loss_h_clean_relation.detach().cpu())
            metrics['grad/h'] = float(torch.as_tensor(grad_h).detach().cpu())
        if loss_s is not None:
            metrics['loss/s_occurrence_rf'] = float(loss_s.detach().cpu())
            metrics['grad/s'] = float(torch.as_tensor(grad_s).detach().cpu())
        return metrics

    def checkpoint_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {'format_version': 1, 'step': self.step_index, 'config': asdict(self.config), 'models': {'d': self.models.d.state_dict(), 'h': self.models.h.state_dict(), 's': None if self.models.s is None else self.models.s.state_dict()}, 'ema': {'d': self.ema_d.state_dict(), 'h': self.ema_h.state_dict(), 's': None if self.ema_s is None else self.ema_s.state_dict()}, 'phase_scales': {'q': self.q_scale.detach().cpu(), 'p': self.p_scale.detach().cpu()}, 'data_provenance': self.data_provenance, 'expert_mode': self.expert_mode, 'optimizer': self.optimizer.state_dict()}
        return payload

    def restore(self, checkpoint: dict[str, object]) -> None:
        if checkpoint.get('expert_mode', 'joint') != self.expert_mode:
            raise ValueError('checkpoint 的 expert_mode 与当前 trainer 不一致')
        validate_checkpoint_payload(checkpoint, expected_config=asdict(self.config), expected_q_scale=self.q_scale, expected_p_scale=self.p_scale, require_optimizer=True)
        if checkpoint.get('data_provenance', {}) != self.data_provenance:
            raise ValueError('checkpoint 的 train/stats provenance 与当前训练数据不一致')
        models = checkpoint['models']
        ema = checkpoint['ema']
        assert isinstance(models, dict) and isinstance(ema, dict)
        self.models.d.load_state_dict(models['d'], strict=True)
        self.models.h.load_state_dict(models['h'], strict=True)
        if self.models.s is not None:
            if models['s'] is None:
                raise ValueError('当前配置启用 S，但 checkpoint 不含 S')
            self.models.s.load_state_dict(models['s'], strict=True)
        self.ema_d.load_state_dict(ema['d'])
        self.ema_h.load_state_dict(ema['h'])
        if self.ema_s is not None:
            if ema['s'] is None:
                raise ValueError('当前配置启用 S，但 checkpoint 不含 S EMA')
            self.ema_s.load_state_dict(ema['s'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_index = int(checkpoint['step'])
