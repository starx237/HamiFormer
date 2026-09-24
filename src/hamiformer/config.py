from __future__ import annotations
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints
import yaml
T = TypeVar('T')

def _strict_dataclass(cls: type[T], raw: dict[str, Any], section: str) -> T:
    if not isinstance(raw, dict):
        raise ValueError(f'配置段 {section!r} 必须是映射')
    known = {item.name for item in fields(cls)}
    unknown = set(raw) - known
    missing = {item.name for item in fields(cls) if item.default is MISSING and item.default_factory is MISSING} - set(raw)
    if unknown:
        raise ValueError(f'配置段 {section!r} 含未知字段: {sorted(unknown)}')
    if missing:
        raise ValueError(f'配置段 {section!r} 缺少字段: {sorted(missing)}')
    hints = get_type_hints(cls)
    for name, expected in hints.items():
        value = raw[name]
        if expected in {bool, int, float, str} and type(value) is not expected:
            raise ValueError(f'配置段 {section!r} 的 {name} 必须为 {expected.__name__}，实际为 {type(value).__name__}')
    return cls(**raw)

@dataclass(frozen=True)
class DataConfig:
    train_manifest: str
    dev_manifest: str
    calibration_fit_manifest: str
    calibration_audit_manifest: str
    test_manifest: str
    long_test_manifest: str
    stats_path: str
    dataset_meta_path: str
    audit_path: str
    num_objects: int
    future_steps: int
    q_dim: int
    attr_dim: int
    num_workers: int
    pin_memory: bool

    @property
    def state_dim(self) -> int:
        return 2 * self.q_dim

@dataclass(frozen=True)
class RectifiedFlowConfig:
    logit_normal_probability: float
    logit_mean: float
    logit_std: float
    tau_max: float
    t_eps: float
    noise_scale: float

@dataclass(frozen=True)
class PhaseDConfig:
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_register_tokens: int
    dropout: float
    qk_norm: bool

@dataclass(frozen=True)
class HamiltonianHConfig:
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    force_float32: bool

@dataclass(frozen=True)
class MatchedSConfig:
    enabled: bool
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float

@dataclass(frozen=True)
class SamplerConfig:
    num_intervals: int
    final_d_euler: bool
    fixed_expert_solver: str

@dataclass(frozen=True)
class RoutingConfig:
    mode: str
    num_bins: int
    thresholds_path: str
    far_limit: float
    cvar_limit: float
    cvar_tail_fraction: float
    minimum_accepted: int

@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    batch_size: int
    max_steps: int
    smoke_steps: int
    learning_rate: float
    warmup_steps: int
    min_learning_rate: float
    weight_decay: float
    grad_clip: float
    ema_decay: float
    log_every: int
    save_every: int
    mixed_precision_d: str
    mixed_precision_h: str
    h_tau_sampling: str
    h_denoise_objective: str
    h_denoise_weight: float
    h_clean_relation_weight: float

@dataclass(frozen=True)
class EvaluationConfig:
    long_rollout_chunks: int
    paired_noise_seed: int

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    data: DataConfig
    rectified_flow: RectifiedFlowConfig
    phase_d: PhaseDConfig
    hamiltonian_h: HamiltonianHConfig
    matched_s: MatchedSConfig
    sampler: SamplerConfig
    routing: RoutingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        if self.data.num_objects <= 0 or self.data.q_dim <= 0 or self.data.future_steps < 2:
            raise ValueError('q_dim 必须为正，future_steps 至少为 2 才有 interior disagreement')
        if self.data.attr_dim != 3:
            raise ValueError('v1.0 attrs 固定为 mass/radius/restitution，因此 attr_dim 必须为 3')
        if self.data.num_workers < 0 or type(self.data.pin_memory) is not bool:
            raise ValueError('num_workers 必须非负，pin_memory 必须是布尔值')
        if not 0.0 <= self.rectified_flow.logit_normal_probability <= 1.0:
            raise ValueError('logit_normal_probability 必须位于 [0,1]')
        if self.rectified_flow.logit_std <= 0.0 or self.rectified_flow.t_eps <= 0.0 or self.rectified_flow.t_eps > 1.0 or (self.rectified_flow.noise_scale <= 0.0):
            raise ValueError('logit_std/noise_scale 必须为正，t_eps 必须位于 (0,1]')
        model_sections = (('phase_d', self.phase_d), ('hamiltonian_h', self.hamiltonian_h), ('matched_s', self.matched_s))
        for name, section in model_sections:
            if section.hidden_size <= 0 or section.depth <= 0 or section.num_heads <= 0:
                raise ValueError(f'{name} 的 hidden_size/depth/num_heads 必须为正')
            if section.mlp_ratio <= 0.0 or not 0.0 <= section.dropout < 1.0:
                raise ValueError(f'{name} 的 mlp_ratio 必须为正且 dropout 位于 [0,1)')
            if section.hidden_size % section.num_heads != 0:
                raise ValueError(f'{name}.hidden_size 必须能被 num_heads 整除')
        if self.phase_d.num_register_tokens < 0 or type(self.phase_d.qk_norm) is not bool:
            raise ValueError('num_register_tokens 必须非负，qk_norm 必须是布尔值')
        if type(self.matched_s.enabled) is not bool or type(self.hamiltonian_h.force_float32) is not bool:
            raise ValueError('matched_s.enabled/force_float32 必须是布尔值')
        if not 0.0 < self.rectified_flow.tau_max < 1.0:
            raise ValueError('tau_max 必须严格位于 (0,1)，避免 RF velocity 除零')
        if self.sampler.num_intervals < 2 or not self.sampler.final_d_euler:
            raise ValueError('v1.0 要求至少两个 interval，且最后一个 interval 固定 D-Euler')
        if self.sampler.fixed_expert_solver not in {'heun', 'clean_euler'}:
            raise ValueError('fixed_expert_solver 只能是 heun 或 clean_euler')
        if self.routing.mode not in {'hard', 'soft_three_zone'}:
            raise ValueError('routing.mode 只能是 hard 或 soft_three_zone')
        if self.routing.num_bins <= 0:
            raise ValueError('routing.num_bins 必须为正')
        if not 0.0 < self.routing.cvar_tail_fraction <= 1.0 or self.routing.minimum_accepted <= 0:
            raise ValueError('cvar_tail_fraction 必须位于 (0,1]，minimum_accepted 必须为正')
        if self.routing.far_limit < 0.0 or self.routing.cvar_limit < 0.0:
            raise ValueError('FAR/CVaR limit 不能为负')
        train = self.training
        if train.batch_size <= 0 or train.max_steps <= 0 or train.smoke_steps <= 0:
            raise ValueError('batch_size/max_steps/smoke_steps 必须为正')
        if train.warmup_steps < 0 or train.log_every <= 0 or train.save_every <= 0:
            raise ValueError('warmup_steps 必须非负，log_every/save_every 必须为正')
        if not 0.0 < train.min_learning_rate <= train.learning_rate:
            raise ValueError('学习率必须满足 0 < min_learning_rate <= learning_rate')
        if train.weight_decay < 0.0 or train.grad_clip <= 0.0 or (not 0.0 < train.ema_decay < 1.0):
            raise ValueError('weight_decay/grad_clip/ema_decay 取值非法')
        if self.training.mixed_precision_d not in {'fp32', 'bf16'}:
            raise ValueError('mixed_precision_d 只能是 fp32 或 bf16')
        if self.training.mixed_precision_h != 'fp32' or not self.hamiltonian_h.force_float32:
            raise ValueError('H expert 含二阶自动微分，v1.0 必须同时启用 force_float32 与 fp32')
        if train.h_tau_sampling not in {'base', 'solver_nodes'}:
            raise ValueError('h_tau_sampling 只能是 base 或 solver_nodes')
        if train.h_denoise_objective not in {'velocity', 'clean'}:
            raise ValueError('h_denoise_objective 只能是 velocity 或 clean')
        if train.h_denoise_weight < 0.0 or train.h_clean_relation_weight < 0.0:
            raise ValueError('H denoise/clean relation 权重不能为负')
        if train.h_denoise_weight + train.h_clean_relation_weight <= 0.0:
            raise ValueError('H 至少需要一个非零训练目标')
        if self.evaluation.long_rollout_chunks <= 0:
            raise ValueError('long_rollout_chunks 必须为正')

def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open('r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f'配置根节点必须是映射: {config_path}')
    expected_root = {'seed', 'device', 'data', 'rectified_flow', 'phase_d', 'hamiltonian_h', 'matched_s', 'sampler', 'routing', 'training', 'evaluation'}
    unknown = set(raw) - expected_root
    missing = expected_root - set(raw)
    if unknown or missing:
        raise ValueError(f'配置根字段错误；未知={sorted(unknown)}，缺失={sorted(missing)}')
    if type(raw['seed']) is not int or type(raw['device']) is not str:
        raise ValueError('根配置 seed/device 必须分别为 int/string，禁止静默转换')
    cfg = ExperimentConfig(seed=raw['seed'], device=raw['device'], data=_strict_dataclass(DataConfig, raw['data'], 'data'), rectified_flow=_strict_dataclass(RectifiedFlowConfig, raw['rectified_flow'], 'rectified_flow'), phase_d=_strict_dataclass(PhaseDConfig, raw['phase_d'], 'phase_d'), hamiltonian_h=_strict_dataclass(HamiltonianHConfig, raw['hamiltonian_h'], 'hamiltonian_h'), matched_s=_strict_dataclass(MatchedSConfig, raw['matched_s'], 'matched_s'), sampler=_strict_dataclass(SamplerConfig, raw['sampler'], 'sampler'), routing=_strict_dataclass(RoutingConfig, raw['routing'], 'routing'), training=_strict_dataclass(TrainingConfig, raw['training'], 'training'), evaluation=_strict_dataclass(EvaluationConfig, raw['evaluation'], 'evaluation'))
    cfg.validate()
    return cfg
