from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import torch
from torch import nn
from hamiformer.flow.rectified_flow import clean_to_velocity
from hamiformer.models.phase_dit import PhaseDiT, initialize_nested_wide_phase_dit, phase_dit_architecture_kwargs

@dataclass(frozen=True)
class HamiBallsDUpdate:
    predicted_clean: torch.Tensor
    tokens: torch.Tensor
    velocity_loss: float
    clean_loss: float
    gradient_norm_preclip: float

def build_hamiballs_d(model_config: dict[str, Any], *, q_dim: int, attr_dim: int, seed: int, device: torch.device, mlp_inner_dim: int | None=None) -> PhaseDiT:
    torch.manual_seed(int(seed) + 101)
    resolved = dict(model_config)
    if mlp_inner_dim is not None:
        resolved['mlp_inner_dim'] = int(mlp_inner_dim)
    return PhaseDiT(state_dim=int(resolved['state_dim']), q_dim=int(q_dim), attr_dim=int(attr_dim), hidden_size=int(resolved['hidden_size']), depth=int(resolved['depth']), num_heads=int(resolved['num_heads']), mlp_ratio=float(resolved['mlp_ratio']), mlp_inner_dim=None if resolved.get('mlp_inner_dim') is None else int(resolved['mlp_inner_dim']), num_register_tokens=int(resolved['num_register_tokens']), dropout=float(resolved['dropout']), qk_norm=bool(resolved['qk_norm']), **{key: value for key, value in phase_dit_architecture_kwargs(resolved).items() if key != 'mlp_inner_dim'}).to(device=device, dtype=torch.float32)

def d_clean_and_tokens(model: PhaseDiT, noisy: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.encode_tokens(noisy, tau, x0=x0, attrs=attrs, physical_time=physical_time)
    batch, frames, objects, width = tokens.shape
    clean = model.output(tokens.reshape(batch, frames * objects, width)).reshape(batch, frames, objects, model.state_dim)
    return (clean, tokens)

def _prepare_gradient_mirror(model: PhaseDiT, optimizer: torch.optim.Optimizer, *, mirror_model: PhaseDiT, mirror_optimizer: torch.optim.Optimizer) -> None:
    source_parameters = dict(model.named_parameters())
    mirror_parameters = dict(mirror_model.named_parameters())
    if tuple(source_parameters) != tuple(mirror_parameters):
        raise ValueError('D gradient mirror parameter names differ')
    source_buffers = dict(model.named_buffers())
    mirror_buffers = dict(mirror_model.named_buffers())
    if tuple(source_buffers) != tuple(mirror_buffers):
        raise ValueError('D gradient mirror buffer names differ')
    if len(optimizer.param_groups) != len(mirror_optimizer.param_groups):
        raise ValueError('D gradient mirror optimizer group count differs')
    for source_group, mirror_group in zip(optimizer.param_groups, mirror_optimizer.param_groups):
        source_fields = {key: value for key, value in source_group.items() if key != 'params'}
        mirror_fields = {key: value for key, value in mirror_group.items() if key != 'params'}
        if source_fields != mirror_fields or len(source_group['params']) != len(mirror_group['params']):
            raise ValueError('D gradient mirror optimizer contract differs')
    mirror_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, source in source_parameters.items():
            mirror = mirror_parameters[name]
            if source.shape != mirror.shape or source.dtype != mirror.dtype:
                raise ValueError(f'D gradient mirror parameter mismatch: {name}')
            if source.grad is None:
                if mirror.grad is not None:
                    raise AssertionError(f'D gradient mirror stale gradient: {name}')
                continue
            mirror.grad = source.grad.detach().clone(memory_format=torch.preserve_format)
        for name, source in source_buffers.items():
            mirror = mirror_buffers[name]
            if source.shape != mirror.shape or source.dtype != mirror.dtype:
                raise ValueError(f'D gradient mirror buffer mismatch: {name}')
            mirror.copy_(source)

def update_hamiballs_d(model: PhaseDiT, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None, *, clean: torch.Tensor, noisy: torch.Tensor, tau: torch.Tensor, target_velocity: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, t_eps: float, grad_clip: float, mirror_model: PhaseDiT | None=None, mirror_optimizer: torch.optim.Optimizer | None=None, mirror_scheduler: torch.optim.lr_scheduler.LRScheduler | None=None) -> HamiBallsDUpdate:
    if (mirror_model is None) != (mirror_optimizer is None):
        raise ValueError('D gradient mirror requires both model and optimizer')
    if mirror_scheduler is not None and mirror_model is None:
        raise ValueError('D gradient mirror scheduler requires a mirror model')
    optimizer.zero_grad(set_to_none=True)
    predicted_clean, tokens = d_clean_and_tokens(model, noisy, tau, x0=x0, attrs=attrs, physical_time=physical_time)
    velocity = clean_to_velocity(predicted_clean, noisy, tau, t_eps=t_eps)
    velocity_loss = (velocity - target_velocity).square().mean()
    clean_loss = (predicted_clean - clean).square().mean()
    if not bool(torch.isfinite(velocity_loss) and torch.isfinite(clean_loss)):
        raise FloatingPointError('non-finite HamiBalls D objective')
    velocity_loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).cpu())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError('non-finite HamiBalls D gradient')
    if mirror_model is not None:
        assert mirror_optimizer is not None
        _prepare_gradient_mirror(model, optimizer, mirror_model=mirror_model, mirror_optimizer=mirror_optimizer)
    optimizer.step()
    if mirror_model is not None:
        assert mirror_optimizer is not None
        mirror_optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if mirror_scheduler is not None:
        mirror_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if mirror_model is not None:
        assert mirror_optimizer is not None
        mirror_optimizer.zero_grad(set_to_none=True)
    return HamiBallsDUpdate(predicted_clean=predicted_clean.detach(), tokens=tokens.detach(), velocity_loss=float(velocity_loss.detach().cpu()), clean_loss=float(clean_loss.detach().cpu()), gradient_norm_preclip=gradient_norm)

def parameter_count(module: nn.Module) -> int:
    return sum((parameter.numel() for parameter in module.parameters()))

def matched_wide_inner_dim(narrow: PhaseDiT, *, deployed_extra_parameters: int) -> dict[str, int]:
    if deployed_extra_parameters < 1:
        raise ValueError('deployed Main extras must be positive')
    inner_values = {int(block.mlp.inner_dim) for block in narrow.blocks}
    if len(inner_values) != 1:
        raise ValueError('narrow D must use a uniform FFN inner width')
    narrow_inner = next(iter(inner_values))
    hidden_size = int(narrow.state_projection.out_features)
    per_unit = len(narrow.blocks) * (3 * hidden_size + 2)
    exact_added = deployed_extra_parameters / per_unit
    candidates = {narrow_inner + max(0, math.floor(exact_added)), narrow_inner + max(0, math.ceil(exact_added))}
    chosen = min(candidates, key=lambda value: (abs((value - narrow_inner) * per_unit - deployed_extra_parameters), value))
    added = (chosen - narrow_inner) * per_unit
    return {'narrow_inner_dim': narrow_inner, 'wide_inner_dim': chosen, 'deployed_extra_parameters': int(deployed_extra_parameters), 'wide_added_parameters': int(added), 'wide_minus_main_parameters': int(added - deployed_extra_parameters), 'parameters_per_inner_unit': int(per_unit)}

def nested_wide_from_narrow(narrow: PhaseDiT, model_config: dict[str, Any], *, q_dim: int, attr_dim: int, seed: int, device: torch.device, wide_inner_dim: int) -> tuple[PhaseDiT, dict[str, int | bool]]:
    wide = build_hamiballs_d(model_config, q_dim=q_dim, attr_dim=attr_dim, seed=seed, device=device, mlp_inner_dim=wide_inner_dim)
    metadata = initialize_nested_wide_phase_dit(narrow, wide)
    return (wide, metadata)
__all__ = ['HamiBallsDUpdate', 'build_hamiballs_d', 'd_clean_and_tokens', 'matched_wide_inner_dim', 'nested_wide_from_narrow', 'parameter_count', 'update_hamiballs_d']
