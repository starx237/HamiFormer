from __future__ import annotations
from typing import Literal
import torch
from hamiformer.types import PhaseObservation
ObservationPattern = Literal['noisy_prefix', 'partial_prefix', 'sparse', 'full_initial']

def _validate_inputs(phase: torch.Tensor, attrs: torch.Tensor, time: torch.Tensor, state_scale: torch.Tensor, q_dim: int) -> None:
    if phase.ndim != 4:
        raise ValueError('phase 必须为 [B,T,K,state_dim]')
    batch, frames, objects, state_dim = phase.shape
    if state_dim != 2 * q_dim:
        raise ValueError('phase 最后一维必须为 2*q_dim')
    if attrs.ndim != 3 or attrs.shape[:2] != (batch, objects):
        raise ValueError('attrs 必须为 [B,K,attr_dim]')
    if time.shape != (batch, frames):
        raise ValueError('time 必须为 [B,T]')
    if state_scale.shape != (state_dim,):
        raise ValueError('state_scale 必须为 [state_dim]')
    if not torch.isfinite(phase).all() or not torch.isfinite(attrs).all():
        raise ValueError('phase/attrs 含 NaN 或 Inf')
    if not bool((time[:, 1:] > time[:, :-1]).all().item()):
        raise ValueError('物理时间必须严格递增')

def make_phase_observation(phase: torch.Tensor, attrs: torch.Tensor, time: torch.Tensor, state_scale: torch.Tensor, *, q_dim: int, pattern: ObservationPattern, prefix_frames: int=8, sparse_probability: float=0.15, sensor_noise_scale: float=0.01, hide_mass: bool=True, generator: torch.Generator | None=None) -> PhaseObservation:
    _validate_inputs(phase, attrs, time, state_scale, q_dim)
    batch, frames, objects, state_dim = phase.shape
    if prefix_frames <= 0:
        raise ValueError('prefix_frames 必须为正')
    if not 0.0 < sparse_probability <= 1.0:
        raise ValueError('sparse_probability 必须位于 (0,1]')
    if sensor_noise_scale < 0.0:
        raise ValueError('sensor_noise_scale 不能为负')
    phase_mask = torch.zeros_like(phase, dtype=torch.bool)
    if pattern == 'noisy_prefix':
        observed_frames = min(prefix_frames, frames)
        phase_mask[:, :observed_frames] = True
    elif pattern == 'partial_prefix':
        observed_frames = min(prefix_frames, frames)
        phase_mask[:, :observed_frames, :, :q_dim] = True
    elif pattern == 'sparse':
        random_mask = torch.rand(phase.shape, device=phase.device, generator=generator)
        phase_mask = random_mask < sparse_probability
        phase_mask[:, 0, :, :q_dim] = True
    elif pattern == 'full_initial':
        phase_mask[:, 0] = True
    else:
        raise ValueError(f'未知 observation pattern: {pattern}')
    state_scale_view = state_scale.to(phase).view(1, 1, 1, state_dim)
    sensor_noise = torch.randn(phase.shape, device=phase.device, dtype=phase.dtype, generator=generator)
    noisy_phase = phase + sensor_noise_scale * state_scale_view * sensor_noise
    observed_phase = torch.where(phase_mask, noisy_phase, torch.zeros_like(noisy_phase))
    attr_mask = torch.ones_like(attrs, dtype=torch.bool)
    if hide_mass:
        if attrs.shape[-1] < 1:
            raise ValueError('attrs 至少需要质量通道')
        attr_mask[..., 0] = False
    observed_attrs = torch.where(attr_mask, attrs, torch.zeros_like(attrs))
    return PhaseObservation(phase=observed_phase, phase_mask=phase_mask, attrs=observed_attrs, attr_mask=attr_mask, time=time)
