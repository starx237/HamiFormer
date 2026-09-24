from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from .rectified_flow import clean_to_velocity
CleanField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass(frozen=True)
class PFRFV1Pair:
    clean: torch.Tensor
    noise: torch.Tensor
    noisy: torch.Tensor
    tau: torch.Tensor
    target_velocity: torch.Tensor

@dataclass(frozen=True)
class PFRFV1Sample:
    trajectory: torch.Tensor
    nfe: int

def validate_pf_rf_v1_steps(*, num_steps: int, t_eps: float) -> None:
    if type(num_steps) is not int or num_steps < 2:
        raise ValueError('num_steps must be an integer >= 2')
    if not 0.0 < t_eps <= 1.0:
        raise ValueError('t_eps must lie in (0, 1]')
    final_remaining = 1.0 / float(num_steps)
    if final_remaining + 1e-12 < t_eps:
        raise ValueError('PF-RF-v1 final Euler interval is shorter than t_eps; lower t_eps or use fewer sampling steps')

def sample_pf_rf_v1_tau(batch_size: int, *, mean: float, std: float, device: torch.device | str, dtype: torch.dtype, generator: torch.Generator | None=None) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not torch.isfinite(torch.tensor((mean, std))).all() or std <= 0.0:
        raise ValueError('mean must be finite and std must be positive')
    normal = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    tau = torch.sigmoid(normal * std + mean)
    lower = torch.nextafter(torch.zeros_like(tau), torch.ones_like(tau))
    upper = torch.nextafter(torch.ones_like(tau), torch.zeros_like(tau))
    return tau.clamp(min=lower, max=upper)

def make_pf_rf_v1_pair(clean: torch.Tensor, tau: torch.Tensor, *, noise_scale: float, t_eps: float, generator: torch.Generator | None=None) -> PFRFV1Pair:
    if clean.ndim < 2:
        raise ValueError('clean must have a batch plus at least one data dimension')
    if tau.shape != (clean.shape[0],):
        raise ValueError('tau must have shape [batch]')
    if not 0.0 < noise_scale:
        raise ValueError('noise_scale must be positive')
    if not bool(torch.isfinite(clean).all().item()):
        raise FloatingPointError('clean contains NaN/Inf')
    noise = noise_scale * torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    tau_view = tau.reshape(tau.shape[0], *[1] * (clean.ndim - 1))
    noisy = tau_view * clean + (1.0 - tau_view) * noise
    target_velocity = clean_to_velocity(clean, noisy, tau, t_eps=t_eps)
    return PFRFV1Pair(clean=clean, noise=noise, noisy=noisy, tau=tau, target_velocity=target_velocity)

@torch.no_grad()
def sample_pf_rf_v1_heun(field: CleanField, source: torch.Tensor, *, num_steps: int, t_eps: float) -> PFRFV1Sample:
    if source.ndim < 2:
        raise ValueError('source must have a batch plus at least one data dimension')
    if not bool(torch.isfinite(source).all().item()):
        raise FloatingPointError('source contains NaN/Inf')
    validate_pf_rf_v1_steps(num_steps=num_steps, t_eps=t_eps)
    batch = source.shape[0]
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=source.device, dtype=source.dtype)
    state = source.clone()
    nfe = 0
    for index in range(num_steps - 1):
        left, right = (grid[index], grid[index + 1])
        tau_left = left.expand(batch)
        clean_left = field(state, tau_left)
        if clean_left.shape != state.shape:
            raise ValueError('field clean prediction shape mismatch')
        velocity_left = clean_to_velocity(clean_left, state, tau_left, t_eps=t_eps)
        proposal = state + (right - left) * velocity_left
        tau_right = right.expand(batch)
        clean_right = field(proposal, tau_right)
        if clean_right.shape != state.shape:
            raise ValueError('field clean prediction shape mismatch')
        velocity_right = clean_to_velocity(clean_right, proposal, tau_right, t_eps=t_eps)
        state = state + 0.5 * (right - left) * (velocity_left + velocity_right)
        nfe += 2
    left, right = (grid[-2], grid[-1])
    tau_left = left.expand(batch)
    clean_left = field(state, tau_left)
    if clean_left.shape != state.shape:
        raise ValueError('field clean prediction shape mismatch')
    velocity_left = clean_to_velocity(clean_left, state, tau_left, t_eps=t_eps)
    state = state + (right - left) * velocity_left
    nfe += 1
    if not bool(torch.isfinite(state).all().item()):
        raise FloatingPointError('PF-RF-v1 sampler produced NaN/Inf')
    return PFRFV1Sample(trajectory=state, nfe=nfe)
__all__ = ['CleanField', 'PFRFV1Pair', 'PFRFV1Sample', 'make_pf_rf_v1_pair', 'sample_pf_rf_v1_heun', 'sample_pf_rf_v1_tau', 'validate_pf_rf_v1_steps']
