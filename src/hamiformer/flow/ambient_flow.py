from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import torch
from .rectified_flow import clean_to_velocity

class AmbientCleanPredictor(Protocol):

    def __call__(self, noisy_trajectory: torch.Tensor, tau: torch.Tensor, *, condition: Any, physical_time: torch.Tensor) -> torch.Tensor:
        ...

@dataclass
class AmbientTrajectorySample:
    trajectory: torch.Tensor
    source: torch.Tensor
    nfe: int

def sample_ambient_trajectory_heun(model: AmbientCleanPredictor, condition: Any, *, batch_size: int, frames: int, objects: int, state_scale: torch.Tensor, physical_time: torch.Tensor, num_sampling_steps: int, noise_scale: float, t_eps: float, device: torch.device | str, dtype: torch.dtype, source: torch.Tensor | None=None, observed_values: torch.Tensor | None=None, observed_mask: torch.Tensor | None=None) -> AmbientTrajectorySample:
    if min(batch_size, frames, objects, num_sampling_steps) <= 0:
        raise ValueError('batch/frames/objects/num_sampling_steps 必须为正')
    if state_scale.ndim != 1 or not bool((state_scale > 0).all().item()):
        raise ValueError('state_scale 必须是一维有限正数')
    state_dim = state_scale.numel()
    expected = (batch_size, frames, objects, state_dim)
    if physical_time.shape != (batch_size, frames):
        raise ValueError('physical_time 必须为 [B,T]')
    if noise_scale <= 0.0 or not 0.0 < t_eps <= 1.0:
        raise ValueError('noise_scale 必须为正，t_eps 必须位于 (0,1]')
    if source is None:
        scale = state_scale.to(device=device, dtype=dtype).view(1, 1, 1, state_dim)
        source = noise_scale * scale * torch.randn(expected, device=device, dtype=dtype)
    elif source.shape != expected:
        raise ValueError('source shape 必须为 [B,T,K,state_dim]')
    else:
        source = source.to(device=device, dtype=dtype)
    if (observed_values is None) != (observed_mask is None):
        raise ValueError('observed_values 与 observed_mask 必须同时提供或同时省略')
    if observed_values is not None and observed_mask is not None:
        if observed_values.shape != expected or observed_mask.shape != expected:
            raise ValueError('observed_values/mask 必须与 trajectory shape 一致')
        observed_values = observed_values.to(device=device, dtype=dtype)
        observed_mask = observed_mask.to(device=device, dtype=torch.bool)

    def clamp_observed(state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if observed_values is None or observed_mask is None:
            return state
        bridge = tau * observed_values + (1.0 - tau) * source
        return torch.where(observed_mask, bridge, state)
    grid = torch.linspace(0.0, 1.0, num_sampling_steps + 1, device=device, dtype=dtype)
    trajectory = clamp_observed(source, grid[0])
    nfe = 0
    for index in range(num_sampling_steps - 1):
        left = grid[index]
        right = grid[index + 1]
        step = right - left
        trajectory = clamp_observed(trajectory, left)
        tau_left = left.expand(batch_size)
        clean_left = model(trajectory, tau_left, condition=condition, physical_time=physical_time)
        velocity_left = clean_to_velocity(clean_left, trajectory, tau_left, t_eps=t_eps)
        proposal = trajectory + step * velocity_left
        proposal = clamp_observed(proposal, right)
        tau_right = right.expand(batch_size)
        clean_right = model(proposal, tau_right, condition=condition, physical_time=physical_time)
        velocity_right = clean_to_velocity(clean_right, proposal, tau_right, t_eps=t_eps)
        trajectory = trajectory + 0.5 * step * (velocity_left + velocity_right)
        trajectory = clamp_observed(trajectory, right)
        nfe += 2
    left = grid[-2]
    right = grid[-1]
    trajectory = clamp_observed(trajectory, left)
    tau_left = left.expand(batch_size)
    clean_left = model(trajectory, tau_left, condition=condition, physical_time=physical_time)
    velocity_left = clean_to_velocity(clean_left, trajectory, tau_left, t_eps=t_eps)
    trajectory = trajectory + (right - left) * velocity_left
    trajectory = clamp_observed(trajectory, right)
    nfe += 1
    return AmbientTrajectorySample(trajectory=trajectory, source=source, nfe=nfe)
