from __future__ import annotations
from typing import Protocol
import torch

class CanonicalVectorField(Protocol):

    def vector_field(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        ...

def midpoint_secant_loss(field: CanonicalVectorField, trajectory: torch.Tensor, theta: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    if trajectory.ndim != 4 or trajectory.shape[1] < 2:
        raise ValueError('trajectory 必须为 [B,T>=2,K,state_dim]')
    batch, frames, objects, state_dim = trajectory.shape
    if theta.ndim != 3 or theta.shape[:2] != (batch, objects):
        raise ValueError('theta 必须为 [B,K,theta_dim]')
    if physical_time.shape != (batch, frames):
        raise ValueError('physical_time 必须为 [B,T]')
    if state_scale.shape != (state_dim,) or not bool(torch.isfinite(state_scale).all().item()):
        raise ValueError('state_scale 必须为 [state_dim] 有限值')
    if not bool((state_scale > 0).all().item()):
        raise ValueError('state_scale 必须严格为正')
    delta = physical_time[:, 1:] - physical_time[:, :-1]
    if not bool((delta > 0).all().item()):
        raise ValueError('physical_time 必须严格递增')
    start = trajectory[:, :-1]
    end = trajectory[:, 1:]
    midpoint = 0.5 * (start + end)
    theta_time = theta[:, None].expand(batch, frames - 1, objects, theta.shape[-1])
    prediction = field.vector_field(midpoint.reshape(batch * (frames - 1), objects, state_dim), theta_time.reshape(batch * (frames - 1), objects, theta.shape[-1])).reshape_as(midpoint)
    target = (end - start) / delta[:, :, None, None]
    scale = state_scale.to(prediction).view(1, 1, 1, state_dim)
    return ((prediction.float() - target.float()) / scale.float()).square().mean()
