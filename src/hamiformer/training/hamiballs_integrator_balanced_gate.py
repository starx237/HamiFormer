from __future__ import annotations
import math
import torch

def position_equivalent_error(candidate: torch.Tensor, target: torch.Tensor, *, attrs: torch.Tensor, state_scale: torch.Tensor, q_dim: int, frame_dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError('balanced gate states must align as [B,F,K,state]')
    state_dim = int(candidate.shape[-1])
    if q_dim <= 0 or state_dim != 2 * q_dim:
        raise ValueError('balanced gate requires equal-dimensional q and p blocks')
    if state_scale.shape != (state_dim,):
        raise ValueError('state_scale must match the canonical state width')
    if attrs.ndim != 3 or attrs.shape[:2] != (candidate.shape[0], candidate.shape[2]):
        raise ValueError('attrs must align as raw [B,K,A]')
    if attrs.shape[-1] < 1:
        raise ValueError('attrs must contain mass in channel zero')
    if not math.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError('frame_dt must be finite and positive')
    mass = attrs[..., 0]
    if not bool(torch.isfinite(mass).all()) or not bool((mass > 0.0).all()):
        raise ValueError('physical mass must be finite and positive')
    error = (candidate - target) * state_scale.to(candidate)
    q_error = error[..., :q_dim]
    mass_view = mass.to(candidate)[:, None, :, None]
    p_displacement = candidate.new_tensor(frame_dt) * error[..., q_dim:] / mass_view
    equivalent = torch.cat((q_error, p_displacement), dim=-1)
    if not bool(torch.isfinite(equivalent).all()):
        raise FloatingPointError('position-equivalent error is non-finite')
    return (q_error, p_displacement, equivalent)

__all__ = ['position_equivalent_error']
