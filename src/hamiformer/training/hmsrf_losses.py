from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from hamiformer.types import FieldOutput

@dataclass
class HMSRFLossOutput:
    total: torch.Tensor
    rf: torch.Tensor
    continuity: torch.Tensor
    dynamics: torch.Tensor
    midpoint_solver_residual: torch.Tensor

def _normalized_masked_mse(residual: torch.Tensor, state_scale: torch.Tensor, *, keep_mask: torch.Tensor | None=None) -> torch.Tensor:
    if state_scale.shape != (residual.shape[-1],):
        raise ValueError('state_scale 必须匹配 residual 最后一维')
    normalized = residual / state_scale.to(residual).view(*[1] * (residual.ndim - 1), -1)
    squared = normalized.square()
    if keep_mask is None:
        return squared.mean()
    if keep_mask.shape != residual.shape or keep_mask.dtype != torch.bool:
        raise ValueError('keep_mask 必须是与 residual 同形的 bool tensor')
    weight = keep_mask.to(dtype=squared.dtype)
    return (squared * weight).sum() / weight.sum().clamp_min(1.0)

def _trajectory_dynamics_loss(clean_full: torch.Tensor, physical_time: torch.Tensor, theta: torch.Tensor, *, vector_field: nn.Module, state_scale: torch.Tensor) -> torch.Tensor:
    if clean_full.ndim != 4:
        raise ValueError('clean_full 必须为 [B,F+1,K,state_dim]')
    batch, frames, objects, state_dim = clean_full.shape
    theta_dim = int(getattr(vector_field, 'theta_dim', -1))
    if theta.shape != (batch, objects, theta_dim):
        raise ValueError('theta shape 与 vector field 不一致')
    if physical_time.shape != (batch, frames):
        raise ValueError('physical_time 必须为 [B,F+1]')
    delta = physical_time[:, 1:] - physical_time[:, :-1]
    if not bool((delta > 0).all().item()):
        raise ValueError('physical_time 必须严格递增')
    start = clean_full[:, :-1]
    end = clean_full[:, 1:]
    midpoint = 0.5 * (start + end)
    flat_midpoint = midpoint.reshape(batch * (frames - 1), objects, state_dim)
    flat_theta = theta[:, None].expand(batch, frames - 1, objects, theta_dim)
    flat_theta = flat_theta.reshape(batch * (frames - 1), objects, theta_dim)
    field = vector_field(flat_midpoint, flat_theta).reshape_as(midpoint)
    secant = (end - start) / delta[:, :, None, None]
    return _normalized_masked_mse(secant - field, state_scale)

def hmsrf_loss(output: FieldOutput, target_velocity: torch.Tensor, *, state_scale: torch.Tensor, known_mask: torch.Tensor | None=None, clean_full: torch.Tensor | None=None, physical_time: torch.Tensor | None=None, theta: torch.Tensor | None=None, vector_field: nn.Module | None=None, continuity_weight: float=1.0, dynamics_weight: float=1.0) -> HMSRFLossOutput:
    if output.velocity.shape != target_velocity.shape:
        raise ValueError('output.velocity/target_velocity shape 不一致')
    if continuity_weight < 0.0 or dynamics_weight < 0.0:
        raise ValueError('loss weights 不能为负')
    unknown_mask = None if known_mask is None else ~known_mask
    rf = _normalized_masked_mse(output.velocity - target_velocity, state_scale, keep_mask=unknown_mask)
    if output.diagnostics is None:
        raise ValueError('HMS-RF output 必须提供 diagnostics')
    continuity_gaps = output.diagnostics.get('continuity_gaps')
    midpoint_residuals = output.diagnostics.get('midpoint_residuals')
    if not isinstance(continuity_gaps, torch.Tensor) or not isinstance(midpoint_residuals, torch.Tensor):
        raise ValueError('diagnostics 缺少 continuity/midpoint tensors')
    if continuity_gaps.numel() == 0:
        continuity = rf.new_zeros(())
    else:
        continuity = _normalized_masked_mse(continuity_gaps, state_scale)
    midpoint_solver_residual = _normalized_masked_mse(midpoint_residuals, state_scale)
    if dynamics_weight == 0.0:
        dynamics = rf.new_zeros(())
    else:
        if clean_full is None or physical_time is None or theta is None or (vector_field is None):
            raise ValueError('非零 dynamics_weight 必须提供 clean/time/theta/vector_field')
        dynamics = _trajectory_dynamics_loss(clean_full, physical_time, theta, vector_field=vector_field, state_scale=state_scale)
    total = rf + continuity_weight * continuity + dynamics_weight * dynamics
    return HMSRFLossOutput(total=total, rf=rf, continuity=continuity, dynamics=dynamics, midpoint_solver_residual=midpoint_solver_residual)
