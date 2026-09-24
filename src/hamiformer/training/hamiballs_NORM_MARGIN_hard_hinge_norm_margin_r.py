from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F
from hamiformer.training import hamiballs_recovery_full_no_regret as full
from hamiformer.training.hamiballs_recovery import gradient_stats
from hamiformer.training import hamiballs_ordered_recovery_ordered_r as ordered
from hamiformer.training.hamiballs_ordered_recovery_ordered_recovery_r import recovery_pseudohuber_loss

def local_hard_hinge_norm_margin_loss(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, scale: torch.Tensor, component: slice) -> torch.Tensor:
    prediction = (hr_candidate[..., component] - h_candidate[..., component]) / scale
    ideal = (target[..., component].detach() - h_candidate[..., component].detach()) / scale
    error = prediction - ideal
    dimension = int(error.shape[-1])
    error_energy = error.square().mean(dim=-1)
    ideal_energy = ideal.square().mean(dim=-1)
    error_rms = torch.linalg.vector_norm(error, dim=-1) / math.sqrt(dimension)
    ideal_rms = torch.linalg.vector_norm(ideal, dim=-1) / math.sqrt(dimension)
    priority = (1.0 + ideal_energy).reciprocal().detach()
    fit = priority * torch.log1p(error_energy)
    margin = F.relu(error_rms - ideal_rms)
    loss = fit + margin
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError('NORM_MARGIN local hard-hinge norm-margin loss is non-finite')
    return loss

def NORM_MARGIN_hard_hinge_norm_margin_residual_update(*, residual: torch.nn.Module, optimizer: torch.optim.Optimizer, carrier: Any, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    if getattr(carrier, 'mode', None) != 'external':
        raise ValueError('NORM_MARGIN r requires an external random-gate carrier')
    full._require_unbounded_residual(residual, q_dim=q_dim)
    if ordered._FROZEN_SCALES is None:
        ordered._FROZEN_SCALES, ordered._SCALE_FIELD_COUNT = ordered.calibrate_ordered_scales(carrier, x0=x0, target=target, state_scale=state_scale, q_dim=q_dim)
    scales = ordered._FROZEN_SCALES
    scale_fields = ordered._SCALE_FIELD_COUNT
    assert scales is not None and scale_fields is not None
    optimizer.zero_grad(set_to_none=True)
    field_index, _ = full.full_no_regret_field_indices(carrier.trace, update_index=update_index)
    field = carrier.trace.traces[field_index]
    mixed = full.replay_full_no_regret_candidates(residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12, online_detached_previous=True)
    previous_g_shape = target.shape[:3] if residual.per_object_previous_g else target.shape[:2]
    previous_g = torch.rand(previous_g_shape, device=target.device, dtype=target.dtype)
    local = ordered.all_edge_local_candidates(residual, field, x0=x0, target=target, attrs=attrs, physical_time=physical_time, state_scale=state_scale, q_dim=q_dim, previous_g=previous_g)
    local_q = local_hard_hinge_norm_margin_loss(local.h_candidate, local.hr_candidate, target, scale=scales.q, component=slice(0, q_dim)).mean()
    local_p = local_hard_hinge_norm_margin_loss(local.h_candidate, local.hr_candidate, target, scale=scales.p, component=slice(q_dim, target.shape[-1])).mean()
    recovery_q = recovery_pseudohuber_loss(mixed.h_candidate, mixed.hr_candidate, target, scale=scales.q, component=slice(0, q_dim)).mean()
    recovery_p = recovery_pseudohuber_loss(mixed.h_candidate, mixed.hr_candidate, target, scale=scales.p, component=slice(q_dim, target.shape[-1])).mean()
    loss = 0.25 * (local_q + local_p + recovery_q + recovery_p)
    loss.backward()
    stats = gradient_stats(residual)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('NORM_MARGIN residual lacks finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(loss.detach().cpu()), 'local_q_loss': float(local_q.detach().cpu()), 'local_p_loss': float(local_p.detach().cpu()), 'recovery_q_loss': float(recovery_q.detach().cpu()), 'recovery_p_loss': float(recovery_p.detach().cpu()), 'q_scale': float(scales.q.detach().cpu()), 'p_scale': float(scales.p.detach().cpu()), 'scale_fit_accepted_fields': int(scale_fields), 'local_previous_g_mean': float(previous_g.detach().mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip, 'gradient_policy': 'ordinary_backward_NORM_MARGIN_hard_hinge_norm_margin_v1', 'window_previous_policy': 'online_current_r_detached_at_12_edges_v1', 'local_previous_policy': 'exact_each_edge_random_previous_g_v1', 'local_loss': 'dimensionless_priority_fit_plus_hard_vector_rms_no_harm_margin_v1', 'mixed_recovery_loss': 'vector_pseudohuber_dimensionless_equal_qp_v1'}
__all__ = ['NORM_MARGIN_hard_hinge_norm_margin_residual_update', 'local_hard_hinge_norm_margin_loss']
