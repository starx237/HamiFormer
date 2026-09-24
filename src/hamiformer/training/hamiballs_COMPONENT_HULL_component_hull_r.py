from __future__ import annotations
from typing import Any
import torch
from hamiformer.training import hamiballs_recovery_full_no_regret as full
from hamiformer.training import hamiballs_ordered_recovery_ordered_r as ordered
from hamiformer.training.hamiballs_NORM_MARGIN_hard_hinge_norm_margin_r import local_hard_hinge_norm_margin_loss
from hamiformer.training.hamiballs_recovery import gradient_stats
from hamiformer.training.hamiballs_ordered_recovery_ordered_recovery_r import recovery_pseudohuber_loss

def component_hull_pseudohuber_loss(hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor, *, scale: torch.Tensor, component: slice) -> tuple[torch.Tensor, torch.Tensor]:
    frozen_d = d_candidate[..., component].detach()
    frozen_target = target[..., component].detach()
    direction = (hr_candidate[..., component] - frozen_d) / scale
    frozen_direction = direction.detach()
    ideal = (frozen_target - frozen_d) / scale
    numerator = (ideal * frozen_direction).sum(dim=-1)
    denominator = frozen_direction.square().sum(dim=-1).clamp_min(torch.finfo(hr_candidate.dtype).eps)
    g_star = (numerator / denominator).clamp(0.0, 1.0).detach()
    error_energy = (g_star[..., None] * direction - ideal).square().mean(dim=-1)
    loss = torch.sqrt(1.0 + error_energy) - 1.0
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError('COMPONENT_HULL component hull loss is non-finite')
    return (loss, g_star)

def COMPONENT_HULL_component_hull_residual_update(*, residual: torch.nn.Module, optimizer: torch.optim.Optimizer, carrier: Any, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    if getattr(carrier, 'mode', None) != 'external':
        raise ValueError('COMPONENT_HULL r requires an external random-gate carrier')
    full._require_unbounded_residual(residual, q_dim=q_dim)
    if ordered._FROZEN_SCALES is None:
        ordered._FROZEN_SCALES, ordered._SCALE_FIELD_COUNT = ordered.calibrate_ordered_scales(carrier, x0=x0, target=target, state_scale=state_scale, q_dim=q_dim)
    scales = ordered._FROZEN_SCALES
    scale_fields = ordered._SCALE_FIELD_COUNT
    assert scales is not None and scale_fields is not None
    optimizer.zero_grad(set_to_none=True)
    field_index, hull_index = full.full_no_regret_field_indices(carrier.trace, update_index=update_index)
    field = carrier.trace.traces[field_index]
    mixed = full.replay_full_no_regret_candidates(residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12, online_detached_previous=True)
    hull = full.replay_full_no_regret_candidates(residual, carrier.trace, field_index=hull_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12, online_detached_previous=True)
    previous_g_shape = target.shape[:3] if residual.per_object_previous_g else target.shape[:2]
    previous_g = torch.rand(previous_g_shape, device=target.device, dtype=target.dtype)
    local = ordered.all_edge_local_candidates(residual, field, x0=x0, target=target, attrs=attrs, physical_time=physical_time, state_scale=state_scale, q_dim=q_dim, previous_g=previous_g)
    local_q = local_hard_hinge_norm_margin_loss(local.h_candidate, local.hr_candidate, target, scale=scales.q, component=slice(0, q_dim)).mean()
    local_p = local_hard_hinge_norm_margin_loss(local.h_candidate, local.hr_candidate, target, scale=scales.p, component=slice(q_dim, target.shape[-1])).mean()
    recovery_q = recovery_pseudohuber_loss(mixed.h_candidate, mixed.hr_candidate, target, scale=scales.q, component=slice(0, q_dim)).mean()
    recovery_p = recovery_pseudohuber_loss(mixed.h_candidate, mixed.hr_candidate, target, scale=scales.p, component=slice(q_dim, target.shape[-1])).mean()
    hull_q_map, hull_g_q = component_hull_pseudohuber_loss(hull.hr_candidate, hull.d_candidate, target, scale=scales.q, component=slice(0, q_dim))
    hull_p_map, hull_g_p = component_hull_pseudohuber_loss(hull.hr_candidate, hull.d_candidate, target, scale=scales.p, component=slice(q_dim, target.shape[-1]))
    hull_q, hull_p = (hull_q_map.mean(), hull_p_map.mean())
    loss = 0.25 * (local_q + local_p) + 0.125 * (recovery_q + recovery_p + hull_q + hull_p)
    loss.backward()
    stats = gradient_stats(residual)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('COMPONENT_HULL residual lacks finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(loss.detach().cpu()), 'local_q_loss': float(local_q.detach().cpu()), 'local_p_loss': float(local_p.detach().cpu()), 'recovery_q_loss': float(recovery_q.detach().cpu()), 'recovery_p_loss': float(recovery_p.detach().cpu()), 'hull_q_loss': float(hull_q.detach().cpu()), 'hull_p_loss': float(hull_p.detach().cpu()), 'hull_g_q_mean': float(hull_g_q.mean().detach().cpu()), 'hull_g_p_mean': float(hull_g_p.mean().detach().cpu()), 'hull_field_index': int(hull_index), 'q_scale': float(scales.q.detach().cpu()), 'p_scale': float(scales.p.detach().cpu()), 'scale_fit_accepted_fields': int(scale_fields), 'local_previous_g_mean': float(previous_g.detach().mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip, 'gradient_policy': 'ordinary_backward_COMPONENT_HULL_component_hull_v1', 'window_previous_policy': 'online_current_r_detached_at_12_edges_v1', 'local_previous_policy': 'exact_each_edge_random_previous_g_v1', 'local_loss': 'NORM_MARGIN_hard_vector_rms_margin_v1', 'mixed_recovery_loss': 'half_direct_half_fixed_scale_component_hull_pseudohuber_v1'}
__all__ = ['COMPONENT_HULL_component_hull_residual_update', 'component_hull_pseudohuber_loss']
