from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
from hamiformer.models.hamiballs_committed import HamiBallsDTokenResidual
from hamiformer.physics.hamiballs_type2 import apply_hamiballs_affine_jet
from hamiformer.training import hamiballs_recovery_full_no_regret as full
from hamiformer.training.hamiballs_recovery import gradient_stats

@dataclass(frozen=True)
class OrderedResidualScales:
    q: torch.Tensor
    p: torch.Tensor

    def validate(self) -> None:
        for name, value in (('q', self.q), ('p', self.p)):
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError(f'ordered residual {name} scale must be finite scalar')
            if not float(value) > 0.0:
                raise ValueError(f'ordered residual {name} scale must be positive')

@dataclass(frozen=True)
class OrderedComponentMaps:
    total: torch.Tensor
    primary: torch.Tensor
    secondary: torch.Tensor
    fit: torch.Tensor
    protection: torch.Tensor
    priority: torch.Tensor
    excess: torch.Tensor
    ideal_energy: torch.Tensor
    error_energy: torch.Tensor

@dataclass(frozen=True)
class OrderedStreamMaps:
    q: OrderedComponentMaps
    p: OrderedComponentMaps

@dataclass(frozen=True)
class OrderedResidualLosses:
    local: OrderedStreamMaps
    mixed: OrderedStreamMaps
    q_loss: torch.Tensor
    p_loss: torch.Tensor
    primary_loss: torch.Tensor
    secondary_loss: torch.Tensor
    total: torch.Tensor
_FROZEN_SCALES: OrderedResidualScales | None = None
_SCALE_FIELD_COUNT: int | None = None

def reset_ordered_scale_state() -> None:
    global _FROZEN_SCALES, _SCALE_FIELD_COUNT
    _FROZEN_SCALES = None
    _SCALE_FIELD_COUNT = None

def frozen_ordered_scale_state() -> tuple[OrderedResidualScales | None, int | None]:
    return (_FROZEN_SCALES, _SCALE_FIELD_COUNT)

def _state_scale_column(state_scale: torch.Tensor, ndim: int) -> torch.Tensor:
    if state_scale.ndim != 1:
        raise ValueError('ordered residual requires one shared state scale vector')
    return state_scale.reshape(*[1] * (ndim - 1), -1)

def all_edge_local_h_candidate(field: Any, *, x0: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('ordered local-H field lacks rollout/tokens/jets')
    if target.ndim != 4 or x0.shape != target.shape[:1] + target.shape[2:]:
        raise ValueError('ordered local-H x0/target shapes do not align')
    frames = int(target.shape[1])
    if int(field.rollout.mixed.shape[1]) != frames:
        raise ValueError('ordered local-H target/field frame mismatch')
    previous = torch.cat((x0[:, None], target[:, :-1]), dim=1).detach()
    scale = _state_scale_column(state_scale, previous.ndim)
    physical = apply_hamiballs_affine_jet(field.jets.matrix.detach(), field.jets.offset.detach(), previous * scale, q_dim=q_dim)
    local_h = physical / scale
    if local_h.shape != target.shape or not bool(torch.isfinite(local_h).all()):
        raise FloatingPointError('ordered all-edge local H is invalid')
    return (previous, local_h)

def all_edge_local_candidates(residual: HamiBallsDTokenResidual, field: Any, *, x0: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, q_dim: int, previous_g: torch.Tensor) -> full.FullNoRegretCandidates:
    full._require_unbounded_residual(residual, q_dim=q_dim)
    previous, local_h = all_edge_local_h_candidate(field, x0=x0, target=target, state_scale=state_scale, q_dim=q_dim)
    batch, frames, objects = target.shape[:3]
    expected_previous_g = (batch, frames, objects) if residual.per_object_previous_g else (batch, frames)
    if previous_g.shape != expected_previous_g:
        raise ValueError('ordered local previous_g does not match the residual architecture')
    if bool(((previous_g < 0.0) | (previous_g > 1.0)).any()):
        raise ValueError('ordered local previous_g must lie in [0,1]')
    d_candidate = field.rollout.d_candidate.detach()
    innovation = residual(field.d_tokens.detach(), field.state.detach(), x0.detach(), previous, local_h.detach(), d_candidate, attrs.detach(), field.tau.detach(), physical_time[:, 1:1 + frames].detach(), previous_g.detach())
    hr_candidate = local_h.detach() + innovation
    if not bool(torch.isfinite(hr_candidate).all()):
        raise FloatingPointError('ordered all-edge H+r is non-finite')
    return full.FullNoRegretCandidates(h_candidate=local_h.detach(), hr_candidate=hr_candidate, d_candidate=d_candidate)

def fit_ordered_scales(local_h_candidates: list[torch.Tensor] | tuple[torch.Tensor, ...], target: torch.Tensor, *, q_dim: int) -> OrderedResidualScales:
    if not local_h_candidates:
        raise ValueError('ordered scale fit requires at least one local-H field')
    state_dim = int(target.shape[-1])
    if not 0 < q_dim < state_dim:
        raise ValueError('q_dim must split ordered residual state')
    q_rows: list[torch.Tensor] = []
    p_rows: list[torch.Tensor] = []
    for local_h in local_h_candidates:
        if local_h.shape != target.shape:
            raise ValueError('ordered scale local-H/target shapes differ')
        ideal = target.detach() - local_h.detach()
        q_rows.append(ideal[..., :q_dim].square().mean(dim=-1).sqrt().reshape(-1))
        p_rows.append(ideal[..., q_dim:].square().mean(dim=-1).sqrt().reshape(-1))
    tiny = torch.finfo(target.dtype).eps
    scales = OrderedResidualScales(q=torch.cat(q_rows).median().clamp_min(tiny), p=torch.cat(p_rows).median().clamp_min(tiny))
    scales.validate()
    return scales

def calibrate_ordered_scales(carrier: Any, *, x0: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int) -> tuple[OrderedResidualScales, int]:
    fields = tuple(carrier.trace.traces)
    local_h = [all_edge_local_h_candidate(field, x0=x0, target=target, state_scale=state_scale, q_dim=q_dim)[1] for field in fields]
    return (fit_ordered_scales(local_h, target, q_dim=q_dim), len(fields))

def _ordered_component_maps(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, component: slice, scale: torch.Tensor) -> OrderedComponentMaps:
    innovation = hr_candidate[..., component] - h_candidate[..., component]
    ideal = target[..., component].detach() - h_candidate[..., component].detach()
    dimension = int(ideal.shape[-1])
    normalized_prediction = innovation / scale
    normalized_ideal = ideal / scale
    error_energy = (normalized_prediction - normalized_ideal).square().mean(dim=-1)
    ideal_energy = normalized_ideal.square().mean(dim=-1)
    fit = torch.log1p(error_energy)
    priority = (1.0 + ideal_energy).reciprocal().detach()
    excess = error_energy - ideal_energy
    protection = priority * F.softplus(excess)
    primary = priority * (fit + F.softplus(excess))
    secondary = (1.0 - priority) * fit
    total = primary + secondary
    for value in (total, primary, secondary, fit, protection, priority, excess):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('ordered residual loss map is non-finite')
    return OrderedComponentMaps(total=total, primary=primary, secondary=secondary, fit=fit, protection=protection, priority=priority, excess=excess, ideal_energy=ideal_energy, error_energy=error_energy)

def ordered_stream_maps(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, scales: OrderedResidualScales, q_dim: int) -> OrderedStreamMaps:
    if h_candidate.shape != hr_candidate.shape or h_candidate.shape != target.shape:
        raise ValueError('ordered H/H+r/target shapes must align')
    scales.validate()
    return OrderedStreamMaps(q=_ordered_component_maps(h_candidate, hr_candidate, target, component=slice(0, q_dim), scale=scales.q), p=_ordered_component_maps(h_candidate, hr_candidate, target, component=slice(q_dim, target.shape[-1]), scale=scales.p))

def ordered_two_stream_losses(local: full.FullNoRegretCandidates, mixed: full.FullNoRegretCandidates, target: torch.Tensor, *, scales: OrderedResidualScales, q_dim: int) -> OrderedResidualLosses:
    local_maps = ordered_stream_maps(local.h_candidate, local.hr_candidate, target, scales=scales, q_dim=q_dim)
    mixed_maps = ordered_stream_maps(mixed.h_candidate, mixed.hr_candidate, target, scales=scales, q_dim=q_dim)
    q_loss = 0.5 * (local_maps.q.total.mean() + mixed_maps.q.total.mean())
    p_loss = 0.5 * (local_maps.p.total.mean() + mixed_maps.p.total.mean())
    primary = 0.25 * sum((row.primary.mean() for row in (local_maps.q, local_maps.p, mixed_maps.q, mixed_maps.p)))
    secondary = 0.25 * sum((row.secondary.mean() for row in (local_maps.q, local_maps.p, mixed_maps.q, mixed_maps.p)))
    total = 0.5 * (q_loss + p_loss)
    if not torch.isclose(total, primary + secondary, rtol=1e-05, atol=1e-07):
        raise AssertionError('ordered primary/secondary decomposition drifted')
    return OrderedResidualLosses(local=local_maps, mixed=mixed_maps, q_loss=q_loss, p_loss=p_loss, primary_loss=primary, secondary_loss=secondary, total=total)

def _map_metrics(prefix: str, row: OrderedComponentMaps) -> dict[str, float]:
    return {f'{prefix}_loss': float(row.total.detach().mean().cpu()), f'{prefix}_primary': float(row.primary.detach().mean().cpu()), f'{prefix}_secondary': float(row.secondary.detach().mean().cpu()), f'{prefix}_priority_mean': float(row.priority.detach().mean().cpu()), f'{prefix}_harm_fraction': float((row.excess.detach() > 0.0).float().mean().cpu()), f'{prefix}_ideal_rms_median': float(row.ideal_energy.detach().sqrt().median().cpu())}

def ordered_recovery_ordered_online_previous_residual_update(*, residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, carrier: Any, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    global _FROZEN_SCALES, _SCALE_FIELD_COUNT
    if getattr(carrier, 'mode', None) != 'external':
        raise ValueError('ordered r requires an external random-gate carrier')
    full._require_unbounded_residual(residual, q_dim=q_dim)
    if _FROZEN_SCALES is None:
        _FROZEN_SCALES, _SCALE_FIELD_COUNT = calibrate_ordered_scales(carrier, x0=x0, target=target, state_scale=state_scale, q_dim=q_dim)
    assert _SCALE_FIELD_COUNT is not None
    optimizer.zero_grad(set_to_none=True)
    field_index, _ = full.full_no_regret_field_indices(carrier.trace, update_index=update_index)
    field = carrier.trace.traces[field_index]
    mixed = full.replay_full_no_regret_candidates(residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12, online_detached_previous=True)
    previous_g_shape = target.shape[:3] if residual.per_object_previous_g else target.shape[:2]
    previous_g = torch.rand(previous_g_shape, device=target.device, dtype=target.dtype)
    local = all_edge_local_candidates(residual, field, x0=x0, target=target, attrs=attrs, physical_time=physical_time, state_scale=state_scale, q_dim=q_dim, previous_g=previous_g)
    losses = ordered_two_stream_losses(local, mixed, target, scales=_FROZEN_SCALES, q_dim=q_dim)
    losses.total.backward()
    stats = gradient_stats(residual)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('ordered residual lacks finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), grad_clip))
    optimizer.step()
    detail: dict[str, Any] = {'loss': float(losses.total.detach().cpu()), 'q_loss': float(losses.q_loss.detach().cpu()), 'p_loss': float(losses.p_loss.detach().cpu()), 'primary_loss': float(losses.primary_loss.detach().cpu()), 'secondary_loss': float(losses.secondary_loss.detach().cpu()), 'q_scale': float(_FROZEN_SCALES.q.detach().cpu()), 'p_scale': float(_FROZEN_SCALES.p.detach().cpu()), 'scale_fit_accepted_fields': int(_SCALE_FIELD_COUNT), 'local_previous_g_mean': float(previous_g.detach().mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip, 'gradient_policy': 'ordinary_backward_after_configured_audit_v1', 'window_previous_policy': 'online_current_r_detached_at_12_edges_v1', 'local_previous_policy': 'exact_each_edge_random_previous_g_v1'}
    for prefix, row in (('local_q', losses.local.q), ('local_p', losses.local.p), ('mixed_q', losses.mixed.q), ('mixed_p', losses.mixed.p)):
        detail.update(_map_metrics(prefix, row))
    return detail
__all__ = ['OrderedComponentMaps', 'OrderedResidualLosses', 'OrderedResidualScales', 'OrderedStreamMaps', 'all_edge_local_candidates', 'all_edge_local_h_candidate', 'calibrate_ordered_scales', 'fit_ordered_scales', 'frozen_ordered_scale_state', 'ordered_stream_maps', 'ordered_two_stream_losses', 'ordered_recovery_ordered_online_previous_residual_update', 'reset_ordered_scale_state']
