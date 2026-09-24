from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

def _cell_mask(object_mask: torch.Tensor, frames: int) -> torch.Tensor:
    if object_mask.ndim != 2:
        raise ValueError('object_mask must be [B,O]')
    return object_mask[:, None].expand(-1, frames, -1).bool()

def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(value)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1)

def _component_slices() -> tuple[slice, slice]:
    return (slice(0, 3), slice(3, 6))

def _component_scale(scales: torch.Tensor, index: int, value: torch.Tensor) -> torch.Tensor:
    if scales.shape != (2,) or not bool((scales > 0).all()):
        raise ValueError('q/p scales must be two positive values')
    return scales[index].to(value)

def local_residual_loss(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, component: slice, scale: torch.Tensor) -> torch.Tensor:
    prediction = (hr_candidate[..., component] - h_candidate[..., component]) / scale
    ideal = (target[..., component].detach() - h_candidate[..., component].detach()) / scale
    error = prediction - ideal
    error_energy = error.square().mean(-1)
    ideal_energy = ideal.square().mean(-1)
    error_rms = torch.linalg.vector_norm(error, dim=-1) / math.sqrt(error.shape[-1])
    ideal_rms = torch.linalg.vector_norm(ideal, dim=-1) / math.sqrt(ideal.shape[-1])
    return torch.log1p(error_energy) / (1.0 + ideal_energy) + F.relu(error_rms - ideal_rms)

def recovery_residual_loss(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, component: slice, scale: torch.Tensor) -> torch.Tensor:
    prediction = (hr_candidate[..., component] - h_candidate[..., component]) / scale
    ideal = (target[..., component].detach() - h_candidate[..., component].detach()) / scale
    return torch.sqrt(1.0 + (prediction - ideal).square().mean(-1)) - 1.0

def hull_residual_loss(hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor, *, component: slice, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frozen_d = d_candidate[..., component].detach()
    direction = (hr_candidate[..., component] - frozen_d) / scale
    frozen_direction = direction.detach()
    ideal = (target[..., component].detach() - frozen_d) / scale
    numerator = (ideal * frozen_direction).sum(-1)
    denominator = frozen_direction.square().sum(-1).clamp_min(torch.finfo(direction.dtype).eps)
    alpha = (numerator / denominator).clamp(0.0, 1.0).detach()
    loss = torch.sqrt(1.0 + (alpha[..., None] * direction - ideal).square().mean(-1)) - 1.0
    return (loss, alpha)

@dataclass(frozen=True)
class CommonResidualLoss:
    total: torch.Tensor
    local_q: torch.Tensor
    local_p: torch.Tensor
    recovery_q: torch.Tensor
    recovery_p: torch.Tensor
    hull_q: torch.Tensor
    hull_p: torch.Tensor

def common_residual_objective(*, local_h: torch.Tensor, local_hr: torch.Tensor, recovery_h: torch.Tensor, recovery_hr: torch.Tensor, hull_hr: torch.Tensor, hull_d: torch.Tensor, target: torch.Tensor, object_mask: torch.Tensor, qp_scales: torch.Tensor) -> CommonResidualLoss:
    tensors = (local_h, local_hr, recovery_h, recovery_hr, hull_hr, hull_d, target)
    if any((value.shape != target.shape for value in tensors)):
        raise ValueError('common-r candidates and target must align')
    mask = _cell_mask(object_mask, target.shape[1])
    local, recovery, hull = ([], [], [])
    for index, component in enumerate(_component_slices()):
        scale = _component_scale(qp_scales, index, target)
        local.append(_masked_mean(local_residual_loss(local_h, local_hr, target, component=component, scale=scale), mask))
        recovery.append(_masked_mean(recovery_residual_loss(recovery_h, recovery_hr, target, component=component, scale=scale), mask))
        hull_map, _ = hull_residual_loss(hull_hr, hull_d, target, component=component, scale=scale)
        hull.append(_masked_mean(hull_map, mask))
    total = 0.25 * (local[0] + local[1]) + 0.125 * sum((*recovery, *hull))
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('common-r objective became non-finite')
    return CommonResidualLoss(total, local[0], local[1], recovery[0], recovery[1], hull[0], hull[1])

def projection_teacher(d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if d_candidate.shape != hr_candidate.shape or d_candidate.shape != target.shape:
        raise ValueError('projection candidates and target must align')
    values = []
    for component in _component_slices():
        delta = hr_candidate[..., component].detach() - d_candidate[..., component].detach()
        error = target[..., component].detach() - d_candidate[..., component].detach()
        denominator = delta.square().sum(-1)
        d_mse = error.square().mean(-1)
        h_mse = (error - delta).square().mean(-1)
        fallback = (h_mse < d_mse).to(target.dtype)
        values.append(torch.where(denominator > torch.finfo(delta.dtype).eps, (error * delta).sum(-1) / denominator.clamp_min(torch.finfo(delta.dtype).eps), fallback).clamp(0.0, 1.0))
    return torch.stack(values, -1).detach()

def global_projection_regret(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor) -> torch.Tensor:
    if gate.ndim == 4:
        gate = gate.mean(-1)
    if gate.shape != target.shape[:-1]:
        raise ValueError('scalar gate must align with [B,F,O]')
    delta = hr_candidate - d_candidate
    target_error = target.detach() - d_candidate.detach()
    frozen_delta = delta.detach()
    denominator = frozen_delta.square().sum(-1)
    teacher = torch.where(denominator > torch.finfo(delta.dtype).eps, (target_error * frozen_delta).sum(-1) / denominator.clamp_min(torch.finfo(delta.dtype).eps), torch.zeros_like(denominator)).clamp(0.0, 1.0).detach()
    mixed = d_candidate + gate[..., None] * delta
    oracle = d_candidate + teacher[..., None] * delta
    midpoint = d_candidate + 0.5 * delta
    error = lambda value: (value - target).square().mean(-1)
    regret = (error(mixed) - error(oracle).detach()).clamp_min(0.0)
    reference = (error(midpoint).detach() - error(oracle).detach()).clamp_min(0.0)
    mask = _cell_mask(object_mask, target.shape[1]).to(regret)
    return (regret * mask).sum() / (reference * mask).sum().clamp_min(torch.finfo(regret.dtype).eps)

def _component_error(candidate: torch.Tensor, target: torch.Tensor, *, attrs: torch.Tensor, state_scale: torch.Tensor, frame_dt: float, component_index: int) -> torch.Tensor:
    component = _component_slices()[component_index]
    physical = (candidate[..., component] - target[..., component]) * state_scale[component].to(candidate)
    if component_index == 1:
        mass = attrs[:, None, :, :1].to(candidate)
        physical = float(frame_dt) * physical / mass.clamp_min(torch.finfo(candidate.dtype).eps)
    return physical

def component_global_recurrent_regret(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, attrs: torch.Tensor, state_scale: torch.Tensor, object_mask: torch.Tensor, frame_dt: float, component_index: int) -> torch.Tensor:
    d_error = _component_error(d_candidate, target, attrs=attrs, state_scale=state_scale, frame_dt=frame_dt, component_index=component_index)
    disagreement = _component_error(hr_candidate, d_candidate, attrs=attrs, state_scale=state_scale, frame_dt=frame_dt, component_index=component_index)
    quadratic = disagreement.square().mean(-1)
    linear = (d_error * disagreement).mean(-1)
    oracle = torch.where(quadratic > 0, -linear / quadratic.clamp_min(torch.finfo(quadratic.dtype).tiny), torch.zeros_like(quadratic)).clamp(0.0, 1.0).detach()
    error = lambda weight: (d_error + weight[..., None] * disagreement).square().mean(-1)
    regret = (error(gate) - error(oracle).detach()).clamp_min(0.0)
    reference = (error(torch.full_like(gate, 0.5)).detach() - error(oracle).detach()).clamp_min(0.0)
    mask = _cell_mask(object_mask, target.shape[1]).to(regret)
    return (regret * mask).sum() / (reference * mask).sum().clamp_min(torch.finfo(regret.dtype).eps)

def endpoint_regret_balanced_bce(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor, component_index: int) -> torch.Tensor:
    component = _component_slices()[component_index]
    d_error = (d_candidate[..., component].detach() - target[..., component].detach()).square().mean(-1)
    h_error = (hr_candidate[..., component].detach() - target[..., component].detach()).square().mean(-1)
    left, right = (d_error < h_error, h_error < d_error)
    mask = _cell_mask(object_mask, target.shape[1]) & (left | right)
    regret = (d_error - h_error).abs()
    eps = torch.finfo(gate.dtype).eps
    left_mass = (regret * (mask & left)).sum()
    right_mass = (regret * (mask & right)).sum()
    if bool(left_mass > eps) and bool(right_mass > eps):
        weight = torch.where(mask & left, 0.5 * regret / left_mass, torch.where(mask & right, 0.5 * regret / right_mass, torch.zeros_like(regret)))
    else:
        mass = (regret * mask).sum()
        weight = torch.where(mask, regret / mass.clamp_min(eps), torch.zeros_like(regret))
    return (weight * F.binary_cross_entropy(gate.clamp(eps, 1.0 - eps), right.to(gate.dtype), reduction='none')).sum()

def source_object_no_harm(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor, component_index: int) -> torch.Tensor:
    component = _component_slices()[component_index]
    delta = hr_candidate[..., component] - d_candidate[..., component]
    mixed = d_candidate[..., component] + gate[..., None] * delta
    target_component = target[..., component]
    d_point = (d_candidate[..., component] - target_component).square().mean(-1)
    mixed_point = (mixed - target_component).square().mean(-1)
    hr_point = (hr_candidate[..., component] - target_component).square().mean(-1)
    harm = (mixed_point - d_point.detach()).clamp_min(0.0).mean(1)
    endpoint = (hr_point.detach() - d_point.detach()).clamp_min(0.0).mean(1)
    ratio = torch.where(endpoint > torch.finfo(endpoint.dtype).eps, harm / endpoint.clamp_min(torch.finfo(endpoint.dtype).eps), torch.zeros_like(harm))
    return _masked_mean(ratio, object_mask)

def source_object_upper_semideviation(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor, component_index: int) -> torch.Tensor:
    component = _component_slices()[component_index]
    delta = hr_candidate[..., component] - d_candidate[..., component]
    mixed = d_candidate[..., component] + gate[..., None] * delta
    target_component = target[..., component]
    d_point = (d_candidate[..., component] - target_component).square().mean(-1)
    mixed_point = (mixed - target_component).square().mean(-1)
    hr_point = (hr_candidate[..., component] - target_component).square().mean(-1)
    harm = (mixed_point - d_point.detach()).mean(1).clamp_min(0.0)
    endpoint = (hr_point.detach() - d_point.detach()).mean(1).clamp_min(0.0)
    valid = object_mask.bool()
    numerator = torch.linalg.vector_norm(harm[valid]) / math.sqrt(max(int(valid.sum()), 1))
    denominator = (torch.linalg.vector_norm(endpoint[valid]) / math.sqrt(max(int(valid.sum()), 1))).detach()
    return torch.where(denominator > torch.finfo(denominator.dtype).eps, numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps), numerator * 0.0)

@dataclass(frozen=True)
class RouterLoss:
    total: torch.Tensor
    q: torch.Tensor
    p: torch.Tensor
    terms: dict[str, torch.Tensor]

def scalar0_objective(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor) -> torch.Tensor:
    return global_projection_regret(gate, d_candidate, hr_candidate, target, object_mask=object_mask)

def scalar1_objective(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, object_mask: torch.Tensor, no_harm_weight: float=0.25) -> RouterLoss:
    scalar = gate.mean(-1) if gate.ndim == 4 else gate
    recurrent = global_projection_regret(scalar, d_candidate, hr_candidate, target, object_mask=object_mask)
    no_harm = [source_object_no_harm(scalar, d_candidate, hr_candidate, target, object_mask=object_mask, component_index=index) for index in range(2)]
    q = 0.5 * recurrent + float(no_harm_weight) * no_harm[0]
    p = 0.5 * recurrent + float(no_harm_weight) * no_harm[1]
    return RouterLoss(0.5 * (q + p), q, p, {'global_recurrent': recurrent, 'q_no_harm': no_harm[0], 'p_no_harm': no_harm[1]})

def final_qp_objective(gate: torch.Tensor, d_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, attrs: torch.Tensor, state_scale: torch.Tensor, object_mask: torch.Tensor, frame_dt: float, no_harm_weight: float=0.25) -> RouterLoss:
    if gate.shape != (*target.shape[:-1], 2):
        raise ValueError('final q/p gate must be [B,F,O,2]')
    local, recurrent, tail, no_harm = ([], [], [], [])
    for index in range(2):
        local.append(endpoint_regret_balanced_bce(gate[..., index], d_candidate, hr_candidate, target, object_mask=object_mask, component_index=index))
        recurrent.append(component_global_recurrent_regret(gate[..., index], d_candidate, hr_candidate, target, attrs=attrs, state_scale=state_scale, object_mask=object_mask, frame_dt=frame_dt, component_index=index))
        tail.append(source_object_upper_semideviation(gate[..., index], d_candidate, hr_candidate, target, object_mask=object_mask, component_index=index))
        no_harm.append(source_object_no_harm(gate[..., index], d_candidate, hr_candidate, target, object_mask=object_mask, component_index=index))
    q = 0.5 * (local[0] + recurrent[0]) + float(no_harm_weight) * no_harm[0]
    p = (local[1] + recurrent[1] + tail[1]) / 3.0 + float(no_harm_weight) * no_harm[1]
    terms = {'q_endpoint': local[0], 'p_endpoint': local[1], 'q_recurrent': recurrent[0], 'p_recurrent': recurrent[1], 'p_upper_semideviation': tail[1], 'q_no_harm': no_harm[0], 'p_no_harm': no_harm[1]}
    total = 0.5 * (q + p)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('final q/p objective became non-finite')
    return RouterLoss(total, q, p, terms)
__all__ = ['CommonResidualLoss', 'RouterLoss', 'common_residual_objective', 'endpoint_regret_balanced_bce', 'final_qp_objective', 'global_projection_regret', 'hull_residual_loss', 'local_residual_loss', 'projection_teacher', 'recovery_residual_loss', 'scalar0_objective', 'scalar1_objective', 'source_object_no_harm', 'source_object_upper_semideviation']
