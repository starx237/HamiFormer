from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
import torch
from torch import nn
from hamiformer.evaluation.hamiballs_formal import HamiBallsExpertFieldTrace, HamiBallsChunkSample, sample_hamiballs_expert_chunk
from hamiformer.models import HamiBallsCommittedGate, HamiBallsCompactCommittedGate, HamiBallsDTokenResidual, HamiBallsPerObjectCompactCommittedGate, rollout_hamiballs_committed_edges
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, apply_hamiballs_affine_jet
from hamiformer.physics.generic_type2 import TokenConditionalTypeIIGenerator
from hamiformer.training.hamiballs_formal import previous_gate_sequence

@dataclass(frozen=True)
class HamiBallsTrajectoryG0:
    traces: tuple[HamiBallsExpertFieldTrace, ...]
    pure_rows: int
    reset_edges: int

def residual_no_harm_loss(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, margin: float=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if h_candidate.shape != hr_candidate.shape or h_candidate.shape != target.shape:
        raise ValueError('H, H+r and target must align')
    if h_candidate.ndim < 3:
        raise ValueError('residual no-harm inputs must include batch/edge/value axes')
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError('no-harm margin must be finite and nonnegative')
    axes = tuple(range(2, h_candidate.ndim))
    h_mse = (h_candidate - target).square().mean(dim=axes)
    hr_mse = (hr_candidate - target).square().mean(dim=axes)
    excess = hr_mse - h_mse
    penalty = torch.nn.functional.relu(excess + h_candidate.new_tensor(margin)).mean()
    return (penalty, excess)

def robust_per_object_pseudo_huber_loss(hr_candidate: torch.Tensor, target: torch.Tensor, *, eps: float=1e-08) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hr_candidate.shape != target.shape or hr_candidate.ndim < 4:
        raise ValueError('robust residual loss requires aligned [B,F,K,state] candidates')
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError('robust residual eps must be finite and positive')
    if not bool(torch.isfinite(hr_candidate).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError('robust residual inputs must be finite')
    error = hr_candidate - target.detach()
    rms = (error.square().mean(dim=-1) + error.new_tensor(eps)).sqrt()
    scale = rms.detach().reshape(-1, rms.shape[-1]).median(dim=0).values
    broadcast_scale = scale.reshape(*[1] * (rms.ndim - 1), rms.shape[-1])
    ratio = rms / (broadcast_scale + error.new_tensor(eps))
    effective_weight = (1.0 + ratio.square()).rsqrt()
    rho = 2.0 * broadcast_scale.square() * ((1.0 + ratio.square()).sqrt() - 1.0)
    loss = rho.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('robust residual loss is non-finite')
    return (loss, rms, scale, effective_weight)

def candidate_hull_utility_loss(hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hr_candidate.shape != d_candidate.shape or hr_candidate.shape != target.shape:
        raise ValueError('candidate hull inputs must align')
    if hr_candidate.ndim < 3:
        raise ValueError('candidate hull inputs must include batch/edge/value axes')
    axes = tuple(range(2, hr_candidate.ndim))
    frozen_d = d_candidate.detach()
    frozen_target = target.detach()
    frozen_delta = hr_candidate.detach() - frozen_d
    numerator = ((frozen_target - frozen_d) * frozen_delta).sum(dim=axes)
    denominator = frozen_delta.square().sum(dim=axes).clamp_min(torch.finfo(hr_candidate.dtype).eps)
    g_star = (numerator / denominator).clamp(0.0, 1.0).detach()
    broadcast = g_star.reshape(*g_star.shape, *[1] * (hr_candidate.ndim - 2))
    projected = frozen_d + broadcast * (hr_candidate - frozen_d)
    return ((projected - frozen_target).square().mean(), g_star)

def per_object_candidate_hull_utility_loss(hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hr_candidate.shape != d_candidate.shape or hr_candidate.shape != target.shape:
        raise ValueError('candidate hull inputs must align')
    if hr_candidate.ndim < 4:
        raise ValueError('per-object candidate hull requires [B,F,K,state] candidate tensors')
    axes = tuple(range(3, hr_candidate.ndim))
    frozen_d = d_candidate.detach()
    frozen_target = target.detach()
    frozen_delta = hr_candidate.detach() - frozen_d
    numerator = ((frozen_target - frozen_d) * frozen_delta).sum(dim=axes)
    denominator = frozen_delta.square().sum(dim=axes).clamp_min(torch.finfo(hr_candidate.dtype).eps)
    g_star = (numerator / denominator).clamp(0.0, 1.0).detach()
    broadcast = g_star.reshape(*g_star.shape, *[1] * (hr_candidate.ndim - 3))
    projected = frozen_d + broadcast * (hr_candidate - frozen_d)
    return ((projected - frozen_target).square().mean(), g_star)

def cyclic_trace_field_index(update_index: int, trace_count: int) -> int:
    if type(update_index) is not int or update_index < 1:
        raise ValueError('cyclic trace update index must be a positive integer')
    if type(trace_count) is not int or trace_count < 1:
        raise ValueError('cyclic trace requires at least one accepted field')
    return (update_index - 1) % trace_count

def _trajectory_residual_field_candidates(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not trace.traces:
        raise ValueError('residual trace replay requires accepted fields')
    if type(field_index) is not int or not 0 <= field_index < len(trace.traces):
        raise ValueError('field_index lies outside the accepted trace')
    if type(window_length) is not int or window_length < 1:
        raise ValueError('window_length must be a positive integer')
    field = trace.traces[field_index]
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('accepted residual field lacks rollout/tokens/jets')
    rollout = field.rollout
    frames = int(rollout.mixed.shape[1])
    if target.shape[:2] != rollout.mixed.shape[:2]:
        raise ValueError('residual replay target does not align with field')
    incoming_g = previous_gate_sequence(rollout.gate.detach(), initial=field.anchor.previous_g)
    object_scale = state_scale.reshape(1, 1, -1)
    hr_candidates: list[torch.Tensor] = []
    d_candidates: list[torch.Tensor] = []
    for start in range(0, frames, window_length):
        stop = min(frames, start + window_length)
        jets = HamiBallsAffineJets(matrix=field.jets.matrix[:, start:stop].detach(), offset=field.jets.offset[:, start:stop].detach(), health=None)

        def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
            raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
            return raw / object_scale
        local = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=None, d_tokens=field.d_tokens[:, start:stop].detach(), noisy=field.state[:, start:stop].detach(), x0=x0.detach(), d_candidate=rollout.d_candidate[:, start:stop].detach(), attrs=attrs.detach(), tau=field.tau.detach(), physical_time=physical_time[:, 1 + start:1 + stop].detach(), initial_previous_mixed=rollout.previous_mixed[:, start].detach(), initial_previous_g=incoming_g[:, start].detach(), exogenous_gate=rollout.gate[:, start:stop].detach())
        hr_candidates.append(local.hr_candidate)
        d_candidates.append(local.d_candidate)
    return (torch.cat(hr_candidates, dim=1), torch.cat(d_candidates, dim=1))

def trajectory_residual_trace_hr_candidates(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=12) -> tuple[torch.Tensor, torch.Tensor]:
    return _trajectory_residual_field_candidates(residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=window_length)

def trajectory_residual_trace_cyclic_hull_loss(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=12) -> tuple[torch.Tensor, torch.Tensor, int]:
    field_index = cyclic_trace_field_index(update_index, len(trace.traces))
    hr_candidate, d_candidate = _trajectory_residual_field_candidates(residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=window_length)
    utility, g_star = candidate_hull_utility_loss(hr_candidate, d_candidate, target)
    return (utility, g_star, field_index)

def trajectory_residual_trace_cyclic_per_object_hull_loss(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=12) -> tuple[torch.Tensor, torch.Tensor, int]:
    field_index = cyclic_trace_field_index(update_index, len(trace.traces))
    hr_candidate, d_candidate = _trajectory_residual_field_candidates(residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=window_length)
    utility, g_star = per_object_candidate_hull_utility_loss(hr_candidate, d_candidate, target)
    return (utility, g_star, field_index)

def trajectory_residual_trace_components(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=8, no_harm_margin: float=0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not trace.traces:
        raise ValueError('residual trace replay requires accepted fields')
    if type(field_index) is not int or not 0 <= field_index < len(trace.traces):
        raise ValueError('field_index lies outside the accepted trace')
    if type(window_length) is not int or window_length < 1:
        raise ValueError('window_length must be a positive integer')
    field = trace.traces[field_index]
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('accepted residual field lacks rollout/tokens/jets')
    rollout = field.rollout
    frames = int(rollout.mixed.shape[1])
    if target.shape[:2] != rollout.mixed.shape[:2]:
        raise ValueError('residual replay target does not align with field')
    incoming_g = previous_gate_sequence(rollout.gate.detach(), initial=field.anchor.previous_g)
    object_scale = state_scale.reshape(1, 1, -1)
    replayed: list[torch.Tensor] = []
    no_harm_terms: list[torch.Tensor] = []
    for start in range(0, frames, window_length):
        stop = min(frames, start + window_length)
        jets = HamiBallsAffineJets(matrix=field.jets.matrix[:, start:stop].detach(), offset=field.jets.offset[:, start:stop].detach(), health=None)

        def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
            raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
            return raw / object_scale
        local = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=None, d_tokens=field.d_tokens[:, start:stop].detach(), noisy=field.state[:, start:stop].detach(), x0=x0.detach(), d_candidate=rollout.d_candidate[:, start:stop].detach(), attrs=attrs.detach(), tau=field.tau.detach(), physical_time=physical_time[:, 1 + start:1 + stop].detach(), initial_previous_mixed=rollout.previous_mixed[:, start].detach(), initial_previous_g=incoming_g[:, start].detach(), exogenous_gate=rollout.gate[:, start:stop].detach())
        replayed.append(local.mixed)
        no_harm, _ = residual_no_harm_loss(local.h_candidate, local.hr_candidate, target[:, start:stop].detach(), margin=no_harm_margin)
        no_harm_terms.append(no_harm)
    replay = torch.cat(replayed, dim=1)
    primary = (replay - target.detach()).square().mean()
    auxiliary = torch.stack(no_harm_terms).mean()
    return (primary, auxiliary, replay)

def trajectory_residual_trace_mse(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=8, no_harm_weight: float=0.0, no_harm_margin: float=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(no_harm_weight) or no_harm_weight < 0.0:
        raise ValueError('no_harm_weight must be finite and nonnegative')
    primary, auxiliary, replay = trajectory_residual_trace_components(residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=window_length, no_harm_margin=no_harm_margin)
    return (primary + primary.new_tensor(no_harm_weight) * auxiliary, replay)

def trajectory_gate_trace_g1_mse(gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    replay = _replay_gate_trace_g1(gate, residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, q_dim=q_dim)
    return ((replay.mixed - target.detach()).square().mean(), replay.gate)

def _replay_gate_trace_g1(gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, q_dim: int):
    if not trace.traces:
        raise ValueError('gate trace G1 requires accepted fields')
    if type(field_index) is not int or not 0 <= field_index < len(trace.traces):
        raise ValueError('field_index lies outside the accepted trace')
    field = trace.traces[field_index]
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('accepted gate field lacks rollout/tokens/jets')
    rollout = field.rollout
    jets = HamiBallsAffineJets(matrix=field.jets.matrix.detach(), offset=field.jets.offset.detach(), health=None)
    object_scale = state_scale.reshape(1, 1, -1)

    def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
        raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
        return raw / object_scale
    replay = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=gate, d_tokens=field.d_tokens.detach(), noisy=field.state.detach(), x0=x0.detach(), d_candidate=rollout.d_candidate.detach(), attrs=attrs.detach(), tau=field.tau.detach(), physical_time=physical_time[:, 1:].detach(), initial_previous_mixed=rollout.previous_mixed[:, 0].detach(), initial_previous_g=None if isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)) else rollout.previous_mixed.new_ones(rollout.previous_mixed.shape[0]) if field.anchor.previous_g is None else field.anchor.previous_g.detach(), initial_gate_hidden=None if (isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False))) or field.anchor.gate_hidden is None else field.anchor.gate_hidden.detach())
    return replay

def soft_projection_bce(gate_value: torch.Tensor, g_star: torch.Tensor, *, direction_balanced: bool=False, conditional_balance_weight: float=0.0) -> torch.Tensor:
    if gate_value.shape != g_star.shape:
        raise ValueError('soft projection gate and teacher must align')
    if not bool(torch.isfinite(gate_value).all()) or not bool(torch.isfinite(g_star).all()):
        raise FloatingPointError('soft projection BCE inputs must be finite')
    if bool((gate_value <= 0.0).any()) or bool((gate_value >= 1.0).any()):
        raise FloatingPointError('soft projection gate saturated outside (0,1)')
    if not math.isfinite(conditional_balance_weight) or conditional_balance_weight < 0.0:
        raise ValueError('conditional balance weight must be finite and nonnegative')
    logits = torch.logit(gate_value)
    uniform = torch.nn.functional.binary_cross_entropy_with_logits(logits, g_star.detach())
    weight: torch.Tensor | None = None
    if direction_balanced:
        left = (0.5 - g_star.detach()).clamp_min(0.0)
        right = (g_star.detach() - 0.5).clamp_min(0.0)
        left_mass = left.sum()
        right_mass = right.sum()
        epsilon = g_star.new_tensor(torch.finfo(g_star.dtype).eps)
        if bool(left_mass > epsilon) and bool(right_mass > epsilon):
            weight = torch.where(left > 0.0, 0.5 / left_mass, torch.where(right > 0.0, 0.5 / right_mass, torch.zeros_like(g_star)))
            weight = weight / weight.mean().clamp_min(epsilon)
    if weight is not None:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, g_star.detach(), weight=weight)
    if conditional_balance_weight == 0.0:
        return uniform
    if gate_value.ndim != 3:
        raise ValueError('conditional direction balance requires [B,F,K]')
    left = (0.5 - g_star.detach()).clamp_min(0.0)
    right = (g_star.detach() - 0.5).clamp_min(0.0)
    left_mass = left.sum(dim=(0, 2), keepdim=True)
    right_mass = right.sum(dim=(0, 2), keepdim=True)
    epsilon = g_star.new_tensor(torch.finfo(g_star.dtype).eps)
    valid = (left_mass > epsilon) & (right_mass > epsilon)
    conditional_weight = torch.where(left > 0.0, 0.5 / left_mass.clamp_min(epsilon), torch.where(right > 0.0, 0.5 / right_mass.clamp_min(epsilon), torch.zeros_like(g_star)))
    conditional_weight = torch.where(valid.expand_as(conditional_weight), conditional_weight, torch.zeros_like(conditional_weight))
    nonzero = conditional_weight > 0.0
    if not bool(nonzero.any()):
        return uniform
    conditional_weight = conditional_weight / conditional_weight[nonzero].mean()
    auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(logits, g_star.detach(), weight=conditional_weight)
    return uniform + uniform.new_tensor(conditional_balance_weight) * auxiliary

def object_pairwise_projection_ranking_loss(gate_value: torch.Tensor, g_star: torch.Tensor) -> torch.Tensor:
    if gate_value.shape != g_star.shape or gate_value.ndim != 3:
        raise ValueError('object pairwise ranking requires aligned [B,F,K] tensors')
    if not bool(torch.isfinite(gate_value).all()) or not bool(torch.isfinite(g_star).all()):
        raise FloatingPointError('object pairwise ranking inputs must be finite')
    if bool((gate_value <= 0.0).any()) or bool((gate_value >= 1.0).any()):
        raise FloatingPointError('object pairwise ranking gate must lie in (0,1)')
    logits = torch.logit(gate_value)
    teacher = g_star.detach()
    logit_gap = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    teacher_gap = teacher.unsqueeze(-1) - teacher.unsqueeze(-2)
    confidence = teacher_gap.abs()
    direction = teacher_gap.sign()
    objects = gate_value.shape[-1]
    mask = ~torch.eye(objects, device=gate_value.device, dtype=torch.bool).reshape(1, 1, objects, objects)
    mask = mask.expand_as(confidence)
    weights = confidence[mask]
    if not bool((weights > 0.0).any()):
        return logits.sum() * 0.0
    losses = torch.nn.functional.softplus(-direction * logit_gap)[mask]
    total = (weights * losses).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('object pairwise ranking loss is non-finite')
    return total

def trajectory_gate_trace_g1_projection_bce(gate: HamiBallsPerObjectCompactCommittedGate, residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, direction_balanced: bool=False, conditional_balance_weight: float=0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    replay = _replay_gate_trace_g1(gate, residual, trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, q_dim=q_dim)
    _unused_mse, g_star = per_object_convex_projection_gate_loss(replay.gate, replay.hr_candidate, replay.d_candidate, target)
    total = soft_projection_bce(replay.gate, g_star, direction_balanced=direction_balanced, conditional_balance_weight=conditional_balance_weight)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('trajectory G1 soft projection BCE is non-finite')
    return (total, replay.gate, g_star)

def _sample_expert(*, d: nn.Module, hamiltonian: TokenConditionalTypeIIGenerator | None, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate | None, source: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, mode: str, reset_gate: torch.Tensor | None=None, reset_rf_tau: torch.Tensor | None=None, detach_d_carrier: bool=False, field_callback: Callable[[HamiBallsExpertFieldTrace], None] | None=None, num_steps: int=20, variable_n_cold_start: bool=False) -> HamiBallsChunkSample:
    return sample_hamiballs_expert_chunk(d, hamiltonian, residual, gate, source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, step_size=frame_dt, num_steps=num_steps, t_eps=t_eps, mode=mode, external_gate=reset_gate, external_gate_tau=reset_rf_tau, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, detach_d_carrier=detach_d_carrier, field_callback=field_callback, variable_n_cold_start=variable_n_cold_start)

def trajectory_residual_mse(*, d: nn.Module, hamiltonian: TokenConditionalTypeIIGenerator | None, residual: HamiBallsDTokenResidual, source: torch.Tensor, x0: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, reset_gate: torch.Tensor, reset_rf_tau: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, num_steps: int=20, variable_n_cold_start: bool=False) -> tuple[torch.Tensor, HamiBallsChunkSample]:
    sample = _sample_expert(d=d, hamiltonian=hamiltonian, residual=residual, gate=None, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, frame_dt=frame_dt, t_eps=t_eps, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, mode='external', reset_gate=reset_gate, reset_rf_tau=reset_rf_tau, num_steps=num_steps, variable_n_cold_start=variable_n_cold_start)
    return ((sample.trajectory - target.detach()).square().mean(), sample)

@torch.no_grad()
def collect_trajectory_g0(*, d: nn.Module, hamiltonian: TokenConditionalTypeIIGenerator | None, residual: HamiBallsDTokenResidual, source: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, reset_gate: torch.Tensor, reset_rf_tau: torch.Tensor, pure_rows: int, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, num_steps: int=20, variable_n_cold_start: bool=False) -> HamiBallsTrajectoryG0:
    traces: list[HamiBallsExpertFieldTrace] = []

    def capture(trace: HamiBallsExpertFieldTrace) -> None:
        if not trace.is_cold and trace.accepted and (trace.side in {'left', 'final'}):
            if trace.rollout is None or trace.d_tokens is None:
                raise RuntimeError('accepted expert field lacks deployment tensors')
            traces.append(trace)
    _sample_expert(d=d, hamiltonian=hamiltonian, residual=residual, gate=None, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, frame_dt=frame_dt, t_eps=t_eps, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, mode='external', reset_gate=reset_gate, reset_rf_tau=reset_rf_tau, field_callback=capture, num_steps=num_steps, variable_n_cold_start=variable_n_cold_start)
    if not traces:
        raise RuntimeError('N20 trajectory G0 captured no accepted expert field')
    return HamiBallsTrajectoryG0(traces=tuple(traces), pure_rows=int(pure_rows), reset_edges=int((reset_gate < 1.0).sum().item()))

@torch.no_grad()
def collect_trajectory_on_policy_g0(*, d: nn.Module, hamiltonian: TokenConditionalTypeIIGenerator | None, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, source: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, num_steps: int=20, variable_n_cold_start: bool=False) -> HamiBallsTrajectoryG0:
    traces: list[HamiBallsExpertFieldTrace] = []

    def capture(trace: HamiBallsExpertFieldTrace) -> None:
        if not trace.is_cold and trace.accepted and (trace.side in {'left', 'final'}):
            if trace.rollout is None or trace.d_tokens is None:
                raise RuntimeError('accepted on-policy field lacks deployment tensors')
            traces.append(trace)
    _sample_expert(d=d, hamiltonian=hamiltonian, residual=residual, gate=gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, frame_dt=frame_dt, t_eps=t_eps, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, mode='main', field_callback=capture, num_steps=num_steps, variable_n_cold_start=variable_n_cold_start)
    if not traces:
        raise RuntimeError('N20 on-policy G0 captured no accepted expert field')
    return HamiBallsTrajectoryG0(traces=tuple(traces), pure_rows=0, reset_edges=0)

def regret_aware_gate_loss(gate_value: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor, *, auxiliary_weight: float, maximum_relative_weight: float=10.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gate_value.ndim == 2:
        gate_prefix_dims = 2
        expected_gate_shape = hr_candidate.shape[:2]
    elif gate_value.ndim == 3:
        gate_prefix_dims = 3
        expected_gate_shape = hr_candidate.shape[:3]
    else:
        raise ValueError('gate_value must be [B,F] or [B,F,K]')
    if gate_value.shape != expected_gate_shape:
        raise ValueError('gate_value does not align with expert candidates')
    if hr_candidate.shape != d_candidate.shape or hr_candidate.shape != target.shape:
        raise ValueError('expert candidates and target must align')
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0.0:
        raise ValueError('auxiliary_weight must be finite and nonnegative')
    mixed_weight = gate_value.reshape(*gate_value.shape, *[1] * (hr_candidate.ndim - gate_prefix_dims))
    mixed = d_candidate + mixed_weight * (hr_candidate - d_candidate)
    mix_mse = (mixed - target).square().mean()
    expert_axes = tuple(range(gate_prefix_dims, hr_candidate.ndim))
    hr_mse = (hr_candidate - target).square().mean(dim=expert_axes)
    d_mse = (d_candidate - target).square().mean(dim=expert_axes)
    regret = (d_mse - hr_mse).detach()
    winner = (regret > 0.0).to(gate_value.dtype)
    scale = regret.abs().mean().clamp_min(torch.finfo(regret.dtype).eps)
    relative = (regret.abs() / scale).clamp(max=maximum_relative_weight)
    ranking = (relative * torch.nn.functional.binary_cross_entropy(gate_value.clamp(1e-06, 1.0 - 1e-06), winner, reduction='none')).mean()
    total = mix_mse + gate_value.new_tensor(auxiliary_weight) * ranking
    return (total, mix_mse, ranking)

def per_object_convex_projection_gate_loss(gate_value: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_value.ndim != 3:
        raise ValueError('projection distillation requires a [B,F,K] per-object gate')
    if hr_candidate.shape != d_candidate.shape or target.shape != d_candidate.shape:
        raise ValueError('projection candidates and target must align')
    if hr_candidate.ndim != gate_value.ndim + 1 or hr_candidate.shape[:-1] != gate_value.shape:
        raise ValueError('projection gate/candidate shapes must be [B,F,K] and [B,F,K,S]')
    delta = hr_candidate.detach() - d_candidate.detach()
    residual = target.detach() - d_candidate.detach()
    denominator = delta.square().sum(dim=-1)
    numerator = (residual * delta).sum(dim=-1)
    epsilon = torch.finfo(delta.dtype).eps
    g_star = torch.where(denominator > epsilon, numerator / denominator, torch.zeros_like(numerator)).clamp(0.0, 1.0).detach()
    loss = (gate_value - g_star).square().mean()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(g_star).all()):
        raise FloatingPointError('projection-distillation loss is non-finite')
    return (loss, g_star)

def trajectory_g0_mse(gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, trace: HamiBallsTrajectoryG0, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, regret_auxiliary_weight: float=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for field in trace.traces:
        if field.rollout is None or field.d_tokens is None:
            raise RuntimeError('trajectory G0 trace is incomplete')
        rollout = field.rollout
        incoming_gate = previous_gate_sequence(rollout.gate.detach(), initial=None if isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)) else field.anchor.previous_g)
        if (isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False))) and incoming_gate.ndim == 2:
            incoming_gate = incoming_gate[:, :, None].expand(-1, -1, rollout.h_candidate.shape[2])
        gate_args = (field.d_tokens.detach(), field.state.detach(), x0.detach(), rollout.previous_mixed.detach(), rollout.h_candidate.detach(), rollout.hr_candidate.detach(), rollout.d_candidate.detach(), attrs.detach(), field.tau.detach(), physical_time[:, 1:].detach(), incoming_gate)
        if isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)):
            value, _ = gate(*gate_args, residual_hidden=rollout.residual_hidden.detach())
        else:
            value, _ = gate(*gate_args, hidden=field.anchor.gate_hidden, residual_hidden=rollout.residual_hidden.detach())
        total, _, _ = regret_aware_gate_loss(value, rollout.hr_candidate.detach(), rollout.d_candidate.detach(), target.detach(), auxiliary_weight=regret_auxiliary_weight)
        losses.append(total)
        values.append(value)
    return (torch.stack(losses).mean(), torch.cat(values, dim=1))

def trajectory_g0_projection_mse(gate: HamiBallsPerObjectCompactCommittedGate, trace: HamiBallsTrajectoryG0, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    for field in trace.traces:
        if field.rollout is None or field.d_tokens is None:
            raise RuntimeError('trajectory projection G0 trace is incomplete')
        rollout = field.rollout
        incoming_gate = previous_gate_sequence(rollout.gate.detach())
        if incoming_gate.ndim == 2:
            incoming_gate = incoming_gate[:, :, None].expand(-1, -1, rollout.h_candidate.shape[2])
        if incoming_gate.ndim != 3:
            raise AssertionError('per-object projection G0 lost object gate provenance')
        gate_args = (field.d_tokens.detach(), field.state.detach(), x0.detach(), rollout.previous_mixed.detach(), rollout.h_candidate.detach(), rollout.hr_candidate.detach(), rollout.d_candidate.detach(), attrs.detach(), field.tau.detach(), physical_time[:, 1:].detach(), incoming_gate)
        value, _ = gate(*gate_args, residual_hidden=rollout.residual_hidden.detach())
        _field_loss, g_star = per_object_convex_projection_gate_loss(value, rollout.hr_candidate, rollout.d_candidate, target)
        values.append(value)
        teachers.append(g_star)
    if not values:
        raise ValueError('trajectory projection G0 trace has no accepted fields')
    joined_value = torch.cat(values, dim=1)
    joined_teacher = torch.cat(teachers, dim=1)
    total = (joined_value - joined_teacher).square().mean()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('trajectory projection G0 loss is non-finite')
    return (total, joined_value, joined_teacher)

def trajectory_g0_projection_bce(gate: HamiBallsPerObjectCompactCommittedGate, trace: HamiBallsTrajectoryG0, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, direction_balanced: bool=False, conditional_balance_weight: float=0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _unused_mse, value, g_star = trajectory_g0_projection_mse(gate, trace, x0=x0, attrs=attrs, physical_time=physical_time, target=target)
    total = soft_projection_bce(value, g_star, direction_balanced=direction_balanced, conditional_balance_weight=conditional_balance_weight)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('trajectory G0 soft projection BCE is non-finite')
    return (total, value, g_star)

def trajectory_g1_mse(*, d: nn.Module, hamiltonian: TokenConditionalTypeIIGenerator | None, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, source: torch.Tensor, x0: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, num_steps: int=20, variable_n_cold_start: bool=False) -> tuple[torch.Tensor, HamiBallsChunkSample]:
    sample = _sample_expert(d=d, hamiltonian=hamiltonian, residual=residual, gate=gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, frame_dt=frame_dt, t_eps=t_eps, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, mode='main', detach_d_carrier=True, num_steps=num_steps, variable_n_cold_start=variable_n_cold_start)
    return ((sample.trajectory - target.detach()).square().mean(), sample)
__all__ = ['candidate_hull_utility_loss', 'per_object_candidate_hull_utility_loss', 'cyclic_trace_field_index', 'HamiBallsTrajectoryG0', 'soft_projection_bce', 'object_pairwise_projection_ranking_loss', 'collect_trajectory_g0', 'collect_trajectory_on_policy_g0', 'trajectory_g0_mse', 'trajectory_g0_projection_mse', 'trajectory_g0_projection_bce', 'trajectory_gate_trace_g1_projection_bce', 'regret_aware_gate_loss', 'per_object_convex_projection_gate_loss', 'residual_no_harm_loss', 'robust_per_object_pseudo_huber_loss', 'trajectory_g1_mse', 'trajectory_gate_trace_g1_mse', 'trajectory_residual_mse', 'trajectory_residual_trace_components', 'trajectory_residual_trace_hr_candidates', 'trajectory_residual_trace_cyclic_hull_loss', 'trajectory_residual_trace_cyclic_per_object_hull_loss', 'trajectory_residual_trace_mse']
