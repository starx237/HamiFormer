from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as functional
from torch import nn
from hamiformer.models.interleaved_hd import precomputed_gated_affine_rollout
from hamiformer.physics.learned_pgf import TypeIIJet, learned_pgf_edge_jets, type2_generator_eom
from hamiformer.routing.objectives import convex_routing_target_prevalidated

@dataclass(frozen=True)
class OneWayGFTangentRollout:
    mixed_clean: torch.Tensor
    h_candidate: torch.Tensor
    jet: TypeIIJet

@dataclass(frozen=True)
class OnPolicyTeacherRollout:
    mixed_clean: torch.Tensor
    h_candidate: torch.Tensor
    oracle_responsibility: torch.Tensor
    training_responsibility: torch.Tensor
    identifiable: torch.Tensor
    relative_gain: torch.Tensor
    jet: TypeIIJet

@dataclass(frozen=True)
class TrainableHObjective:
    total: torch.Tensor
    predictive: torch.Tensor
    generating_relation: torch.Tensor
    hessian_barrier: torch.Tensor
    rollout: OnPolicyTeacherRollout

def oneway_gf_tangent_rollout(generator: nn.Module, *, anchor_future: torch.Tensor, d_clean: torch.Tensor, responsibility: torch.Tensor, initial_state: torch.Tensor, theta_sys: torch.Tensor, step_size: float, mixed_hessian_floor: float) -> OneWayGFTangentRollout:
    if anchor_future.shape != d_clean.shape:
        raise ValueError('anchor_future 与 d_clean 必须同形')
    if anchor_future.ndim != 3:
        raise ValueError('anchor_future 必须为 [B,T,D]')
    if responsibility.shape != (*anchor_future.shape[:-1], 1):
        raise ValueError('responsibility 必须为 [B,T,1]')
    if initial_state.shape != (anchor_future.shape[0], anchor_future.shape[-1]):
        raise ValueError('initial_state 必须为 [B,D]')
    anchor = anchor_future.detach()
    fallback = d_clean.detach()
    fixed_responsibility = responsibility.detach()
    fixed_initial = initial_state.detach()
    fixed_theta = theta_sys.detach()
    jet = learned_pgf_edge_jets(generator, anchor, fixed_initial, fixed_theta, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, create_graph=True)
    mixed_clean = precomputed_gated_affine_rollout(fixed_initial, fallback, jet.matrix, jet.offset, fixed_responsibility)
    mixed_sources = torch.cat([fixed_initial[:, None], mixed_clean[:, :-1]], dim=1)
    h_candidate = torch.matmul(jet.matrix, mixed_sources.unsqueeze(-1)).squeeze(-1) + jet.offset
    return OneWayGFTangentRollout(mixed_clean=mixed_clean, h_candidate=h_candidate, jet=jet)

def annealed_teacher_responsibility(oracle_responsibility: torch.Tensor, *, teacher_strength: float, prior_responsibility: float=0.5) -> torch.Tensor:
    if not math.isfinite(teacher_strength) or not 0.0 <= teacher_strength <= 1.0:
        raise ValueError('teacher_strength 必须是 [0,1] 内的 finite 数')
    if not math.isfinite(prior_responsibility) or not 0.0 <= prior_responsibility <= 1.0:
        raise ValueError('prior_responsibility 必须是 [0,1] 内的 finite 数')
    if oracle_responsibility.ndim < 1 or oracle_responsibility.shape[-1] != 1:
        raise ValueError('oracle_responsibility 最后一维必须为 1')
    if not bool(torch.isfinite(oracle_responsibility).all().item()) or bool(((oracle_responsibility < 0.0) | (oracle_responsibility > 1.0)).any().item()):
        raise ValueError('oracle_responsibility 必须位于 [0,1]')
    oracle = oracle_responsibility.detach()
    return ((1.0 - teacher_strength) * prior_responsibility + teacher_strength * oracle).detach()

def linear_teacher_strength(step: int, *, start_step: int, end_step: int) -> float:
    if step < 0 or start_step < 0 or end_step < start_step:
        raise ValueError('step/start_step/end_step 合同非法')
    if end_step == start_step:
        return float(step >= end_step)
    return float(min(1.0, max(0.0, (step - start_step) / (end_step - start_step))))

def onpolicy_convex_teacher_rollout(generator: nn.Module, *, anchor_future: torch.Tensor, d_clean: torch.Tensor, clean_target: torch.Tensor, initial_state: torch.Tensor, theta_sys: torch.Tensor, phase_scale: torch.Tensor, step_size: float, mixed_hessian_floor: float, teacher_strength: float, prior_responsibility: float=0.5) -> OnPolicyTeacherRollout:
    if anchor_future.shape != d_clean.shape or d_clean.shape != clean_target.shape:
        raise ValueError('anchor_future、d_clean 与 clean_target 必须同形')
    if anchor_future.ndim != 3 or anchor_future.shape[-1] != 2:
        raise ValueError('第一版 phase 必须为 [B,T,2]')
    batch, edges, state_dim = anchor_future.shape
    if edges < 1:
        raise ValueError('至少需要一条物理边')
    if initial_state.shape != (batch, state_dim):
        raise ValueError('initial_state 必须为 [B,2]')
    if theta_sys.ndim != 2 or theta_sys.shape[0] != batch:
        raise ValueError('theta_sys 必须为 [B,theta_dim]')
    if phase_scale.shape != (state_dim,):
        raise ValueError('phase_scale 必须为 [2]')
    if not bool(torch.isfinite(phase_scale).all().item()) or bool((phase_scale <= 0.0).any().item()):
        raise ValueError('phase_scale 必须全部为 finite 正数')
    anchor = anchor_future.detach()
    fallback = d_clean.detach()
    target = clean_target.detach()
    fixed_initial = initial_state.detach()
    fixed_theta = theta_sys.detach()
    fixed_scale = phase_scale.detach()
    jet = learned_pgf_edge_jets(generator, anchor, fixed_initial, fixed_theta, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, create_graph=True)
    current = fixed_initial
    mixed_states: list[torch.Tensor] = []
    h_candidates: list[torch.Tensor] = []
    oracle_weights: list[torch.Tensor] = []
    training_weights: list[torch.Tensor] = []
    identifiable: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []
    for edge in range(edges):
        h_candidate = torch.matmul(jet.matrix[:, edge], current.unsqueeze(-1)).squeeze(-1) + jet.offset[:, edge]
        teacher = convex_routing_target_prevalidated(fallback[:, edge:edge + 1], h_candidate.detach()[:, None], target[:, edge:edge + 1], fixed_scale)
        oracle = teacher.responsibility[:, 0]
        training_weight = annealed_teacher_responsibility(oracle, teacher_strength=teacher_strength, prior_responsibility=prior_responsibility)
        current = fallback[:, edge] + training_weight * (h_candidate - fallback[:, edge])
        h_candidates.append(h_candidate)
        mixed_states.append(current)
        oracle_weights.append(oracle)
        training_weights.append(training_weight)
        identifiable.append(teacher.identifiable[:, 0])
        gains.append(teacher.relative_gain[:, 0])
    return OnPolicyTeacherRollout(mixed_clean=torch.stack(mixed_states, dim=1), h_candidate=torch.stack(h_candidates, dim=1), oracle_responsibility=torch.stack(oracle_weights, dim=1).detach(), training_responsibility=torch.stack(training_weights, dim=1).detach(), identifiable=torch.stack(identifiable, dim=1).detach(), relative_gain=torch.stack(gains, dim=1).detach(), jet=jet)

def responsibility_with_floor(responsibility: torch.Tensor, *, floor: float) -> torch.Tensor:
    if not 0.0 <= floor <= 1.0:
        raise ValueError('floor 必须位于 [0,1]')
    if responsibility.shape[-1] != 1:
        raise ValueError('responsibility 最后一维必须为 1')
    if not bool(torch.isfinite(responsibility).all().item()):
        raise ValueError('responsibility 必须为 finite')
    detached = responsibility.detach().clamp(0.0, 1.0)
    return floor + (1.0 - floor) * detached

def weighted_type2_relation_loss(generator: nn.Module, *, initial_state: torch.Tensor, clean_future: torch.Tensor, theta_sys: torch.Tensor, edge_weight: torch.Tensor, step_size: float, q_scale: float, p_scale: float, robust_delta: float | None=None) -> torch.Tensor:
    if clean_future.ndim != 3 or clean_future.shape[-1] != 2:
        raise ValueError('clean_future 必须为 [B,T,2]')
    batch, edges, _ = clean_future.shape
    if initial_state.shape != (batch, 2):
        raise ValueError('initial_state 必须为 [B,2]')
    if theta_sys.ndim != 2 or theta_sys.shape[0] != batch:
        raise ValueError('theta_sys 必须为 [B,theta_dim]')
    if edge_weight.shape != (batch, edges, 1):
        raise ValueError('edge_weight 必须为 [B,T,1]')
    if step_size <= 0.0 or q_scale <= 0.0 or p_scale <= 0.0:
        raise ValueError('step_size/q_scale/p_scale 必须为正')
    if robust_delta is not None and robust_delta <= 0.0:
        raise ValueError('robust_delta 必须为正或 None')
    if not bool(torch.isfinite(edge_weight).all().item()) or bool((edge_weight < 0.0).any().item()):
        raise ValueError('edge_weight 必须为 finite 非负数')
    fixed_initial = initial_state.detach()
    fixed_clean = clean_future.detach()
    fixed_theta = theta_sys.detach()
    source = torch.cat([fixed_initial[:, None], fixed_clean[:, :-1]], dim=1)
    source_q = source[..., 0]
    source_p = source[..., 1]
    target_q = fixed_clean[..., 0]
    target_p = fixed_clean[..., 1]
    theta_edges = fixed_theta[:, None, :].expand(batch, edges, fixed_theta.shape[-1])
    predicted_p, predicted_q = type2_generator_eom(generator, source_q, target_p, theta_edges, step_size=step_size, create_graph=True)
    p_residual = (predicted_p - source_p) / (step_size * p_scale)
    q_residual = (predicted_q - target_q) / (step_size * q_scale)
    if robust_delta is None:
        per_edge = p_residual.square() + q_residual.square()
    else:
        per_edge = functional.huber_loss(p_residual, torch.zeros_like(p_residual), reduction='none', delta=robust_delta) + functional.huber_loss(q_residual, torch.zeros_like(q_residual), reduction='none', delta=robust_delta)
    weight = edge_weight.detach().squeeze(-1)
    denominator = weight.sum().clamp_min(torch.as_tensor(1e-12, device=weight.device, dtype=weight.dtype))
    return (weight * per_edge).sum() / denominator

def mixed_hessian_barrier(jet: TypeIIJet, *, floor: float) -> torch.Tensor:
    if not torch.isfinite(torch.as_tensor(floor)) or floor <= 0.0:
        raise ValueError('floor 必须为正')
    return functional.relu(floor - jet.mixed_hessian.abs()).square().mean()

def trainable_h_objective(generator: nn.Module, *, anchor_future: torch.Tensor, d_clean: torch.Tensor, clean_target: torch.Tensor, initial_state: torch.Tensor, theta_sys: torch.Tensor, phase_scale: torch.Tensor, q_scale: float, p_scale: float, step_size: float, mixed_hessian_floor: float, teacher_strength: float, prior_responsibility: float, predictive_weight: float, relation_weight: float, barrier_weight: float, robust_delta: float | None) -> TrainableHObjective:
    weights = (predictive_weight, relation_weight, barrier_weight)
    if any((not math.isfinite(value) or value < 0.0 for value in weights)):
        raise ValueError('H objective 权重必须为 finite 非负数')
    if sum(weights) <= 0.0:
        raise ValueError('H objective 至少需要一个正权重')
    rollout = onpolicy_convex_teacher_rollout(generator, anchor_future=anchor_future, d_clean=d_clean, clean_target=clean_target, initial_state=initial_state, theta_sys=theta_sys, phase_scale=phase_scale, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, teacher_strength=teacher_strength, prior_responsibility=prior_responsibility)
    scale = phase_scale.detach().to(clean_target)
    target = clean_target.detach()
    per_edge = ((rollout.h_candidate - target) / scale).square().sum(dim=-1)
    edge_weight = rollout.training_responsibility.squeeze(-1)
    denominator = edge_weight.sum().clamp_min(torch.as_tensor(1e-12, device=edge_weight.device, dtype=edge_weight.dtype))
    predictive = (edge_weight * per_edge).sum() / denominator
    relation = weighted_type2_relation_loss(generator, initial_state=initial_state, clean_future=clean_target, theta_sys=theta_sys, edge_weight=rollout.training_responsibility, step_size=step_size, q_scale=q_scale, p_scale=p_scale, robust_delta=robust_delta)
    barrier = mixed_hessian_barrier(rollout.jet, floor=mixed_hessian_floor)
    total = predictive_weight * predictive + relation_weight * relation + barrier_weight * barrier
    return TrainableHObjective(total=total, predictive=predictive, generating_relation=relation, hessian_barrier=barrier, rollout=rollout)
__all__ = ['OneWayGFTangentRollout', 'OnPolicyTeacherRollout', 'TrainableHObjective', 'annealed_teacher_responsibility', 'linear_teacher_strength', 'mixed_hessian_barrier', 'oneway_gf_tangent_rollout', 'onpolicy_convex_teacher_rollout', 'responsibility_with_floor', 'trainable_h_objective', 'weighted_type2_relation_loss']
