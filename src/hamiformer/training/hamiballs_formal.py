from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HamiBallsCompactCommittedGate, HamiBallsCommittedGate, HamiBallsCommittedRollout, HamiBallsDTokenResidual, HamiBallsGatePolicy, HamiBallsPerObjectCompactCommittedGate, rollout_hamiballs_committed_edges
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, HamiBallsMixedJetAnchor, apply_hamiballs_affine_jet, flatten_hamiballs_phase, hamiballs_anchor_from_candidates, hamiballs_anchor_from_d_candidate, hamiballs_leapfrog_affine_jets, hamiballs_type2_affine_jets
from hamiformer.physics.generic_type2 import GenericTypeIIHealthBarrier, type2_vector_health_barrier
from hamiformer.training.unlabeled_type2_location import UnlabeledTypeIILocationObjective, UnlabeledTypeIIRelationObjective, multivariate_student_t_location, unlabeled_type2_free_rollout_objective, unlabeled_type2_relation_objective

def canonical_state_scale(per_object_state_scale: torch.Tensor, *, num_objects: int, q_dim: int) -> torch.Tensor:
    if per_object_state_scale.shape != (2 * q_dim,):
        raise ValueError('per-object state scale must have shape [2*q_dim]')
    if not bool(torch.isfinite(per_object_state_scale).all()) or bool((per_object_state_scale <= 0).any()):
        raise ValueError('state scale must be finite and positive')
    return torch.cat([per_object_state_scale[:q_dim].repeat(num_objects), per_object_state_scale[q_dim:].repeat(num_objects)], dim=0)

def normalise_object_context(attrs: torch.Tensor, attr_scale: torch.Tensor) -> torch.Tensor:
    if attrs.ndim != 3 or attr_scale.shape != (attrs.shape[-1],):
        raise ValueError('attrs/attr_scale must align as [B,K,C] and [C]')
    if not bool(torch.isfinite(attrs).all() and torch.isfinite(attr_scale).all()):
        raise ValueError('attrs and attr_scale must be finite')
    if bool((attr_scale <= 0).any()):
        raise ValueError('attr_scale must be positive')
    return (attrs / attr_scale.to(attrs))[..., None, :]

def _state_scale_views(state_scale: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if state_scale.ndim != 1 or state_scale.shape[0] != reference.shape[-1]:
        raise ValueError("state_scale must match one object's [q,p] state")
    scale = state_scale.to(reference)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError('state_scale must be finite and positive')
    return (scale.reshape(1, 1, -1), scale.reshape(1, 1, 1, -1))

def hamiballs_relation_loss(generator: nn.Module, phase: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, student_t_dof: float=4.0) -> UnlabeledTypeIIRelationObjective:
    if phase.ndim != 4 or phase.shape[0] != attrs.shape[0]:
        raise ValueError('phase/attrs must be [B,T,K,state] and [B,K,C]')
    batch, frames, objects, state_dim = phase.shape
    if frames < 2 or state_dim != 2 * q_dim or attrs.shape[1] != objects:
        raise ValueError('invalid HamiBalls phase/context dimensions')
    canonical = flatten_hamiballs_phase(phase, q_dim=q_dim)
    edges = frames - 1
    context = normalise_object_context(attrs, attr_scale)
    context = context[:, None].expand(batch, edges, *context.shape[1:])
    return unlabeled_type2_relation_objective(generator, canonical[:, :-1].reshape(batch * edges, -1), canonical[:, 1:].reshape(batch * edges, -1), context.reshape(batch * edges, *context.shape[2:]), state_scale=canonical_state_scale(state_scale, num_objects=objects, q_dim=q_dim), step_size=step_size, student_t_dof=student_t_dof)

@dataclass(frozen=True)
class HamiBallsChannelSeparatedRelationObjective:
    location: torch.Tensor
    p_location: torch.Tensor
    q_location: torch.Tensor
    residual: torch.Tensor
    predicted_source_p: torch.Tensor
    predicted_target_q: torch.Tensor

@dataclass(frozen=True)
class HamiBallsAnisotropicIRLSRelationObjective:
    location: torch.Tensor
    residual: torch.Tensor
    equation_residual_scale: torch.Tensor
    standardized_squared_mean: torch.Tensor
    retention_weight: torch.Tensor
    small_residual_multiplier: float
    predicted_source_p: torch.Tensor
    predicted_target_q: torch.Tensor

def hamiballs_identity_relation_residual_scale(phase: torch.Tensor, *, state_scale: torch.Tensor, q_dim: int, minimum_scale: float=1e-08) -> torch.Tensor:
    if phase.ndim != 4 or phase.shape[1] < 2 or phase.shape[-1] != 2 * q_dim:
        raise ValueError('phase must be [B,T,K,2*q_dim] with at least two frames')
    if state_scale.shape != (2 * q_dim,):
        raise ValueError("state_scale must match one object's phase")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError('minimum_scale must be finite and positive')
    scale = state_scale.to(phase)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError('state_scale must be finite and positive')
    source, target = (phase[:, :-1], phase[:, 1:])
    p_identity = (source[..., q_dim:] - target[..., q_dim:]) / scale[q_dim:]
    q_identity = (target[..., :q_dim] - source[..., :q_dim]) / scale[:q_dim]
    p_scale = torch.quantile(p_identity.detach().abs().reshape(-1, q_dim), 0.5, dim=0)
    q_scale = torch.quantile(q_identity.detach().abs().reshape(-1, q_dim), 0.5, dim=0)
    result = torch.cat([p_scale, q_scale], dim=0)
    if not bool(torch.isfinite(result).all()) or bool((result <= minimum_scale).any()):
        raise ValueError('identity relation residual scale is degenerate')
    return result.detach()

def hamiballs_anisotropic_irls_relation_loss(generator: nn.Module, phase: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, equation_residual_scale: torch.Tensor, q_dim: int, step_size: float, student_t_dof: float=4.0) -> HamiBallsAnisotropicIRLSRelationObjective:
    relation = hamiballs_relation_loss(generator, phase, attrs, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, step_size=step_size, student_t_dof=student_t_dof)
    if not math.isfinite(student_t_dof) or student_t_dof <= 0.0:
        raise ValueError('student_t_dof must be finite and positive')
    if equation_residual_scale.shape != (2 * q_dim,):
        raise ValueError('equation_residual_scale must have shape [2*q_dim]')
    fixed_scale = equation_residual_scale.to(relation.residual)
    if fixed_scale.requires_grad:
        raise ValueError('equation_residual_scale must be fixed')
    if not bool(torch.isfinite(fixed_scale).all()) or bool((fixed_scale <= 0).any()):
        raise ValueError('equation_residual_scale must be finite and positive')
    objects = int(phase.shape[2])
    expanded_scale = torch.cat([fixed_scale[:q_dim].repeat(objects), fixed_scale[q_dim:].repeat(objects)], dim=0)
    if expanded_scale.shape != (relation.residual.shape[-1],):
        raise RuntimeError('expanded anisotropic scale does not match relation residual')
    standardized_squared_mean = (relation.residual / expanded_scale).square().mean(dim=-1)
    retention_weight = (relation.residual.new_tensor(student_t_dof) / (student_t_dof + standardized_squared_mean)).detach()
    dimension = relation.residual.shape[-1]
    small_residual_multiplier = (student_t_dof + dimension) / student_t_dof
    canonical_half_mse = 0.5 * relation.residual.square().mean(dim=-1)
    location = small_residual_multiplier * (retention_weight * canonical_half_mse).mean()
    return HamiBallsAnisotropicIRLSRelationObjective(location=location, residual=relation.residual, equation_residual_scale=fixed_scale, standardized_squared_mean=standardized_squared_mean, retention_weight=retention_weight, small_residual_multiplier=small_residual_multiplier, predicted_source_p=relation.predicted_source_p, predicted_target_q=relation.predicted_target_q)

@dataclass(frozen=True)
class HamiBallsOneStepSolveThroughObjective:
    total: torch.Tensor
    location: torch.Tensor
    q_location: torch.Tensor
    p_location: torch.Tensor
    health_weighted: torch.Tensor
    residual: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    newton_residual_max: torch.Tensor
    health: GenericTypeIIHealthBarrier

def hamiballs_one_step_solve_through_loss(generator: nn.Module, phase: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, newton_iterations: int=3, student_t_dof: float=4.0, health_weight: float=0.1, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, health_safety_margin: float=1.25, edge_mask: torch.Tensor | None=None) -> HamiBallsOneStepSolveThroughObjective:
    if phase.ndim != 4 or phase.shape[0] != attrs.shape[0]:
        raise ValueError('phase/attrs must be [B,T,K,state] and [B,K,C]')
    batch, frames, objects, state_dim = phase.shape
    if frames < 2 or state_dim != 2 * q_dim or attrs.shape[1] != objects:
        raise ValueError('invalid HamiBalls phase/context dimensions')
    if edge_mask is not None:
        if edge_mask.dtype != torch.bool or edge_mask.shape != (batch, frames - 1):
            raise ValueError('edge_mask must be bool [B,T-1]')
        if not bool(edge_mask.any()):
            raise ValueError('edge_mask must select at least one edge')
    canonical = flatten_hamiballs_phase(phase, q_dim=q_dim)
    edges = frames - 1
    source = canonical[:, :-1].reshape(batch * edges, -1)
    target = canonical[:, 1:].reshape(batch * edges, -1)
    context = normalise_object_context(attrs, attr_scale)
    context = context[:, None].expand(batch, edges, *context.shape[1:])
    context = context.reshape(batch * edges, *context.shape[2:])
    if edge_mask is not None:
        selected = edge_mask.reshape(batch * edges)
        source = source[selected]
        target = target[selected]
        context = context[selected]
    generic = unlabeled_type2_free_rollout_objective(generator, source, target[:, None], context, state_scale=canonical_state_scale(state_scale, num_objects=objects, q_dim=q_dim), step_size=step_size, newton_iterations=newton_iterations, student_t_dof=student_t_dof, health_weight=health_weight, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, health_safety_margin=health_safety_margin, solver='newton')
    canonical_scale = canonical_state_scale(state_scale, num_objects=objects, q_dim=q_dim).to(target)
    residual = (generic.decoded.state - target) / canonical_scale
    dimension = int(generator.state_dim)
    if residual.shape[-1] != 2 * dimension:
        raise RuntimeError('solve-through residual has the wrong state dimension')
    q_location = multivariate_student_t_location(residual[..., :dimension], degrees_of_freedom=student_t_dof)
    p_location = multivariate_student_t_location(residual[..., dimension:], degrees_of_freedom=student_t_dof)
    manual_joint = multivariate_student_t_location(residual, degrees_of_freedom=student_t_dof)
    if not torch.allclose(generic.location, manual_joint, atol=1e-07, rtol=1e-06):
        raise RuntimeError('generic and HamiBalls joint solve-through locations differ')
    return HamiBallsOneStepSolveThroughObjective(total=generic.total, location=generic.location, q_location=q_location, p_location=p_location, health_weighted=generic.health_weighted, residual=residual, prediction=generic.decoded.state, target=target, newton_residual_max=generic.decoded.residual_max, health=generic.health)

def hamiballs_channel_separated_relation_loss(generator: nn.Module, phase: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, student_t_dof: float=4.0) -> HamiBallsChannelSeparatedRelationObjective:
    relation = hamiballs_relation_loss(generator, phase, attrs, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, step_size=step_size, student_t_dof=student_t_dof)
    dimension = int(generator.state_dim)
    if relation.residual.shape[-1] != 2 * dimension:
        raise RuntimeError('Type-II relation residual has the wrong state dimension')
    p_residual = relation.residual[..., :dimension]
    q_residual = relation.residual[..., dimension:]
    p_location = multivariate_student_t_location(p_residual, degrees_of_freedom=student_t_dof)
    q_location = multivariate_student_t_location(q_residual, degrees_of_freedom=student_t_dof)
    return HamiBallsChannelSeparatedRelationObjective(location=0.5 * (p_location + q_location), p_location=p_location, q_location=q_location, residual=relation.residual, predicted_source_p=relation.predicted_source_p, predicted_target_q=relation.predicted_target_q)

@dataclass(frozen=True)
class HamiBallsDeployPConsistencyObjective:
    location: torch.Tensor
    residual: torch.Tensor
    first_prediction: torch.Tensor
    prediction: torch.Tensor

def _rollout_hamiballs_affine_h_only(jets: HamiBallsAffineJets, initial_state: torch.Tensor, *, q_dim: int) -> torch.Tensor:
    if initial_state.ndim != 3 or initial_state.shape[-1] != 2 * q_dim:
        raise ValueError('initial_state must be [B,K,2*q_dim]')
    batch, objects, state_dim = initial_state.shape
    if jets.matrix.ndim != 4 or jets.offset.ndim != 3:
        raise ValueError('affine jets must be [B,F,S,S] and [B,F,S]')
    if jets.matrix.shape[0] != batch or jets.offset.shape[:2] != jets.matrix.shape[:2]:
        raise ValueError('affine jets and initial state batch dimensions differ')
    canonical_dim = objects * q_dim
    expected_matrix = (batch, jets.matrix.shape[1], 2 * canonical_dim, 2 * canonical_dim)
    expected_offset = (batch, jets.matrix.shape[1], 2 * canonical_dim)
    if jets.matrix.shape != expected_matrix or jets.offset.shape != expected_offset:
        raise ValueError('affine jets and initial state dimensions differ')
    if state_dim != 2 * q_dim:
        raise ValueError('initial state dimension differs from q_dim')
    previous = initial_state
    predictions: list[torch.Tensor] = []
    for edge in range(jets.matrix.shape[1]):
        prediction = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous, q_dim=q_dim)
        predictions.append(prediction)
        previous = prediction
    return torch.stack(predictions, dim=1)

def hamiballs_deploy_p_consistency_loss(generator: nn.Module, initial_state: torch.Tensor, d_carrier: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, student_t_dof: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float) -> HamiBallsDeployPConsistencyObjective:
    if initial_state.ndim != 3 or d_carrier.ndim != 4 or target.ndim != 4:
        raise ValueError('initial/D-carrier/target dimensions are invalid')
    batch, frames, objects, state_dim = d_carrier.shape
    if initial_state.shape != (batch, objects, state_dim) or target.shape != d_carrier.shape or state_dim != 2 * q_dim or (attrs.shape[0] != batch) or (attrs.shape[1] != objects):
        raise ValueError('deploy p consistency phase/context shapes differ')
    if state_scale.shape != (state_dim,) or not bool(torch.isfinite(state_scale).all()):
        raise ValueError('state_scale must match one object state and be finite')
    if bool((state_scale <= 0.0).any()) or frames < 1:
        raise ValueError('state_scale must be positive and carrier must be nonempty')
    detached_carrier = d_carrier.detach()
    first_jets = learned_hamiballs_affine_jets(generator, hamiballs_anchor_from_d_candidate(initial_state.detach(), detached_carrier, q_dim=q_dim), attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit)
    first_prediction = _rollout_hamiballs_affine_h_only(first_jets, initial_state.detach(), q_dim=q_dim)
    inherited_anchor = hamiballs_anchor_from_candidates(initial_state.detach(), first_prediction.detach(), first_prediction.detach(), q_dim=q_dim).detached()
    second_jets = learned_hamiballs_affine_jets(generator, inherited_anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=True)
    prediction = _rollout_hamiballs_affine_h_only(second_jets, initial_state.detach(), q_dim=q_dim)
    p_scale = state_scale[q_dim:].to(prediction).reshape(1, 1, 1, -1)
    residual = (prediction[..., q_dim:] - target.detach()[..., q_dim:]) / p_scale
    location = multivariate_student_t_location(residual.reshape(batch * frames, -1), degrees_of_freedom=student_t_dof)
    return HamiBallsDeployPConsistencyObjective(location=location, residual=residual, first_prediction=first_prediction, prediction=prediction)

def hamiballs_free_rollout_loss(generator: nn.Module, phase: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, horizon: int=4, gfjp_iterations: int=2, student_t_dof: float=4.0, health_weight: float=0.1, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, health_safety_margin: float=1.25, start: int | None=None, rng: torch.Generator | None=None) -> UnlabeledTypeIILocationObjective:
    if phase.ndim != 4 or horizon < 1 or phase.shape[1] <= horizon:
        raise ValueError('phase must contain an initial state plus rollout horizon')
    _, frames, objects, _ = phase.shape
    maximum_start = frames - horizon - 1
    if start is None:
        start = int(torch.randint(maximum_start + 1, (), device=phase.device, generator=rng).item())
    if not 0 <= start <= maximum_start:
        raise ValueError('free-rollout start lies outside the phase window')
    canonical = flatten_hamiballs_phase(phase, q_dim=q_dim)
    return unlabeled_type2_free_rollout_objective(generator, canonical[:, start], canonical[:, start + 1:start + 1 + horizon], normalise_object_context(attrs, attr_scale), state_scale=canonical_state_scale(state_scale, num_objects=objects, q_dim=q_dim), step_size=step_size, newton_iterations=gfjp_iterations, student_t_dof=student_t_dof, health_weight=health_weight, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, health_safety_margin=health_safety_margin, solver='gfjp', gfjp_iterations=gfjp_iterations, gfjp_detach_refinement_anchors=True)

def identity_hamiballs_affine_jets(reference: torch.Tensor, *, q_dim: int) -> HamiBallsAffineJets:
    if reference.ndim != 4 or reference.shape[-1] != 2 * q_dim:
        raise ValueError('reference must be [B,F,K,2*q_dim]')
    batch, edges, objects, _ = reference.shape
    canonical_dim = 2 * objects * q_dim
    matrix = torch.eye(canonical_dim, device=reference.device, dtype=reference.dtype).expand(batch, edges, canonical_dim, canonical_dim)
    offset = reference.new_zeros(batch, edges, canonical_dim)
    return HamiBallsAffineJets(matrix=matrix, offset=offset, health=None)

def learned_hamiballs_affine_jets(generator: nn.Module, anchor: HamiBallsMixedJetAnchor, attrs: torch.Tensor, *, attr_scale: torch.Tensor, step_size: float, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, differentiable: bool=False) -> HamiBallsAffineJets:
    return hamiballs_type2_affine_jets(generator, anchor, normalise_object_context(attrs, attr_scale), step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable)

def learned_hamiballs_leapfrog_affine_jets(generator: nn.Module, anchor: HamiBallsMixedJetAnchor, attrs: torch.Tensor, *, attr_scale: torch.Tensor, step_size: float, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, differentiable: bool=False) -> HamiBallsAffineJets:
    return hamiballs_leapfrog_affine_jets(generator, anchor, normalise_object_context(attrs, attr_scale), step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable)

def hamiballs_deployment_anchor_health_barrier(generator: nn.Module, anchor: HamiBallsMixedJetAnchor, attrs: torch.Tensor, *, attr_scale: torch.Tensor, step_size: float, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, safety_margin: float=1.25) -> GenericTypeIIHealthBarrier:
    if anchor.source_q.shape != anchor.target_p.shape or anchor.source_q.ndim != 3:
        raise ValueError('deployment health anchor must be [B,F,canonical_q]')
    batch, edges, dimension = anchor.source_q.shape
    context = normalise_object_context(attrs.detach(), attr_scale)
    context = context[:, None].expand(batch, edges, *context.shape[1:])
    return type2_vector_health_barrier(generator, anchor.source_q.detach().reshape(batch * edges, dimension), anchor.target_p.detach().reshape(batch * edges, dimension), context.reshape(batch * edges, *context.shape[2:]), step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, safety_margin=safety_margin)

def rollout_with_affine_jets(jets: HamiBallsAffineJets, *, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate | None, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, q_dim: int, exogenous_gate: torch.Tensor | None=None, initial_previous_mixed: torch.Tensor | None=None, initial_previous_g: torch.Tensor | None=None, initial_gate_hidden: torch.Tensor | None=None, state_scale: torch.Tensor | None=None, gate_policy: HamiBallsGatePolicy | None=None, per_object_gate_policy: bool=False, component_gate_policy: bool=False) -> HamiBallsCommittedRollout:
    if jets.matrix.shape[:2] != d_candidate.shape[:2]:
        raise ValueError('affine jets and D candidate edge dimensions differ')

    def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
        if state_scale is None:
            return apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous, q_dim=q_dim)
        object_scale, _ = _state_scale_views(state_scale, previous)
        raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
        return raw / object_scale
    return rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=gate, d_tokens=d_tokens, noisy=noisy, x0=x0, d_candidate=d_candidate, attrs=attrs, tau=tau, physical_time=physical_time, initial_previous_mixed=initial_previous_mixed, initial_previous_g=initial_previous_g, initial_gate_hidden=initial_gate_hidden, exogenous_gate=exogenous_gate, gate_policy=gate_policy, per_object_gate_policy=per_object_gate_policy, component_gate_policy=component_gate_policy)

@dataclass(frozen=True)
class HamiBallsPseudoAnchorResult:
    first: HamiBallsCommittedRollout
    second: HamiBallsCommittedRollout
    inherited_anchor: HamiBallsMixedJetAnchor | None

@dataclass(frozen=True)
class HamiBallsHardHealthAudit:
    edges_examined: int
    sigma_min: float
    condition_max: float
    tangent_norm_max: float
    passed: bool

def audit_hamiballs_hard_health(generator: nn.Module, phase_batches: list[torch.Tensor] | tuple[torch.Tensor, ...], attrs_batches: list[torch.Tensor] | tuple[torch.Tensor, ...], *, attr_scale: torch.Tensor, q_dim: int, step_size: float, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0) -> HamiBallsHardHealthAudit:
    if len(phase_batches) != len(attrs_batches) or not phase_batches:
        raise ValueError('health audit requires aligned nonempty phase/attrs batches')
    edges_examined = 0
    sigma_min = math.inf
    condition_max = 0.0
    tangent_max = 0.0
    for phase, attrs in zip(phase_batches, attrs_batches):
        if phase.ndim != 4 or attrs.ndim != 3 or phase.shape[0] != attrs.shape[0]:
            raise ValueError('health phase/attrs batches have invalid shapes')
        canonical = flatten_hamiballs_phase(phase, q_dim=q_dim)
        dimension = generator.state_dim
        anchor = HamiBallsMixedJetAnchor(source_q=canonical[:, :-1, :dimension], target_p=canonical[:, 1:, dimension:])
        with torch.enable_grad():
            jets = learned_hamiballs_affine_jets(generator, anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit)
        if jets.health is None:
            raise RuntimeError('learned-H health audit did not return jet health')
        edges_examined += phase.shape[0] * (phase.shape[1] - 1)
        sigma_min = min(sigma_min, float(jets.health.mixed_singular_min.detach().cpu()))
        condition_max = max(condition_max, float(jets.health.mixed_condition.detach().cpu()))
        tangent_max = max(tangent_max, float(jets.health.tangent_spectral_norm.detach().cpu()))
    passed = sigma_min >= mixed_singular_floor and condition_max <= mixed_condition_limit and (tangent_max <= tangent_spectral_norm_limit)
    return HamiBallsHardHealthAudit(edges_examined=edges_examined, sigma_min=sigma_min, condition_max=condition_max, tangent_norm_max=tangent_max, passed=passed)

def two_sweep_pseudo_anchor(*, generator: nn.Module | None, residual: HamiBallsDTokenResidual, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, reset_gate: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, state_scale: torch.Tensor | None=None, differentiable_second_jets: bool=False) -> HamiBallsPseudoAnchorResult:
    d_tokens, noisy, x0, d_candidate = (d_tokens.detach(), noisy.detach(), x0.detach(), d_candidate.detach())
    attrs, tau, physical_time, reset_gate = (attrs.detach(), tau.detach(), physical_time.detach(), reset_gate.detach())
    inherited_anchor: HamiBallsMixedJetAnchor | None = None
    object_scale: torch.Tensor | None = None
    sequence_scale: torch.Tensor | None = None
    if state_scale is not None:
        object_scale, sequence_scale = _state_scale_views(state_scale, x0)
    raw_x0 = x0 if object_scale is None else x0 * object_scale
    raw_d = d_candidate if sequence_scale is None else d_candidate * sequence_scale
    if generator is None:
        first_jets = identity_hamiballs_affine_jets(d_candidate, q_dim=q_dim)
    else:
        first_jets = learned_hamiballs_affine_jets(generator, hamiballs_anchor_from_d_candidate(raw_x0, raw_d, q_dim=q_dim), attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit)
    with torch.no_grad():
        first = rollout_with_affine_jets(first_jets, residual=residual, gate=None, d_tokens=d_tokens, noisy=noisy, x0=x0, d_candidate=d_candidate, attrs=attrs, tau=tau, physical_time=physical_time, q_dim=q_dim, exogenous_gate=reset_gate, state_scale=state_scale)
    if generator is None:
        second_jets = identity_hamiballs_affine_jets(d_candidate, q_dim=q_dim)
    else:
        inherited_anchor = hamiballs_anchor_from_candidates(raw_x0, first.mixed.detach() if sequence_scale is None else first.mixed.detach() * sequence_scale, first.h_candidate.detach() if sequence_scale is None else first.h_candidate.detach() * sequence_scale, q_dim=q_dim).detached()
        second_jets = learned_hamiballs_affine_jets(generator, inherited_anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable_second_jets)
    second = rollout_with_affine_jets(second_jets, residual=residual, gate=None, d_tokens=d_tokens, noisy=noisy, x0=x0, d_candidate=d_candidate, attrs=attrs, tau=tau, physical_time=physical_time, q_dim=q_dim, exogenous_gate=reset_gate, state_scale=state_scale)
    return HamiBallsPseudoAnchorResult(first, second, inherited_anchor)

def previous_gate_sequence(gate: torch.Tensor, *, initial: torch.Tensor | None=None) -> torch.Tensor:
    if gate.ndim not in {2, 3, 4} or gate.shape[1] < 1:
        raise ValueError('gate must be [B,F], [B,F,K], or [B,F,K,2]')
    if gate.ndim == 4 and gate.shape[-1] != 2:
        raise ValueError('component gate must be [B,F,K,2]')
    first = gate.new_ones(gate.shape[0], 1, *gate.shape[2:]) if initial is None else initial[..., None].expand(*initial.shape, 2)[:, None] if gate.ndim == 4 and initial.shape == gate.shape[:1] + gate.shape[2:3] else initial[:, None]
    if first.shape != (gate.shape[0], 1, *gate.shape[2:]):
        raise ValueError('initial previous gate must match system, per-object, or component gate')
    return torch.cat([first, gate[:, :-1]], dim=1)

def pointwise_gate_mse(gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate, rollout: HamiBallsCommittedRollout, *, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, provenance_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, hr, d = (rollout.h_candidate.detach(), rollout.hr_candidate.detach(), rollout.d_candidate.detach())
    incoming_gate = previous_gate_sequence(provenance_gate.detach())
    if (isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False))) and incoming_gate.ndim == 2:
        incoming_gate = incoming_gate[:, :, None].expand(-1, -1, rollout.h_candidate.shape[2])
    value, _ = gate(d_tokens.detach(), noisy.detach(), x0.detach(), rollout.previous_mixed.detach(), h, hr, d, attrs.detach(), tau.detach(), physical_time.detach(), incoming_gate, residual_hidden=rollout.residual_hidden.detach())
    if value.ndim == 2:
        mixed_weight = value[:, :, None, None]
    elif value.ndim == 3:
        mixed_weight = value[:, :, :, None]
    else:
        raise ValueError('learned gate must return [B,F] or [B,F,K]')
    mixed = d + mixed_weight * (hr - d)
    return ((mixed - target.detach()).square().mean(), value)

def closed_loop_gate_mse(gate: HamiBallsCommittedGate, jets: HamiBallsAffineJets, *, residual: HamiBallsDTokenResidual, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, q_dim: int, state_scale: torch.Tensor | None=None) -> tuple[torch.Tensor, HamiBallsCommittedRollout]:
    rollout = rollout_with_affine_jets(jets, residual=residual, gate=gate, d_tokens=d_tokens.detach(), noisy=noisy.detach(), x0=x0, d_candidate=d_candidate.detach(), attrs=attrs.detach(), tau=tau.detach(), physical_time=physical_time.detach(), q_dim=q_dim, state_scale=state_scale)
    return ((rollout.mixed - target.detach()).square().mean(), rollout)

def convex_oracle_gate(hr_candidate: torch.Tensor, d_candidate: torch.Tensor, target: torch.Tensor, *, granularity: Literal['system', 'per_object']='system') -> torch.Tensor:
    if hr_candidate.shape != d_candidate.shape or target.shape != d_candidate.shape:
        raise ValueError('oracle candidates/target must align')
    if granularity not in {'system', 'per_object'}:
        raise ValueError("oracle granularity must be 'system' or 'per_object'")
    delta = hr_candidate - d_candidate
    axes = tuple(range(2, delta.ndim)) if granularity == 'system' else (-1,)
    numerator = ((target - d_candidate) * delta).sum(dim=axes)
    denominator = delta.square().sum(dim=axes)
    value = torch.where(denominator > torch.finfo(delta.dtype).eps, numerator / denominator, torch.zeros_like(numerator))
    return value.clamp(0.0, 1.0)

def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)

def split_hamiballs_coadapt_gradients(h_total: torch.Tensor, hr_loss: torch.Tensor, h_parameters: tuple[nn.Parameter, ...], residual_parameters: tuple[nn.Parameter, ...]) -> tuple[tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]]:
    h_gradients = torch.autograd.grad(h_total, h_parameters, retain_graph=True, allow_unused=True)
    residual_gradients = torch.autograd.grad(hr_loss, residual_parameters, retain_graph=False, allow_unused=True)
    return (h_gradients, residual_gradients)
__all__ = ['HamiBallsChannelSeparatedRelationObjective', 'HamiBallsDeployPConsistencyObjective', 'HamiBallsHardHealthAudit', 'HamiBallsPseudoAnchorResult', 'audit_hamiballs_hard_health', 'canonical_state_scale', 'closed_loop_gate_mse', 'convex_oracle_gate', 'hamiballs_channel_separated_relation_loss', 'hamiballs_deploy_p_consistency_loss', 'hamiballs_deployment_anchor_health_barrier', 'hamiballs_free_rollout_loss', 'hamiballs_one_step_solve_through_loss', 'hamiballs_relation_loss', 'identity_hamiballs_affine_jets', 'learned_hamiballs_affine_jets', 'learned_hamiballs_leapfrog_affine_jets', 'normalise_object_context', 'pointwise_gate_mse', 'previous_gate_sequence', 'rollout_with_affine_jets', 'set_module_trainable', 'split_hamiballs_coadapt_gradients', 'two_sweep_pseudo_anchor']
