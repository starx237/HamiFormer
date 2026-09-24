from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from hamiformer.physics.generic_type2 import GenericTypeIIHealthBarrier, GenericTypeIIUnrolledResult, solve_generic_type2_step_unrolled, type2_vector_eom, type2_vector_health_barrier
from hamiformer.physics.generic_gfjp import differentiable_gfjp_refinement

@dataclass(frozen=True)
class UnlabeledTypeIILocationObjective:
    total: torch.Tensor
    location: torch.Tensor
    health_weighted: torch.Tensor
    decoded: GenericTypeIIUnrolledResult
    health: GenericTypeIIHealthBarrier

@dataclass(frozen=True)
class UnlabeledTypeIIRelationObjective:
    location: torch.Tensor
    residual: torch.Tensor
    predicted_source_p: torch.Tensor
    predicted_target_q: torch.Tensor

def multivariate_student_t_location(residual: torch.Tensor, *, degrees_of_freedom: float) -> torch.Tensor:
    if residual.ndim < 1 or residual.shape[-1] < 1:
        raise ValueError('multivariate Student-t residual must have a feature axis')
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError('Student-t degrees of freedom must be finite and positive')
    dimension = residual.shape[-1]
    squared_norm = residual.square().sum(dim=-1)
    coefficient = 0.5 * (degrees_of_freedom + dimension) / dimension
    return (residual.new_tensor(coefficient) * torch.log1p(squared_norm / degrees_of_freedom)).mean()

def unlabeled_type2_relation_objective(generator: nn.Module, source_state: torch.Tensor, target_state: torch.Tensor, context: torch.Tensor, *, state_scale: torch.Tensor, step_size: float, student_t_dof: float) -> UnlabeledTypeIIRelationObjective:
    if source_state.shape != target_state.shape or source_state.ndim < 2:
        raise ValueError('source/target states must be aligned batched tensors')
    dimension = int(generator.state_dim)
    if source_state.shape[-1] != 2 * dimension:
        raise ValueError('state dimension does not match Type-II generator')
    if state_scale.shape != (2 * dimension,):
        raise ValueError('state_scale must have shape [2 * generator.state_dim]')
    if not bool(torch.isfinite(state_scale).all()) or bool((state_scale <= 0).any()):
        raise ValueError('state_scale must be finite and positive')
    q_source = source_state[..., :dimension]
    p_source = source_state[..., dimension:]
    q_target = target_state[..., :dimension]
    p_target = target_state[..., dimension:]
    predicted_source_p, predicted_target_q = type2_vector_eom(generator, q_source, p_target, context, step_size=step_size, create_graph=True)
    q_scale = state_scale[:dimension].to(source_state)
    p_scale = state_scale[dimension:].to(source_state)
    residual = torch.cat([(p_source - predicted_source_p) / p_scale, (q_target - predicted_target_q) / q_scale], dim=-1)
    return UnlabeledTypeIIRelationObjective(location=multivariate_student_t_location(residual, degrees_of_freedom=student_t_dof), residual=residual, predicted_source_p=predicted_source_p, predicted_target_q=predicted_target_q)

def unlabeled_type2_location_objective(generator: nn.Module, source_state: torch.Tensor, target_state: torch.Tensor, context: torch.Tensor, *, state_scale: torch.Tensor, step_size: float, newton_iterations: int, student_t_dof: float, health_weight: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, health_safety_margin: float=1.25) -> UnlabeledTypeIILocationObjective:
    if source_state.shape != target_state.shape or source_state.ndim < 1:
        raise ValueError('source/target states must have identical nonempty shapes')
    if source_state.shape[-1] != 2 * generator.state_dim:
        raise ValueError('state dimension does not match Type-II generator')
    if state_scale.shape != (2 * generator.state_dim,) or not bool(torch.isfinite(state_scale).all()) or bool((state_scale <= 0.0).any()):
        raise ValueError('state_scale must be finite positive [2 * state_dim]')
    if not all((math.isfinite(value) and value > 0.0 for value in (student_t_dof, health_weight))):
        raise ValueError('Student-t and health weights must be finite positive')
    decoded = solve_generic_type2_step_unrolled(generator, source_state, context, step_size=step_size, iterations=newton_iterations)
    residual = (target_state - decoded.state) / state_scale.to(target_state)
    location = 0.5 * (student_t_dof + 1.0) * torch.log1p(residual.square() / student_t_dof)
    location = location.mean()
    dimension = generator.state_dim
    health = type2_vector_health_barrier(generator, source_state[..., :dimension], decoded.p_next, context, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, safety_margin=health_safety_margin)
    health_weighted = source_state.new_tensor(health_weight) * health.penalty
    return UnlabeledTypeIILocationObjective(total=location + health_weighted, location=location, health_weighted=health_weighted, decoded=decoded, health=health)

def unlabeled_type2_free_rollout_objective(generator: nn.Module, source_state: torch.Tensor, target_states: torch.Tensor, context: torch.Tensor, *, state_scale: torch.Tensor, step_size: float, newton_iterations: int, student_t_dof: float, health_weight: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, health_safety_margin: float=1.25, solver: str='newton', gfjp_iterations: int | None=None, gfjp_detach_refinement_anchors: bool=True) -> UnlabeledTypeIILocationObjective:
    if target_states.ndim != source_state.ndim + 1 or target_states.shape[0] != source_state.shape[0] or target_states.shape[2:] != source_state.shape[1:]:
        raise ValueError('target_states must be [batch,positive_horizon,*source_state.shape[1:]]')
    horizon = target_states.shape[1]
    if horizon < 1:
        raise ValueError('free-rollout objective horizon must be positive')
    if solver not in {'newton', 'gfjp'}:
        raise ValueError('unlabelled Type-II solver must be newton or gfjp')
    if solver == 'gfjp':
        iterations = newton_iterations if gfjp_iterations is None else gfjp_iterations
        if iterations < 1:
            raise ValueError('GFJP iteration count must be positive')
        refined = differentiable_gfjp_refinement(generator, source_state, context, edges=horizon, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, iterations=iterations, detach_refinement_anchors=gfjp_detach_refinement_anchors)
        residual = (target_states - refined.states) / state_scale.to(target_states)
        location = multivariate_student_t_location(residual, degrees_of_freedom=student_t_dof)
        jets = refined.final_scan.jets
        protected_floor = jets.mixed_singular_min.new_tensor(mixed_singular_floor * health_safety_margin)
        protected_condition = jets.mixed_condition.new_tensor(mixed_condition_limit / health_safety_margin)
        protected_tangent = jets.tangent_spectral_norm.new_tensor(tangent_spectral_norm_limit / health_safety_margin)
        penalty = torch.nn.functional.relu((protected_floor - jets.mixed_singular_min) / protected_floor).square() + torch.nn.functional.relu(jets.mixed_condition / protected_condition - 1.0).square() + torch.nn.functional.relu(jets.tangent_spectral_norm / protected_tangent - 1.0).square()
        health = GenericTypeIIHealthBarrier(penalty=penalty, mixed_singular_min=jets.mixed_singular_min, mixed_singular_max=jets.mixed_singular_max, mixed_condition=jets.mixed_condition, tangent_spectral_norm=jets.tangent_spectral_norm)
        health_weighted = source_state.new_tensor(health_weight) * penalty
        decoded = GenericTypeIIUnrolledResult(state=refined.states[:, -1], p_next=refined.p_next_anchor[:, -1], residual_max=residual.abs().amax())
        return UnlabeledTypeIILocationObjective(total=location + health_weighted, location=location, health_weighted=health_weighted, decoded=decoded, health=health)
    current = source_state
    location = source_state.new_zeros(())
    health_penalty = source_state.new_zeros(())
    final_decoded: GenericTypeIIUnrolledResult | None = None
    final_health: GenericTypeIIHealthBarrier | None = None
    dimension = generator.state_dim
    for target in target_states.unbind(dim=1):
        decoded = solve_generic_type2_step_unrolled(generator, current, context, step_size=step_size, iterations=newton_iterations)
        residual = (target - decoded.state) / state_scale.to(target)
        location = location + multivariate_student_t_location(residual, degrees_of_freedom=student_t_dof)
        health = type2_vector_health_barrier(generator, current[..., :dimension], decoded.p_next, context, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, safety_margin=health_safety_margin)
        health_penalty = health_penalty + health.penalty
        current = decoded.state
        final_decoded, final_health = (decoded, health)
    assert final_decoded is not None and final_health is not None
    location = location / horizon
    health_weighted = source_state.new_tensor(health_weight) * health_penalty / horizon
    return UnlabeledTypeIILocationObjective(total=location + health_weighted, location=location, health_weighted=health_weighted, decoded=final_decoded, health=final_health)
