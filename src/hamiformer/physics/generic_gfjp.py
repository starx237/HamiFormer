from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from hamiformer.physics.generic_type2 import GenericTypeIIJet, type2_vector_jets
from hamiformer.physics.pgf_scan import AffinePrefix, apply_prefix, parallel_doubling_prefix, serial_prefix

@dataclass(frozen=True)
class GenericGFJPResult:
    states: torch.Tensor
    prefix: AffinePrefix
    jets: GenericTypeIIJet

@dataclass(frozen=True)
class DifferentiableGFJPRefinementResult:
    states: torch.Tensor
    q_anchor: torch.Tensor
    p_next_anchor: torch.Tensor
    final_scan: GenericGFJPResult
    iterations: int

def generic_gfjp_scan(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, context: torch.Tensor, initial_state: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, method: Literal['serial', 'parallel']='parallel', differentiable: bool=False, compute_tangent_spectral_norm: bool=True) -> GenericGFJPResult:
    if q_anchor.shape != p_next_anchor.shape or q_anchor.ndim != 3:
        raise ValueError('GFJP q/P anchors must align as [batch, edges, state_dim]')
    batch, edges, state_dim = q_anchor.shape
    if state_dim != generator.state_dim or edges < 1:
        raise ValueError('GFJP anchor state dimension/edge count is invalid')
    if context.shape[:2] != (batch, edges):
        raise ValueError('GFJP context must share [batch, edges] anchor prefix')
    if initial_state.shape != (batch, 2 * state_dim):
        raise ValueError('GFJP initial_state must be [batch, 2 * state_dim]')
    if method not in {'serial', 'parallel'}:
        raise ValueError('GFJP method must be serial or parallel')
    if not bool(torch.isfinite(q_anchor).all() and torch.isfinite(p_next_anchor).all() and torch.isfinite(context).all() and torch.isfinite(initial_state).all()):
        raise ValueError('GFJP anchors, context, and initial state must be finite')
    flat_q = q_anchor.reshape(batch * edges, state_dim)
    flat_p = p_next_anchor.reshape(batch * edges, state_dim)
    flat_context = context.reshape(batch * edges, *context.shape[2:])
    jets = type2_vector_jets(generator, flat_q, flat_p, flat_context, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, create_graph=differentiable, detach=not differentiable, compute_tangent_spectral_norm=compute_tangent_spectral_norm)
    matrix = jets.matrix.reshape(batch, edges, 2 * state_dim, 2 * state_dim)
    offset = jets.offset.reshape(batch, edges, 2 * state_dim)
    prefix = serial_prefix(matrix, offset) if method == 'serial' else parallel_doubling_prefix(matrix, offset)
    states = apply_prefix(prefix, initial_state)
    return GenericGFJPResult(states=states, prefix=prefix, jets=GenericTypeIIJet(matrix=matrix, offset=offset, source_graph=jets.source_graph.reshape(batch, edges, 2 * state_dim), target_graph=jets.target_graph.reshape(batch, edges, 2 * state_dim), mixed_jacobian=jets.mixed_jacobian.reshape(batch, edges, state_dim, state_dim), mixed_singular_min=jets.mixed_singular_min, mixed_singular_max=jets.mixed_singular_max, mixed_condition=jets.mixed_condition, tangent_spectral_norm=jets.tangent_spectral_norm, mixed_singular_min_per_map=jets.mixed_singular_min_per_map.reshape(batch, edges), mixed_singular_max_per_map=jets.mixed_singular_max_per_map.reshape(batch, edges), mixed_condition_per_map=jets.mixed_condition_per_map.reshape(batch, edges), tangent_spectral_norm_per_map=jets.tangent_spectral_norm_per_map.reshape(batch, edges), tangent_spectral_norm_computed=jets.tangent_spectral_norm_computed))

def differentiable_gfjp_refinement(generator: nn.Module, initial_state: torch.Tensor, context: torch.Tensor, *, edges: int, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, iterations: int, detach_refinement_anchors: bool=True, differentiable: bool=True) -> DifferentiableGFJPRefinementResult:
    if initial_state.ndim != 2 or initial_state.shape[-1] != 2 * generator.state_dim:
        raise ValueError('initial_state must be [batch, 2 * state_dim]')
    if context.shape[0] != initial_state.shape[0] or edges < 1 or iterations < 1:
        raise ValueError('GFJP refinement batch, edge count, and iterations are invalid')
    if not bool(torch.isfinite(initial_state).all() and torch.isfinite(context).all()):
        raise ValueError('GFJP refinement inputs must be finite')
    batch, dimension = (initial_state.shape[0], generator.state_dim)
    context_edges = context.unsqueeze(1).expand(batch, edges, *context.shape[1:])
    q_anchor = initial_state[:, None, :dimension].expand(batch, edges, dimension)
    p_anchor = initial_state[:, None, dimension:].expand(batch, edges, dimension)
    final_scan: GenericGFJPResult | None = None
    for sweep in range(iterations):
        final_scan = generic_gfjp_scan(generator, q_anchor, p_anchor, context_edges, initial_state, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, method='parallel', differentiable=differentiable)
        if sweep + 1 < iterations:
            states = final_scan.states
            source_q = torch.cat([initial_state[:, None, :dimension], states[:, :-1, :dimension]], dim=1)
            target_p = states[:, :, dimension:]
            q_anchor = source_q.detach() if detach_refinement_anchors else source_q
            p_anchor = target_p.detach() if detach_refinement_anchors else target_p
    assert final_scan is not None
    return DifferentiableGFJPRefinementResult(states=final_scan.states, q_anchor=q_anchor, p_next_anchor=p_anchor, final_scan=final_scan, iterations=iterations)
