from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from hamiformer.physics.generic_type2 import GenericTypeIIJet, type2_vector_jets
from hamiformer.physics.pgf_scan import apply_prefix, parallel_doubling_prefix

class _RotatedHamiltonian(nn.Module):

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        for name in ('state_dim', 'num_objects', 'coordinate_dim', 'spatial_tokens', 'token_context_dim'):
            setattr(self, name, getattr(base, name))

    def _validate(self, q_tilde: torch.Tensor, p_tilde: torch.Tensor, context: torch.Tensor) -> None:
        self.base._validate(-p_tilde, q_tilde, context)

    def forward(self, q_tilde: torch.Tensor, p_tilde: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.base(-p_tilde, q_tilde, context)

@dataclass(frozen=True)
class CompositeLeapfrogHealth:
    mixed_singular_min: torch.Tensor
    mixed_singular_max: torch.Tensor
    mixed_condition: torch.Tensor
    tangent_spectral_norm: torch.Tensor

@dataclass(frozen=True)
class LeapfrogAffineJets:
    matrix: torch.Tensor
    offset: torch.Tensor
    first_matrix: torch.Tensor
    first_offset: torch.Tensor
    first_site: GenericTypeIIJet
    second_site_rotated: GenericTypeIIJet
    health: CompositeLeapfrogHealth

@dataclass(frozen=True)
class LeapfrogGFJPRefinementResult:
    states: torch.Tensor
    midpoint_states: torch.Tensor
    final_jets: LeapfrogAffineJets
    iterations: int

def _rotation(dimension: int, *, reference: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(dimension, device=reference.device, dtype=reference.dtype)
    zero = torch.zeros_like(identity)
    return torch.cat((torch.cat((zero, identity), dim=-1), torch.cat((-identity, zero), dim=-1)), dim=-2)

def generalized_leapfrog_affine_jets(hamiltonian: nn.Module, source_state_anchor: torch.Tensor, target_state_anchor: torch.Tensor, midpoint_p_anchor: torch.Tensor, context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, differentiable: bool=False) -> LeapfrogAffineJets:
    if source_state_anchor.shape != target_state_anchor.shape:
        raise ValueError('Leapfrog source/target anchors must align')
    if source_state_anchor.ndim != 2 or source_state_anchor.shape[-1] % 2:
        raise ValueError('Leapfrog anchors must be [rows,2*canonical_dim]')
    dimension = source_state_anchor.shape[-1] // 2
    if midpoint_p_anchor.shape != (source_state_anchor.shape[0], dimension):
        raise ValueError('Leapfrog midpoint momentum anchor has wrong shape')
    if context.shape[0] != source_state_anchor.shape[0]:
        raise ValueError('Leapfrog context rows differ from anchors')
    half = 0.5 * float(step_size)
    source_q = source_state_anchor[:, :dimension]
    target_q = target_state_anchor[:, :dimension]
    first = type2_vector_jets(hamiltonian, source_q, midpoint_p_anchor, context, step_size=half, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, create_graph=differentiable, detach=not differentiable)
    rotated = _RotatedHamiltonian(hamiltonian)
    second_rotated = type2_vector_jets(rotated, midpoint_p_anchor, -target_q, context, step_size=half, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, create_graph=differentiable, detach=not differentiable)
    rotation = _rotation(dimension, reference=source_state_anchor)
    inverse_rotation = rotation.transpose(-1, -2)
    second_matrix = inverse_rotation @ second_rotated.matrix @ rotation
    second_offset = (inverse_rotation @ second_rotated.offset.unsqueeze(-1)).squeeze(-1)
    matrix = second_matrix @ first.matrix
    offset = (second_matrix @ first.offset.unsqueeze(-1)).squeeze(-1) + second_offset
    tangent = torch.linalg.svdvals(matrix).amax()
    if not bool(torch.isfinite(tangent)) or float(tangent.detach()) > tangent_spectral_norm_limit:
        raise RuntimeError(f'GFJP Leapfrog composed tangent health failed: {float(tangent.detach()):.3e} > {tangent_spectral_norm_limit:.3e}')
    return LeapfrogAffineJets(matrix=matrix, offset=offset, first_matrix=first.matrix, first_offset=first.offset, first_site=first, second_site_rotated=second_rotated, health=CompositeLeapfrogHealth(mixed_singular_min=torch.minimum(first.mixed_singular_min, second_rotated.mixed_singular_min), mixed_singular_max=torch.maximum(first.mixed_singular_max, second_rotated.mixed_singular_max), mixed_condition=torch.maximum(first.mixed_condition, second_rotated.mixed_condition), tangent_spectral_norm=tangent))

def generalized_leapfrog_gfjp_refinement(hamiltonian: nn.Module, initial_state: torch.Tensor, context: torch.Tensor, *, edges: int, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, iterations: int, differentiable: bool=False) -> LeapfrogGFJPRefinementResult:
    if initial_state.ndim != 2 or initial_state.shape[-1] % 2:
        raise ValueError('initial_state must be [batch,2*canonical_dim]')
    if context.shape[0] != initial_state.shape[0] or edges < 1 or iterations < 1:
        raise ValueError('Leapfrog GFJP batch/edge/iteration contract is invalid')
    batch, canonical = initial_state.shape
    dimension = canonical // 2
    edge_context = context[:, None].expand(batch, edges, *context.shape[1:])
    source = initial_state[:, None].expand(batch, edges, canonical)
    target = source
    midpoint_p = initial_state[:, None, dimension:].expand(batch, edges, dimension)
    final_jets = None
    midpoint_states = source
    states = source
    for sweep in range(iterations):
        flat = generalized_leapfrog_affine_jets(hamiltonian, source.reshape(batch * edges, canonical), target.reshape(batch * edges, canonical), midpoint_p.reshape(batch * edges, dimension), edge_context.reshape(batch * edges, *context.shape[1:]), step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable)
        matrix = flat.matrix.reshape(batch, edges, canonical, canonical)
        offset = flat.offset.reshape(batch, edges, canonical)
        prefix = parallel_doubling_prefix(matrix, offset)
        states = apply_prefix(prefix, initial_state)
        source = torch.cat((initial_state[:, None], states[:, :-1]), dim=1)
        first_matrix = flat.first_matrix.reshape(batch, edges, canonical, canonical)
        first_offset = flat.first_offset.reshape(batch, edges, canonical)
        midpoint_states = (first_matrix @ source.unsqueeze(-1)).squeeze(-1) + first_offset
        if sweep + 1 < iterations:
            target = states.detach()
            source = source.detach()
            midpoint_p = midpoint_states[..., dimension:].detach()
        final_jets = flat
    assert final_jets is not None
    return LeapfrogGFJPRefinementResult(states=states, midpoint_states=midpoint_states, final_jets=final_jets, iterations=iterations)
__all__ = ['CompositeLeapfrogHealth', 'LeapfrogAffineJets', 'LeapfrogGFJPRefinementResult', 'generalized_leapfrog_affine_jets', 'generalized_leapfrog_gfjp_refinement']
