from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .generic_type2 import GenericTypeIIJet, type2_vector_jets
from .generalized_leapfrog_gfjp import CompositeLeapfrogHealth, generalized_leapfrog_affine_jets

def flatten_hamiballs_phase(phase: torch.Tensor, *, q_dim: int) -> torch.Tensor:
    if phase.ndim < 2 or q_dim < 1 or phase.shape[-1] != 2 * q_dim:
        raise ValueError('phase must end in a valid [q,p] state axis')
    prefix, objects = (phase.shape[:-2], phase.shape[-2])
    q = phase[..., :q_dim].reshape(*prefix, objects * q_dim)
    p = phase[..., q_dim:].reshape(*prefix, objects * q_dim)
    return torch.cat([q, p], dim=-1)

def unflatten_hamiballs_phase(canonical: torch.Tensor, *, num_objects: int, q_dim: int) -> torch.Tensor:
    dimension = num_objects * q_dim
    if canonical.ndim < 1 or canonical.shape[-1] != 2 * dimension:
        raise ValueError('canonical phase has an incompatible final dimension')
    q = canonical[..., :dimension].reshape(*canonical.shape[:-1], num_objects, q_dim)
    p = canonical[..., dimension:].reshape(*canonical.shape[:-1], num_objects, q_dim)
    return torch.cat([q, p], dim=-1)

@dataclass(frozen=True)
class HamiBallsMixedJetAnchor:
    source_q: torch.Tensor
    target_p: torch.Tensor
    source_p: torch.Tensor | None = None
    target_q: torch.Tensor | None = None
    previous_g: torch.Tensor | None = None
    gate_hidden: torch.Tensor | None = None

    def detached(self) -> 'HamiBallsMixedJetAnchor':
        return HamiBallsMixedJetAnchor(source_q=self.source_q.detach(), target_p=self.target_p.detach(), source_p=None if self.source_p is None else self.source_p.detach(), target_q=None if self.target_q is None else self.target_q.detach(), previous_g=None if self.previous_g is None else self.previous_g.detach(), gate_hidden=None if self.gate_hidden is None else self.gate_hidden.detach())

def hamiballs_anchor_from_candidates(initial_state: torch.Tensor, committed_mixed: torch.Tensor, h_candidate: torch.Tensor, *, q_dim: int, previous_g: torch.Tensor | None=None, gate_hidden: torch.Tensor | None=None) -> HamiBallsMixedJetAnchor:
    if committed_mixed.shape != h_candidate.shape or committed_mixed.ndim != 4:
        raise ValueError('mixed/H candidates must align as [B,F,K,state_dim]')
    batch, frames, objects, state_dim = committed_mixed.shape
    if state_dim != 2 * q_dim or initial_state.shape != (batch, objects, state_dim):
        raise ValueError('initial/candidate phase shapes are incompatible')
    if previous_g is not None:
        if previous_g.shape != (batch,):
            raise ValueError('previous_g provenance must be [B]')
        if not bool(torch.isfinite(previous_g).all().item()) or bool(((previous_g < 0.0) | (previous_g > 1.0)).any().item()):
            raise ValueError('previous_g provenance must be finite in [0,1]')
    if gate_hidden is not None and gate_hidden.ndim < 2:
        raise ValueError('gate_hidden provenance must include layer and batch axes')
    if gate_hidden is not None and gate_hidden.shape[1] != batch:
        raise ValueError('gate_hidden provenance batch dimension differs from anchor')
    previous = torch.cat([initial_state[:, None], committed_mixed[:, :-1]], dim=1)
    previous_flat = flatten_hamiballs_phase(previous, q_dim=q_dim)
    h_flat = flatten_hamiballs_phase(h_candidate, q_dim=q_dim)
    dimension = objects * q_dim
    return HamiBallsMixedJetAnchor(source_q=previous_flat[..., :dimension], target_p=h_flat[..., dimension:], source_p=previous_flat[..., dimension:], target_q=h_flat[..., :dimension], previous_g=previous_g, gate_hidden=gate_hidden)

def hamiballs_anchor_from_d_candidate(initial_state: torch.Tensor, d_candidate: torch.Tensor, *, q_dim: int, previous_g: torch.Tensor | None=None, gate_hidden: torch.Tensor | None=None, detach: bool=True) -> HamiBallsMixedJetAnchor:
    if type(detach) is not bool:
        raise TypeError('detach must be bool')
    anchor = hamiballs_anchor_from_candidates(initial_state, d_candidate, d_candidate, q_dim=q_dim, previous_g=previous_g, gate_hidden=gate_hidden)
    return anchor.detached() if detach else anchor

@dataclass(frozen=True)
class HamiBallsAffineJets:
    matrix: torch.Tensor
    offset: torch.Tensor
    health: GenericTypeIIJet | CompositeLeapfrogHealth | None

def hamiballs_type2_affine_jets(generator: nn.Module, anchor: HamiBallsMixedJetAnchor, object_context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, differentiable: bool) -> HamiBallsAffineJets:
    if anchor.source_q.shape != anchor.target_p.shape or anchor.source_q.ndim != 3:
        raise ValueError('HamiBalls anchors must align as [B,F,canonical_q]')
    batch, edges, dimension = anchor.source_q.shape
    if dimension != generator.state_dim:
        raise ValueError('anchor canonical dimension differs from generator')
    expected_context = (batch, generator.num_objects, generator.spatial_tokens, generator.token_context_dim)
    if object_context.shape != expected_context:
        raise ValueError(f'object_context must have shape {expected_context}, got {tuple(object_context.shape)}')
    context = object_context[:, None].expand(batch, edges, *object_context.shape[1:])
    flat_context = context.reshape(batch * edges, *context.shape[2:])
    jets = type2_vector_jets(generator, anchor.source_q.reshape(batch * edges, dimension), anchor.target_p.reshape(batch * edges, dimension), flat_context, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, create_graph=differentiable, detach=not differentiable)
    return HamiBallsAffineJets(matrix=jets.matrix.reshape(batch, edges, 2 * dimension, 2 * dimension), offset=jets.offset.reshape(batch, edges, 2 * dimension), health=jets)

def hamiballs_leapfrog_affine_jets(generator: nn.Module, anchor: HamiBallsMixedJetAnchor, object_context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, differentiable: bool) -> HamiBallsAffineJets:
    if anchor.source_p is None or anchor.target_q is None:
        raise ValueError('Leapfrog GFJP requires full source/target endpoint anchors')
    if not anchor.source_q.shape == anchor.source_p.shape == anchor.target_q.shape == anchor.target_p.shape:
        raise ValueError('Leapfrog GFJP endpoint anchors must align')
    batch, edges, dimension = anchor.source_q.shape
    context = object_context[:, None].expand(batch, edges, *object_context.shape[1:])
    source = torch.cat((anchor.source_q, anchor.source_p), dim=-1)
    target = torch.cat((anchor.target_q, anchor.target_p), dim=-1)
    midpoint_p = 0.5 * (anchor.source_p + anchor.target_p)
    jets = generalized_leapfrog_affine_jets(generator, source.reshape(batch * edges, 2 * dimension), target.reshape(batch * edges, 2 * dimension), midpoint_p.reshape(batch * edges, dimension), context.reshape(batch * edges, *context.shape[2:]), step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable)
    return HamiBallsAffineJets(matrix=jets.matrix.reshape(batch, edges, 2 * dimension, 2 * dimension), offset=jets.offset.reshape(batch, edges, 2 * dimension), health=jets.health)

def apply_hamiballs_affine_jet(matrix: torch.Tensor, offset: torch.Tensor, state: torch.Tensor, *, q_dim: int) -> torch.Tensor:
    canonical = flatten_hamiballs_phase(state, q_dim=q_dim)
    if matrix.shape != (*canonical.shape[:-1], canonical.shape[-1], canonical.shape[-1]):
        raise ValueError('affine matrix/state shapes are incompatible')
    if offset.shape != canonical.shape:
        raise ValueError('affine offset/state shapes are incompatible')
    output = (matrix @ canonical.unsqueeze(-1)).squeeze(-1) + offset
    return unflatten_hamiballs_phase(output, num_objects=state.shape[-2], q_dim=q_dim)
__all__ = ['HamiBallsAffineJets', 'HamiBallsMixedJetAnchor', 'hamiballs_leapfrog_affine_jets', 'apply_hamiballs_affine_jet', 'flatten_hamiballs_phase', 'hamiballs_anchor_from_candidates', 'hamiballs_anchor_from_d_candidate', 'hamiballs_type2_affine_jets', 'unflatten_hamiballs_phase']
