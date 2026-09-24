from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn
from hamiformer.physics.learned_pgf import type2_generator_jets

@dataclass(frozen=True)
class MixedJetAnchor:
    source_q: torch.Tensor
    target_p: torch.Tensor
    residual_cache: torch.Tensor | None = None
    clean_cache: torch.Tensor | None = None

    def detached(self) -> 'MixedJetAnchor':
        return MixedJetAnchor(self.source_q.detach(), self.target_p.detach(), None if self.residual_cache is None else self.residual_cache.detach(), None if self.clean_cache is None else self.clean_cache.detach())

@dataclass(frozen=True)
class StatefulResidualRollout:
    mixed_clean: torch.Tensor
    h_candidate: torch.Tensor
    residual: torch.Tensor
    next_anchor: MixedJetAnchor
ReadoutFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def initial_mixed_anchor(phase: torch.Tensor, initial_state: torch.Tensor) -> MixedJetAnchor:
    if phase.ndim != 3 or phase.shape[-1] != 2:
        raise ValueError('phase 必须为 [B,T,2]')
    if initial_state.shape != (phase.shape[0], 2):
        raise ValueError('initial_state 必须为 [B,2]')
    source_q = torch.cat([initial_state[:, None, 0], phase[:, :-1, 0]], dim=1)
    return MixedJetAnchor(source_q=source_q, target_p=phase[..., 1]).detached()

def committed_mixed_anchor(initial_state: torch.Tensor, mixed_clean: torch.Tensor, h_candidate: torch.Tensor, residual_cache: torch.Tensor | None=None, clean_cache: torch.Tensor | None=None) -> MixedJetAnchor:
    if mixed_clean.ndim != 3 or mixed_clean.shape[-1] != 2:
        raise ValueError('mixed_clean 必须为 [B,T,2]')
    if h_candidate.shape != mixed_clean.shape:
        raise ValueError('h_candidate 必须与 mixed_clean 同形')
    if initial_state.shape != (mixed_clean.shape[0], 2):
        raise ValueError('initial_state 必须为 [B,2]')
    if residual_cache is not None and residual_cache.shape != mixed_clean.shape:
        raise ValueError('residual_cache 必须与 mixed_clean 同形')
    if clean_cache is not None and clean_cache.shape != mixed_clean.shape:
        raise ValueError('clean_cache 必须与 mixed_clean 同形')
    source_q = torch.cat([initial_state[:, None, 0], mixed_clean[:, :-1, 0]], dim=1)
    return MixedJetAnchor(source_q=source_q, target_p=h_candidate[..., 1], residual_cache=residual_cache, clean_cache=clean_cache).detached()

def detached_affine_jets(generator: nn.Module, anchor: MixedJetAnchor, theta_sys: torch.Tensor, *, step_size: float, mixed_hessian_floor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchor.source_q.shape != anchor.target_p.shape:
        raise ValueError('anchor.source_q 与 anchor.target_p 必须同形')
    if anchor.source_q.ndim != 2:
        raise ValueError('anchor tensors 必须为 [B,T]')
    batch, edges = anchor.source_q.shape
    if theta_sys.ndim != 2 or theta_sys.shape[0] != batch:
        raise ValueError('theta_sys 必须为 [B,theta_dim]')
    theta = theta_sys[:, None, :].expand(batch, edges, theta_sys.shape[-1])
    jet = type2_generator_jets(generator, anchor.source_q, anchor.target_p, theta, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, create_graph=False)
    return (jet.matrix.detach(), jet.offset.detach(), jet.unsafe_fraction.detach())

def interleaved_residual_rollout(initial_state: torch.Tensor, matrix: torch.Tensor, offset: torch.Tensor, tokens: torch.Tensor, readout: ReadoutFn, tau: torch.Tensor, theta_sys: torch.Tensor, residual_cache: torch.Tensor | None=None, detach_readout_state_inputs: bool=False) -> StatefulResidualRollout:
    if matrix.ndim != 4 or matrix.shape[-2:] != (2, 2):
        raise ValueError('matrix 必须为 [B,T,2,2]')
    batch, edges = matrix.shape[:2]
    if offset.shape != (batch, edges, 2):
        raise ValueError('offset 必须为 [B,T,2]')
    if tokens.ndim != 3 or tokens.shape[:2] != (batch, edges):
        raise ValueError('tokens 必须为 [B,T,H]')
    if initial_state.shape != (batch, 2):
        raise ValueError('initial_state 必须为 [B,2]')
    if tau.shape != (batch,):
        raise ValueError('tau 必须为 [B]')
    if theta_sys.ndim != 2 or theta_sys.shape[0] != batch:
        raise ValueError('theta_sys 必须为 [B,theta_dim]')
    if residual_cache is not None and residual_cache.shape != (batch, edges, 2):
        raise ValueError('residual_cache 必须为 [B,T,2]')
    current = initial_state
    mixed_states: list[torch.Tensor] = []
    h_candidates: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    for edge in range(edges):
        h_candidate = torch.matmul(matrix[:, edge], current.unsqueeze(-1)).squeeze(-1) + offset[:, edge]
        readout_source = current.detach() if detach_readout_state_inputs else current
        readout_h = h_candidate.detach() if detach_readout_state_inputs else h_candidate
        residual = readout(tokens[:, edge], readout_source, readout_h, tau, theta_sys)
        if residual.shape != current.shape:
            raise ValueError('readout 必须返回 [B,2]')
        if residual_cache is not None:
            residual = residual + residual_cache[:, edge]
        current = h_candidate + residual
        h_candidates.append(h_candidate)
        residuals.append(residual)
        mixed_states.append(current)
    mixed_clean = torch.stack(mixed_states, dim=1)
    h_candidate = torch.stack(h_candidates, dim=1)
    return StatefulResidualRollout(mixed_clean=mixed_clean, h_candidate=h_candidate, residual=torch.stack(residuals, dim=1), next_anchor=committed_mixed_anchor(initial_state, mixed_clean, h_candidate, residual_cache=torch.stack(residuals, dim=1), clean_cache=mixed_clean))

def _select_anchor(use_candidate: torch.Tensor, previous: MixedJetAnchor, candidate: MixedJetAnchor) -> MixedJetAnchor:
    mask = use_candidate[:, None]
    residual_cache = None
    if previous.residual_cache is not None or candidate.residual_cache is not None:
        if previous.residual_cache is None or candidate.residual_cache is None:
            raise ValueError('cannot select between missing/present residual caches')
        residual_cache = torch.where(mask[:, :, None], candidate.residual_cache, previous.residual_cache)
    clean_cache = None
    if previous.clean_cache is not None or candidate.clean_cache is not None:
        if previous.clean_cache is None or candidate.clean_cache is None:
            raise ValueError('cannot select between missing/present clean caches')
        clean_cache = torch.where(mask[:, :, None], candidate.clean_cache, previous.clean_cache)
    return MixedJetAnchor(source_q=torch.where(mask, candidate.source_q, previous.source_q), target_p=torch.where(mask, candidate.target_p, previous.target_p), residual_cache=residual_cache, clean_cache=clean_cache).detached()

def iterated_interleaved_residual_rollout(generator: nn.Module, anchor: MixedJetAnchor, initial_state: torch.Tensor, tokens: torch.Tensor, readout: ReadoutFn, tau: torch.Tensor, theta_sys: torch.Tensor, iterations: int | torch.Tensor, *, step_size: float, mixed_hessian_floor: float) -> tuple[StatefulResidualRollout, torch.Tensor]:
    batch = initial_state.shape[0]
    if isinstance(iterations, int):
        if iterations < 1:
            raise ValueError('iterations 必须为正')
        count = torch.full((batch,), iterations, device=initial_state.device, dtype=torch.long)
    else:
        if iterations.shape != (batch,):
            raise ValueError('iterations tensor 必须为 [B]')
        count = iterations.to(device=initial_state.device, dtype=torch.long)
        if bool((count < 1).any().item()):
            raise ValueError('iterations 必须为正')
    current_anchor = anchor.detached()
    selected: StatefulResidualRollout | None = None
    unsafe_values: list[torch.Tensor] = []
    for index in range(int(count.max().item())):
        matrix, offset, unsafe = detached_affine_jets(generator, current_anchor, theta_sys, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor)
        candidate = interleaved_residual_rollout(initial_state, matrix, offset, tokens, readout, tau, theta_sys)
        unsafe_values.append(unsafe)
        if selected is None:
            selected = candidate
        else:
            use_candidate = count > index
            state_mask = use_candidate[:, None, None]
            selected = StatefulResidualRollout(mixed_clean=torch.where(state_mask, candidate.mixed_clean, selected.mixed_clean), h_candidate=torch.where(state_mask, candidate.h_candidate, selected.h_candidate), residual=torch.where(state_mask, candidate.residual, selected.residual), next_anchor=_select_anchor(use_candidate, selected.next_anchor, candidate.next_anchor))
        current_anchor = candidate.next_anchor
    assert selected is not None
    return (selected, torch.stack(unsafe_values).amax())
__all__ = ['MixedJetAnchor', 'StatefulResidualRollout', 'committed_mixed_anchor', 'detached_affine_jets', 'initial_mixed_anchor', 'interleaved_residual_rollout', 'iterated_interleaved_residual_rollout']
