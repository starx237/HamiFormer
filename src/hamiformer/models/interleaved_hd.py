from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import torch
from hamiformer.physics.pgf_scan import apply_prefix, parallel_doubling_prefix, validate_convex_responsibility

@dataclass(frozen=True)
class InterleavedCleanRollout:
    mixed_clean: torch.Tensor
    h_candidate: torch.Tensor
    responsibility: torch.Tensor
ProposalFn = Callable[[int, torch.Tensor], torch.Tensor]
ResponsibilityFn = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]

def interleaved_clean_rollout(initial_state: torch.Tensor, d_clean: torch.Tensor, proposal_fn: ProposalFn, responsibility_fn: ResponsibilityFn) -> InterleavedCleanRollout:
    if initial_state.ndim != 2:
        raise ValueError('initial_state 必须为 [B,D]')
    if d_clean.ndim != 3:
        raise ValueError('d_clean 必须为 [B,T,D]')
    batch, edges, state_dim = d_clean.shape
    if initial_state.shape != (batch, state_dim):
        raise ValueError('initial_state 与 d_clean 的 batch/state_dim 不一致')
    if edges < 1:
        raise ValueError('至少需要一条物理边')
    current = initial_state
    mixed_states: list[torch.Tensor] = []
    h_candidates: list[torch.Tensor] = []
    responsibilities: list[torch.Tensor] = []
    for edge in range(edges):
        h_candidate = proposal_fn(edge, current)
        if h_candidate.shape != current.shape:
            raise ValueError('proposal_fn 必须返回 [B,D]')
        responsibility = responsibility_fn(edge, h_candidate, current)
        if responsibility.shape != (batch, 1):
            raise ValueError('responsibility_fn 必须返回逐样本标量 [B,1]')
        current = d_clean[:, edge] + responsibility * (h_candidate - d_clean[:, edge])
        h_candidates.append(h_candidate)
        responsibilities.append(responsibility)
        mixed_states.append(current)
    stacked_responsibility = torch.stack(responsibilities, dim=1)
    validate_convex_responsibility(stacked_responsibility, expected_shape=(batch, edges, 1))
    return InterleavedCleanRollout(mixed_clean=torch.stack(mixed_states, dim=1), h_candidate=torch.stack(h_candidates, dim=1), responsibility=stacked_responsibility)

def precomputed_gated_affine_rollout(initial_state: torch.Tensor, d_clean: torch.Tensor, matrix: torch.Tensor, offset: torch.Tensor, responsibility: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 4 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('matrix 必须为 [B,T,D,D]')
    if offset.shape != matrix.shape[:-1]:
        raise ValueError('offset 必须为 [B,T,D]')
    if d_clean.shape != offset.shape:
        raise ValueError('d_clean 必须与 offset 同形')
    validate_convex_responsibility(responsibility, expected_shape=(*offset.shape[:-1], 1))
    if initial_state.shape != (matrix.shape[0], matrix.shape[-1]):
        raise ValueError('initial_state 必须为 [B,D]')
    gated_matrix = responsibility.unsqueeze(-1) * matrix
    gated_offset = responsibility * offset + (1.0 - responsibility) * d_clean
    return apply_prefix(parallel_doubling_prefix(gated_matrix, gated_offset), initial_state)
__all__ = ['InterleavedCleanRollout', 'interleaved_clean_rollout', 'precomputed_gated_affine_rollout']
