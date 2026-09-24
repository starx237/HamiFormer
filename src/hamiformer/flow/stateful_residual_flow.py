from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import torch
from hamiformer.models.stateful_gfjp_residual import MixedJetAnchor

@dataclass(frozen=True)
class StatefulFieldEvaluation:
    clean: torch.Tensor
    velocity: torch.Tensor
    next_anchor: MixedJetAnchor
    h_candidate: torch.Tensor

@dataclass(frozen=True)
class StatefulFlowSample:
    trajectory: torch.Tensor
    final_anchor: MixedJetAnchor
    field_evaluations: int
StatefulField = Callable[[torch.Tensor, float, MixedJetAnchor], StatefulFieldEvaluation]

def sample_stateful_heun(field: StatefulField, source: torch.Tensor, initial_anchor: MixedJetAnchor, grid: torch.Tensor) -> StatefulFlowSample:
    if source.ndim != 3:
        raise ValueError('source 必须为 [B,T,D]')
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError('grid 必须是一维且至少含两个点')
    if not bool((grid[1:] > grid[:-1]).all().item()):
        raise ValueError('grid 必须严格递增')
    if not 0.0 <= float(grid[0].item()) < float(grid[-1].item()) < 1.0:
        raise ValueError('grid 必须位于 [0,1) 且严格覆盖正区间')
    phase = source.clone()
    committed = initial_anchor.detached()
    evaluations = 0
    for index in range(grid.numel() - 1):
        left = float(grid[index].item())
        right = float(grid[index + 1].item())
        delta = right - left
        frozen = committed
        left_eval = field(phase, left, frozen)
        evaluations += 1
        predictor = phase + delta * left_eval.velocity
        right_eval = field(predictor, right, frozen)
        evaluations += 1
        phase = phase + 0.5 * delta * (left_eval.velocity + right_eval.velocity)
        committed = right_eval.next_anchor.detached()
    tau_max = float(grid[-1].item())
    final_eval = field(phase, tau_max, committed)
    evaluations += 1
    trajectory = phase + (1.0 - tau_max) * final_eval.velocity
    return StatefulFlowSample(trajectory=trajectory, final_anchor=final_eval.next_anchor.detached(), field_evaluations=evaluations)
__all__ = ['StatefulFieldEvaluation', 'StatefulFlowSample', 'sample_stateful_heun']
