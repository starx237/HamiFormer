from __future__ import annotations
from collections.abc import Callable
import torch
from .pf_rf_v1 import validate_pf_rf_v1_steps
from .rectified_flow import clean_to_velocity
from .stateful_residual_flow import MixedJetAnchor, StatefulFieldEvaluation, StatefulFlowSample
PFRFV1StatefulField = Callable[[torch.Tensor, torch.Tensor, MixedJetAnchor], StatefulFieldEvaluation]
PFRFV1EvaluationCallback = Callable[[int, str, bool, bool, torch.Tensor, torch.Tensor, MixedJetAnchor, StatefulFieldEvaluation], None]

def default_pf_rf_v1_cold_start_intervals(num_steps: int) -> int:
    if type(num_steps) is not int or num_steps < 2:
        raise ValueError('num_steps must be an integer >= 2')
    return min(num_steps - 1, max(1, int(0.1 * num_steps + 0.5)))

def hamiballs_formal_cold_start_intervals(num_steps: int) -> int:
    if type(num_steps) is not int or num_steps < 2:
        raise ValueError('num_steps must be an integer >= 2')
    return min(2, num_steps - 1)

def _evaluate_clean(field: PFRFV1StatefulField, state: torch.Tensor, tau: torch.Tensor, anchor: MixedJetAnchor, *, t_eps: float) -> tuple[StatefulFieldEvaluation, torch.Tensor]:
    evaluation = field(state, tau, anchor)
    if evaluation.clean.shape != state.shape:
        raise ValueError('stateful PF-RF-v1 field clean shape mismatch')
    if evaluation.velocity.shape != state.shape:
        raise ValueError('stateful PF-RF-v1 field velocity shape mismatch')
    if not bool(torch.isfinite(evaluation.clean).all().item()):
        raise FloatingPointError('stateful PF-RF-v1 clean field is non-finite')
    velocity = clean_to_velocity(evaluation.clean, state, tau, t_eps=t_eps)
    if not bool(torch.isfinite(velocity).all().item()):
        raise FloatingPointError('stateful PF-RF-v1 velocity is non-finite')
    return (evaluation, velocity)

def sample_stateful_pf_rf_v1_heun(field: PFRFV1StatefulField, source: torch.Tensor, initial_anchor: MixedJetAnchor, *, num_steps: int, t_eps: float, cold_start_field: PFRFV1StatefulField | None=None, cold_start_intervals: int=0, commit_anchor_from: str='right_proposal', detach_accepted_anchor: bool=True, evaluation_callback: PFRFV1EvaluationCallback | None=None) -> StatefulFlowSample:
    if source.ndim < 2:
        raise ValueError('source must have a batch plus data dimensions')
    if not bool(torch.isfinite(source).all().item()):
        raise FloatingPointError('source contains NaN/Inf')
    validate_pf_rf_v1_steps(num_steps=num_steps, t_eps=t_eps)
    if type(cold_start_intervals) is not int:
        raise ValueError('cold_start_intervals must be an integer')
    if cold_start_field is None:
        if cold_start_intervals != 0:
            raise ValueError('cold_start_intervals requires cold_start_field')
    elif not 1 <= cold_start_intervals < num_steps:
        raise ValueError('cold_start_intervals must lie in [1, num_steps)')
    if commit_anchor_from not in {'right_proposal', 'accepted_left'}:
        raise ValueError('commit_anchor_from must be right_proposal or accepted_left')
    if type(detach_accepted_anchor) is not bool:
        raise TypeError('detach_accepted_anchor must be bool')
    batch = source.shape[0]
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=source.device, dtype=source.dtype)
    state = source.clone()
    committed = initial_anchor.detached() if detach_accepted_anchor else initial_anchor
    evaluations = 0
    for index in range(num_steps - 1):
        left, right = (grid[index], grid[index + 1])
        tau_left = left.expand(batch)
        tau_right = right.expand(batch)
        frozen = committed
        interval_field = cold_start_field if cold_start_field is not None and index < cold_start_intervals else field
        is_cold = cold_start_field is not None and index < cold_start_intervals
        left_eval, velocity_left = _evaluate_clean(interval_field, state, tau_left, frozen, t_eps=t_eps)
        if evaluation_callback is not None:
            evaluation_callback(index, 'left', is_cold, commit_anchor_from == 'accepted_left', state, tau_left, frozen, left_eval)
        predictor = state + (right - left) * velocity_left
        right_eval, velocity_right = _evaluate_clean(interval_field, predictor, tau_right, frozen, t_eps=t_eps)
        if evaluation_callback is not None:
            evaluation_callback(index, 'right', is_cold, commit_anchor_from == 'right_proposal', predictor, tau_right, frozen, right_eval)
        state = state + 0.5 * (right - left) * (velocity_left + velocity_right)
        committed = right_eval.next_anchor if commit_anchor_from == 'right_proposal' else left_eval.next_anchor
        if detach_accepted_anchor:
            committed = committed.detached()
        evaluations += 2
    tau_left = grid[-2].expand(batch)
    final_eval, final_velocity = _evaluate_clean(field, state, tau_left, committed, t_eps=t_eps)
    if evaluation_callback is not None:
        evaluation_callback(num_steps - 1, 'final', False, True, state, tau_left, committed, final_eval)
    state = state + (grid[-1] - grid[-2]) * final_velocity
    evaluations += 1
    if not bool(torch.isfinite(state).all().item()):
        raise FloatingPointError('stateful PF-RF-v1 sampler produced NaN/Inf')
    return StatefulFlowSample(trajectory=state, final_anchor=final_eval.next_anchor.detached() if detach_accepted_anchor else final_eval.next_anchor, field_evaluations=evaluations)
__all__ = ['PFRFV1StatefulField', 'PFRFV1EvaluationCallback', 'default_pf_rf_v1_cold_start_intervals', 'hamiballs_formal_cold_start_intervals', 'sample_stateful_pf_rf_v1_heun']
