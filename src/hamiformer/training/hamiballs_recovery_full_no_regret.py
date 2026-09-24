from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual, rollout_hamiballs_committed_edges
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, apply_hamiballs_affine_jet
from hamiformer.training.hamiballs_formal import previous_gate_sequence
from hamiformer.training.hamiballs_trajectory import HamiBallsTrajectoryG0, cyclic_trace_field_index, per_object_candidate_hull_utility_loss

@dataclass(frozen=True)
class FullNoRegretCandidates:
    h_candidate: torch.Tensor
    hr_candidate: torch.Tensor
    d_candidate: torch.Tensor
    residual_hidden: torch.Tensor | None = None
    previous_mixed: torch.Tensor | None = None

@dataclass(frozen=True)
class FullNoRegretLosses:
    baseline_loss: torch.Tensor
    trajectory_loss: torch.Tensor
    hull_loss: torch.Tensor
    q_no_regret_loss: torch.Tensor
    p_no_regret_loss: torch.Tensor
    q_harm_fraction: torch.Tensor
    p_harm_fraction: torch.Tensor
    no_regret_loss: torch.Tensor
    hull_g_star: torch.Tensor

def _require_unbounded_residual(residual: HamiBallsDTokenResidual, *, q_dim: int) -> None:
    if not isinstance(residual, HamiBallsDTokenResidual):
        raise TypeError('full no-regret requires HamiBallsDTokenResidual')
    if residual.parameterization != HAMIBALLS_RESIDUAL_UNBOUNDED_V1:
        raise ValueError('full no-regret requires unbounded_v1 residual')
    if residual.gain_head is not None or float(residual.output_scale) != 1.0:
        raise ValueError('full no-regret forbids gain/output scaling')
    if not 0 < q_dim < residual.state_dim:
        raise ValueError('q_dim must split the residual state')

def replay_full_no_regret_candidates(residual: HamiBallsDTokenResidual, trace: HamiBallsTrajectoryG0, *, field_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, window_length: int=12, online_detached_previous: bool=False) -> FullNoRegretCandidates:
    _require_unbounded_residual(residual, q_dim=q_dim)
    if type(online_detached_previous) is not bool:
        raise TypeError('online_detached_previous must be bool')
    if not trace.traces:
        raise ValueError('full no-regret replay requires accepted fields')
    if type(field_index) is not int or not 0 <= field_index < len(trace.traces):
        raise ValueError('field_index lies outside accepted trace')
    if type(window_length) is not int or window_length < 1:
        raise ValueError('window_length must be positive')
    field = trace.traces[field_index]
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('accepted full-no-regret field lacks rollout/tokens/jets')
    rollout = field.rollout
    frames = int(rollout.mixed.shape[1])
    if target.shape[:2] != rollout.mixed.shape[:2]:
        raise ValueError('full-no-regret target does not align with field')
    incoming_g = previous_gate_sequence(rollout.gate.detach(), initial=field.anchor.previous_g)
    object_scale = state_scale.reshape(1, 1, -1)
    h_rows: list[torch.Tensor] = []
    hr_rows: list[torch.Tensor] = []
    d_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    previous_rows: list[torch.Tensor] = []
    online_previous_mixed = rollout.previous_mixed[:, 0].detach()
    online_residual_context: object | None = None
    for start in range(0, frames, window_length):
        stop = min(frames, start + window_length)
        jets = HamiBallsAffineJets(matrix=field.jets.matrix[:, start:stop].detach(), offset=field.jets.offset[:, start:stop].detach(), health=None)

        def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
            raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
            return raw / object_scale
        local = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=None, d_tokens=field.d_tokens[:, start:stop].detach(), noisy=field.state[:, start:stop].detach(), x0=x0.detach(), d_candidate=rollout.d_candidate[:, start:stop].detach(), attrs=attrs.detach(), tau=field.tau.detach(), physical_time=physical_time[:, 1 + start:1 + stop].detach(), initial_previous_mixed=online_previous_mixed if online_detached_previous else rollout.previous_mixed[:, start].detach(), initial_previous_g=incoming_g[:, start].mean(dim=-1).detach() if incoming_g.ndim == 4 else incoming_g[:, start].detach(), exogenous_gate=rollout.gate[:, start:stop].detach(), initial_residual_context=online_residual_context)
        h_rows.append(local.h_candidate)
        hr_rows.append(local.hr_candidate)
        d_rows.append(local.d_candidate)
        hidden_rows.append(local.residual_hidden)
        previous_rows.append(local.previous_mixed)
        if online_detached_previous:
            online_previous_mixed = local.final_state.detach()
        if local.final_residual_context is not None:
            detach_context = getattr(residual, 'detach_field_context', None)
            if detach_context is None:
                raise TypeError('contextual residual must implement detach_field_context')
            online_residual_context = detach_context(local.final_residual_context)
    return FullNoRegretCandidates(h_candidate=torch.cat(h_rows, dim=1), hr_candidate=torch.cat(hr_rows, dim=1), d_candidate=torch.cat(d_rows, dim=1), residual_hidden=torch.cat(hidden_rows, dim=1), previous_mixed=torch.cat(previous_rows, dim=1))

def _component_no_regret_loss(h_candidate: torch.Tensor, hr_candidate: torch.Tensor, target: torch.Tensor, *, component: slice) -> tuple[torch.Tensor, torch.Tensor]:
    if h_candidate.shape != hr_candidate.shape or h_candidate.shape != target.shape:
        raise ValueError('no-regret candidates must align')
    h_error = (h_candidate[..., component] - target[..., component].detach()).square().mean(dim=-1)
    hr_error = (hr_candidate[..., component] - target[..., component].detach()).square().mean(dim=-1)
    excess = hr_error - h_error
    return (torch.nn.functional.relu(excess).mean(), (excess > 0.0).detach().float().mean())

def full_no_regret_losses(*, terminal: FullNoRegretCandidates, hull: FullNoRegretCandidates, target: torch.Tensor, q_dim: int) -> FullNoRegretLosses:
    if terminal.h_candidate.shape != terminal.hr_candidate.shape or terminal.hr_candidate.shape != terminal.d_candidate.shape or terminal.hr_candidate.shape != target.shape:
        raise ValueError('terminal full-no-regret candidates must align')
    if hull.hr_candidate.shape != hull.d_candidate.shape or hull.hr_candidate.shape != target.shape:
        raise ValueError('hull full-no-regret candidates must align')
    state_dim = int(terminal.hr_candidate.shape[-1])
    if not 0 < q_dim < state_dim:
        raise ValueError('q_dim must split the full-no-regret state')
    trajectory = (terminal.hr_candidate - target.detach()).square().mean()
    hull_loss, hull_g_star = per_object_candidate_hull_utility_loss(hull.hr_candidate, hull.d_candidate, target)
    baseline = 0.5 * trajectory + 0.5 * hull_loss
    q_no_regret, q_harm_fraction = _component_no_regret_loss(terminal.h_candidate, terminal.hr_candidate, target, component=slice(0, q_dim))
    p_no_regret, p_harm_fraction = _component_no_regret_loss(terminal.h_candidate, terminal.hr_candidate, target, component=slice(q_dim, state_dim))
    q_fraction = baseline.new_tensor(float(q_dim) / float(state_dim))
    p_fraction = baseline.new_tensor(float(state_dim - q_dim) / float(state_dim))
    no_regret = 0.5 * (q_fraction * q_no_regret + p_fraction * p_no_regret)
    for name, value in (('baseline', baseline), ('q_no_regret', q_no_regret), ('p_no_regret', p_no_regret), ('no_regret', no_regret)):
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f'full no-regret {name} is non-finite')
    return FullNoRegretLosses(baseline_loss=baseline, trajectory_loss=trajectory, hull_loss=hull_loss, q_no_regret_loss=q_no_regret, p_no_regret_loss=p_no_regret, q_harm_fraction=q_harm_fraction, p_harm_fraction=p_harm_fraction, no_regret_loss=no_regret, hull_g_star=hull_g_star)

def full_no_regret_field_indices(trace: HamiBallsTrajectoryG0, *, update_index: int) -> tuple[int, int]:
    if not trace.traces:
        raise ValueError('full no-regret requires accepted fields')
    return (len(trace.traces) - 1, cyclic_trace_field_index(update_index, len(trace.traces)))

def _gradient_group_stats(residual: HamiBallsDTokenResidual, *, q_dim: int) -> dict[str, float]:
    network = residual.network
    if not isinstance(network, nn.Sequential) or len(network) != 5:
        raise ValueError('full no-regret residual network schema drifted')
    if not all((isinstance(network[index], nn.Linear) for index in (0, 2, 4))):
        raise ValueError('full no-regret expects three Linear layers')
    output = network[4]
    assert isinstance(output, nn.Linear)
    groups = {'shared_hidden_abs_sum': (network[0].weight.grad, network[0].bias.grad, network[2].weight.grad, network[2].bias.grad), 'q_output_row_abs_sum': (output.weight.grad[:q_dim] if output.weight.grad is not None else None, output.bias.grad[:q_dim] if output.bias.grad is not None else None), 'p_output_row_abs_sum': (output.weight.grad[q_dim:] if output.weight.grad is not None else None, output.bias.grad[q_dim:] if output.bias.grad is not None else None)}
    result: dict[str, float] = {}
    for name, values in groups.items():
        if any((value is None for value in values)):
            raise AssertionError(f'full no-regret missing gradient in {name}')
        result[name] = float(sum((value.detach().abs().double().sum().cpu() for value in values)))
    return result

def assign_full_no_regret_gradients(residual: HamiBallsDTokenResidual, *, baseline_loss: torch.Tensor, no_regret_loss: torch.Tensor, q_dim: int) -> dict[str, Any]:
    _require_unbounded_residual(residual, q_dim=q_dim)
    parameters = tuple(residual.parameters())
    if not parameters or any((parameter.grad is not None for parameter in parameters)):
        raise AssertionError('residual gradients must be cleared before backpropagation')
    output = residual.network[4]
    if not isinstance(output, nn.Linear):
        raise ValueError('full no-regret residual output schema drifted')
    output_head_zero_at_entry = bool(torch.count_nonzero(output.weight.detach()).item() == 0 and torch.count_nonzero(output.bias.detach()).item() == 0)
    objective = baseline_loss + no_regret_loss
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError('full no-regret objective is non-finite')
    objective.backward()
    gradients = tuple((parameter.grad for parameter in parameters))
    if any((gradient is None for gradient in gradients)):
        raise AssertionError('full no-regret left a residual parameter without gradient')
    if not all((bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)):
        raise FloatingPointError('full no-regret gradient contains NaN/Inf')
    groups = _gradient_group_stats(residual, q_dim=q_dim)
    for name in ('q_output_row_abs_sum', 'p_output_row_abs_sum'):
        if groups[name] <= 0.0:
            raise AssertionError(f'full no-regret did not reach {name}')
    if not output_head_zero_at_entry and groups['shared_hidden_abs_sum'] <= 0.0:
        raise AssertionError('full no-regret did not reach shared hidden gradient group')
    return {'objective': float(objective.detach().cpu()), 'baseline_loss': float(baseline_loss.detach().cpu()), 'no_regret_loss': float(no_regret_loss.detach().cpu()), 'shared_hidden_zero_permitted_at_zero_head': output_head_zero_at_entry, **groups}

def recovery_full_no_regret_residual_update(*, residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, carrier: Any, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    if getattr(carrier, 'mode', None) != 'external':
        raise ValueError('full no-regret r update requires an external/reset carrier')
    trace = getattr(carrier, 'trace', None)
    if trace is None or not trace.traces:
        raise ValueError('full no-regret r carrier is empty')
    optimizer.zero_grad(set_to_none=True)
    terminal_index, hull_index = full_no_regret_field_indices(trace, update_index=update_index)
    terminal = replay_full_no_regret_candidates(residual, trace, field_index=terminal_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12)
    hull = replay_full_no_regret_candidates(residual, trace, field_index=hull_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12)
    losses = full_no_regret_losses(terminal=terminal, hull=hull, target=target, q_dim=q_dim)
    assignment = assign_full_no_regret_gradients(residual, baseline_loss=losses.baseline_loss, no_regret_loss=losses.no_regret_loss, q_dim=q_dim)
    from hamiformer.training.hamiballs_recovery import gradient_stats
    stats = gradient_stats(residual)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('full no-regret residual lacks finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), grad_clip))
    optimizer.step()
    return {'loss': assignment['objective'], 'baseline_loss': float(losses.baseline_loss.detach().cpu()), 'trajectory_loss': float(losses.trajectory_loss.detach().cpu()), 'hull_loss': float(losses.hull_loss.detach().cpu()), 'hull_field_index': int(hull_index), 'g_star_mean': float(losses.hull_g_star.detach().mean().cpu()), 'g_star_std': float(losses.hull_g_star.detach().float().std(unbiased=False).cpu()), 'q_no_regret_loss': float(losses.q_no_regret_loss.detach().cpu()), 'p_no_regret_loss': float(losses.p_no_regret_loss.detach().cpu()), 'no_regret_loss': float(losses.no_regret_loss.detach().cpu()), 'q_harm_fraction': float(losses.q_harm_fraction.detach().cpu()), 'p_harm_fraction': float(losses.p_harm_fraction.detach().cpu()), 'assignment': assignment, 'gradient': stats, 'gradient_norm_preclip': preclip}

def gradient_max_abs(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> float:
    if len(first) != len(second):
        raise ValueError('gradient collections must have equal lengths')
    return max((float((left.detach() - right.detach()).abs().max().cpu()) for left, right in zip(first, second, strict=True)))
__all__ = ['FullNoRegretCandidates', 'FullNoRegretLosses', 'assign_full_no_regret_gradients', 'full_no_regret_field_indices', 'full_no_regret_losses', 'gradient_max_abs', 'recovery_full_no_regret_residual_update', 'replay_full_no_regret_candidates']
