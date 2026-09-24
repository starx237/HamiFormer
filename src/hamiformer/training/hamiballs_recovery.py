from __future__ import annotations
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Literal
import torch
from torch import nn
from hamiformer.evaluation.hamiballs_formal import HamiBallsExpertFieldTrace, sample_hamiballs_expert_chunk
from hamiformer.models import HamiBallsCommittedRollout, HamiBallsDTokenResidual, HamiBallsPerObjectCompactCommittedGate
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, HamiBallsMixedJetAnchor
from hamiformer.training.hamiballs_trajectory import HamiBallsTrajectoryG0, trajectory_g0_mse, trajectory_g0_projection_bce, trajectory_g0_projection_mse, trajectory_gate_trace_g1_projection_bce, trajectory_gate_trace_g1_mse, trajectory_residual_trace_components, trajectory_residual_trace_cyclic_per_object_hull_loss
CarrierMode = Literal['external', 'main']

@dataclass(frozen=True)
class RecoveryCarrier:
    trace: HamiBallsTrajectoryG0
    num_steps: int
    mode: CarrierMode
    wall_seconds: float
    tangent_max: float | None
    tangent_min: float | None
    accepted_fields: int
    cold_left_intervals: tuple[int, ...]
    h_anchor_continuous: bool
    selector_provenance_isolated: bool

def module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def gradient_stats(module: nn.Module) -> dict[str, float | bool | int]:
    gradients = [parameter.grad for parameter in module.parameters()]
    present = [value for value in gradients if value is not None]
    if not present:
        return {'parameter_gradients_present': False, 'all_finite': True, 'nonzero': False, 'abs_sum': 0.0}
    abs_sum = sum((float(value.detach().abs().sum().cpu()) for value in present))
    return {'parameter_gradients_present': True, 'all_finite': all((bool(torch.isfinite(value).all()) for value in present)), 'nonzero': abs_sum > 0.0, 'abs_sum': abs_sum}

def set_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)

def clear_frozen_gradients(*modules: nn.Module) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)
        if any((parameter.grad is not None for parameter in module.parameters())):
            raise AssertionError('frozen module retained a gradient')

def frozen_gradients_absent(*modules: nn.Module) -> bool:
    return all((parameter.grad is None for module in modules for parameter in module.parameters()))

def _tensor_equal_digest(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))

@torch.no_grad()
def collect_recovery_carrier(*, d: nn.Module, hamiltonian: nn.Module | None, residual: HamiBallsDTokenResidual, gate: HamiBallsPerObjectCompactCommittedGate | None, source: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, frame_dt: float, t_eps: float, num_steps: int, mode: CarrierMode, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, external_gate: torch.Tensor | None=None, external_gate_tau: torch.Tensor | None=None, continuous_integrator_method: Literal['euler', 'rk4', 'explicit_euler', 'symplectic_euler', 'leapfrog', 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'] | None=None) -> RecoveryCarrier:
    if mode == 'main' and gate is None:
        raise ValueError('main Recovery carrier requires the per-object gate')
    if mode == 'external' and gate is not None:
        raise ValueError('external Recovery carrier must not consume a learned gate')
    if source.shape != x0[:, None].expand_as(source).shape:
        raise ValueError('source must be [B,F,K,state] aligned with x0')
    per_object = gate is not None
    modules: dict[str, nn.Module] = {'d': d, 'r': residual}
    if hamiltonian is not None:
        modules['h'] = hamiltonian
    if gate is not None:
        modules['g'] = gate
    before = {name: module_digest(module) for name, module in modules.items()}
    traces: list[HamiBallsExpertFieldTrace] = []
    tangent_rows: list[torch.Tensor] = []
    cold_left: list[int] = []
    selector_isolated = True
    h_anchor_continuous = True
    prior_left_next: HamiBallsMixedJetAnchor | None = None

    def callback(field: HamiBallsExpertFieldTrace) -> None:
        nonlocal selector_isolated, h_anchor_continuous, prior_left_next
        if field.next_anchor is None:
            raise AssertionError('Recovery callback did not expose next anchor')
        anchor, next_anchor = (field.anchor, field.next_anchor)
        if per_object and (anchor.previous_g is not None or anchor.gate_hidden is not None or next_anchor.previous_g is not None or (next_anchor.gate_hidden is not None)):
            selector_isolated = False
        if field.side == 'left':
            if prior_left_next is not None and (not (_tensor_equal_digest(anchor.source_q, prior_left_next.source_q) and _tensor_equal_digest(anchor.target_p, prior_left_next.target_p))):
                h_anchor_continuous = False
            prior_left_next = next_anchor
            if field.is_cold:
                cold_left.append(int(field.interval))
        if field.accepted and (not field.is_cold) and (field.side in {'left', 'final'}):
            if field.rollout is None or field.d_tokens is None:
                raise RuntimeError('accepted Recovery expert field lacks replay tensors')
            gfjp_jets = continuous_integrator_method in {None, 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'}
            if gfjp_jets and field.jets is None:
                raise RuntimeError('finite-H Recovery expert field lacks affine jets')
            if not gfjp_jets and field.jets is not None:
                raise RuntimeError('continuous-H Recovery field unexpectedly stored jets')
            traces.append(field)
            if hamiltonian is not None and gfjp_jets:
                assert field.jets is not None
                if field.jets.health is None:
                    raise RuntimeError('learned-H carrier lacks GFJP health')
                tangent_rows.append(field.jets.health.tangent_spectral_norm.detach())
    started = time.perf_counter()
    sample_hamiballs_expert_chunk(d, hamiltonian, residual, gate, source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, step_size=frame_dt, num_steps=num_steps, t_eps=t_eps, mode=mode, external_gate=external_gate, external_gate_tau=external_gate_tau, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, field_callback=callback, variable_n_cold_start=True, continuous_integrator_method=continuous_integrator_method, differentiable_continuous_state=False)
    wall_seconds = time.perf_counter() - started
    if not traces:
        raise RuntimeError('Recovery carrier captured no accepted expert field')
    expected_cold = min(num_steps - 1, max(1, int(0.1 * num_steps + 0.5)))
    if tuple(sorted(set(cold_left))) != tuple(range(expected_cold)):
        raise AssertionError(f'variable-N cold ledger mismatch: {sorted(set(cold_left))}')
    if per_object and (not selector_isolated):
        raise AssertionError('field-local gate provenance leaked through H anchor')
    if not h_anchor_continuous:
        raise AssertionError('accepted-left H anchor did not continue across RF fields')
    after = {name: module_digest(module) for name, module in modules.items()}
    if before != after:
        raise AssertionError('no-grad Recovery collection mutated a model')
    tangent: torch.Tensor | None = None
    if tangent_rows:
        tangent = torch.cat([value.reshape(-1) for value in tangent_rows])
        if not bool(torch.isfinite(tangent).all()):
            raise FloatingPointError('Recovery GFJP health contains NaN/Inf')
        if float(tangent.max().cpu()) > tangent_spectral_norm_limit:
            raise RuntimeError(f'Recovery strict GFJP health failed: {float(tangent.max().cpu())} > {tangent_spectral_norm_limit}')
    return RecoveryCarrier(trace=HamiBallsTrajectoryG0(traces=tuple(traces), pure_rows=0 if external_gate is None else int((external_gate == 1.0).all(dim=tuple(range(1, external_gate.ndim))).sum().item()), reset_edges=0 if external_gate is None else int((external_gate < 1.0).sum().item())), num_steps=int(num_steps), mode=mode, wall_seconds=float(wall_seconds), tangent_max=None if tangent is None else float(tangent.max().cpu()), tangent_min=None if tangent is None else float(tangent.min().cpu()), accepted_fields=len(traces), cold_left_intervals=tuple(sorted(set(cold_left))), h_anchor_continuous=h_anchor_continuous, selector_provenance_isolated=selector_isolated)

def _slice_anchor(anchor: HamiBallsMixedJetAnchor, start: int, stop: int) -> HamiBallsMixedJetAnchor:
    hidden = anchor.gate_hidden
    if hidden is not None:
        if hidden.ndim >= 2 and hidden.shape[1] >= stop:
            hidden = hidden[:, start:stop]
        elif hidden.shape[0] >= stop:
            hidden = hidden[start:stop]
        else:
            raise ValueError('gate hidden has no identifiable batch dimension')
    return HamiBallsMixedJetAnchor(source_q=anchor.source_q[start:stop], target_p=anchor.target_p[start:stop], source_p=None if anchor.source_p is None else anchor.source_p[start:stop], target_q=None if anchor.target_q is None else anchor.target_q[start:stop], previous_g=None if anchor.previous_g is None else anchor.previous_g[start:stop], gate_hidden=hidden)

def _slice_rollout(rollout: HamiBallsCommittedRollout, start: int, stop: int) -> HamiBallsCommittedRollout:
    return HamiBallsCommittedRollout(h_candidate=rollout.h_candidate[start:stop], innovation=rollout.innovation[start:stop], residual_gain=rollout.residual_gain[start:stop], residual_direction_rms=rollout.residual_direction_rms[start:stop], innovation_rms=rollout.innovation_rms[start:stop], residual_hidden=rollout.residual_hidden[start:stop], hr_candidate=rollout.hr_candidate[start:stop], d_candidate=rollout.d_candidate[start:stop], gate=rollout.gate[start:stop], mixed=rollout.mixed[start:stop], previous_mixed=rollout.previous_mixed[start:stop], final_state=rollout.final_state[start:stop], final_previous_g=rollout.final_previous_g[start:stop], gate_hidden=None if rollout.gate_hidden is None else _slice_anchor(HamiBallsMixedJetAnchor(source_q=rollout.final_state, target_p=rollout.final_state, gate_hidden=rollout.gate_hidden), start, stop).gate_hidden)

def _slice_trace_field(field: HamiBallsExpertFieldTrace, start: int, stop: int) -> HamiBallsExpertFieldTrace:
    if field.rollout is None or field.d_tokens is None or field.jets is None:
        raise ValueError('Recovery sharding requires a complete accepted field')
    next_anchor = None if field.next_anchor is None else _slice_anchor(field.next_anchor, start, stop)
    return HamiBallsExpertFieldTrace(interval=field.interval, side=field.side, is_cold=field.is_cold, accepted=field.accepted, state=field.state[start:stop], tau=field.tau[start:stop], anchor=_slice_anchor(field.anchor, start, stop), rollout=_slice_rollout(field.rollout, start, stop), d_tokens=field.d_tokens[start:stop], jets=HamiBallsAffineJets(matrix=field.jets.matrix[start:stop], offset=field.jets.offset[start:stop], health=None), clean=field.clean[start:stop], next_anchor=next_anchor)

def shard_recovery_trace(trace: HamiBallsTrajectoryG0, *, shard_size: int) -> tuple[HamiBallsTrajectoryG0, ...]:
    if not trace.traces:
        raise ValueError('cannot shard an empty Recovery trace')
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError('shard_size must be a positive integer')
    batch = trace.traces[0].state.shape[0]
    if batch % shard_size:
        raise ValueError('carrier batch must divide exactly into Recovery shards')
    if any((field.state.shape[0] != batch for field in trace.traces)):
        raise ValueError('accepted fields have inconsistent batch sizes')
    return tuple((HamiBallsTrajectoryG0(traces=tuple((_slice_trace_field(field, start, start + shard_size) for field in trace.traces)), pure_rows=trace.pure_rows // (batch // shard_size) if trace.pure_rows else 0, reset_edges=trace.reset_edges // (batch // shard_size)) for start in range(0, batch, shard_size)))

def recovery_residual_update(*, residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, carrier: RecoveryCarrier, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    if carrier.mode != 'external':
        raise ValueError('Recovery r update requires an external/reset carrier')
    if not carrier.trace.traces:
        raise ValueError('Recovery r carrier is empty')
    field_index = len(carrier.trace.traces) - 1
    optimizer.zero_grad(set_to_none=True)
    primary, _, _ = trajectory_residual_trace_components(residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12)
    hull, g_star, hull_field = trajectory_residual_trace_cyclic_per_object_hull_loss(residual, carrier.trace, update_index=update_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, window_length=12)
    total = 0.5 * primary + 0.5 * hull
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Recovery r trajectory/hull loss is non-finite')
    total.backward()
    stats = gradient_stats(residual)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery r has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(residual.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'trajectory_loss': float(primary.detach().cpu()), 'hull_loss': float(hull.detach().cpu()), 'hull_field_index': int(hull_field), 'g_star_mean': float(g_star.detach().mean().cpu()), 'g_star_std': float(g_star.detach().float().std(unbiased=False).cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}

def recovery_g0_update(*, gate: HamiBallsPerObjectCompactCommittedGate, optimizer: torch.optim.Optimizer, trace: HamiBallsTrajectoryG0, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, regret_auxiliary_weight: float, grad_clip: float) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    total, value = trajectory_g0_mse(gate, trace, x0=x0, attrs=attrs, physical_time=physical_time, target=target, regret_auxiliary_weight=regret_auxiliary_weight)
    if value.ndim != 3 or not bool(torch.isfinite(total)):
        raise FloatingPointError('Recovery per-object G0 output is invalid')
    total.backward()
    stats = gradient_stats(gate)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery G0 has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'gate_mean': float(value.detach().mean().cpu()), 'gate_std': float(value.detach().float().std(unbiased=False).cpu()), 'gate_object_std': float(value.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}

def recovery_g0_projection_update(*, gate: HamiBallsPerObjectCompactCommittedGate, optimizer: torch.optim.Optimizer, trace: HamiBallsTrajectoryG0, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, grad_clip: float) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    total, value, g_star = trajectory_g0_projection_mse(gate, trace, x0=x0, attrs=attrs, physical_time=physical_time, target=target)
    if value.ndim != 3 or g_star.shape != value.shape or (not bool(torch.isfinite(total))):
        raise FloatingPointError('Recovery projection G0 output is invalid')
    total.backward()
    stats = gradient_stats(gate)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery projection G0 has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'gate_mean': float(value.detach().mean().cpu()), 'gate_std': float(value.detach().float().std(unbiased=False).cpu()), 'gate_object_std': float(value.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'g_star_mean': float(g_star.detach().mean().cpu()), 'g_star_std': float(g_star.detach().float().std(unbiased=False).cpu()), 'g_star_object_std': float(g_star.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}

def recovery_g0_projection_bce_update(*, gate: HamiBallsPerObjectCompactCommittedGate, optimizer: torch.optim.Optimizer, trace: HamiBallsTrajectoryG0, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, grad_clip: float, direction_balanced: bool=False, conditional_balance_weight: float=0.0) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    total, value, g_star = trajectory_g0_projection_bce(gate, trace, x0=x0, attrs=attrs, physical_time=physical_time, target=target, direction_balanced=direction_balanced, conditional_balance_weight=conditional_balance_weight)
    if value.ndim != 3 or g_star.shape != value.shape or (not bool(torch.isfinite(total))):
        raise FloatingPointError('Recovery projection-BCE G0 output is invalid')
    total.backward()
    stats = gradient_stats(gate)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery projection-BCE G0 has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'objective': 'per_object_direction_balanced_soft_projection_bce_v1' if direction_balanced else 'per_edge_conditional_direction_aux_soft_projection_bce_v1' if conditional_balance_weight > 0.0 else 'per_object_soft_projection_bce_v1', 'gate_mean': float(value.detach().mean().cpu()), 'gate_std': float(value.detach().float().std(unbiased=False).cpu()), 'gate_object_std': float(value.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'g_star_mean': float(g_star.detach().mean().cpu()), 'g_star_std': float(g_star.detach().float().std(unbiased=False).cpu()), 'g_star_object_std': float(g_star.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}

def recovery_g1_update(*, gate: HamiBallsPerObjectCompactCommittedGate, residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, carrier: RecoveryCarrier, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float) -> dict[str, Any]:
    if carrier.mode != 'main' or not carrier.trace.traces:
        raise ValueError('Recovery G1 requires a fresh on-policy carrier')
    field_index = (update_index - 1) % len(carrier.trace.traces)
    optimizer.zero_grad(set_to_none=True)
    total, value = trajectory_gate_trace_g1_mse(gate, residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim)
    if value.ndim != 3 or not bool(torch.isfinite(total)):
        raise FloatingPointError('Recovery per-object G1 output is invalid')
    total.backward()
    stats = gradient_stats(gate)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery G1 has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'field_index': int(field_index), 'gate_mean': float(value.detach().mean().cpu()), 'gate_std': float(value.detach().float().std(unbiased=False).cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}

def recovery_g1_projection_bce_update(*, gate: HamiBallsPerObjectCompactCommittedGate, residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, carrier: RecoveryCarrier, update_index: int, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, q_dim: int, grad_clip: float, direction_balanced: bool=False, conditional_balance_weight: float=0.0) -> dict[str, Any]:
    if carrier.mode != 'main' or not carrier.trace.traces:
        raise ValueError('Recovery projection-BCE G1 requires a fresh on-policy carrier')
    field_index = (update_index - 1) % len(carrier.trace.traces)
    optimizer.zero_grad(set_to_none=True)
    total, value, g_star = trajectory_gate_trace_g1_projection_bce(gate, residual, carrier.trace, field_index=field_index, x0=x0, attrs=attrs, physical_time=physical_time, target=target, state_scale=state_scale, q_dim=q_dim, direction_balanced=direction_balanced, conditional_balance_weight=conditional_balance_weight)
    if value.ndim != 3 or g_star.shape != value.shape or (not bool(torch.isfinite(total))):
        raise FloatingPointError('Recovery projection-BCE G1 output is invalid')
    total.backward()
    stats = gradient_stats(gate)
    if not bool(stats['all_finite']) or not bool(stats['nonzero']):
        raise AssertionError('Recovery projection-BCE G1 has no finite nonzero gradient')
    preclip = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), grad_clip))
    optimizer.step()
    return {'loss': float(total.detach().cpu()), 'objective': 'same_history_per_object_direction_balanced_soft_projection_bce_v1' if direction_balanced else 'same_history_per_edge_conditional_direction_aux_soft_projection_bce_v1' if conditional_balance_weight > 0.0 else 'same_history_per_object_soft_projection_bce_v1', 'field_index': int(field_index), 'gate_mean': float(value.detach().mean().cpu()), 'gate_std': float(value.detach().float().std(unbiased=False).cpu()), 'gate_object_std': float(value.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'g_star_mean': float(g_star.detach().mean().cpu()), 'g_star_std': float(g_star.detach().float().std(unbiased=False).cpu()), 'g_star_object_std': float(g_star.detach().float().std(dim=-1, unbiased=False).mean().cpu()), 'gradient': stats, 'gradient_norm_preclip': preclip}
__all__ = ['CarrierMode', 'RecoveryCarrier', 'clear_frozen_gradients', 'collect_recovery_carrier', 'frozen_gradients_absent', 'gradient_stats', 'module_digest', 'recovery_g0_update', 'recovery_g0_projection_update', 'recovery_g0_projection_bce_update', 'recovery_g1_update', 'recovery_g1_projection_bce_update', 'recovery_residual_update', 'set_trainable', 'shard_recovery_trace']
