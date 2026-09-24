from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
import torch
from torch import nn
from hamiformer.flow.pf_rf_v1 import sample_pf_rf_v1_heun
from hamiformer.flow.rectified_flow import clean_to_velocity
from hamiformer.flow.stateful_pf_rf_v1 import default_pf_rf_v1_cold_start_intervals, hamiballs_formal_cold_start_intervals, sample_stateful_pf_rf_v1_heun
from hamiformer.flow.stateful_residual_flow import StatefulFieldEvaluation
from hamiformer.models import HamiBallsCompactCommittedGate, HamiBallsCommittedGate, HamiBallsCommittedRollout, HamiBallsDTokenResidual, HamiBallsPerObjectCompactCommittedGate, rollout_hamiballs_committed_edges
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, HamiBallsMixedJetAnchor, hamiballs_anchor_from_candidates, hamiballs_anchor_from_d_candidate
from hamiformer.physics.continuous_hamiltonian import TokenConditionalContinuousHamiltonian, normalized_continuous_hamiltonian_step
from hamiformer.training.hamiballs_d import d_clean_and_tokens
from hamiformer.training.hamiballs_formal import _rollout_hamiballs_affine_h_only, convex_oracle_gate, identity_hamiballs_affine_jets, learned_hamiballs_affine_jets, learned_hamiballs_leapfrog_affine_jets, rollout_with_affine_jets
FormalExpertMode = Literal['main', 'pure_hr', 'oracle', 'external']
OracleGranularity = Literal['system', 'per_object', 'per_object_qp']

@dataclass(frozen=True)
class HamiBallsExpertFieldTrace:
    interval: int
    side: str
    is_cold: bool
    accepted: bool
    state: torch.Tensor
    tau: torch.Tensor
    anchor: HamiBallsMixedJetAnchor
    rollout: HamiBallsCommittedRollout | None
    d_tokens: torch.Tensor | None
    jets: HamiBallsAffineJets | None
    clean: torch.Tensor
    next_anchor: HamiBallsMixedJetAnchor | None = None
    previous_heun_defect: torch.Tensor | None = None
HamiBallsExpertFieldCallback = Callable[[HamiBallsExpertFieldTrace], None]

def _heun_predictor_corrector_defect(*, left_state: torch.Tensor, left_tau: torch.Tensor, left_clean: torch.Tensor, right_state: torch.Tensor, right_tau: torch.Tensor, right_clean: torch.Tensor, t_eps: float) -> torch.Tensor:
    left_velocity = clean_to_velocity(left_clean, left_state, left_tau, t_eps=t_eps)
    right_velocity = clean_to_velocity(right_clean, right_state, right_tau, t_eps=t_eps)
    step = (right_tau - left_tau).reshape(right_tau.shape[0], *[1] * (right_state.ndim - 1))
    return 0.5 * step * (right_velocity - left_velocity)

@dataclass(frozen=True)
class HamiBallsChunkSample:
    trajectory: torch.Tensor
    final_previous_g: torch.Tensor
    final_gate_hidden: torch.Tensor | None
    final_field_rollout: HamiBallsCommittedRollout
    field_evaluations: int

@dataclass(frozen=True)
class HamiBallsLongSample:
    trajectory: torch.Tensor
    gate: torch.Tensor
    chunks: tuple[HamiBallsChunkSample, ...]
    final_previous_g: torch.Tensor
    final_gate_hidden: torch.Tensor | None
    field_evaluations: int

def _scale_views(state_scale: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if state_scale.shape != (reference.shape[-1],):
        raise ValueError("state scale differs from one object's state")
    scale = state_scale.to(reference)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError('state scale must be finite positive')
    return (scale.reshape(1, 1, -1), scale.reshape(1, 1, 1, -1))

@torch.no_grad()
def sample_hamiballs_d_chunk(d_model: nn.Module, source: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, num_steps: int, t_eps: float) -> torch.Tensor:

    def field(state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        clean, _ = d_clean_and_tokens(d_model, state, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        return clean
    return sample_pf_rf_v1_heun(field, source, num_steps=num_steps, t_eps=t_eps).trajectory

def sample_hamiballs_expert_chunk(d_model: nn.Module, hamiltonian: nn.Module | None, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate | None, source: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, num_steps: int, t_eps: float, mode: FormalExpertMode, initial_previous_g: torch.Tensor | None=None, initial_gate_hidden: torch.Tensor | None=None, oracle_target: torch.Tensor | None=None, external_gate: torch.Tensor | None=None, external_gate_tau: torch.Tensor | None=None, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, detach_d_carrier: bool=False, detach_accepted_anchor: bool=True, differentiable_h_jets: bool=False, field_callback: HamiBallsExpertFieldCallback | None=None, variable_n_cold_start: bool=False, oracle_granularity: OracleGranularity='system', continuous_integrator_method: Literal['euler', 'rk4', 'explicit_euler', 'symplectic_euler', 'leapfrog', 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'] | None=None, differentiable_continuous_state: bool=False) -> HamiBallsChunkSample:
    if mode == 'main' and gate is None:
        raise ValueError('main mode requires a learned gate')
    if mode != 'main' and gate is not None:
        raise ValueError('pure/oracle/external modes do not consume a learned gate')
    if mode == 'oracle' and oracle_target is None:
        raise ValueError('oracle mode requires the held-out clean chunk')
    if mode != 'oracle' and oracle_target is not None:
        raise ValueError('GT target is permitted only in oracle mode')
    if oracle_granularity not in {'system', 'per_object', 'per_object_qp'}:
        raise ValueError('unknown oracle granularity')
    if mode != 'oracle' and oracle_granularity != 'system':
        raise ValueError('per-object oracle granularity is valid only in oracle mode')
    if source.ndim != 4 or physical_time.shape[:1] != source.shape[:1]:
        raise ValueError('source/time must be [B,F,K,S] and [B,F+1]')
    batch, edges, objects = source.shape[:3]
    if physical_time.shape != (batch, edges + 1):
        raise ValueError('physical time must include the chunk initial node')
    if oracle_target is not None and oracle_target.shape != source.shape:
        raise ValueError('oracle target must align with the RF state')
    if mode == 'external':
        if external_gate is None or tuple(external_gate.shape) not in {(batch, edges), (batch, edges, objects)}:
            raise ValueError('external mode requires external_gate [B,F] or [B,F,K]')
        if not bool(torch.isfinite(external_gate).all()) or bool((external_gate < 0.0).any() or (external_gate > 1.0).any()):
            raise ValueError('external gate must be finite and lie in [0,1]')
        if external_gate_tau is not None and (external_gate_tau.shape != (batch,) or not bool(torch.isfinite(external_gate_tau).all()) or bool((external_gate_tau < 0.0).any() or (external_gate_tau > 1.0).any())):
            raise ValueError('external_gate_tau must be finite [B] in [0,1]')
    elif external_gate is not None:
        raise ValueError('external_gate is permitted only in external mode')
    elif external_gate_tau is not None:
        raise ValueError('external_gate_tau is permitted only in external mode')
    if type(detach_d_carrier) is not bool:
        raise TypeError('detach_d_carrier must be bool')
    if type(detach_accepted_anchor) is not bool:
        raise TypeError('detach_accepted_anchor must be bool')
    if type(differentiable_h_jets) is not bool:
        raise TypeError('differentiable_h_jets must be bool')
    if type(differentiable_continuous_state) is not bool:
        raise TypeError('differentiable_continuous_state must be bool')
    if continuous_integrator_method not in {None, 'euler', 'rk4', 'explicit_euler', 'symplectic_euler', 'leapfrog', 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'}:
        raise ValueError('unknown continuous integrator')
    if continuous_integrator_method is not None and (not isinstance(hamiltonian, TokenConditionalContinuousHamiltonian)):
        raise TypeError('continuous integration requires a continuous Hamiltonian')
    if continuous_integrator_method is None and differentiable_continuous_state:
        raise ValueError('continuous-state differentiation requires an integrator')
    if continuous_integrator_method not in {None, 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'} and differentiable_h_jets:
        raise ValueError('continuous integration does not construct Type-II jets')
    if type(variable_n_cold_start) is not bool:
        raise TypeError('variable_n_cold_start must be bool')
    per_object_oracle = mode == 'oracle' and oracle_granularity in {'per_object', 'per_object_qp'}
    component_oracle = mode == 'oracle' and oracle_granularity == 'per_object_qp'
    per_object_gate = isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)) or per_object_oracle or (mode == 'external' and external_gate is not None and (external_gate.ndim == 3))
    if per_object_gate and (initial_previous_g is not None or initial_gate_hidden is not None):
        raise ValueError('per-object reset selector rejects externally supplied selector provenance')
    if variable_n_cold_start and gate is not None and (not per_object_gate):
        raise ValueError('variable-N cold start is registered only with the per-object reset selector')
    object_scale, sequence_scale = _scale_views(state_scale, x0)
    raw_x0 = x0 * object_scale
    initial_anchor = hamiballs_anchor_from_d_candidate(raw_x0, source * sequence_scale, q_dim=q_dim, previous_g=None if per_object_gate else initial_previous_g, gate_hidden=None if per_object_gate else initial_gate_hidden, detach=detach_accepted_anchor)
    last_rollout: HamiBallsCommittedRollout | None = None
    last_d_tokens: torch.Tensor | None = None
    last_jets: HamiBallsAffineJets | None = None
    expert_field_evaluations = 0
    heun_left_state: torch.Tensor | None = None
    heun_left_tau: torch.Tensor | None = None
    heun_left_clean: torch.Tensor | None = None
    previous_heun_defect: torch.Tensor | None = None

    def d_field(state: torch.Tensor, tau: torch.Tensor, _anchor: HamiBallsMixedJetAnchor) -> StatefulFieldEvaluation:
        if detach_d_carrier:
            with torch.no_grad():
                d_candidate, _ = d_clean_and_tokens(d_model, state, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        else:
            d_candidate, _ = d_clean_and_tokens(d_model, state, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        if detach_d_carrier:
            d_candidate = d_candidate.detach()
        next_anchor = hamiballs_anchor_from_d_candidate(raw_x0, d_candidate * sequence_scale, q_dim=q_dim, previous_g=None if per_object_gate else _anchor.previous_g, gate_hidden=None if per_object_gate else _anchor.gate_hidden, detach=detach_accepted_anchor)
        return StatefulFieldEvaluation(clean=d_candidate, velocity=clean_to_velocity(d_candidate, state, tau, t_eps=t_eps), next_anchor=next_anchor, h_candidate=d_candidate)

    def expert_field(state: torch.Tensor, tau: torch.Tensor, anchor: HamiBallsMixedJetAnchor) -> StatefulFieldEvaluation:
        nonlocal last_rollout, last_d_tokens, last_jets, expert_field_evaluations
        if detach_d_carrier:
            with torch.no_grad():
                d_candidate, d_tokens = d_clean_and_tokens(d_model, state, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        else:
            d_candidate, d_tokens = d_clean_and_tokens(d_model, state, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        if detach_d_carrier:
            d_candidate = d_candidate.detach()
            d_tokens = d_tokens.detach()
        if continuous_integrator_method in {'gfjp_leapfrog', 'gfjp_leapfrog_cold2'}:
            if hamiltonian is None:
                raise TypeError('GFJP Leapfrog requires a learned continuous H')
            jets = learned_hamiballs_leapfrog_affine_jets(hamiltonian, anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable_h_jets)
            if continuous_integrator_method == 'gfjp_leapfrog_cold2' and expert_field_evaluations < 2:
                first_prediction = _rollout_hamiballs_affine_h_only(jets, raw_x0, q_dim=q_dim)
                refined_anchor = hamiballs_anchor_from_candidates(raw_x0, first_prediction.detach(), first_prediction.detach(), q_dim=q_dim, previous_g=anchor.previous_g, gate_hidden=anchor.gate_hidden).detached()
                jets = learned_hamiballs_leapfrog_affine_jets(hamiltonian, refined_anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable_h_jets)
            expert_field_evaluations += 1
        elif continuous_integrator_method is not None:
            jets = None
        elif hamiltonian is None:
            jets = identity_hamiballs_affine_jets(d_candidate, q_dim=q_dim)
        else:
            jets = learned_hamiballs_affine_jets(hamiltonian, anchor, attrs, attr_scale=attr_scale, step_size=step_size, mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, differentiable=differentiable_h_jets)
        exogenous_gate = None
        gate_policy = None
        active_gate = gate
        if mode == 'pure_hr':
            active_gate = None
            exogenous_gate = state.new_ones(batch, edges)
        elif mode == 'oracle':
            active_gate = None
            assert oracle_target is not None

            def oracle_policy(edge: int, _previous: torch.Tensor, _h: torch.Tensor, hr: torch.Tensor, d: torch.Tensor, _previous_g: torch.Tensor) -> torch.Tensor:
                target = oracle_target[:, edge:edge + 1]
                if component_oracle:
                    return torch.stack([convex_oracle_gate(hr[:, None, :, :q_dim], d[:, None, :, :q_dim], target[..., :q_dim], granularity='per_object')[:, 0], convex_oracle_gate(hr[:, None, :, q_dim:], d[:, None, :, q_dim:], target[..., q_dim:], granularity='per_object')[:, 0]], dim=-1)
                return convex_oracle_gate(hr[:, None], d[:, None], target, granularity=oracle_granularity)[:, 0]
            gate_policy = oracle_policy
        elif mode == 'external':
            active_gate = None
            assert external_gate is not None
            if external_gate_tau is None:
                exogenous_gate = external_gate
            else:
                at_reset_node = torch.isclose(tau, external_gate_tau.to(device=tau.device, dtype=tau.dtype), rtol=0.0, atol=8.0 * torch.finfo(tau.dtype).eps)
                reset_selector = at_reset_node[:, None, None] if external_gate.ndim == 3 else at_reset_node[:, None]
                exogenous_gate = torch.where(reset_selector, external_gate, torch.ones_like(external_gate))
        common_rollout = {'residual': residual, 'gate': active_gate, 'd_tokens': d_tokens, 'noisy': state, 'x0': x0, 'd_candidate': d_candidate, 'attrs': attrs, 'tau': tau, 'physical_time': physical_time[:, 1:], 'exogenous_gate': exogenous_gate, 'initial_previous_g': None if per_object_gate else anchor.previous_g, 'initial_gate_hidden': None if per_object_gate else anchor.gate_hidden, 'gate_policy': gate_policy, 'per_object_gate_policy': per_object_oracle, 'component_gate_policy': component_oracle}
        if continuous_integrator_method in {None, 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'}:
            assert jets is not None
            rollout = rollout_with_affine_jets(jets, q_dim=q_dim, state_scale=state_scale, **common_rollout)
        else:
            assert isinstance(hamiltonian, TokenConditionalContinuousHamiltonian)

            def continuous_h_builder(_edge: int, previous: torch.Tensor) -> torch.Tensor:
                return normalized_continuous_hamiltonian_step(hamiltonian, previous, attrs, state_scale=state_scale, attr_scale=attr_scale, step_size=step_size, method=continuous_integrator_method, create_graph=differentiable_continuous_state)
            rollout = rollout_hamiballs_committed_edges(h_builder=continuous_h_builder, **common_rollout)
        last_rollout = rollout
        last_d_tokens = d_tokens
        last_jets = jets
        next_anchor = hamiballs_anchor_from_candidates(raw_x0, rollout.mixed * sequence_scale, rollout.h_candidate * sequence_scale, q_dim=q_dim, previous_g=None if per_object_gate else rollout.final_previous_g, gate_hidden=None if per_object_gate else rollout.gate_hidden)
        if detach_accepted_anchor:
            next_anchor = next_anchor.detached()
        return StatefulFieldEvaluation(clean=rollout.mixed, velocity=clean_to_velocity(rollout.mixed, state, tau, t_eps=t_eps), next_anchor=next_anchor, h_candidate=rollout.h_candidate)

    def trace_callback(interval: int, side: str, is_cold: bool, accepted: bool, state: torch.Tensor, tau: torch.Tensor, anchor: HamiBallsMixedJetAnchor, evaluation: StatefulFieldEvaluation) -> None:
        nonlocal heun_left_state, heun_left_tau, heun_left_clean
        nonlocal previous_heun_defect
        if field_callback is None:
            return
        defect_for_trace = previous_heun_defect if side in {'left', 'final'} else None
        if side == 'left':
            heun_left_state = state.detach()
            heun_left_tau = tau.detach()
            heun_left_clean = evaluation.clean.detach()
        elif side == 'right':
            if heun_left_state is None or heun_left_tau is None or heun_left_clean is None:
                raise RuntimeError('Heun right trace lacks its causal left evaluation')
            previous_heun_defect = _heun_predictor_corrector_defect(left_state=heun_left_state, left_tau=heun_left_tau, left_clean=heun_left_clean, right_state=state.detach(), right_tau=tau.detach(), right_clean=evaluation.clean.detach(), t_eps=t_eps).detach()
            heun_left_state = None
            heun_left_tau = None
            heun_left_clean = None
        field_callback(HamiBallsExpertFieldTrace(interval=interval, side=side, is_cold=is_cold, accepted=accepted, state=state, tau=tau, anchor=anchor, rollout=None if is_cold else last_rollout, d_tokens=None if is_cold else last_d_tokens, jets=None if is_cold else last_jets, clean=evaluation.clean, next_anchor=evaluation.next_anchor, previous_heun_defect=defect_for_trace))
    sample = sample_stateful_pf_rf_v1_heun(expert_field, source, initial_anchor, num_steps=num_steps, t_eps=t_eps, cold_start_field=d_field, cold_start_intervals=default_pf_rf_v1_cold_start_intervals(num_steps) if variable_n_cold_start else hamiballs_formal_cold_start_intervals(num_steps), commit_anchor_from='accepted_left', detach_accepted_anchor=detach_accepted_anchor, evaluation_callback=trace_callback if field_callback is not None else None)
    if last_rollout is None:
        raise RuntimeError('formal sampler did not execute a post-cold-start expert field')
    return HamiBallsChunkSample(trajectory=sample.trajectory, final_previous_g=sample.final_anchor.previous_g.detach() if sample.final_anchor.previous_g is not None else last_rollout.final_previous_g.detach(), final_gate_hidden=None if per_object_gate or sample.final_anchor.gate_hidden is None else sample.final_anchor.gate_hidden.detach(), final_field_rollout=last_rollout, field_evaluations=sample.field_evaluations)

@torch.no_grad()
def sample_hamiballs_expert_chunks(d_model: nn.Module, hamiltonian: nn.Module | None, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate | None, source: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, q_dim: int, step_size: float, num_steps: int, t_eps: float, mode: FormalExpertMode, initial_previous_g: torch.Tensor | None=None, initial_gate_hidden: torch.Tensor | None=None, oracle_target: torch.Tensor | None=None, mixed_singular_floor: float=0.2, mixed_condition_limit: float=10.0, tangent_spectral_norm_limit: float=5.0, variable_n_cold_start: bool=False, oracle_granularity: OracleGranularity='system', continuous_integrator_method: Literal['euler', 'rk4', 'explicit_euler', 'symplectic_euler', 'leapfrog', 'gfjp_leapfrog', 'gfjp_leapfrog_cold2'] | None=None) -> HamiBallsLongSample:
    if source.ndim != 5:
        raise ValueError('chunk source must be [B,C,F,K,S]')
    batch, chunk_count, chunk_edges = source.shape[:3]
    if chunk_count < 1 or chunk_edges < 1:
        raise ValueError('chunk source must contain at least one edge')
    if physical_time.shape != (batch, chunk_count * chunk_edges + 1):
        raise ValueError('long physical time must cover all chunk nodes')
    expected_target = (batch, chunk_count * chunk_edges, source.shape[3], source.shape[4])
    if mode == 'oracle':
        if oracle_target is None or oracle_target.shape != expected_target:
            raise ValueError('long oracle target must cover every physical edge')
    elif oracle_target is not None:
        raise ValueError('GT target is permitted only in oracle mode')
    if oracle_granularity not in {'system', 'per_object', 'per_object_qp'}:
        raise ValueError('unknown oracle granularity')
    if mode != 'oracle' and oracle_granularity != 'system':
        raise ValueError('per-object oracle granularity is valid only in oracle mode')
    per_object_oracle = mode == 'oracle' and oracle_granularity in {'per_object', 'per_object_qp'}
    per_object_gate = isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)) or per_object_oracle
    if per_object_gate and (initial_previous_g is not None or initial_gate_hidden is not None):
        raise ValueError('per-object reset selector rejects externally supplied selector provenance')
    if variable_n_cold_start and gate is not None and (not per_object_gate):
        raise ValueError('variable-N cold start is registered only with the per-object reset selector')
    current = x0
    previous_g = initial_previous_g
    hidden = initial_gate_hidden
    samples: list[HamiBallsChunkSample] = []
    trajectories: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    evaluations = 0
    for chunk in range(chunk_count):
        left, right = (chunk_edges * chunk, chunk_edges * (chunk + 1))
        sampled = sample_hamiballs_expert_chunk(d_model, hamiltonian, residual, gate, source[:, chunk], x0=current, attrs=attrs, physical_time=physical_time[:, left:right + 1], state_scale=state_scale, attr_scale=attr_scale, q_dim=q_dim, step_size=step_size, num_steps=num_steps, t_eps=t_eps, mode=mode, initial_previous_g=None if per_object_gate else previous_g, initial_gate_hidden=None if per_object_gate else hidden, oracle_target=None if oracle_target is None else oracle_target[:, left:right], mixed_singular_floor=mixed_singular_floor, mixed_condition_limit=mixed_condition_limit, tangent_spectral_norm_limit=tangent_spectral_norm_limit, variable_n_cold_start=variable_n_cold_start, oracle_granularity=oracle_granularity, continuous_integrator_method=continuous_integrator_method)
        samples.append(sampled)
        trajectories.append(sampled.trajectory)
        gates.append(sampled.final_field_rollout.gate)
        evaluations += sampled.field_evaluations
        current = sampled.trajectory[:, -1]
        if not per_object_gate:
            previous_g = sampled.final_previous_g
            hidden = sampled.final_gate_hidden
    return HamiBallsLongSample(trajectory=torch.cat(trajectories, dim=1), gate=torch.cat(gates, dim=1), chunks=tuple(samples), final_previous_g=samples[-1].final_previous_g, final_gate_hidden=None if per_object_gate else hidden, field_evaluations=evaluations)
__all__ = ['FormalExpertMode', 'OracleGranularity', 'HamiBallsChunkSample', 'HamiBallsExpertFieldTrace', 'HamiBallsLongSample', 'HamiBallsExpertFieldCallback', 'sample_hamiballs_d_chunk', 'sample_hamiballs_expert_chunk', 'sample_hamiballs_expert_chunks']
