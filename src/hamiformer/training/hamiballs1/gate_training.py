from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
ROOT = project_root()
SELF_POLICY_REFRESH_CACHE_OBSERVER = None
SELF_POLICY_REFRESH_COLLECTOR_ADAPTER = None
SELF_POLICY_PRECOLLECT_STAGE_PARENT_REFRESHES = False
SELF_POLICY_PRECOLLECT_FIXED_CARRIERS = False
SELF_POLICY_MIXED_NUM_STEPS_CYCLE: tuple[int, ...] | None = None
NONFATAL_ONLINE_REPLAY_AUDIT = False
NONFATAL_NONFINITE_OPTIMIZER_STEP = False
DEBUG_AUTOGRAD_ANOMALY_FROM_UPDATE: int | None = None
BATCH_PAIRED_NOISE_COLLECTION = False
DISJOINT_PAIRED_NOISE_SOURCES = False
DEFER_CARRIER_SYNCHRONIZE = False
PARALLEL_MIXED_N_CARRIER_STREAMS = False
FIXED_MIXED_N_SUPERBATCH_FACTOR = 1
from hamiformer.data import load_phase_scales
from hamiformer.models.hamiballs_gate_responsibility import HamiBallsResponsibilitySeparatedGate
from hamiformer.models.hamiballs_gate_component_risk_veto import HamiBallsComponentRiskVetoGate
from hamiformer.models.hamiballs_gate_dual_channel import HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate, HamiBallsStagedFrozenRiskPositiveConeCapPerObjectCompactCommittedGate, HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate, HamiBallsObservableAdditiveRecoveryPerObjectCompactCommittedGate, HamiBallsDualStatePerObjectCompactCommittedGate

def _collect_shared_main_carrier_deferred_sync(*, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module, residual: torch.nn.Module, gate: torch.nn.Module, train: Any, stream: Any, state_scale: torch.Tensor, attr_scale: torch.Tensor, num_steps: int, batch_size: int, source_rng: torch.Generator, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream.batch_size != batch_size:
        raise AssertionError('main carrier stream has an incorrect batch size')
    x0, target, attrs, physical_time = carrier_cache.stage_a._carrier_batch(train, stream, state_scale=state_scale, device=device)
    source = carrier_cache.stage_a._random_source(target, generator=source_rng, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
    carrier = carrier_cache.recovery.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']))
    return (carrier, x0, target, attrs, physical_time)

def _collect_recovery_from_prepared(*, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module, residual: torch.nn.Module, gate: torch.nn.Module, source: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor, attr_scale: torch.Tensor, num_steps: int):
    carrier = carrier_cache.recovery.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']))
    return carrier

def _collect_fixed_mixed_n_superbatch_rows(*, indices: torch.Tensor, base_cycle: tuple[int, ...], factor: int, seed_offset: int, noise_seed: int, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module, residual: torch.nn.Module, gate: torch.nn.Module, train: Any, state_scale: torch.Tensor, attr_scale: torch.Tensor, device: torch.device) -> list[TensorCache]:
    base = len(base_cycle)
    if factor < 1 or base < 1 or int(indices.numel()) % base:
        raise ValueError('fixed mixed-N superbatch received an incomplete base pool')
    assignment = torch.tensor(base_cycle, device=indices.device)
    if set(base_cycle) != {8, 12, 20}:
        raise ValueError('fixed mixed-N superbatch requires N8/N12/N20')
    blocks = []
    for block_index, begin in enumerate(range(0, int(indices.numel()), base)):
        group_indices = indices[begin:begin + base]
        x0, target, attrs, physical_time = carrier_cache.stage_a._carrier_batch(train, fixed._FixedStream(group_indices), state_scale=state_scale, device=device)
        generator = torch.Generator(device=device).manual_seed(int(noise_seed) + int(seed_offset) + block_index)
        records = []
        for num_steps in (8, 12, 20):
            mask = assignment == num_steps
            value_mask = mask.to(target.device)
            local_target = target[value_mask]
            source = carrier_cache.stage_a._random_source(local_target, generator=generator, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
            records.append({'num_steps': num_steps, 'indices': group_indices[mask], 'slots': torch.arange(base, device=device, dtype=torch.long)[mask], 'x0': x0[value_mask], 'target': local_target, 'attrs': attrs[value_mask], 'physical_time': physical_time[value_mask], 'source': source})
        blocks.append(records)
    output: list[TensorCache] = []
    logical_gate_calls = [0 for _ in blocks]
    for block_begin in range(0, len(blocks), factor):
        local_blocks = blocks[block_begin:block_begin + factor]
        block_rows: list[list[TensorCache]] = [[] for _ in local_blocks]
        for component, num_steps in enumerate((8, 12, 20)):
            records = [block[component] for block in local_blocks]
            joined = {name: torch.cat([record[name] for record in records], dim=0) for name in ('x0', 'target', 'attrs', 'physical_time', 'source')}
            random_collector = isinstance(gate, _UniformRandomCarrierGate)
            before_gate_calls = int(getattr(gate, 'calls', 0))
            if random_collector:
                gate._logical_rng_parts = [(int(record['indices'].numel()), gate.seed + 1000003 * (block_begin + j), logical_gate_calls[block_begin + j]) for j, record in enumerate(records)]
                gate._logical_rng_call_origin = before_gate_calls
            carrier = _collect_recovery_from_prepared(config=config, contract=contract, d=d, hamiltonian=hamiltonian, residual=residual, gate=gate, source=joined['source'], x0=joined['x0'], attrs=joined['attrs'], physical_time=joined['physical_time'], state_scale=state_scale, attr_scale=attr_scale, num_steps=num_steps)
            if random_collector:
                added_calls = gate.calls - before_gate_calls
                for j in range(len(records)):
                    logical_gate_calls[block_begin + j] += added_calls
                del gate._logical_rng_parts
                del gate._logical_rng_call_origin
            offset = 0
            for local_block, record in enumerate(records):
                count = int(record['indices'].numel())
                end = offset + count
                for field_order, trace in enumerate(carrier.trace.traces):
                    row = _field_tensors(trace, x0=joined['x0'], target=joined['target'], attrs=joined['attrs'], physical_time=joined['physical_time'])
                    row = {name: value[offset:end] for name, value in row.items()}
                    row.update({'trajectory_slot': record['slots'], 'source_index': record['indices'].to(device=device, dtype=torch.long), 'rf_field_order': torch.full((count,), field_order, device=device, dtype=torch.long), 'rf_trace_index': torch.full((count,), field_order, device=device, dtype=torch.long)})
                    block_rows[local_block].append(row)
                offset = end
        for rows in block_rows:
            output.extend(rows)
    return output
from hamiformer.models.hamiballs_committed import rollout_hamiballs_committed_edges
from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets, apply_hamiballs_affine_jet
from hamiformer.training.training_schedule import set_optimizer_learning_rate
from hamiformer.training.hamiballs_formal import previous_gate_sequence
from hamiformer.training.hamiballs_recovery import module_digest
from hamiformer.training.hamiballs_integrator_balanced_gate import position_equivalent_error
from hamiformer.training.hamiballs_trajectory import per_object_convex_projection_gate_loss
from hamiformer.training.hamiballs1 import fixed_carrier as fixed
from hamiformer.training import base as stage_a
from hamiformer.training.hamiballs1 import residual_models as gate_tools
from hamiformer.training.hamiballs1 import gate_models as gate_training_gate
from hamiformer.training.hamiballs1 import carrier_cache as carrier_cache
from hamiformer.training.hamiballs1.carrier_data import _sha256_file, validate_selector_dataset_contract

def _hamiltonian_digest(module: torch.nn.Module | None) -> str:
    return 'identity_affine_map' if module is None else module_digest(module)
TensorCache = dict[str, torch.Tensor]

def _packed_previous_gate_sequence(committed_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if committed_gate.ndim == 4:
        if committed_gate.shape[-1] != 2:
            raise ValueError('component gate cache requires q/p as its final axis')
        incoming = torch.cat((committed_gate.new_ones(committed_gate.shape[0], 1, committed_gate.shape[2], committed_gate.shape[3]), committed_gate[:, :-1]), dim=1)
        return (incoming, committed_gate.mean(dim=-1))
    incoming = previous_gate_sequence(committed_gate)
    if incoming.ndim == 2:
        incoming = incoming[:, :, None].expand_as(committed_gate)
    return (incoming, committed_gate)

def _effective_previous_gate_sequence(committed_gate: torch.Tensor, initial: torch.Tensor, *, component_gate: bool) -> torch.Tensor:
    if component_gate:
        if committed_gate.ndim != 4 or initial.shape != committed_gate[:, 0].shape:
            raise ValueError('component replay initial gate must be [B,K,2]')
        return torch.cat((initial[:, None], committed_gate[:, :-1]), dim=1)
    return previous_gate_sequence(committed_gate, initial=initial)

def _shared_history_initial_previous_g(gate: torch.nn.Module, cached: torch.Tensor) -> torch.Tensor:
    if bool(getattr(gate, 'component_history_gate', False)):
        return cached
    if cached.ndim == 3 and cached.shape[-1] == 2:
        return cached.mean(dim=-1)
    return cached

def _objective_for_update(arm: str, update: int, updates_per_stage: int) -> str | None:
    if arm != 'uniform_then_calibrated':
        return arm if update <= updates_per_stage else None
    return 'uniform' if update <= updates_per_stage else 'calibrated_utility_bce'

def _stage_update(update: int, updates_per_stage: int) -> int:
    return (update - 1) % updates_per_stage + 1

def _field_tensors(field: Any, *, x0: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> TensorCache:
    rollout = field.rollout
    if rollout is None or field.d_tokens is None or field.jets is None:
        raise RuntimeError('packed gate cache received an incomplete field')
    committed_gate = rollout.gate.detach()
    incoming, teacher_gate = _packed_previous_gate_sequence(committed_gate)
    _unused, teacher = per_object_convex_projection_gate_loss(teacher_gate, rollout.hr_candidate, rollout.d_candidate, target)
    previous_heun_defect = getattr(field, 'previous_heun_defect', None)
    if previous_heun_defect is None:
        previous_heun_defect = torch.zeros_like(field.state)
    if previous_heun_defect.shape != field.state.shape:
        raise ValueError('previous Heun defect must align with the RF field state')
    base_hr_candidate = getattr(rollout, 'base_hr_candidate', None)
    if base_hr_candidate is None:
        base_hr_candidate = rollout.hr_candidate
    base_gate = getattr(rollout, 'base_gate', None)
    if base_gate is None:
        base_gate = rollout.gate
    return {'d_tokens': field.d_tokens.detach(), 'noisy': field.state.detach(), 'x0': x0.detach(), 'previous_mixed': rollout.previous_mixed.detach(), 'h_candidate': rollout.h_candidate.detach(), 'hr_candidate': rollout.hr_candidate.detach(), 'base_hr_candidate': base_hr_candidate.detach(), 'base_gate': base_gate.detach(), 'd_candidate': rollout.d_candidate.detach(), 'mixed': rollout.mixed.detach(), 'attrs': attrs.detach(), 'tau': field.tau.detach(), 'physical_time': physical_time[:, 1:].detach(), 'previous_g': incoming.detach(), 'residual_hidden': rollout.residual_hidden.detach(), 'previous_heun_defect': previous_heun_defect.detach(), 'teacher': teacher.detach(), 'target': target.detach(), 'field_start_previous_mixed': rollout.previous_mixed[:, 0].detach(), 'field_start_previous_g': incoming[:, 0].detach(), 'jet_matrix': field.jets.matrix.detach(), 'jet_offset': field.jets.offset.detach()}

def _cat(rows: list[TensorCache]) -> TensorCache:
    if not rows:
        raise ValueError('cannot pack an empty gate cache')
    keys = set(rows[0])
    if any((set(row) != keys for row in rows)):
        raise ValueError('packed gate cache keys drifted')
    return {name: torch.cat([row[name] for row in rows], dim=0) for name in rows[0]}

def _forward(gate: torch.nn.Module, cache: TensorCache) -> torch.Tensor:
    value, _hidden = gate(cache['d_tokens'], cache['noisy'], cache['x0'], cache['previous_mixed'], cache['h_candidate'], cache['hr_candidate'], cache['d_candidate'], cache['attrs'], cache['tau'], cache['physical_time'], cache['previous_g'], residual_hidden=cache['residual_hidden'])
    return value

def _veto_forward(gate: HamiBallsComponentRiskVetoGate, cache: TensorCache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, _hidden, base, risk = gate.forward_with_risk(cache['d_tokens'], cache['noisy'], cache['x0'], cache['previous_mixed'], cache['h_candidate'], cache['hr_candidate'], cache['d_candidate'], cache['attrs'], cache['tau'], cache['physical_time'], cache['previous_g'], residual_hidden=cache['residual_hidden'])
    return (value, base, risk)

def _online_forward(gate: torch.nn.Module, residual: torch.nn.Module, cache: TensorCache, *, state_scale: torch.Tensor, q_dim: int) -> tuple[torch.Tensor, TensorCache]:
    jets = HamiBallsAffineJets(matrix=cache['jet_matrix'], offset=cache['jet_offset'], health=None)
    object_scale = state_scale.reshape(1, 1, -1)

    def h_builder(edge: int, previous: torch.Tensor) -> torch.Tensor:
        raw = apply_hamiballs_affine_jet(jets.matrix[:, edge], jets.offset[:, edge], previous * object_scale, q_dim=q_dim)
        return raw / object_scale
    initial_previous_g = _shared_history_initial_previous_g(gate, cache['field_start_previous_g'])
    replay = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=gate, d_tokens=cache['d_tokens'], noisy=cache['noisy'], x0=cache['x0'], d_candidate=cache['d_candidate'], attrs=cache['attrs'], tau=cache['tau'], physical_time=cache['physical_time'], initial_previous_mixed=cache['field_start_previous_mixed'], initial_previous_g=initial_previous_g)
    component_gate = bool(getattr(gate, 'component_gate', False))
    teacher_gate = replay.gate.mean(dim=-1) if component_gate else replay.gate
    effective_initial_g = initial_previous_g
    if component_gate and effective_initial_g.ndim == 2:
        effective_initial_g = effective_initial_g[..., None].expand(-1, -1, 2)
    _unused, teacher = per_object_convex_projection_gate_loss(teacher_gate, replay.hr_candidate, replay.d_candidate, cache['target'])
    effective = dict(cache)
    effective.update({'previous_mixed': replay.previous_mixed, 'h_candidate': replay.h_candidate, 'hr_candidate': replay.hr_candidate, 'mixed': replay.mixed, 'previous_g': _effective_previous_gate_sequence(replay.gate, effective_initial_g, component_gate=component_gate), 'residual_hidden': replay.residual_hidden, 'teacher': teacher})
    return (replay.gate, effective)

def _online_observable_disjoint_logits_forward(gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, residual: torch.nn.Module, cache: TensorCache, *, state_scale: torch.Tensor, q_dim: int, semi_gradient: bool=False) -> tuple[torch.Tensor, TensorCache, torch.Tensor]:
    captured_q: list[torch.Tensor] = []
    captured_p: list[torch.Tensor] = []
    captured_q_recovery: list[torch.Tensor] = []
    captured_p_recovery: list[torch.Tensor] = []
    captured_common: list[torch.Tensor] = []

    def capture(rows: list[torch.Tensor]):

        def hook(_module: torch.nn.Module, _arguments: tuple[Any, ...], output: torch.Tensor) -> None:
            rows.append(output)
        return hook
    if semi_gradient:
        with torch.no_grad():
            _sampled_value, sampled_effective = _online_forward(gate, residual, cache, state_scale=state_scale, q_dim=q_dim)
        effective = {name: tensor.detach() for name, tensor in sampled_effective.items()}
        if effective['previous_g'].ndim == 4:
            effective['previous_g'] = effective['previous_g'].mean(dim=-1)
    else:
        effective = cache
    hooks = [gate.q_delta_output.register_forward_hook(capture(captured_q)), gate.p_delta_output.register_forward_hook(capture(captured_p)), gate.common_output.register_forward_hook(capture(captured_common))]
    staged = isinstance(gate, HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate)
    if staged:
        hooks.extend((gate.q_recovery_output.register_forward_hook(capture(captured_q_recovery)), gate.p_recovery_output.register_forward_hook(capture(captured_p_recovery))))
    try:
        if semi_gradient:
            value = _forward(gate, effective)
        else:
            value, effective = _online_forward(gate, residual, cache, state_scale=state_scale, q_dim=q_dim)
    finally:
        for hook in hooks:
            hook.remove()
    batch, edges, objects = value.shape[:3]
    if not all((len(rows) == edges for rows in (captured_q, captured_p, captured_common))):
        raise ValueError('observable disjoint logit capture count changed')

    def stack(rows: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([row.reshape(batch, objects) for row in rows], dim=1)
    common = stack(captured_common)
    q_delta = stack(captured_q)
    p_delta = stack(captured_p)
    if staged:
        if not all((len(rows) == edges for rows in (captured_q_recovery, captured_p_recovery))):
            raise ValueError('staged recovery logit capture count changed')
        q_delta = q_delta + stack(captured_q_recovery)
        p_delta = p_delta + stack(captured_p_recovery)
    temperature = gate.log_temperature.clamp(-6.0, 6.0).exp().detach()
    local_logits = torch.stack((common + stack(captured_q), common + stack(captured_p)), dim=-1) / temperature
    logits = torch.stack((common + q_delta, common + p_delta), dim=-1) / temperature
    if logits.shape != (batch, edges, objects, 2):
        raise ValueError('observable disjoint effective logit shape changed')
    if local_logits.shape != logits.shape:
        raise ValueError('observable disjoint local logit shape changed')
    return (value, effective, logits, local_logits)

def _select(cache: TensorCache, indices: torch.Tensor) -> TensorCache:
    return {name: value[indices] for name, value in cache.items()}

def _sample_aggregate_replay(blocks: tuple[tuple[TensorCache, ...], ...], *, batch_size: int, generator: torch.Generator) -> tuple[tuple[TensorCache, ...], int]:
    if not blocks or batch_size < 1:
        raise ValueError('aggregate replay requires blocks and a positive batch')
    noise_count = len(blocks[0])
    if noise_count < 1 or any((len(block) != noise_count for block in blocks)):
        raise ValueError('aggregate replay noise-cache count drifted')
    sequence_count = int(blocks[0][0]['teacher'].shape[0])
    if sequence_count < 1:
        raise ValueError('aggregate replay block is empty')
    reference_keys = set(blocks[0][0])
    reference_shapes = {name: tuple(value.shape[1:]) for name, value in blocks[0][0].items()}
    for block in blocks:
        for cache in block:
            if set(cache) != reference_keys:
                raise ValueError('aggregate replay cache keys drifted')
            if int(cache['teacher'].shape[0]) != sequence_count or any((int(value.shape[0]) != sequence_count or tuple(value.shape[1:]) != reference_shapes[name] for name, value in cache.items())):
                raise ValueError('aggregate replay cache shape drifted')
    block_ids = torch.randint(len(blocks), (batch_size,), generator=generator, device='cpu')
    row_ids = torch.randint(sequence_count, (batch_size,), generator=generator, device='cpu')
    sampled: list[TensorCache] = []
    for noise_index in range(noise_count):
        output = {name: value.new_empty((batch_size, *value.shape[1:])) for name, value in blocks[0][noise_index].items()}
        for block_index, block in enumerate(blocks):
            positions_cpu = torch.nonzero(block_ids == block_index, as_tuple=False).flatten()
            if positions_cpu.numel() == 0:
                continue
            cache = block[noise_index]
            for name, value in cache.items():
                positions = positions_cpu.to(device=value.device)
                indices = row_ids[positions_cpu].to(device=value.device)
                output[name].index_copy_(0, positions, value[indices])
        sampled.append(output)
    return (tuple(sampled), len(blocks) * sequence_count)

def _global_sequence_projection_regret_loss(value: torch.Tensor, cache: TensorCache) -> torch.Tensor:
    if value.shape != cache['teacher'].shape:
        raise ValueError('global projection regret gate and teacher must align')
    delta = cache['hr_candidate'] - cache['d_candidate']
    mixed = cache['d_candidate'] + value[..., None] * delta
    oracle = cache['d_candidate'] + cache['teacher'].detach()[..., None] * delta
    midpoint = cache['d_candidate'] + 0.5 * delta
    target = cache['target']
    mixed_error = (mixed - target).square().mean(dim=-1)
    oracle_error = (oracle - target).square().mean(dim=-1).detach()
    midpoint_error = (midpoint - target).square().mean(dim=-1).detach()
    regret = (mixed_error - oracle_error).clamp_min(0.0)
    midpoint_regret = (midpoint_error - oracle_error).clamp_min(0.0)
    reduction_dims = tuple(range(1, regret.ndim))
    sequence_regret = regret.mean(dim=reduction_dims)
    reference = midpoint_regret.mean(dim=reduction_dims).detach()
    epsilon = torch.finfo(regret.dtype).eps
    return sequence_regret.sum() / (reference.sum() + epsilon)

def _dual_qp_global_sequence_projection_regret_losses(value: torch.Tensor, cache: TensorCache, *, state_scale: torch.Tensor, q_dim: int, frame_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != (*cache['target'].shape[:-1], 2):
        raise ValueError('dual global recurrent gate must be [B,F,K,2]')
    d_q, d_p, _ = position_equivalent_error(cache['d_candidate'], cache['target'], attrs=cache['attrs'], state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt)
    delta_q, delta_p, _ = position_equivalent_error(cache['hr_candidate'], cache['d_candidate'], attrs=cache['attrs'], state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt)

    def component_loss(gate: torch.Tensor, d_error: torch.Tensor, disagreement: torch.Tensor) -> torch.Tensor:
        quadratic = disagreement.square().mean(dim=-1)
        linear = (d_error * disagreement).mean(dim=-1)
        oracle_gate = torch.where(quadratic > 0.0, -linear / quadratic.clamp_min(torch.finfo(quadratic.dtype).tiny), torch.zeros_like(quadratic)).clamp(0.0, 1.0).detach()

        def error_at(weight: torch.Tensor | float) -> torch.Tensor:
            if isinstance(weight, torch.Tensor):
                mixed = d_error + weight[..., None] * disagreement
            else:
                mixed = d_error + float(weight) * disagreement
            return mixed.square().mean(dim=-1)
        oracle_error = error_at(oracle_gate).detach()
        midpoint_error = error_at(0.5).detach()
        regret = (error_at(gate) - oracle_error).clamp_min(0.0)
        reference = (midpoint_error - oracle_error).clamp_min(0.0)
        epsilon = gate.new_tensor(torch.finfo(gate.dtype).eps)
        return regret.sum() / (reference.sum() + epsilon)
    q_loss = component_loss(value[..., 0], d_q, delta_q)
    p_loss = component_loss(value[..., 1], d_p, delta_p)
    if not bool(torch.isfinite(q_loss)) or not bool(torch.isfinite(p_loss)):
        raise FloatingPointError('dual global recurrent loss is non-finite')
    return (q_loss, p_loss)

def _dual_qp_source_object_upper_semideviation_losses(value: torch.Tensor, cache: TensorCache, *, state_scale: torch.Tensor, q_dim: int, frame_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != (*cache['target'].shape[:-1], 2):
        raise ValueError('dual upper-semideviation gate must be [B,F,K,2]')
    d_q, d_p, _ = position_equivalent_error(cache['d_candidate'], cache['target'], attrs=cache['attrs'], state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt)
    delta_q, delta_p, _ = position_equivalent_error(cache['hr_candidate'], cache['d_candidate'], attrs=cache['attrs'], state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt)

    def component_loss(gate: torch.Tensor, d_error: torch.Tensor, disagreement: torch.Tensor) -> torch.Tensor:
        mixed_error = (d_error + gate[..., None] * disagreement).square().mean(-1)
        d_mse = d_error.square().mean(-1).detach()
        hr_mse = (d_error + disagreement).square().mean(-1).detach()
        source_object_harm = (mixed_error - d_mse).mean(dim=1).clamp_min(0.0)
        endpoint_harm = (hr_mse - d_mse).mean(dim=1).clamp_min(0.0)
        numerator = torch.linalg.vector_norm(source_object_harm) / math.sqrt(source_object_harm.numel())
        denominator = (torch.linalg.vector_norm(endpoint_harm) / math.sqrt(endpoint_harm.numel())).detach()
        epsilon = gate.new_tensor(torch.finfo(gate.dtype).eps)
        return torch.where(denominator > epsilon, numerator / denominator.clamp_min(epsilon), numerator * 0.0)
    q_loss = component_loss(value[..., 0], d_q, delta_q)
    p_loss = component_loss(value[..., 1], d_p, delta_p)
    if not bool(torch.isfinite(q_loss)) or not bool(torch.isfinite(p_loss)):
        raise FloatingPointError('dual upper-semideviation loss is non-finite')
    return (q_loss, p_loss)
QP_NO_HARM_OBJECTIVE = 'global_sequence_projection_regret_with_source_object_qp_no_harm'
QP_NO_HARM_DEFINITION = 'global_sequence_projection_regret_plus_equal_qp_source_object_edge_mean_positive_mixed_vs_d_over_positive_hr_vs_d_v1'
DUAL_OBSERVABLE_DISJOINT_ENDPOINT_RECURRENT_OBJECTIVE = 'dual_qp_observable_disjoint_endpoint_plus_global_recurrent_regret'
DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE = 'dual_qp_observable_disjoint_endpoint_global_recurrent_upper_semideviation'
DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_DEFINITION = 'frozen_scalar_global_sequence_policy_plus_component_disjoint_width4_observable_deltas_equal_mean_endpoint_bce_global_recurrent_regret_and_source_object_positive_mixed_vs_d_upper_rms_semideviation_independent_qp_v1'
COMPONENT_TAIL_COMPONENT_SPECIFIC_Q_NO_TAIL_P_TAIL = False
NO_HARM_FINAL_QP_NO_HARM_WEIGHT = 0.0

def _source_object_balanced_qp_no_harm_regret_losses(value: torch.Tensor, cache: TensorCache, *, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != cache['teacher'].shape:
        raise ValueError('q/p no-harm gate and teacher must align')
    state_dim = int(cache['target'].shape[-1])
    if not 0 < q_dim < state_dim:
        raise ValueError('q/p no-harm q_dim does not split the state')
    d_candidate = cache['d_candidate']
    hr_candidate = cache['hr_candidate']
    delta = hr_candidate - d_candidate
    mixed = d_candidate + value[..., None] * delta
    target = cache['target']
    losses: list[torch.Tensor] = []
    for component in (slice(0, q_dim), slice(q_dim, state_dim)):
        mixed_error = (mixed[..., component] - target[..., component]).square().mean(dim=-1)
        d_error = (d_candidate[..., component] - target[..., component]).square().mean(dim=-1).detach()
        hr_error = (hr_candidate[..., component] - target[..., component]).square().mean(dim=-1).detach()
        positive_harm = (mixed_error - d_error).clamp_min(0.0).mean(dim=1)
        endpoint_scale = (hr_error - d_error).clamp_min(0.0).mean(dim=1)
        epsilon = torch.finfo(positive_harm.dtype).eps
        ratio = torch.where(endpoint_scale > epsilon, positive_harm / endpoint_scale.clamp_min(epsilon), torch.zeros_like(positive_harm))
        losses.append(ratio.mean())
    return (losses[0], losses[1])

def _dual_qp_source_object_balanced_no_harm_regret_losses(value: torch.Tensor, cache: TensorCache, *, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != (*cache['target'].shape[:-1], 2):
        raise ValueError('dual q/p no-harm gate must be [B,F,K,2]')
    state_dim = int(cache['target'].shape[-1])
    if not 0 < q_dim < state_dim:
        raise ValueError('dual q/p no-harm q_dim does not split the state')
    d_candidate = cache['d_candidate']
    hr_candidate = cache['hr_candidate']
    target = cache['target']
    delta = hr_candidate - d_candidate
    losses: list[torch.Tensor] = []
    for gate, component in ((value[..., 0], slice(0, q_dim)), (value[..., 1], slice(q_dim, state_dim))):
        mixed = d_candidate[..., component] + gate[..., None] * delta[..., component]
        mixed_error = (mixed - target[..., component]).square().mean(dim=-1)
        d_error = (d_candidate[..., component] - target[..., component]).square().mean(dim=-1).detach()
        hr_error = (hr_candidate[..., component] - target[..., component]).square().mean(dim=-1).detach()
        positive_harm = (mixed_error - d_error).clamp_min(0.0).mean(dim=1)
        endpoint_scale = (hr_error - d_error).clamp_min(0.0).mean(dim=1)
        epsilon = torch.finfo(positive_harm.dtype).eps
        ratio = torch.where(endpoint_scale > epsilon, positive_harm / endpoint_scale.clamp_min(epsilon), torch.zeros_like(positive_harm))
        losses.append(ratio.mean())
    return (losses[0], losses[1])

def _loss_parameter_gradient_geometry(base_loss: torch.Tensor, auxiliary_loss: torch.Tensor, module: torch.nn.Module) -> dict[str, float]:
    parameters = tuple((parameter for parameter in module.parameters() if parameter.requires_grad))
    if not parameters:
        raise ValueError('component gradient audit requires trainable parameters')
    base_gradients = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
    auxiliary_gradients = torch.autograd.grad(auxiliary_loss, parameters, retain_graph=True, allow_unused=True)

    def vector(rows: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
        return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1) for parameter, gradient in zip(parameters, rows, strict=True)])
    base = vector(base_gradients)
    auxiliary = vector(auxiliary_gradients)
    base_norm = base.norm()
    auxiliary_norm = auxiliary.norm()
    denominator = base_norm * auxiliary_norm
    cosine = torch.dot(base, auxiliary) / denominator if float(denominator.detach().cpu()) > 0.0 else torch.full((), float('nan'), device=base.device, dtype=base.dtype)
    return {'base_gradient_norm': float(base_norm.cpu()), 'auxiliary_gradient_norm': float(auxiliary_norm.cpu()), 'base_auxiliary_gradient_cosine': float(cosine.cpu())}

def _dual_qp_endpoint_regret_balanced_raw_logit_bce(logits: torch.Tensor, cache: TensorCache, *, q_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != (*cache['teacher'].shape, 2):
        raise ValueError('dual raw endpoint logits must be [B,F,K,2]')
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('dual raw endpoint logits became non-finite')
    state_dim = int(cache['target'].shape[-1])
    if not 0 < q_dim < state_dim:
        raise ValueError('dual raw endpoint q_dim does not split the state')

    def component_loss(index: int, component: slice) -> torch.Tensor:
        target = cache['target'].detach()[..., component]
        d_error = (cache['d_candidate'].detach()[..., component] - target).square().mean(dim=-1)
        hr_error = (cache['hr_candidate'].detach()[..., component] - target).square().mean(dim=-1)
        d_side = d_error < hr_error
        hr_side = hr_error < d_error
        active = d_side | hr_side
        if not bool(active.any()):
            return logits[..., index].sum() * 0.0
        regret = (d_error - hr_error).abs()
        total_cells = logits.new_tensor(float(logits[..., index].numel()))
        epsilon = logits.new_tensor(torch.finfo(logits.dtype).eps)
        d_mass = (regret * d_side).sum()
        hr_mass = (regret * hr_side).sum()
        if bool(d_mass > epsilon) and bool(hr_mass > epsilon):
            weight = torch.where(d_side, 0.5 * total_cells * regret / d_mass, torch.where(hr_side, 0.5 * total_cells * regret / hr_mass, torch.zeros_like(regret)))
        else:
            active_mass = (regret * active).sum()
            weight = torch.where(active, total_cells * regret / active_mass.clamp_min(epsilon), torch.zeros_like(regret))
        return F.binary_cross_entropy_with_logits(logits[..., index], hr_side.to(logits.dtype), weight=weight, reduction='sum') / total_cells
    return (component_loss(0, slice(0, q_dim)), component_loss(1, slice(q_dim, state_dim)))

def _metrics(gate: torch.nn.Module, caches: tuple[TensorCache, ...], *, online_context: tuple[torch.nn.Module, torch.Tensor, int] | None=None) -> dict[str, float]:
    values: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    mixed_errors: list[np.ndarray] = []
    d_errors: list[np.ndarray] = []
    oracle_errors: list[np.ndarray] = []
    component_errors: dict[str, dict[str, list[np.ndarray]]] = {name: {kind: [] for kind in ('d', 'mixed', 'oracle')} for name in ('q', 'p')}
    gate.eval()
    with torch.no_grad():
        for cache in caches:
            rows: list[torch.Tensor] = []
            teacher_rows: list[torch.Tensor] = []
            mixed_error_rows: list[torch.Tensor] = []
            d_error_rows: list[torch.Tensor] = []
            oracle_error_rows: list[torch.Tensor] = []
            for start in range(0, cache['teacher'].shape[0], 256):
                selected = _select(cache, torch.arange(start, min(start + 256, cache['teacher'].shape[0]), device=cache['teacher'].device))
                if online_context is None:
                    value, effective = (_forward(gate, selected), selected)
                else:
                    residual, state_scale, q_dim = online_context
                    value, effective = _online_forward(gate, residual, selected, state_scale=state_scale, q_dim=q_dim)
                teacher = effective['teacher']
                delta = effective['hr_candidate'] - effective['d_candidate']
                if bool(getattr(gate, 'component_gate', False)):
                    if value.shape != (*teacher.shape, 2):
                        raise ValueError('dual metric gate shape changed')
                    assert online_context is not None
                    q_dim = int(online_context[2])
                    d_error = effective['d_candidate'] - effective['target']

                    def component_teacher(component: slice) -> torch.Tensor:
                        local_d = d_error[..., component]
                        local_delta = delta[..., component]
                        denominator = local_delta.square().mean(dim=-1)
                        numerator = -(local_d * local_delta).mean(dim=-1)
                        return torch.where(denominator > torch.finfo(denominator.dtype).eps, numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps), torch.zeros_like(denominator)).clamp(0.0, 1.0)
                    teacher = torch.stack([component_teacher(slice(0, q_dim)), component_teacher(slice(q_dim, delta.shape[-1]))], dim=-1)
                    weight = torch.cat([value[..., 0, None].expand(*value.shape[:-1], q_dim), value[..., 1, None].expand(*value.shape[:-1], delta.shape[-1] - q_dim)], dim=-1)
                    oracle_weight = torch.cat([teacher[..., 0, None].expand(*teacher.shape[:-1], q_dim), teacher[..., 1, None].expand(*teacher.shape[:-1], delta.shape[-1] - q_dim)], dim=-1)
                    mixed = effective['d_candidate'] + weight * delta
                    oracle = effective['d_candidate'] + oracle_weight * delta
                else:
                    mixed = effective['d_candidate'] + value[..., None] * delta
                    oracle = effective['d_candidate'] + teacher[..., None] * delta
                target = effective['target']
                if online_context is not None:
                    local_q_dim = int(online_context[2])
                    for component_name, component_slice in (('q', slice(0, local_q_dim)), ('p', slice(local_q_dim, target.shape[-1]))):
                        component_errors[component_name]['d'].append((effective['d_candidate'][..., component_slice] - target[..., component_slice]).square().mean(dim=-1).cpu().numpy())
                        component_errors[component_name]['mixed'].append((mixed[..., component_slice] - target[..., component_slice]).square().mean(dim=-1).cpu().numpy())
                        component_errors[component_name]['oracle'].append((oracle[..., component_slice] - target[..., component_slice]).square().mean(dim=-1).cpu().numpy())
                rows.append(value)
                teacher_rows.append(teacher)
                mixed_error_rows.append((mixed - target).square().mean(dim=-1))
                d_error_rows.append((effective['d_candidate'] - target).square().mean(dim=-1))
                oracle_error_rows.append((oracle - target).square().mean(dim=-1))
            value = torch.cat(rows, dim=0)
            teacher = torch.cat(teacher_rows, dim=0)
            values.append(value.cpu().numpy())
            teachers.append(teacher.cpu().numpy())
            mixed_errors.append(torch.cat(mixed_error_rows, dim=0).cpu().numpy())
            d_errors.append(torch.cat(d_error_rows, dim=0).cpu().numpy())
            oracle_errors.append(torch.cat(oracle_error_rows, dim=0).cpu().numpy())
    value = np.stack(values)
    teacher = np.stack(teachers)
    flat_value, flat_teacher = (value.reshape(-1), teacher.reshape(-1))
    left, right = (flat_teacher < 0.5, flat_teacher > 0.5)
    predicted = flat_value > 0.5
    object_axis = -2 if bool(getattr(gate, 'component_gate', False)) else -1
    value_centered = value - value.mean(axis=object_axis, keepdims=True)
    teacher_centered = teacher - teacher.mean(axis=object_axis, keepdims=True)
    d_mse = float(np.stack(d_errors).mean())
    mixed_mse = float(np.stack(mixed_errors).mean())
    oracle_mse = float(np.stack(oracle_errors).mean())
    headroom = d_mse - oracle_mse
    result = {'pearson': float(np.corrcoef(flat_value, flat_teacher)[0, 1]), 'object_centered_pearson': float(np.corrcoef(value_centered.reshape(-1), teacher_centered.reshape(-1))[0, 1]), 'side_balanced_accuracy': 0.5 * (float((~predicted[left]).mean()) + float(predicted[right].mean())), 'side_accuracy': float((predicted[left | right] == right[left | right]).mean()), 'gate_mean': float(value.mean()), 'gate_std': float(value.std()), 'gate_object_std': float(value.std(axis=object_axis).mean()), 'teacher_mean': float(teacher.mean()), 'teacher_std': float(teacher.std()), 'teacher_object_std': float(teacher.std(axis=object_axis).mean()), 'd_mse': d_mse, 'mixed_mse': mixed_mse, 'oracle_mse': oracle_mse, 'headroom_recovery': float((d_mse - mixed_mse) / headroom) if headroom > 0 else float('nan')}
    if online_context is not None:
        for index, name in enumerate(('q', 'p')):
            if bool(getattr(gate, 'component_gate', False)):
                local_value = value[..., index].reshape(-1)
                local_teacher = teacher[..., index].reshape(-1)
                local_left = local_teacher < 0.5
                local_right = local_teacher > 0.5
                local_predicted = local_value > 0.5
                result[f'{name}_gate_mean'] = float(local_value.mean())
                result[f'{name}_gate_std'] = float(local_value.std())
                result[f'{name}_teacher_mean'] = float(local_teacher.mean())
                result[f'{name}_pearson'] = float(np.corrcoef(local_value, local_teacher)[0, 1])
                result[f'{name}_side_balanced_accuracy'] = 0.5 * (float((~local_predicted[local_left]).mean()) + float(local_predicted[local_right].mean()))
                local_active = local_left | local_right
                result[f'{name}_side_accuracy'] = float((local_predicted[local_active] == local_right[local_active]).mean())
            local_d_mse = float(np.concatenate(component_errors[name]['d']).mean())
            local_mixed_mse = float(np.concatenate(component_errors[name]['mixed']).mean())
            local_oracle_mse = float(np.concatenate(component_errors[name]['oracle']).mean())
            local_headroom = local_d_mse - local_oracle_mse
            result[f'{name}_d_mse'] = local_d_mse
            result[f'{name}_mixed_mse'] = local_mixed_mse
            result[f'{name}_oracle_mse'] = local_oracle_mse
            result[f'{name}_headroom_recovery'] = float((local_d_mse - local_mixed_mse) / local_headroom) if local_headroom > 0.0 else float('nan')
    gate.train()
    return result

def _sample_permutation_block(*, size: int, count: int, seed: int, offset: int) -> torch.Tensor:
    if offset < 0 or count < 1 or offset + count > size:
        raise ValueError('sample-offset block exceeds the train split')
    permutation = torch.randperm(size, generator=torch.Generator(device='cpu').manual_seed(seed))
    return permutation[offset:offset + count]

def _sample_permutation_block_excluding_scenes(*, size: int, count: int, seed: int, offset: int, scene_ids: list[str], excluded_scenes: set[str]) -> tuple[torch.Tensor, list[int]]:
    if len(scene_ids) != size:
        raise ValueError('scene-excluded sampling requires a complete scene ledger')
    if offset < 0 or count < 1 or offset >= size:
        raise ValueError('scene-excluded sample offset is outside the train split')
    permutation = torch.randperm(size, generator=torch.Generator(device='cpu').manual_seed(seed))
    selected: list[int] = []
    positions: list[int] = []
    for position in range(offset, size):
        index = int(permutation[position])
        if scene_ids[index] in excluded_scenes:
            continue
        selected.append(index)
        positions.append(position)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('scene exclusions exhausted the train permutation')
    return (torch.tensor(selected, dtype=torch.long), positions)

def _self_policy_refresh_schedule(*, updates: int, refresh_every: int, refresh_sources: int) -> tuple[tuple[int, int, int, int], ...]:
    if refresh_every == 0 and refresh_sources == 0:
        return tuple()
    if updates < 1 or refresh_every < 1 or refresh_sources < 1:
        raise ValueError('self-policy refresh requires positive updates, refresh-every and sources')
    blocks = math.ceil(updates / refresh_every)
    return tuple(((block, 1 + block * refresh_every, block * refresh_sources, (block + 1) * refresh_sources) for block in range(blocks)))

def _scaled_staged_recovery_collector(candidate: torch.nn.Module, *, scale: float) -> HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate:
    if type(candidate) is not HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate:
        raise TypeError('recovery-carrier scaling requires the exact staged recovery gate')
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError('recovery carrier scale must lie in [0,1]')
    collector = copy.deepcopy(candidate)
    with torch.no_grad():
        for output in (collector.q_recovery_output, collector.p_recovery_output):
            output.weight.mul_(scale)
            output.bias.mul_(scale)
    return collector

class _UniformRandomCarrierGate(torch.nn.Module):

    def __init__(self, candidate: torch.nn.Module, *, seed: int) -> None:
        super().__init__()
        if seed < 1:
            raise ValueError('uniform random carrier seed must be positive')
        self.candidate = candidate
        self.seed = int(seed)
        self.calls = 0
        self.per_object_gate = True
        self.requires_residual_hidden = True
        for name in ('component_gate', 'component_history_gate', 'requires_previous_d_token'):
            setattr(self, name, bool(getattr(candidate, name, False)))

    def forward_step(self, *args: Any, **kwargs: Any):
        _value, hidden = self.candidate.forward_step(*args, **kwargs)
        parts = getattr(self, '_logical_rng_parts', None)
        if parts is not None:
            call = self.calls - self._logical_rng_call_origin
            values = []
            for count, seed, offset in parts:
                generator = torch.Generator(device=_value.device).manual_seed(seed + offset + call)
                values.append(torch.rand((count, *_value.shape[1:]), dtype=_value.dtype, device=_value.device, generator=generator))
            self.calls += 1
            value = torch.cat(values, dim=0)
            if value.shape != _value.shape:
                raise AssertionError('logical random-route batch shape drift')
            return (value, hidden)
        generator = torch.Generator(device=_value.device).manual_seed(self.seed + self.calls)
        self.calls += 1
        value = torch.rand(_value.shape, dtype=_value.dtype, device=_value.device, generator=generator)
        return (value, hidden)

def _self_policy_collector_provenance(candidate: torch.nn.Module, collector: torch.nn.Module, *, uniform_random: bool, recovery_scale: float | None, stage_parent: bool=False, ema_parent: bool=False, ema_beta: float | None=None, ema_teacher_updates: int | None=None) -> dict[str, Any]:
    candidate_digest = module_digest(candidate)
    collector_digest = module_digest(collector)
    if stage_parent and ema_parent:
        raise AssertionError('collector cannot be both frozen-parent and EMA-parent')
    if ema_parent:
        if collector is candidate or uniform_random or recovery_scale is not None or (ema_beta is None) or (ema_teacher_updates is None):
            raise AssertionError('EMA-parent collector aliases candidate or lost metadata')
        behavior = 'same_run_ema_teacher'
        collector_is_current_candidate = False
    elif stage_parent:
        if collector is candidate or uniform_random or recovery_scale is not None:
            raise AssertionError('stage-parent collector aliases candidate or mixed modes')
        behavior = 'fixed_same_run_stage_parent'
        collector_is_current_candidate = False
    elif uniform_random:
        if type(collector) is not _UniformRandomCarrierGate or collector.candidate is not candidate or module_digest(collector.candidate) != candidate_digest or (recovery_scale is not None):
            raise AssertionError('uniform-random collector lost its live candidate')
        behavior = 'seeded_uniform_random_with_live_candidate_hidden'
        collector_is_current_candidate = False
    elif recovery_scale is not None:
        if collector is candidate:
            raise AssertionError('scaled recovery collector unexpectedly aliases candidate')
        behavior = 'scaled_staged_recovery_candidate_copy'
        collector_is_current_candidate = False
    else:
        if collector is not candidate or collector_digest != candidate_digest:
            raise AssertionError('self-policy collector is not the live candidate')
        behavior = 'current_candidate'
        collector_is_current_candidate = True
    return {'candidate_digest_before_collection': candidate_digest, 'collector_digest': collector_digest, 'collector_behavior': behavior, 'collector_is_current_candidate': collector_is_current_candidate, 'collector_is_stage_parent': bool(stage_parent), 'collector_is_ema_teacher': bool(ema_parent), 'ema_beta': float(ema_beta) if ema_parent else None, 'ema_teacher_updates': int(ema_teacher_updates) if ema_parent else None, 'ema_student_digest_before_collection': candidate_digest if ema_parent else None}

def _scene_exclusion_mode_allowed(*, self_policy_refresh: bool, capacity_audit: bool, disjoint_gate_resume: bool) -> bool:
    return bool(self_policy_refresh or capacity_audit or disjoint_gate_resume)

def _observable_disjoint_clone_reference(parent_value: torch.Tensor, *, disjoint_gate_resume: bool) -> torch.Tensor:
    if disjoint_gate_resume:
        if parent_value.ndim < 1 or parent_value.shape[-1] != 2:
            raise ValueError('observable-disjoint resume parent is not q/p-valued')
        return parent_value
    return parent_value[..., None]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument('--gate-checkpoint', type=Path)
    parser.add_argument('--dataset-root', type=Path, default=None, help='Independent selector-calibration root; parent state/attribute scales are reused.')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--fit-samples', type=int, default=256)
    parser.add_argument('--holdout-samples', type=int, default=64)
    parser.add_argument('--collection-batch', type=int, default=64)
    parser.add_argument('--sequence-batch', type=int, default=64)
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--self-policy-refresh-every', type=int, default=0, help='Recollect one committed training carrier under the current single-arm candidate before every N optimizer updates; zero uses one fixed carrier cache.')
    parser.add_argument('--self-policy-refresh-sources', type=int, default=0, help='Number of permutation-disjoint training sources in each self-policy refresh block.')
    parser.add_argument('--self-policy-aggregate-replay', action='store_true', help='Keep every in-run self-policy refresh cache and sample updates from their accumulated DAgger dataset instead of replacing it.')
    parser.add_argument('--self-policy-uniform-random-carriers', action='store_true', help="Collect each refresh block with iid Uniform[0,1] scalar or q/p commitments while preserving the current candidate's hidden-state evolution.")
    parser.add_argument('--self-policy-stage-parent-carriers', action='store_true', help='Refresh fresh source blocks under the frozen same-run stage-parent gate instead of the live candidate; this is a target-policy stability diagnostic and does not retain old source caches.')
    parser.add_argument('--noise-seeds', default='270491,270492')
    parser.add_argument('--field-indices', default='0,5,11,17')
    parser.add_argument('--arms', required=True, help='Comma-separated subset of global_sequence_projection_regret, global_sequence_projection_regret_with_source_object_qp_no_harm, and dual_qp_observable_disjoint_endpoint_global_recurrent_upper_semideviation.')
    parser.add_argument('--qp-no-harm-weight', type=float, default=0.0, help='Fixed auxiliary coefficient for the registered global-regret + equal q/p source-object no-harm objective; must be zero for every other arm.')
    parser.add_argument('--sample-seed', type=int, default=270489)
    parser.add_argument('--optimizer-sample-seed', type=int, default=270493, help='Independent deterministic RNG for optimizer sequence minibatches. Training configurations pass a run-seed-42 namespaced derivative explicitly.')
    parser.add_argument('--sample-offset', type=int, default=0, help='Offset into the single sample-seed permutation. This permits deterministic disjoint policy-refresh blocks without changing seeds.')
    parser.add_argument('--sample-excluded-scene-configuration', type=Path, default=None, help='Optional immutable result-free train-audit configuration whose scene IDs are excluded from every fit/hold source row.')
    parser.add_argument('--maximum-lr', type=float, default=0.0003)
    parser.add_argument('--minimum-lr', type=float, default=3e-06)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--online-recurrence', action='store_true', help='Cache only the RF-field start/frozen observations and rebuild all 48 physical-edge mixed/previous-g states with the current gate.')
    parser.add_argument('--observable-disjoint-component-gate-expansion', action='store_true', help='Freeze a scalar checkpoint and train two parameter-disjoint zero-output width-4 q/p deltas over residual observables.')
    parser.add_argument('--observable-disjoint-delta-width', type=int, default=4, help='Hidden width of each parameter-disjoint observable q/p tower; the default width is four.')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('refusing to overwrite gate training output')
    if args.fit_samples < 1 or args.holdout_samples < 1 or args.updates < 1:
        raise ValueError('packed-cache sizes must be positive')
    if not 0.0 < args.minimum_lr <= args.maximum_lr:
        raise ValueError('packed-cache learning rates must be positive and ordered')
    if args.fit_samples % args.collection_batch or (args.holdout_samples > args.collection_batch and args.holdout_samples % args.collection_batch):
        raise ValueError('fit and holdout samples must divide into collection batches')
    noise_seeds = [int(value) for value in args.noise_seeds.split(',')]
    field_indices = [int(value) for value in args.field_indices.split(',')]
    if len(noise_seeds) not in {1, 2} or len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError('gate training requires one or two distinct noise seeds')
    if len(field_indices) != len(set(field_indices)) or min(field_indices) < 0:
        raise ValueError('field-indices must be unique nonnegative integers')
    arms = [value.strip() for value in args.arms.split(',') if value.strip()]
    allowed_arms = {'global_sequence_projection_regret', 'global_sequence_projection_regret_with_source_object_qp_no_harm', 'dual_qp_observable_disjoint_endpoint_global_recurrent_upper_semideviation'}
    if not arms or len(arms) != len(set(arms)) or (not set(arms) <= allowed_arms):
        raise ValueError('arms must be a unique nonempty subset of the supported objectives')
    self_policy_refresh_schedule = _self_policy_refresh_schedule(updates=args.updates, refresh_every=args.self_policy_refresh_every, refresh_sources=args.self_policy_refresh_sources)
    self_policy_refresh = bool(self_policy_refresh_schedule)
    refresh_blocks = len(self_policy_refresh_schedule)
    if self_policy_refresh:
        expected_fit_samples = refresh_blocks * args.self_policy_refresh_sources
        if args.fit_samples != expected_fit_samples:
            raise ValueError('self-policy refresh fit-samples must equal ceil(updates/refresh-every)*refresh-sources')
        if args.self_policy_refresh_sources % args.collection_batch:
            raise ValueError('self-policy refresh sources must divide into collection batches')
        if len(arms) != 1:
            raise ValueError('self-policy refresh requires exactly one one-stage arm')
        if not args.online_recurrence:
            raise ValueError('self-policy refresh requires online recurrence')
    if args.self_policy_aggregate_replay and (not self_policy_refresh):
        raise ValueError('aggregate replay requires self-policy refresh')
    if args.self_policy_uniform_random_carriers and (not self_policy_refresh):
        raise ValueError('uniform random carriers require self-policy refresh')
    if args.self_policy_uniform_random_carriers and args.self_policy_stage_parent_carriers:
        raise ValueError('uniform random carriers cannot be combined with other collector modes')
    if args.self_policy_stage_parent_carriers and (not self_policy_refresh):
        raise ValueError('stage-parent carriers require refresh and exclude stratification')
    recovery_carrier_scales = tuple()
    if args.sample_excluded_scene_configuration is not None and (not _scene_exclusion_mode_allowed(self_policy_refresh=self_policy_refresh, capacity_audit=False, disjoint_gate_resume=False)):
        raise ValueError('scene exclusion is restricted to registered train-only modes')
    if arms in ([QP_NO_HARM_OBJECTIVE],):
        if not math.isfinite(args.qp_no_harm_weight) or args.qp_no_harm_weight <= 0.0:
            raise ValueError('q/p no-harm objective requires one positive finite weight')
    elif args.qp_no_harm_weight != 0.0:
        raise ValueError('q/p no-harm weight is forbidden for every other objective')
    if args.observable_disjoint_component_gate_expansion:
        if args.observable_disjoint_delta_width < 1:
            raise ValueError('observable-disjoint delta width must be positive')
        if args.gate_checkpoint is None:
            raise ValueError('observable disjoint gate requires scalar checkpoint')
        if arms not in ([DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE],):
            raise ValueError('observable disjoint gate requires its endpoint or recurrent objective')
        if not args.online_recurrence:
            raise ValueError('observable disjoint gate requires online recurrence')
    if args.observable_disjoint_delta_width != 4 and (not args.observable_disjoint_component_gate_expansion):
        raise ValueError('observable-disjoint delta width requires its expansion')
    device = torch.device(args.device)
    registration = carrier_cache._load_registration(args.registration)
    main_parent, parent_contract, d, h, residual, frozen = gate_training_gate._load_gate_training_models(registration, device=device)
    config = main_parent['config']
    contract = parent_contract['train_contract']
    frame_dt = float(contract['frame_dt'])
    if not math.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError('packed-cache parent frame_dt is invalid')
    parent_root = Path(contract['root'])
    dataset_root = args.dataset_root if args.dataset_root is not None else parent_root
    dataset_contract = validate_selector_dataset_contract(parent_root=parent_root, dataset_root=dataset_root, split='train', require_independent=args.dataset_root is not None)
    state_scale = main_parent['state_scale'].to(device=device, dtype=torch.float32)
    attr_scale = main_parent['attr_scale'].to(device=device, dtype=torch.float32)
    loaded_scale = load_phase_scales(Path(contract['root']) / 'stats' / 'phase_scales.json', int(config['dataset']['q_dim']), expected_train_manifest_sha256=str(contract['train_manifest_sha256'])).state(device=device, dtype=torch.float32)
    if not torch.equal(state_scale, loaded_scale):
        raise ValueError('parent and train state scales differ')
    dataset_config = copy.deepcopy(config)
    dataset_config['dataset']['root'] = dataset_contract['dataset_root']
    train = stage_a._load_train_cache(dataset_config, device=device)
    total_sources = args.fit_samples + args.holdout_samples
    if total_sources > train.size:
        raise ValueError('gate training exceeds train split')
    history_registration = registration
    history_gate = gate_tools._build_gate(config, history_registration, device=device)
    dual_sequence_start_parent: HamiBallsDualStatePerObjectCompactCommittedGate | None = None
    if args.gate_checkpoint is not None:
        payload = torch.load(args.gate_checkpoint, map_location='cpu', weights_only=False)
        initial_gate_state = payload['gate_state_dict']
        initial_gate_source = {'kind': 'checkpoint', 'path': str(args.gate_checkpoint.expanduser().resolve()), 'sha256': _sha256_file(args.gate_checkpoint.expanduser().resolve())}
        history_gate.load_state_dict(initial_gate_state, strict=True)
    else:
        initial_gate_state = copy.deepcopy(history_gate.state_dict())
        initial_gate_source = {'kind': 'neutral_g_equals_0.5'}
    history_gate.eval()
    if args.observable_disjoint_component_gate_expansion:
        candidate = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate.from_scalar(history_gate, delta_width=args.observable_disjoint_delta_width).to(device)
        candidates = {arms[0]: candidate}
    else:
        candidates = {name: gate_tools._build_gate(config, registration, device=device) for name in arms}
        for candidate in candidates.values():
            candidate.load_state_dict(copy.deepcopy(initial_gate_state), strict=True)
    for candidate in candidates.values():
        candidate.train()
    audit_candidate_source: dict[str, Any] | None = None
    history_digest = module_digest(history_gate)
    ema_teacher: torch.nn.Module | None = None
    ema_teacher_update_count = 0
    ema_teacher_initial_digest: str | None = None
    dual_sequence_start_parent_digest = None
    excluded_scene_source: dict[str, Any] = {'scene_count': 0}
    if args.sample_excluded_scene_configuration is not None:
        exclusion_path = args.sample_excluded_scene_configuration.expanduser().resolve()
        from hamiformer.training.source_partitions import load_scene_exclusions
        excluded_scenes = load_scene_exclusions(exclusion_path)
        adapter_manifest = (Path(dataset_contract['dataset_root']) / 'manifests' / 'train.jsonl').resolve()
        adapter_pack = json.loads(adapter_manifest.read_text(encoding='utf-8'))
        ledger_path = adapter_manifest.parent / adapter_pack['row_ledger']['file']
        if _sha256_file(ledger_path) != adapter_pack['row_ledger']['sha256']:
            raise ValueError('scene exclusion train row ledger SHA changed')
        ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(ledger_rows) != train.size or any((row.get('row') != index for index, row in enumerate(ledger_rows))):
            raise ValueError('scene exclusion train row ledger is malformed')
        permutation, permutation_positions = _sample_permutation_block_excluding_scenes(size=train.size, count=total_sources, seed=args.sample_seed, offset=args.sample_offset, scene_ids=[str(row['scene_id']) for row in ledger_rows], excluded_scenes={str(value) for value in excluded_scenes})
        excluded_scene_source = {'path': str(exclusion_path), 'sha256': _sha256_file(exclusion_path), 'scene_count': len(excluded_scenes), 'row_ledger_sha256': _sha256_file(ledger_path)}
    else:
        permutation = _sample_permutation_block(size=train.size, count=total_sources, seed=args.sample_seed, offset=args.sample_offset)
        permutation_positions = list(range(args.sample_offset, args.sample_offset + total_sources))
    fit_indices = permutation[:args.fit_samples]
    holdout_indices = permutation[args.fit_samples:]

    def collect_cache(indices: torch.Tensor, *, seed_offset: int, collector_gate: torch.nn.Module | None=None) -> tuple[tuple[TensorCache, ...], float, str]:
        noise_rows: list[list[TensorCache]] = [[] for _ in noise_seeds]
        started = time.perf_counter()
        collector = history_gate if collector_gate is None else collector_gate
        collector_digest = module_digest(collector)
        collector_was_training = collector.training
        collector.eval()
        generators = [torch.Generator(device=device).manual_seed(seed + seed_offset) for seed in noise_seeds]
        if FIXED_MIXED_N_SUPERBATCH_FACTOR > 1 and (not BATCH_PAIRED_NOISE_COLLECTION) and (len(noise_seeds) == 1) and (SELF_POLICY_MIXED_NUM_STEPS_CYCLE is not None) and (int(indices.numel()) > len(SELF_POLICY_MIXED_NUM_STEPS_CYCLE)):
            try:
                with torch.no_grad():
                    noise_rows[0].extend(_collect_fixed_mixed_n_superbatch_rows(indices=indices, base_cycle=tuple(SELF_POLICY_MIXED_NUM_STEPS_CYCLE), factor=int(FIXED_MIXED_N_SUPERBATCH_FACTOR), seed_offset=seed_offset, noise_seed=int(noise_seeds[0]), config=config, contract=contract, d=d, hamiltonian=h, residual=residual, gate=collector, train=train, state_scale=state_scale, attr_scale=attr_scale, device=device))
            finally:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                collector.train(collector_was_training)
            if module_digest(collector) != collector_digest:
                raise AssertionError('superbatch collection mutated its collector gate')
            return (tuple((_cat(rows) for rows in noise_rows)), time.perf_counter() - started, collector_digest)
        try:
            with torch.no_grad():
                for start in range(0, indices.numel(), args.collection_batch):
                    local = indices[start:start + args.collection_batch]
                    if BATCH_PAIRED_NOISE_COLLECTION and len(generators) == 2 and (SELF_POLICY_MIXED_NUM_STEPS_CYCLE is not None):
                        if not DISJOINT_PAIRED_NOISE_SOURCES:
                            raise ValueError('paired mixed-N collection requires disjoint sources')
                        cycle = tuple((int(value) for value in SELF_POLICY_MIXED_NUM_STEPS_CYCLE))
                        if int(local.numel()) != args.collection_batch or len(cycle) != int(local.numel()) or int(local.numel()) % 2 or (set(cycle) != {8, 12, 20}):
                            raise ValueError('paired mixed-N collection requires one full even cycle')
                        source_indices = local.chunk(2)
                        batch = int(source_indices[0].numel())
                        if any((int(part.numel()) != batch for part in source_indices)):
                            raise AssertionError('paired mixed-N source halves drifted')
                        assignment = torch.tensor(cycle, device=local.device).chunk(2)
                        counts = [tuple((int((part == n).sum()) for n in (8, 12, 20))) for part in assignment]
                        if counts[0] != counts[1] or any((value <= 0 for value in counts[0])):
                            raise ValueError('each paired-noise half must have identical positive N counts')
                        physical_batches = [carrier_cache.stage_a._carrier_batch(train, fixed._FixedStream(part), state_scale=state_scale, device=device) for part in source_indices]
                        half_slots = torch.arange(start, start + batch, device=device, dtype=torch.long)
                        parallel_prepared = []
                        for num_steps in (8, 12, 20):
                            selected_physical = []
                            selected_sources = []
                            selected_slots = []
                            for noise_index, (values, generator) in enumerate(zip(physical_batches, generators, strict=True)):
                                source_mask = assignment[noise_index] == num_steps
                                value_mask = source_mask.to(values[0].device)
                                local_x0 = values[0][value_mask]
                                local_target = values[1][value_mask]
                                local_attrs = values[2][value_mask]
                                local_time = values[3][value_mask]
                                selected_physical.append((local_x0, local_target, local_attrs, local_time))
                                selected_sources.append(carrier_cache.stage_a._random_source(local_target, generator=generator, noise_scale=float(config['rectified_flow']['phase_noise_scale'])))
                                selected_slots.append(half_slots[value_mask])
                            joined_x0 = torch.cat([values[0] for values in selected_physical], dim=0)
                            joined_target = torch.cat([values[1] for values in selected_physical], dim=0)
                            joined_attrs = torch.cat([values[2] for values in selected_physical], dim=0)
                            joined_time = torch.cat([values[3] for values in selected_physical], dim=0)
                            source = torch.cat(selected_sources, dim=0)
                            if PARALLEL_MIXED_N_CARRIER_STREAMS and device.type == 'cuda':
                                parallel_prepared.append((num_steps, joined_x0, joined_target, joined_attrs, joined_time, source, tuple(selected_slots)))
                                continue
                            if source.device.type == 'cuda' and (not DEFER_CARRIER_SYNCHRONIZE):
                                torch.cuda.synchronize(source.device)
                            carrier = carrier_cache.recovery.collect_recovery_carrier(d=d, hamiltonian=h, residual=residual, gate=collector, source=source, x0=joined_x0, attrs=joined_attrs, physical_time=joined_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']))
                            if source.device.type == 'cuda' and (not DEFER_CARRIER_SYNCHRONIZE):
                                torch.cuda.synchronize(source.device)
                            selected_fields = list(range(len(carrier.trace.traces)))
                            per_noise = counts[0][(8, 12, 20).index(num_steps)]
                            for field_order, field_index in enumerate(selected_fields):
                                joined = _field_tensors(carrier.trace.traces[field_index], x0=joined_x0, target=joined_target, attrs=joined_attrs, physical_time=joined_time)
                                for noise_index in range(2):
                                    begin = noise_index * per_noise
                                    end = begin + per_noise
                                    row = {name: value[begin:end] for name, value in joined.items()}
                                    source_mask = assignment[noise_index] == num_steps
                                    row.update({'trajectory_slot': selected_slots[noise_index], 'source_index': source_indices[noise_index][source_mask].to(device=device, dtype=torch.long), 'rf_field_order': torch.full((per_noise,), field_order, device=device, dtype=torch.long), 'rf_trace_index': torch.full((per_noise,), field_index, device=device, dtype=torch.long)})
                                    noise_rows[noise_index].append(row)
                        if parallel_prepared:
                            parent_stream = torch.cuda.current_stream(device)
                            streams = [torch.cuda.Stream(device=device) for _ in parallel_prepared]
                            carriers = []
                            for values, child_stream in zip(parallel_prepared, streams, strict=True):
                                num_steps, joined_x0, joined_target, joined_attrs, joined_time, source, selected_slots = values
                                child_stream.wait_stream(parent_stream)
                                with torch.cuda.stream(child_stream):
                                    carrier = _collect_recovery_from_prepared(config=config, contract=contract, d=d, hamiltonian=h, residual=residual, gate=collector, source=source, x0=joined_x0, attrs=joined_attrs, physical_time=joined_time, state_scale=state_scale, attr_scale=attr_scale, num_steps=num_steps)
                                carriers.append((values, carrier))
                            for child_stream in streams:
                                parent_stream.wait_stream(child_stream)
                            for values, carrier in carriers:
                                num_steps, joined_x0, joined_target, joined_attrs, joined_time, _source, selected_slots = values
                                selected_fields = list(range(len(carrier.trace.traces)))
                                per_noise = counts[0][(8, 12, 20).index(num_steps)]
                                for field_order, field_index in enumerate(selected_fields):
                                    joined = _field_tensors(carrier.trace.traces[field_index], x0=joined_x0, target=joined_target, attrs=joined_attrs, physical_time=joined_time)
                                    for noise_index in range(2):
                                        begin = noise_index * per_noise
                                        end = begin + per_noise
                                        row = {name: value[begin:end] for name, value in joined.items()}
                                        source_mask = assignment[noise_index] == num_steps
                                        row.update({'trajectory_slot': selected_slots[noise_index], 'source_index': source_indices[noise_index][source_mask].to(device=device, dtype=torch.long), 'rf_field_order': torch.full((per_noise,), field_order, device=device, dtype=torch.long), 'rf_trace_index': torch.full((per_noise,), field_index, device=device, dtype=torch.long)})
                                        noise_rows[noise_index].append(row)
                        continue
                    if BATCH_PAIRED_NOISE_COLLECTION and len(generators) == 2:
                        if DISJOINT_PAIRED_NOISE_SOURCES:
                            if int(local.numel()) % 2:
                                raise ValueError('disjoint paired-noise source batch must be even')
                            source_indices = local.chunk(2)
                            physical_batches = [carrier_cache.stage_a._carrier_batch(train, fixed._FixedStream(part), state_scale=state_scale, device=device) for part in source_indices]
                            batch = int(source_indices[0].numel())
                            if any((int(part.numel()) != batch for part in source_indices)):
                                raise AssertionError('disjoint paired-noise source halves drifted')
                            sources = [carrier_cache.stage_a._random_source(values[1], generator=generator, noise_scale=float(config['rectified_flow']['phase_noise_scale'])) for values, generator in zip(physical_batches, generators, strict=True)]
                            joined_x0 = torch.cat([values[0] for values in physical_batches], dim=0)
                            joined_target = torch.cat([values[1] for values in physical_batches], dim=0)
                            joined_attrs = torch.cat([values[2] for values in physical_batches], dim=0)
                            joined_time = torch.cat([values[3] for values in physical_batches], dim=0)
                        else:
                            x0, target, attrs, physical_time = carrier_cache.stage_a._carrier_batch(train, fixed._FixedStream(local), state_scale=state_scale, device=device)
                            source_indices = (local, local)
                            sources = [carrier_cache.stage_a._random_source(target, generator=generator, noise_scale=float(config['rectified_flow']['phase_noise_scale'])) for generator in generators]
                            repeat = lambda value: torch.cat((value, value), dim=0)
                            joined_x0 = repeat(x0)
                            joined_target = repeat(target)
                            joined_attrs = repeat(attrs)
                            joined_time = repeat(physical_time)
                            batch = int(local.numel())
                        source = torch.cat(sources, dim=0)
                        if source.device.type == 'cuda':
                            torch.cuda.synchronize(source.device)
                        carrier = carrier_cache.recovery.collect_recovery_carrier(d=d, hamiltonian=h, residual=residual, gate=collector, source=source, x0=joined_x0, attrs=joined_attrs, physical_time=joined_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=20, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']))
                        if source.device.type == 'cuda':
                            torch.cuda.synchronize(source.device)
                        if max(field_indices) >= len(carrier.trace.traces):
                            raise ValueError('packed-cache field index is not accepted')
                        for field_order, field_index in enumerate(field_indices):
                            joined = _field_tensors(carrier.trace.traces[field_index], x0=joined_x0, target=joined_target, attrs=joined_attrs, physical_time=joined_time)
                            for noise_index in range(2):
                                begin = noise_index * batch
                                end = begin + batch
                                row = {name: value[begin:end] for name, value in joined.items()}
                                row.update({'trajectory_slot': torch.arange(start, start + batch, device=device, dtype=torch.long), 'source_index': source_indices[noise_index].to(device=device, dtype=torch.long), 'rf_field_order': torch.full((batch,), field_order, device=device, dtype=torch.long), 'rf_trace_index': torch.full((batch,), field_index, device=device, dtype=torch.long)})
                                noise_rows[noise_index].append(row)
                        continue
                    if SELF_POLICY_MIXED_NUM_STEPS_CYCLE is None:
                        groups = [(local, torch.arange(start, start + int(local.numel()), device=device, dtype=torch.long), 20)]
                    else:
                        cycle = tuple((int(value) for value in SELF_POLICY_MIXED_NUM_STEPS_CYCLE))
                        if not cycle or len(cycle) != int(local.numel()) or set(cycle) != {8, 12, 20} or (int(local.numel()) != len(cycle)) or (len(generators) != 1):
                            raise ValueError('mixed-N fixed cache requires one noise and one full configured cycle per collection batch')
                        assignment = torch.tensor(cycle, device=local.device)
                        slots = torch.arange(start, start + int(local.numel()), device=device, dtype=torch.long)
                        groups = [(local[assignment == num_steps], slots[(assignment == num_steps).to(device)], num_steps) for num_steps in (8, 12, 20)]
                    collected_groups = []
                    if PARALLEL_MIXED_N_CARRIER_STREAMS and SELF_POLICY_MIXED_NUM_STEPS_CYCLE is not None and (device.type == 'cuda'):
                        if len(generators) != 1:
                            raise ValueError('parallel fixed mixed-N requires one noise')
                        prepared = []
                        for group_indices, trajectory_slots, num_steps in groups:
                            x0, target, attrs, physical_time = carrier_cache.stage_a._carrier_batch(train, fixed._FixedStream(group_indices), state_scale=state_scale, device=device)
                            source = carrier_cache.stage_a._random_source(target, generator=generators[0], noise_scale=float(config['rectified_flow']['phase_noise_scale']))
                            prepared.append((group_indices, trajectory_slots, num_steps, source, x0, target, attrs, physical_time))
                        parent_stream = torch.cuda.current_stream(device)
                        streams = [torch.cuda.Stream(device=device) for _ in prepared]
                        for values, child_stream in zip(prepared, streams, strict=True):
                            group_indices, trajectory_slots, num_steps, source, x0, target, attrs, physical_time = values
                            child_stream.wait_stream(parent_stream)
                            with torch.cuda.stream(child_stream):
                                carrier = _collect_recovery_from_prepared(config=config, contract=contract, d=d, hamiltonian=h, residual=residual, gate=collector, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, num_steps=num_steps)
                            collected_groups.append((group_indices, trajectory_slots, num_steps, ((carrier, x0, target, attrs, physical_time),)))
                        for child_stream in streams:
                            parent_stream.wait_stream(child_stream)
                    else:
                        for group_indices, trajectory_slots, num_steps in groups:
                            batches = []
                            for generator in generators:
                                collector_fn = _collect_shared_main_carrier_deferred_sync if DEFER_CARRIER_SYNCHRONIZE else carrier_cache._collect_shared_main_carrier
                                batches.append(collector_fn(config=config, contract=contract, d=d, hamiltonian=h, residual=residual, gate=collector, train=train, stream=fixed._FixedStream(group_indices), state_scale=state_scale, attr_scale=attr_scale, num_steps=num_steps, batch_size=int(group_indices.numel()), source_rng=generator, device=device))
                            collected_groups.append((group_indices, trajectory_slots, num_steps, tuple(batches)))
                    for group_indices, trajectory_slots, num_steps, batches in collected_groups:
                        for noise_index, (carrier, x0, target, attrs, physical_time) in enumerate(batches):
                            selected_fields = list(range(len(carrier.trace.traces))) if SELF_POLICY_MIXED_NUM_STEPS_CYCLE is not None else field_indices
                            if max(selected_fields) >= len(carrier.trace.traces):
                                raise ValueError('packed-cache field index is not accepted')
                            for field_order, field_index in enumerate(selected_fields):
                                row = _field_tensors(carrier.trace.traces[field_index], x0=x0, target=target, attrs=attrs, physical_time=physical_time)
                                row.update({'trajectory_slot': trajectory_slots, 'source_index': group_indices.to(device=device, dtype=torch.long), 'rf_field_order': torch.full((int(group_indices.numel()),), field_order, device=device, dtype=torch.long), 'rf_trace_index': torch.full((int(group_indices.numel()),), field_index, device=device, dtype=torch.long)})
                                noise_rows[noise_index].append(row)
                        for paired in batches[1:]:
                            if any((not torch.equal(a, b) for a, b in zip(batches[0][1:], paired[1:]))):
                                raise AssertionError('packed paired caches lost source/target alignment')
        finally:
            if DEFER_CARRIER_SYNCHRONIZE and device.type == 'cuda':
                torch.cuda.synchronize(device)
            collector.train(collector_was_training)
        if module_digest(collector) != collector_digest:
            raise AssertionError('cache collection mutated its collector gate')
        return (tuple((_cat(rows) for rows in noise_rows)), time.perf_counter() - started, collector_digest)
    initial_fit_indices = fit_indices[:args.self_policy_refresh_sources] if self_policy_refresh else fit_indices

    def self_policy_collector(block: int) -> torch.nn.Module:
        candidate = next(iter(candidates.values()))
        if args.self_policy_uniform_random_carriers:
            collector = _UniformRandomCarrierGate(candidate, seed=args.optimizer_sample_seed + 1000003 * (block + 1))
        elif args.self_policy_stage_parent_carriers:
            collector = history_gate
        elif not recovery_carrier_scales:
            collector = candidate
        else:
            collector = _scaled_staged_recovery_collector(candidate, scale=recovery_carrier_scales[block])
        if SELF_POLICY_REFRESH_COLLECTOR_ADAPTER is not None:
            collector = SELF_POLICY_REFRESH_COLLECTOR_ADAPTER(block=block, candidate=candidate, collector=collector)
        return collector
    initial_fit_collector = self_policy_collector(0) if self_policy_refresh else history_gate
    initial_fit_provenance = _self_policy_collector_provenance(next(iter(candidates.values())), initial_fit_collector, uniform_random=bool(args.self_policy_uniform_random_carriers), stage_parent=bool(args.self_policy_stage_parent_carriers), ema_parent=False, ema_beta=None, ema_teacher_updates=0, recovery_scale=recovery_carrier_scales[0] if recovery_carrier_scales else None) if self_policy_refresh else None
    fit_caches, fit_collection_wall, initial_fit_collector_digest = collect_cache(initial_fit_indices, seed_offset=0, collector_gate=initial_fit_collector)
    if SELF_POLICY_REFRESH_CACHE_OBSERVER is not None and self_policy_refresh:
        SELF_POLICY_REFRESH_CACHE_OBSERVER(block=0, candidate=next(iter(candidates.values())), caches=fit_caches)
    if initial_fit_provenance is not None and initial_fit_collector_digest != initial_fit_provenance['collector_digest']:
        raise AssertionError('initial refresh collector digest changed during collection')
    fit_replay_blocks: tuple[tuple[TensorCache, ...], ...] = (fit_caches,) if args.self_policy_aggregate_replay else tuple()
    hold_caches, hold_collection_wall, hold_collector_digest = collect_cache(holdout_indices, seed_offset=1000)
    self_policy_refresh_ledger: list[dict[str, Any]] = []
    if self_policy_refresh:
        self_policy_refresh_ledger.append({'block': 0, 'first_update': 1, 'source_indices': initial_fit_indices.tolist(), 'noise_seed_offset': 0, 'collector_digest': initial_fit_collector_digest, 'collection_wall_seconds': fit_collection_wall, 'aggregate_replay_block_count': 1 if args.self_policy_aggregate_replay else None, 'aggregate_replay_source_count': int(initial_fit_indices.numel()) if args.self_policy_aggregate_replay else None, 'recovery_carrier_scale': recovery_carrier_scales[0] if recovery_carrier_scales else None, 'uniform_random_carrier_seed': args.optimizer_sample_seed + 1000003 if args.self_policy_uniform_random_carriers else None, **initial_fit_provenance})
    precollected_stage_parent_refreshes = False
    if SELF_POLICY_PRECOLLECT_STAGE_PARENT_REFRESHES or SELF_POLICY_PRECOLLECT_FIXED_CARRIERS:
        if not (self_policy_refresh and (args.self_policy_stage_parent_carriers or args.self_policy_uniform_random_carriers) and args.self_policy_aggregate_replay):
            raise ValueError('precollection requires fixed stage-parent or seeded-random aggregate replay')
        for refresh_block, update, source_start, source_stop in self_policy_refresh_schedule[1:]:
            refresh_indices = fit_indices[source_start:source_stop]
            if refresh_indices.numel() != args.self_policy_refresh_sources:
                raise AssertionError('self-policy refresh source ledger was exhausted')
            refresh_collector = self_policy_collector(refresh_block)
            refresh_provenance = _self_policy_collector_provenance(next(iter(candidates.values())), refresh_collector, uniform_random=bool(args.self_policy_uniform_random_carriers), stage_parent=bool(args.self_policy_stage_parent_carriers), ema_parent=False, ema_beta=None, ema_teacher_updates=0, recovery_scale=recovery_carrier_scales[refresh_block] if recovery_carrier_scales else None)
            refresh_caches, refresh_wall, refresh_digest = collect_cache(refresh_indices, seed_offset=refresh_block, collector_gate=refresh_collector)
            if refresh_digest != refresh_provenance['collector_digest']:
                raise AssertionError('refresh collector digest changed during collection')
            fit_caches = refresh_caches
            fit_cache0 = fit_caches[0]
            if SELF_POLICY_REFRESH_CACHE_OBSERVER is not None:
                SELF_POLICY_REFRESH_CACHE_OBSERVER(block=refresh_block, candidate=next(iter(candidates.values())), caches=fit_caches)
            fit_replay_blocks = (*fit_replay_blocks, refresh_caches)
            fit_collection_wall += refresh_wall
            self_policy_refresh_ledger.append({'block': refresh_block, 'first_update': update, 'source_indices': refresh_indices.tolist(), 'noise_seed_offset': refresh_block, 'collector_digest': refresh_digest, 'collection_wall_seconds': refresh_wall, 'aggregate_replay_block_count': len(fit_replay_blocks), 'aggregate_replay_source_count': sum((len(row['source_indices']) for row in self_policy_refresh_ledger)) + int(refresh_indices.numel()), 'recovery_carrier_scale': recovery_carrier_scales[refresh_block] if recovery_carrier_scales else None, 'uniform_random_carrier_seed': args.optimizer_sample_seed + 1000003 * (refresh_block + 1) if args.self_policy_uniform_random_carriers else None, **refresh_provenance})
        precollected_stage_parent_refreshes = True
    fit_cache0 = fit_caches[0]
    hold_cache0 = hold_caches[0]
    q_dim = int(config['dataset']['q_dim'])
    online_context = (residual, state_scale, q_dim) if args.online_recurrence else None
    fit_trajectory_layouts: tuple[torch.Tensor, ...] | None = None
    hold_trajectory_layouts: tuple[torch.Tensor, ...] | None = None
    if any((cache['teacher'].shape != fit_cache0['teacher'].shape for cache in fit_caches[1:])):
        raise AssertionError('paired fit caches lost shape alignment')
    with torch.no_grad():
        prefix = torch.arange(min(args.collection_batch, hold_cache0['teacher'].shape[0]), device=device)
        packed = _forward(history_gate, _select(hold_cache0, prefix))
        singleton = torch.cat([_forward(history_gate, _select(hold_cache0, row[None])) for row in prefix], dim=0)
        packing_max_abs = float((packed - singleton).abs().max().cpu())
    packing_scale = float(torch.maximum(packed.abs().amax(), singleton.abs().amax()).cpu())
    packing_relative_max_abs = packing_max_abs / max(packing_scale, torch.finfo(packed.dtype).eps)
    packing_tolerance = max(0.0005, 16.0 * torch.finfo(packed.dtype).eps * float(packed.shape[1]) * float(getattr(history_gate, 'rank', 1)))
    if packing_max_abs > packing_tolerance or packing_relative_max_abs > packing_tolerance:
        raise AssertionError(f'packed field-local gate drifted by {packing_max_abs} (relative {packing_relative_max_abs})')
    online_step0_gate_max_abs: float | None = None
    online_step0_gate_relative_max_abs: float | None = None
    online_step0_state_max_abs: float | None = None
    online_step0_state_relative_max_abs: float | None = None
    online_field_start_previous_g_max_abs_from_one: float | None = None
    if args.online_recurrence:
        selected = _select(hold_cache0, prefix)
        with torch.no_grad():
            cached_value = _forward(history_gate, selected)
            online_value, online_cache = _online_forward(history_gate, residual, selected, state_scale=state_scale, q_dim=q_dim)
            online_step0_gate_max_abs = float((online_value - cached_value).abs().max().cpu())
            online_gate_scale = float(torch.maximum(online_value.abs().amax(), cached_value.abs().amax()).cpu())
            online_step0_gate_relative_max_abs = online_step0_gate_max_abs / max(online_gate_scale, torch.finfo(online_value.dtype).eps)
            online_field_start_previous_g_max_abs_from_one = float((selected['field_start_previous_g'] - 1.0).abs().max().cpu())
            state_differences = [(online_cache[name] - selected[name]).abs().max() for name in ('previous_mixed', 'h_candidate', 'hr_candidate', 'previous_g', 'residual_hidden', 'teacher')]
            online_step0_state_max_abs = float(torch.stack(state_differences).max().cpu())
            state_scale_max = max((float(online_cache[name].abs().amax().cpu()) for name in ('previous_mixed', 'h_candidate', 'hr_candidate', 'previous_g', 'residual_hidden', 'teacher')))
            state_scale_max = max(state_scale_max, max((float(selected[name].abs().amax().cpu()) for name in ('previous_mixed', 'h_candidate', 'hr_candidate', 'previous_g', 'residual_hidden', 'teacher'))))
            online_step0_state_relative_max_abs = online_step0_state_max_abs / max(state_scale_max, torch.finfo(online_value.dtype).eps)
        online_gate_tolerance = 0.0005
        online_state_relative_tolerance = 0.002
        if online_step0_gate_max_abs > 0.0005 or online_step0_gate_relative_max_abs > 0.0005 or online_step0_state_relative_max_abs > 0.002 or (online_field_start_previous_g_max_abs_from_one != 0.0):
            message = f'online field replay failed current-history gate equivalence: gate={online_step0_gate_max_abs}/{online_step0_gate_relative_max_abs}, state={online_step0_state_max_abs}/{online_step0_state_relative_max_abs}, field_start_g={online_field_start_previous_g_max_abs_from_one}'
            if NONFATAL_ONLINE_REPLAY_AUDIT:
                print(json.dumps({'warning': message}), flush=True)
            else:
                raise AssertionError(message)
    function_preserving_max_abs: float | None = None
    dual_clone_max_abs: float | None = None
    dual_history_adapter_nonzero: int | None = None
    veto_initial_risk_mean: float | None = None
    veto_initial_monotonic_max_violation: float | None = None
    if args.observable_disjoint_component_gate_expansion:
        candidate = next(iter(candidates.values()))
        if type(candidate) not in {HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate, HamiBallsObservableAdditiveRecoveryPerObjectCompactCommittedGate}:
            raise TypeError('observable disjoint expansion built wrong model')
        with torch.no_grad():
            parent_value = _forward(history_gate, _select(hold_cache0, prefix))
            dual_value = _forward(candidate, _select(hold_cache0, prefix))
        clone_reference = _observable_disjoint_clone_reference(parent_value, disjoint_gate_resume=False)
        dual_clone_max_abs = float((dual_value - clone_reference).abs().max().cpu())
        if dual_clone_max_abs > 2e-06:
            message = f'observable disjoint clone changed step-0 gate by {dual_clone_max_abs}'
            if NONFATAL_ONLINE_REPLAY_AUDIT:
                print(json.dumps({'warning': message}), flush=True)
            else:
                raise AssertionError(message)
        counts = {'total': sum((value.numel() for value in candidate.parameters())), 'trainable': sum((value.numel() for value in candidate.parameters() if value.requires_grad))}
        if type(candidate) is HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate:
            trainable_count = 2 * (candidate.residual_hidden_dim * candidate.delta_width + candidate.delta_width + 1)
        else:
            trainable_count = sum((value.numel() for value in candidate.parameters() if value.requires_grad))
        expected_counts = {'total': sum((value.numel() for value in history_gate.parameters())) + trainable_count, 'trainable': trainable_count}
        declared_extra = int(getattr(candidate, '_declared_extra_trainable_parameters', 0))
        if declared_extra:
            expected_counts = {'total': expected_counts['total'] + declared_extra, 'trainable': expected_counts['trainable'] + declared_extra}
        declared_trainable = getattr(candidate, '_declared_trainable_parameters_override', None)
        if declared_trainable is not None:
            expected_counts['trainable'] = int(declared_trainable)
        if counts != expected_counts:
            raise AssertionError(f'observable disjoint counts changed: {counts}, expected {expected_counts}')
    before_gate = history_gate
    if before_gate is None:
        raise AssertionError('gate training lost its before-policy')
    before = _metrics(before_gate, hold_caches, online_context=online_context)
    fit_before = _metrics(before_gate, fit_caches, online_context=online_context)
    optimizers: dict[str, torch.optim.AdamW] = {}
    for name, candidate in candidates.items():
        optimizers[name] = torch.optim.AdamW([parameter for parameter in candidate.parameters() if parameter.requires_grad], lr=args.maximum_lr, weight_decay=float(config['optimization']['weight_decay']))
    if args.optimizer_sample_seed < 1:
        raise ValueError('optimizer-sample-seed must be positive')
    cpu_generator = torch.Generator(device='cpu').manual_seed(args.optimizer_sample_seed)
    sequence_count = int(fit_cache0['teacher'].shape[0])
    history: list[dict[str, Any]] = []
    nonlinear_constraint_step_ledger: list[dict[str, Any]] = []
    initial_pcgrad_losses: dict[str, dict[str, float]] = {}
    actual_optimizer_steps = {name: 0 for name in arms}
    skipped_nonfinite_optimizer_steps = {name: 0 for name in arms}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_updates = args.updates * 1
    for update in range(1, total_updates + 1):
        refresh_row = None if precollected_stage_parent_refreshes else next((row for row in self_policy_refresh_schedule[1:] if row[1] == update), None)
        if refresh_row is not None:
            refresh_block, _first_update, source_start, source_stop = refresh_row
            refresh_indices = fit_indices[source_start:source_stop]
            if refresh_indices.numel() != args.self_policy_refresh_sources:
                raise AssertionError('self-policy refresh source ledger was exhausted')
            refresh_collector = self_policy_collector(refresh_block)
            refresh_provenance = _self_policy_collector_provenance(next(iter(candidates.values())), refresh_collector, uniform_random=bool(args.self_policy_uniform_random_carriers), stage_parent=bool(args.self_policy_stage_parent_carriers), ema_parent=False, ema_beta=None, ema_teacher_updates=0, recovery_scale=recovery_carrier_scales[refresh_block] if recovery_carrier_scales else None)
            refresh_caches, refresh_wall, refresh_digest = collect_cache(refresh_indices, seed_offset=refresh_block, collector_gate=refresh_collector)
            if refresh_digest != refresh_provenance['collector_digest']:
                raise AssertionError('refresh collector digest changed during collection')
            fit_caches = refresh_caches
            fit_cache0 = fit_caches[0]
            if SELF_POLICY_REFRESH_CACHE_OBSERVER is not None:
                SELF_POLICY_REFRESH_CACHE_OBSERVER(block=refresh_block, candidate=next(iter(candidates.values())), caches=fit_caches)
            if args.self_policy_aggregate_replay:
                fit_replay_blocks = (*fit_replay_blocks, refresh_caches)
                sequence_count = sum((int(block[0]['teacher'].shape[0]) for block in fit_replay_blocks))
            else:
                sequence_count = int(fit_cache0['teacher'].shape[0])
            fit_collection_wall += refresh_wall
            self_policy_refresh_ledger.append({'block': refresh_block, 'first_update': update, 'source_indices': refresh_indices.tolist(), 'noise_seed_offset': refresh_block, 'collector_digest': refresh_digest, 'collection_wall_seconds': refresh_wall, 'aggregate_replay_block_count': len(fit_replay_blocks) if args.self_policy_aggregate_replay else None, 'aggregate_replay_source_count': sum((len(row['source_indices']) for row in self_policy_refresh_ledger)) + int(refresh_indices.numel()) if args.self_policy_aggregate_replay else None, 'recovery_carrier_scale': recovery_carrier_scales[refresh_block] if recovery_carrier_scales else None, 'uniform_random_carrier_seed': args.optimizer_sample_seed + 1000003 * (refresh_block + 1) if args.self_policy_uniform_random_carriers else None, **refresh_provenance})
        replay_visible_sequences = sequence_count
        if args.self_policy_aggregate_replay:
            local_caches, replay_visible_sequences = _sample_aggregate_replay(fit_replay_blocks, batch_size=min(args.sequence_batch, sequence_count), generator=cpu_generator)
            indices = torch.empty(0, dtype=torch.long, device=device)
        else:
            indices = torch.randint(sequence_count, (min(args.sequence_batch, sequence_count),), generator=cpu_generator, device='cpu').to(device)
            local_caches = tuple((_select(cache, indices) for cache in fit_caches))
        local_update = _stage_update(update, args.updates)
        maximum, minimum = (args.maximum_lr, args.minimum_lr)
        if local_update <= 20:
            lr = maximum * local_update / 20
        else:
            progress = (local_update - 20) / max(1, args.updates - 20)
            lr = minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))
        ledger: dict[str, float | int | str] = {'update': update, 'stage_update': local_update, 'learning_rate': lr}
        if args.self_policy_aggregate_replay:
            ledger.update({'aggregate_replay_block_count': len(fit_replay_blocks), 'aggregate_replay_source_count': sum((len(row['source_indices']) for row in self_policy_refresh_ledger)), 'aggregate_replay_sequence_count': replay_visible_sequences})
        for name, candidate in candidates.items():
            objective = _objective_for_update(name, update, args.updates)
            optimizer = optimizers[name]
            set_optimizer_learning_rate(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            responsibility_rows: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...] | None = None
            trajectory_loss: torch.Tensor | None = None
            component_constraint_losses: dict[str, torch.Tensor] | None = None
            private_component_losses: tuple[torch.Tensor, ...] | None = None
            adaptive_component_losses: tuple[torch.Tensor, torch.Tensor] | None = None
            capped_direct_component_losses: tuple[torch.Tensor, torch.Tensor] | None = None
            pcgrad_component_losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            raw_pcgrad_component_losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            regime_component_losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            staged_recovery_component_losses: tuple[torch.Tensor, torch.Tensor] | None = None
            allocation_scores: tuple[torch.Tensor, ...] | None = None
            crossing_logits: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
            disjoint_logits: tuple[torch.Tensor, ...] | None = None
            disjoint_local_logits: tuple[torch.Tensor, ...] | None = None
            if isinstance(candidate, HamiBallsComponentRiskVetoGate):
                raise ValueError('veto gate received an unregistered objective')
                if args.online_recurrence:
                    online_rows = tuple((_online_forward(candidate, residual, cache, state_scale=state_scale, q_dim=q_dim) for cache in local_caches))
                    deployed_values = tuple((row[0] for row in online_rows))
                    objective_caches = tuple((row[1] for row in online_rows))
                else:
                    deployed_values = tuple((_forward(candidate, cache) for cache in local_caches))
                    objective_caches = local_caches
                veto_rows = tuple((_veto_forward(candidate, cache) for cache in objective_caches))
                values = tuple((1.0 - row[2] for row in veto_rows))
                if local_update == 1:
                    replay_difference = max((float((deployed - row[0]).abs().max().detach().cpu()) for deployed, row in zip(deployed_values, veto_rows, strict=True)))
                    ledger[f'{name}_veto_replay_max_abs'] = replay_difference
                    if replay_difference > 0.0005:
                        raise AssertionError('component-risk veto replay changed deployed values')
                ledger[f'{name}_risk_mean'] = float(torch.stack([row[2].mean() for row in veto_rows]).mean().detach().cpu())
                ledger[f'{name}_base_gate_mean'] = float(torch.stack([row[1].mean() for row in veto_rows]).mean().detach().cpu())
                ledger[f'{name}_deployed_gate_mean'] = float(torch.stack([row[0].mean() for row in veto_rows]).mean().detach().cpu())
            elif args.online_recurrence:
                if objective in (DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE,):
                    expected_online_type = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate
                    if type(candidate) is not expected_online_type:
                        raise TypeError('raw endpoint objective received the wrong gate')
                    online_rows = tuple((_online_observable_disjoint_logits_forward(candidate, residual, cache, state_scale=state_scale, q_dim=q_dim, semi_gradient=False) for cache in local_caches))
                    disjoint_logits = tuple((row[2] for row in online_rows))
                else:
                    online_rows = tuple((_online_forward(candidate, residual, cache, state_scale=state_scale, q_dim=q_dim) for cache in local_caches))
                values = tuple((row[0] for row in online_rows))
                objective_caches = tuple((row[1] for row in online_rows))
            else:
                values = tuple((_forward(candidate, cache) for cache in local_caches))
                objective_caches = local_caches
            if objective == 'global_sequence_projection_regret':
                loss = sum((_global_sequence_projection_regret_loss(value, cache) for value, cache in zip(values, objective_caches))) / len(objective_caches)
            elif objective == QP_NO_HARM_OBJECTIVE:
                base_loss = sum((_global_sequence_projection_regret_loss(value, cache) for value, cache in zip(values, objective_caches))) / len(objective_caches)
                component_rows = tuple((_source_object_balanced_qp_no_harm_regret_losses(value, cache, q_dim=q_dim) for value, cache in zip(values, objective_caches)))
                q_auxiliary = sum((row[0] for row in component_rows)) / len(component_rows)
                p_auxiliary = sum((row[1] for row in component_rows)) / len(component_rows)
                auxiliary_loss = 0.5 * (q_auxiliary + p_auxiliary)
                loss = base_loss + args.qp_no_harm_weight * auxiliary_loss
                ledger[f'{name}_base_loss'] = float(base_loss.detach().cpu())
                ledger[f'{name}_q_no_harm_loss'] = float(q_auxiliary.detach().cpu())
                ledger[f'{name}_p_no_harm_loss'] = float(p_auxiliary.detach().cpu())
                if local_update == 1:
                    geometry = _loss_parameter_gradient_geometry(base_loss, auxiliary_loss, candidate)
                    qp_geometry = _loss_parameter_gradient_geometry(q_auxiliary, p_auxiliary, candidate)
                    for key, item in geometry.items():
                        ledger[f'{name}_{key}'] = item
                    for key, item in qp_geometry.items():
                        ledger[f'{name}_qp_{key}'] = item
                    ledger[f'{name}_weighted_auxiliary_to_base_gradient_norm'] = args.qp_no_harm_weight * geometry['auxiliary_gradient_norm'] / max(geometry['base_gradient_norm'], 1e-30)
            elif objective in (DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE,):
                if type(candidate) is not HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate:
                    raise TypeError('observable-disjoint endpoint+recurrent objective received wrong gate')
                if disjoint_logits is None:
                    raise AssertionError('endpoint+recurrent objective lost component logits')
                local_rows = tuple((_dual_qp_endpoint_regret_balanced_raw_logit_bce(logits, cache, q_dim=q_dim) for logits, cache in zip(disjoint_logits, objective_caches, strict=True)))
                local_label = 'raw_endpoint_bce'
                recurrent_rows = tuple((_dual_qp_global_sequence_projection_regret_losses(value, cache, state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt) for value, cache in zip(values, objective_caches, strict=True)))
                semideviation_rows = tuple((_dual_qp_source_object_upper_semideviation_losses(value, cache, state_scale=state_scale, q_dim=q_dim, frame_dt=frame_dt) for value, cache in zip(values, objective_caches, strict=True))) if objective in {DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE} else None
                parent_semideviation_rows = None
                q_local = sum((row[0] for row in local_rows)) / len(local_rows)
                p_local = sum((row[1] for row in local_rows)) / len(local_rows)
                q_recurrent = sum((row[0] for row in recurrent_rows)) / len(recurrent_rows)
                p_recurrent = sum((row[1] for row in recurrent_rows)) / len(recurrent_rows)
                if semideviation_rows is None:
                    q_semideviation = q_local.new_zeros(())
                    p_semideviation = p_local.new_zeros(())
                    q_loss = 0.5 * (q_local + q_recurrent)
                    p_loss = 0.5 * (p_local + p_recurrent)
                else:
                    q_semideviation = sum((row[0] for row in semideviation_rows)) / len(semideviation_rows)
                    p_semideviation = sum((row[1] for row in semideviation_rows)) / len(semideviation_rows)
                    if COMPONENT_TAIL_COMPONENT_SPECIFIC_Q_NO_TAIL_P_TAIL:
                        q_loss = 0.5 * (q_local + q_recurrent)
                        p_loss = (p_local + p_recurrent + p_semideviation) / 3.0
                    else:
                        q_loss = (q_local + q_recurrent + q_semideviation) / 3.0
                        p_loss = (p_local + p_recurrent + p_semideviation) / 3.0
                if NO_HARM_FINAL_QP_NO_HARM_WEIGHT > 0.0 and objective == DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE:
                    no_harm_rows = tuple((_dual_qp_source_object_balanced_no_harm_regret_losses(value, cache, q_dim=q_dim) for value, cache in zip(values, objective_caches, strict=True)))
                    q_no_harm = sum((row[0] for row in no_harm_rows)) / len(no_harm_rows)
                    p_no_harm = sum((row[1] for row in no_harm_rows)) / len(no_harm_rows)
                    q_loss = q_loss + NO_HARM_FINAL_QP_NO_HARM_WEIGHT * q_no_harm
                    p_loss = p_loss + NO_HARM_FINAL_QP_NO_HARM_WEIGHT * p_no_harm
                    ledger[f'{name}_q_source_object_no_harm'] = float(q_no_harm.detach().cpu())
                    ledger[f'{name}_p_source_object_no_harm'] = float(p_no_harm.detach().cpu())
                    ledger[f'{name}_source_object_no_harm_weight'] = float(NO_HARM_FINAL_QP_NO_HARM_WEIGHT)
                loss = 0.5 * (q_loss + p_loss)
                ledger[f'{name}_q_{local_label}'] = float(q_local.detach().cpu())
                ledger[f'{name}_p_{local_label}'] = float(p_local.detach().cpu())
                ledger[f'{name}_q_global_recurrent_regret'] = float(q_recurrent.detach().cpu())
                ledger[f'{name}_p_global_recurrent_regret'] = float(p_recurrent.detach().cpu())
                ledger[f'{name}_q_upper_semideviation'] = float(q_semideviation.detach().cpu())
                ledger[f'{name}_p_upper_semideviation'] = float(p_semideviation.detach().cpu())
                if local_update == 1:
                    for component, local_loss, recurrent_loss in (('q', q_local, q_recurrent), ('p', p_local, p_recurrent)):
                        parameters = tuple((parameter for parameter_name, parameter in candidate.named_parameters() if parameter.requires_grad and parameter_name.startswith(f'{component}_delta_')))
                        if not parameters:
                            parameters = tuple((parameter for parameter in candidate.parameters() if parameter.requires_grad))
                        local_gradients = torch.autograd.grad(local_loss, parameters, retain_graph=True, allow_unused=True)
                        recurrent_gradients = torch.autograd.grad(recurrent_loss, parameters, retain_graph=True, allow_unused=True)
                        local_vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1).float() for parameter, gradient in zip(parameters, local_gradients, strict=True)])
                        recurrent_vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1).float() for parameter, gradient in zip(parameters, recurrent_gradients, strict=True)])
                        local_norm = local_vector.norm()
                        recurrent_norm = recurrent_vector.norm()
                        denominator = local_norm * recurrent_norm
                        cosine = torch.dot(local_vector, recurrent_vector) / denominator if float(denominator.cpu()) > 0.0 else local_norm.new_tensor(float('nan'))
                        retention_denominator = local_norm + recurrent_norm
                        retention = (local_vector + recurrent_vector).norm() / retention_denominator if float(retention_denominator.cpu()) > 0.0 else local_norm.new_tensor(float('nan'))
                        ledger[f'{name}_{component}_{local_label}_gradient_norm'] = float(local_norm.cpu())
                        ledger[f'{name}_{component}_recurrent_gradient_norm'] = float(recurrent_norm.cpu())
                        ledger[f'{name}_{component}_{local_label}_recurrent_cosine'] = float(cosine.cpu())
                        ledger[f'{name}_{component}_combined_norm_retention'] = float(retention.cpu())
                    if semideviation_rows is not None:
                        for component, endpoint_loss, recurrent_loss, tail_loss in (('q', q_local, q_recurrent, q_semideviation), ('p', p_local, p_recurrent, p_semideviation)):
                            parameters = tuple((parameter for parameter_name, parameter in candidate.named_parameters() if parameter.requires_grad and parameter_name.startswith(f'{component}_delta_')))
                            if not parameters:
                                parameters = tuple((parameter for parameter in candidate.parameters() if parameter.requires_grad))
                            base_loss = 0.5 * (endpoint_loss + recurrent_loss)
                            base_gradients = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
                            tail_gradients = torch.autograd.grad(tail_loss, parameters, retain_graph=True, allow_unused=True)
                            base_vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1).float() for parameter, gradient in zip(parameters, base_gradients, strict=True)])
                            tail_vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1).float() for parameter, gradient in zip(parameters, tail_gradients, strict=True)])
                            base_norm = base_vector.norm()
                            tail_norm = tail_vector.norm()
                            denominator = base_norm * tail_norm
                            cosine = torch.dot(base_vector, tail_vector) / denominator if float(denominator.cpu()) > 0.0 else base_norm.new_tensor(float('nan'))
                            retention_denominator = base_norm + tail_norm
                            retention = (base_vector + tail_vector).norm() / retention_denominator if float(retention_denominator.cpu()) > 0.0 else base_norm.new_tensor(float('nan'))
                            ledger[f'{name}_{component}_base_gradient_norm'] = float(base_norm.cpu())
                            ledger[f'{name}_{component}_semideviation_gradient_norm'] = float(tail_norm.cpu())
                            ledger[f'{name}_{component}_base_semideviation_cosine'] = float(cosine.cpu())
                            ledger[f'{name}_{component}_base_semideviation_retention'] = float(retention.cpu())
            projected_constraint_gradients: tuple[dict[str, torch.Tensor], ...] | None = None
            nonlinear_constraint_before_values: dict[str, float] | None = None
            nonlinear_constraint_before_parameters: dict[str, torch.Tensor] | None = None
            if DEBUG_AUTOGRAD_ANOMALY_FROM_UPDATE is not None and update >= DEBUG_AUTOGRAD_ANOMALY_FROM_UPDATE:
                with torch.autograd.detect_anomaly(check_nan=True):
                    loss.backward()
            else:
                loss.backward()
            if isinstance(candidate, HamiBallsResponsibilitySeparatedGate):
                if any((parameter.grad is not None for parameter in candidate.base_gate.parameters())):
                    raise AssertionError('responsibility loss reached the frozen base gate')
                for direction, tower in (('h', candidate.h_evidence), ('d', candidate.d_evidence)):
                    gradients = [parameter.grad for parameter in tower.parameters() if parameter.grad is not None]
                    if not gradients or not all((bool(torch.isfinite(gradient).all()) for gradient in gradients)):
                        raise AssertionError(f'responsibility {direction} tower lacks finite gradients')
                    gradient_abs_sum = sum((float(gradient.detach().abs().sum().cpu()) for gradient in gradients))
                    if gradient_abs_sum == 0.0:
                        raise AssertionError(f'responsibility {direction} tower has zero gradient')
                    ledger[f'{name}_{direction}_tower_gradient_abs_sum'] = gradient_abs_sum
            grad = float(torch.nn.utils.clip_grad_norm_([parameter for parameter in candidate.parameters() if parameter.requires_grad], 1.0))
            if not math.isfinite(grad):
                nonfinite_parameters = [parameter_name for parameter_name, parameter in candidate.named_parameters() if parameter.requires_grad and parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all()))]
                message = {'warning': 'nonfinite optimizer gradient; update skipped', 'candidate': name, 'update': update, 'stage_update': local_update, 'loss': float(loss.detach().cpu()), 'nonfinite_parameters': nonfinite_parameters}
                print(json.dumps(message, sort_keys=True), flush=True)
                ledger[f'{name}_optimizer_step_skipped_nonfinite'] = True
                skipped_nonfinite_optimizer_steps[name] += 1
                optimizer.zero_grad(set_to_none=True)
                if not NONFATAL_NONFINITE_OPTIMIZER_STEP:
                    raise FloatingPointError(f'{name} optimizer gradient is non-finite at update {update}')
            else:
                optimizer.step()
                actual_optimizer_steps[name] += 1
            if isinstance(candidate, HamiBallsStagedFrozenRiskPositiveConeCapPerObjectCompactCommittedGate):
                candidate.project_cap_parameters_()
                ledger[f'{name}_positive_cone_projection'] = True
            ledger[f'{name}_objective'] = objective
            ledger[f'{name}_loss'] = float(loss.detach().cpu())
            ledger[f'{name}_grad'] = grad
        if update == 1 or update % 50 == 0 or update in {args.updates, args.updates + 1, total_updates}:
            history.append(ledger)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_wall = time.perf_counter() - started
    train_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    train_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else None
    after = {name: _metrics(candidate, hold_caches, online_context=online_context) for name, candidate in candidates.items()}
    fit_after = {name: _metrics(candidate, fit_caches, online_context=online_context) for name, candidate in candidates.items()}
    veto_holdout_metrics: dict[str, float] | None = None
    veto_fit_metrics: dict[str, float] | None = None
    private_holdout_source_object_risk: dict[str, Any] | None = None
    private_holdout_parent_relative: dict[str, Any] | None = None
    frozen_after = {'d': module_digest(d), 'h': _hamiltonian_digest(h), 'r': module_digest(residual), 'history_gate': module_digest(history_gate)}
    expected_frozen = {**frozen, 'history_gate': history_digest}
    if frozen_after != expected_frozen:
        raise AssertionError('gate training mutated a frozen model')
    result = {
        'dataset_contract': dataset_contract,
        'objectives': arms,
        'updates': args.updates,
        'actual_optimizer_steps': actual_optimizer_steps,
        'skipped_nonfinite_optimizer_steps': skipped_nonfinite_optimizer_steps,
        'learning_rate': {'maximum': args.maximum_lr, 'minimum': args.minimum_lr},
        'fit_samples': args.fit_samples,
        'holdout_samples': args.holdout_samples,
        'noise_seeds': noise_seeds,
        'field_indices': field_indices,
        'sequence_batch': args.sequence_batch,
        'qp_no_harm_weight': args.qp_no_harm_weight,
        'metrics': {'fit_before': fit_before, 'fit_after': fit_after, 'before': before, 'after': after},
        'packing_error': {'absolute': packing_max_abs, 'relative': packing_relative_max_abs},
        'wall_seconds': {'fit_collection': fit_collection_wall, 'holdout_collection': hold_collection_wall, 'training': train_wall},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    for name, candidate in candidates.items():
        torch.save({'gate_state_dict': candidate.state_dict(), 'summary': result}, args.output_dir / f'{name}_gate.pt')
    (args.output_dir / 'COMPLETE').write_text('PASS\n', encoding='utf8')
    print(json.dumps(result, indent=2))
