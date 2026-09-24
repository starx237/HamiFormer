from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any
import numpy as np
import torch
from hamiformer.data.hamiballs2 import HamiBalls2CropPackDataset
from hamiformer.flow.rectified_flow import clean_to_velocity
from hamiformer.models.hamiballs2_dual_expert import hamiballs2_node_graph_features
from hamiformer.models.hamiballs2_hamiltonian import graph_context
from hamiformer.models.hamiballs2_posthd_formal import HamiBalls2FormalPostHD
from hamiformer.physics.generic_gfjp import generic_gfjp_scan
from hamiformer.physics.hamiballs_type2 import flatten_hamiballs_phase, unflatten_hamiballs_phase
from hamiformer.physics.pgf_scan import apply_prefix, parallel_doubling_prefix, serial_prefix
from hamiformer.training.hamiballs2_posthd_fitting import HierarchicalRidgeAccumulator, apply_serialized_tree, calibrate_component_alpha, cart_responsibility, fit_source_balanced_cart, source_fold
from hamiformer.training.hamiballs2_posthd_formal import common_residual_objective, final_qp_objective, scalar0_objective, scalar1_objective
from hamiformer.training.hamiballs2_posthd_protocol import common_r_refresh_plan, final_refresh_plan, namespaced_seed, scalar_pool_plan
from hamiformer.inference.mixed_rollout import MixedRollout
ROOT = project_root()

def _atomic_torch(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)

def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')
        handle.flush()

def _fetch(dataset: HamiBalls2CropPackDataset, rows: np.ndarray, device: torch.device):
    names = ('phase', 'attrs', 'time', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'physical_seed')
    return {name: torch.from_numpy(np.array(dataset.arrays[name][rows], copy=True)).to(device) for name in names}

class CarrierCollector(MixedRollout):

    @torch.no_grad()
    def collect(self, assignment, *, source_seed, noise_seed, route_seed, parent_mode, store_affine_jets=None):
        if store_affine_jets is None:
            store_affine_jets = bool(self.cfg['protocol'].get('replay_affine_jets_during_training', False))
        assignment = np.asarray(assignment, dtype=np.int16)
        rng = np.random.default_rng(int(source_seed))
        selected_rows = rng.choice(len(self.dataset), len(assignment), replace=False)
        records = []
        for rf_steps in sorted(np.unique(assignment)):
            positions = np.flatnonzero(assignment == rf_steps)
            rows = selected_rows[positions]
            batch = _fetch(self.dataset, rows, self.device)
            true_previous = torch.cat((batch['phase'][:, :1].float(), batch['phase'][:, 1:-1].float()), 1)
            valid = batch['object_mask'].bool()[:, None, :, None]
            noise_generator = torch.Generator(device=self.device).manual_seed(int(noise_seed) + int(rf_steps))
            route_generator = torch.Generator(device=self.device).manual_seed(int(route_seed) + int(rf_steps))
            state = 0.1 * self.scale[None, None, None] * torch.randn(len(rows), 48, 10, 6, device=self.device, generator=noise_generator)
            state *= valid
            grid = torch.linspace(0.0, 1.0, int(rf_steps) + 1, device=self.device)
            group_records = []
            committed = None
            cold_intervals = self._cold_start_intervals(int(rf_steps))
            for step in range(int(rf_steps)):
                left, right = (grid[step], grid[step + 1])
                tau = left.expand(len(rows))
                is_cold = step < cold_intervals
                if is_cold:
                    d, _, next_anchor = self._d_only_field(state, tau, batch)
                    v_left = clean_to_velocity(d, state, tau, t_eps=self.t_eps)
                    if step == int(rf_steps) - 1:
                        state = state + (right - left) * v_left
                        continue
                    proposal = state + (right - left) * v_left
                    right_d, _, _ = self._d_only_field(proposal, right.expand(len(rows)), batch)
                    v_right = clean_to_velocity(right_d, proposal, right.expand(len(rows)), t_eps=self.t_eps)
                    state = state + 0.5 * (right - left) * (v_left + v_right)
                    committed = next_anchor
                    continue
                field_result = self._field(state, tau, batch, parent_mode=parent_mode, route_generator=route_generator, anchor=committed, return_affine_jets=store_affine_jets)
                output, d, tokens, local_h, forced, tangent, d_reset_edges, d_reset_mask, tangent_exact_eigensolve_edges = field_result[:9]
                group_records.append({'state': state.cpu(), 'd_candidate': d.cpu(), 'h_candidate': output['h_candidate'].cpu(), 'local_h': local_h.cpu(), 'local_previous': true_previous.cpu(), 'd_tokens': tokens.cpu(), 'target': batch['phase'][:, 1:].float().cpu(), 'x0': batch['phase'][:, 0].float().cpu(), 'attrs': batch['attrs'].float().cpu(), 'node_graph': hamiballs2_node_graph_features(batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float()).cpu(), 'physical_time': batch['time'][:, 1:].float().cpu(), 'tau': tau.cpu(), 'physical_fraction': torch.linspace(0.0, 1.0, 48, device=self.device)[None].expand(len(rows), -1).cpu(), 'object_mask': batch['object_mask'].bool().cpu(), 'source': torch.from_numpy(rows.astype(np.int64)), 'carrier_source': torch.from_numpy(positions.astype(np.int64)), 'rf_num_steps': torch.full((len(rows),), int(rf_steps), dtype=torch.int16), 'rf_field_index': torch.full((len(rows),), int(step), dtype=torch.int16), 'forced_gate': (forced if forced is not None else torch.zeros_like(output['gate'])).cpu(), 'carrier_previous': torch.cat((batch['phase'][:, :1].float(), output['mixed'][:, :-1]), 1).cpu(), 'tangent': tangent.cpu(), 'tangent_diagnostic_computed': torch.full((len(rows),), self.exact_tangent_gate, dtype=torch.bool), 'plas_d_reset_edges': d_reset_edges.cpu(), 'plas_d_reset_mask': d_reset_mask.cpu(), 'tangent_exact_eigensolve_edges': tangent_exact_eigensolve_edges.cpu()})
                if store_affine_jets:
                    group_records[-1]['jet_matrix'] = field_result[9].cpu()
                    group_records[-1]['jet_offset'] = field_result[10].cpu()
                clean = output['mixed']
                v_left = clean_to_velocity(clean, state, tau, t_eps=self.t_eps)
                next_anchor = tuple((value.detach() for value in output['next_anchor']))
                if step == int(rf_steps) - 1:
                    state = state + (right - left) * v_left
                    continue
                proposal = state + (right - left) * v_left
                right_tau = right.expand(len(rows))
                right_output, *_ = self._field(proposal, right_tau, batch, parent_mode=parent_mode, route_generator=route_generator, anchor=committed)
                v_right = clean_to_velocity(right_output['mixed'], proposal, right_tau, t_eps=self.t_eps)
                state = state + 0.5 * (right - left) * (v_left + v_right)
                committed = next_anchor
            records.extend(group_records)
        keys = records[0].keys()
        return {name: torch.cat([row[name] for row in records], 0) for name in keys}

def _subset(pool: dict[str, torch.Tensor], index: np.ndarray, device: torch.device):
    chosen = torch.from_numpy(np.asarray(index, dtype=np.int64))
    return {name: value[chosen].to(device, non_blocking=True) for name, value in pool.items()}

def _batches(row_count: int, *, seed: int, count: int=8):
    order = np.random.default_rng(int(seed)).permutation(row_count)
    return [part for part in np.array_split(order, count) if len(part)]

def _concat_pools(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        raise ValueError('carrier concatenation requires at least one pool')
    keys = set(parts[0])
    if any((set(part) != keys for part in parts)):
        raise ValueError('carrier halves do not share the same fields')
    adjusted = []
    offset = 0
    for part in parts:
        local = dict(part)
        slots = local['carrier_source'].long()
        local['carrier_source'] = slots + offset
        offset += int(slots.max().item()) + 1
        adjusted.append(local)
    return {name: torch.cat([part[name] for part in adjusted], 0) for name in parts[0]}

def _source_axis_batches(pool: dict[str, torch.Tensor], *, seed: int, count: int=8, expected_sources: int=64) -> list[np.ndarray]:
    slots = pool['carrier_source'].cpu().numpy().astype(np.int64, copy=False)
    unique = np.unique(slots)
    if len(unique) != int(expected_sources):
        raise ValueError(f'source-axis carrier requires {expected_sources} sources, got {len(unique)}')
    rng = np.random.default_rng(int(seed))
    chosen = np.empty((count, len(unique)), dtype=np.int64)
    for column, source_slot in enumerate(unique):
        rows = np.flatnonzero(slots == source_slot)
        if not len(rows):
            raise ValueError(f'source slot {source_slot} exposes no accepted hybrid fields')
        chosen[:, column] = rng.permutation(rows)[:count] if len(rows) >= count else rng.choice(rows, size=count, replace=True)
    return [row[rng.permutation(len(row))] for row in chosen]

def _next_same_source(pool: dict[str, torch.Tensor]) -> np.ndarray:
    source = pool.get('carrier_source', pool['source']).cpu().numpy().astype(np.int64, copy=False)
    result = np.arange(len(source), dtype=np.int64)
    for value in np.unique(source):
        rows = np.flatnonzero(source == value)
        if len(rows) < 2:
            raise ValueError('common-r hull pairing requires at least two accepted RF fields per source')
        result[rows] = np.roll(rows, -1)
    return result

def _validate_common_hull_pair(primary: dict[str, torch.Tensor], hull: dict[str, torch.Tensor], primary_indices: np.ndarray, hull_indices: np.ndarray) -> None:
    if np.any(np.asarray(primary_indices) == np.asarray(hull_indices)):
        raise RuntimeError('common-r hull field must differ from the primary RF field')
    for name in ('source', 'target', 'x0', 'object_mask'):
        if not torch.equal(primary[name], hull[name]):
            raise RuntimeError(f'common-r hull pairing changed same-source field {name}')
    if 'carrier_source' in primary and (not torch.equal(primary['carrier_source'], hull['carrier_source'])):
        raise RuntimeError('common-r hull pairing changed carrier source slot')

def _lr(update: int, total: int, maximum: float, minimum: float) -> float:
    progress = (update - 1) / max(total - 1, 1)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))

def _optimizer(model, cfg, stage):
    trainable = model.set_training_stage(stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or trainable != sum((p.numel() for p in parameters)):
        raise RuntimeError(f'{stage} trainable-parameter contract failed')
    spec = cfg['optimizer'][stage]
    return torch.optim.AdamW(parameters, lr=float(spec['maximum_lr']), weight_decay=float(spec['weight_decay']), fused=torch.cuda.is_available())

def _local_common_candidates(model, batch):
    scale = model.phase_scale.to(batch['state'])
    state, d, h = (batch['state'] / scale, batch['d_candidate'] / scale, batch['local_h'] / scale)
    x0 = batch['x0'] / scale
    previous = batch['local_previous'] / scale
    static = (torch.cat((batch['attrs'], batch['node_graph']), -1) - model.static_feature_mean) / model.static_feature_scale
    static = static * batch['object_mask'][..., None]
    rows = []
    for edge in range(state.shape[1]):
        time = batch['physical_fraction'][:, edge]
        previous_gate = batch.get('previous_gate_start', torch.ones_like(batch['forced_gate'][:, edge])) if edge == 0 else batch['forced_gate'][:, edge - 1]
        residual, _ = model.common_r(model._r_features(batch['d_tokens'][:, edge], state[:, edge], d[:, edge], h[:, edge], previous[:, edge], x0, static, batch['tau'], time, previous_gate))
        rows.append((h[:, edge] + residual) * scale)
    return torch.stack(rows, 1)

def _forward(model, batch, *, force_random=False):
    forced = batch['forced_gate'] if force_random else None
    return model(batch['state'], batch['d_candidate'], batch['h_candidate'], batch['d_tokens'], x0=batch['x0'], attrs=batch['attrs'], node_graph=batch['node_graph'], physical_time=batch['physical_time'], tau=batch['tau'], object_mask=batch['object_mask'], force_gate=forced, previous_start=batch.get('previous_start'), previous_gate_start=batch.get('previous_gate_start'), physical_fraction=batch.get('physical_fraction'), detach_physical_history=bool(batch.get('detach_physical_history', False)), jet_matrix=batch.get('jet_matrix'), jet_offset=batch.get('jet_offset'))

def _common_window(batch: dict[str, torch.Tensor], *, update: int, length: int) -> dict[str, torch.Tensor]:
    edges = int(batch['state'].shape[1])
    if not 1 <= length <= edges:
        raise ValueError('common-r window length is outside the physical horizon')
    start = (int(update) - 1) % (edges - length + 1)
    stop = start + length
    result = dict(batch)
    for name in ('state', 'd_candidate', 'h_candidate', 'local_h', 'local_previous', 'd_tokens', 'target', 'physical_time', 'physical_fraction', 'forced_gate', 'carrier_previous', 'jet_matrix', 'jet_offset'):
        if name in batch:
            result[name] = batch[name][:, start:stop]
    result['previous_start'] = batch['carrier_previous'][:, start]
    result['previous_gate_start'] = torch.ones_like(batch['forced_gate'][:, 0]) if start == 0 else batch['forced_gate'][:, start - 1]
    result['detach_physical_history'] = True
    return result

def _train_block(model, optimizer, cfg, stage, pool, update_start, seed, metrics_path, warnings_path):
    spec = cfg['optimizer'][stage]
    total = int(cfg['protocol'][f'{stage}_updates'])
    effective = 0
    hull_lookup = _next_same_source(pool) if stage == 'common_r' else None
    for local, indices in enumerate(_source_axis_batches(pool, seed=seed), 1):
        update = update_start + local
        batch = _subset(pool, indices, next(model.parameters()).device)
        if stage == 'common_r':
            batch = _common_window(batch, update=update, length=int(cfg['protocol']['common_r_window_edges']))
        optimizer.zero_grad(set_to_none=True)
        try:
            phase_scale = model.phase_scale.to(batch['state'])
            normalized_d = batch['d_candidate'] / phase_scale
            normalized_target = batch['target'] / phase_scale
            if stage == 'common_r':
                output = _forward(model, batch, force_random=True)
                local_hr = _local_common_candidates(model, batch)
                assert hull_lookup is not None
                hull_batch = _common_window(_subset(pool, hull_lookup[indices], next(model.parameters()).device), update=update, length=int(cfg['protocol']['common_r_window_edges']))
                _validate_common_hull_pair(batch, hull_batch, indices, hull_lookup[indices])
                hull_output = _forward(model, hull_batch, force_random=True)
                loss_result = common_residual_objective(local_h=batch['local_h'] / phase_scale, local_hr=local_hr / phase_scale, recovery_h=output['h_candidate'] / phase_scale, recovery_hr=output['hr_candidate'] / phase_scale, hull_hr=hull_output['hr_candidate'] / phase_scale, hull_d=hull_batch['d_candidate'] / phase_scale, target=normalized_target, object_mask=batch['object_mask'], qp_scales=torch.tensor(cfg['protocol']['q_p_training_scales'], device=batch['state'].device))
                loss = loss_result.total
            else:
                output = _forward(model, batch)
                normalized_hr = output['hr_candidate'] / phase_scale
                if stage == 'scalar0':
                    loss = scalar0_objective(output['gate'], normalized_d, normalized_hr, normalized_target, object_mask=batch['object_mask'])
                elif stage == 'scalar1':
                    loss = scalar1_objective(output['gate'], normalized_d, normalized_hr, normalized_target, object_mask=batch['object_mask'], no_harm_weight=float(cfg['protocol']['scalar1_no_harm_weight'])).total
                else:
                    loss = final_qp_objective(output['gate'], normalized_d, normalized_hr, normalized_target, attrs=batch['attrs'], state_scale=model.phase_scale, object_mask=batch['object_mask'], frame_dt=float(cfg['hamiltonian']['frame_dt']), no_harm_weight=float(cfg['protocol']['final_qp_no_harm_weight'])).total
            loss.backward()
            finite = bool(torch.isfinite(loss)) and all((p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad))
            if not finite:
                raise FloatingPointError('non-finite loss/gradient')
            preclip = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(spec['grad_clip'])))
            lr = _lr(update, total, float(spec['maximum_lr']), float(spec['minimum_lr']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            effective += 1
            _append_jsonl(metrics_path, {'stage': stage, 'requested_update': update, 'effective': True, 'loss': float(loss.detach().cpu()), 'lr': lr, 'gradient_norm_preclip': preclip})
        except Exception as exc:
            optimizer.zero_grad(set_to_none=True)
            _append_jsonl(warnings_path, {'stage': stage, 'requested_update': update, 'effective': False, 'error': repr(exc), 'action': 'skip_update_continue'})
    return effective

def _tree_rows(model, pools, device):
    rows = {name: [] for name in ('observable', 'd', 'h', 'hr', 'target', 'source', 'mask', 'edge')}
    model.eval()
    with torch.no_grad():
        for stored in pools:
            pool = stored if isinstance(stored, dict) else torch.load(stored, map_location='cpu', weights_only=False)
            for indices in _batches(len(pool['state']), seed=1, count=max(1, math.ceil(len(pool['state']) / 96))):
                batch = _subset(pool, indices, device)
                output = _forward(model, batch)
                for name, value in (('observable', output['observable']), ('d', batch['d_candidate']), ('h', output['h_candidate']), ('hr', output['hr_candidate']), ('target', batch['target'])):
                    rows[name].append(value.cpu())
                rows['source'].append(batch['source'].cpu())
                rows['mask'].append(batch['object_mask'].cpu())
                rows['edge'].append(torch.arange(output['observable'].shape[1])[None, :, None].expand(output['observable'].shape[:-1]).cpu())
    joined = {name: torch.cat(values, 0) for name, values in rows.items()}
    valid = joined['mask'][:, None].expand(joined['observable'].shape[:-1])
    source = joined['source'][:, None, None].expand(valid.shape)
    result = {name: joined[name][valid].numpy() for name in ('observable', 'd', 'h', 'hr', 'target')}
    result['source'] = source[valid].numpy()
    result['edge'] = joined['edge'][valid].numpy()
    return result

def _install_tree_and_initial_ridge(model, rows, cfg):
    scale = np.asarray(cfg['model']['phase_scale'], np.float64)
    response = cart_responsibility(rows['d'] / scale, rows['hr'] / scale, rows['h'] / scale, rows['target'] / scale, qp_scale=np.ones(2), include_endpoint_advantage=bool(cfg['protocol'].get('tree_endpoint_advantage', False)))
    fitted = fit_source_balanced_cart(rows['observable'], response, rows['source'], master_seed=int(cfg['protocol']['master_seed']), max_depth=int(cfg['protocol']['tree_max_depth']), max_leaves=int(cfg['protocol']['tree_max_leaves']), feature_budget=int(cfg['protocol']['tree_feature_budget']))
    folds = source_fold(rows['source'], master_seed=int(cfg['protocol']['master_seed']))
    head = (folds == 2) & (rows['edge'] > 0)
    mean = rows['observable'][head].mean(0)
    std = np.maximum(rows['observable'][head].std(0), 1e-05)
    design = np.concatenate((np.clip((rows['observable'] - mean) / std, -8, 8), np.ones((len(head), 1))), -1)
    leaf = apply_serialized_tree(fitted.tree, rows['observable'])
    residual = (rows['target'] - rows['h']) / scale
    accumulator = HierarchicalRidgeAccumulator(design.shape[1], max_leaves=int(cfg['protocol']['tree_max_leaves']))
    accumulator.add(design[head], residual[head], leaf[head], rows['source'][head])
    weight, ridge_report = accumulator.solve(ridge_relative=float(cfg['protocol']['ridge_relative']), intercept_multiplier=float(cfg['protocol']['ridge_intercept_multiplier']))
    prediction = np.zeros_like(residual)
    for region in np.unique(leaf):
        use = leaf == region
        prediction[use, :3] = design[use] @ weight[int(region), 0]
        prediction[use, 3:] = design[use] @ weight[int(region), 1]
    calibration_mask = (folds == 3) & (rows['edge'] > 0)
    alpha, calibration = calibrate_component_alpha(prediction, residual, rows['source'], calibration_mask)
    model.install_fitted_state(tree=fitted.tree, feature_mean=torch.from_numpy(mean).float().to(model.phase_scale.device), feature_scale=torch.from_numpy(std).float().to(model.phase_scale.device), ridge_weight=torch.from_numpy(weight).float().to(model.phase_scale.device), component_alpha=torch.from_numpy(alpha).float().to(model.phase_scale.device))
    calibration_store = [(design[calibration_mask], residual[calibration_mask], leaf[calibration_mask], rows['source'][calibration_mask])]
    return (accumulator, calibration_store, {'tree': fitted.tree, 'tree_report': fitted.fold_report, 'ridge': ridge_report, 'calibration': calibration})

def _update_ridge(model, accumulator, calibration_store, pool, cfg, device):
    rows = _tree_rows(model, [pool], device)
    folds = source_fold(rows['source'], master_seed=int(cfg['protocol']['master_seed']))
    mean = model.ridge_feature_mean.detach().cpu().numpy()
    std = model.ridge_feature_scale.detach().cpu().numpy()
    design = np.concatenate((np.clip((rows['observable'] - mean) / std, -8, 8), np.ones((len(rows['source']), 1))), -1)
    leaf = apply_serialized_tree(model.tree, rows['observable'])
    scale = np.asarray(cfg['model']['phase_scale'], np.float64)
    residual = (rows['target'] - rows['h']) / scale
    use = (folds == 2) & (rows['edge'] > 0)
    accumulator.add(design[use], residual[use], leaf[use], rows['source'][use])
    weight, report = accumulator.solve(ridge_relative=float(cfg['protocol']['ridge_relative']), intercept_multiplier=float(cfg['protocol']['ridge_intercept_multiplier']))
    calibrate = (folds == 3) & (rows['edge'] > 0)
    calibration_store.append((design[calibrate], residual[calibrate], leaf[calibrate], rows['source'][calibrate]))
    x = np.concatenate([row[0] for row in calibration_store])
    y = np.concatenate([row[1] for row in calibration_store])
    leaves = np.concatenate([row[2] for row in calibration_store])
    sources = np.concatenate([row[3] for row in calibration_store])
    prediction = np.zeros_like(y)
    for region in np.unique(leaves):
        local = leaves == region
        prediction[local, :3] = x[local] @ weight[int(region), 0]
        prediction[local, 3:] = x[local] @ weight[int(region), 1]
    alpha, calibration = calibrate_component_alpha(prediction, y, sources, np.ones(len(sources), bool))
    model.ridge_weight.copy_(torch.from_numpy(weight).float().to(device))
    model.component_alpha.copy_(torch.from_numpy(alpha).float().to(device))
    return {'ridge': report, 'calibration': calibration}

def execute_training(cfg: dict[str, Any], config_path: Path) -> None:
    from hamiformer.training.hamiballs2.downstream import _load_models, _validate_config, prepare
    contract = _validate_config(cfg)
    output = Path(cfg['output_dir'])
    prepared = output / 'preparation_manifest.json'
    if not prepared.is_file():
        prepare(cfg, config_path)
    initialization = torch.load(output / 'formal_posthd_initialization.pt', map_location='cpu', weights_only=False)
    if initialization.get('optimizer_updates') != 0 or initialization.get('tree') is not None:
        raise RuntimeError('formal initialization is not a pristine zero-update boundary')
    if (output / 'COMPLETE').exists():
        raise FileExistsError(output / 'COMPLETE')
    prepared_marker = output / 'PREPARED_NOT_STARTED'
    if prepared_marker.exists():
        prepared_marker.unlink()
    (output / 'RUNNING').write_text('formal post-HD training explicitly launched\n', encoding='utf-8')
    device = torch.device('cuda')
    wide, h, _, _ = _load_models(cfg, device)
    model = HamiBalls2FormalPostHD(**cfg['model']).to(device)
    model.load_state_dict(initialization['model_state_dict'], strict=True)
    dataset = HamiBalls2CropPackDataset(cfg['train_pack'])
    collector = CarrierCollector(cfg=cfg, dataset=dataset, wide=wide, hamiltonian=h, model=model, device=device)
    metrics = output / 'training_metrics.jsonl'
    warnings = output / cfg['operations']['warning_ledger']
    errors = output / cfg['operations']['error_ledger']
    started = time.perf_counter()
    stage_report = {}
    requested_total = effective_total = 0
    def run_stage(stage, plans, parent_mode, persist):
        nonlocal requested_total, effective_total
        optimizer = _optimizer(model, cfg, stage)
        effective = 0
        pools = []
        transient_count = 0
        health_scene_edges = health_d_resets = health_exact_eigensolves = 0
        for block, plan in enumerate(plans):
            try:
                if 'noise_halves' in plan:
                    parts = [collector.collect(half['rf_num_steps'], source_seed=half['source_seed'], noise_seed=half['noise_seed'], route_seed=namespaced_seed(int(cfg['protocol']['master_seed']), 'final_unused_route', block * 2 + half['noise_slot']), parent_mode=parent_mode) for half in plan['noise_halves']]
                    pool = _concat_pools(parts)
                else:
                    pool = collector.collect(plan['rf_num_steps'], source_seed=plan['source_seed'], noise_seed=plan['noise_seed'], route_seed=plan['route_seed'], parent_mode=parent_mode)
                scene_edges = int(pool['plas_d_reset_mask'].numel())
                d_resets = int(pool['plas_d_reset_mask'].sum())
                eigensolves = int(pool['tangent_exact_eigensolve_edges'].sum())
                health_scene_edges += scene_edges
                health_d_resets += d_resets
                health_exact_eigensolves += eigensolves
                _append_jsonl(metrics, {'stage': f'{stage}_carrier', 'block': block, 'accepted_fields': int(len(pool['state'])), 'scene_edges': scene_edges, 'd_reset_scene_edges': d_resets, 'tangent_exact_eigensolve_edges': eigensolves})
                if persist:
                    pools.append(pool)
                else:
                    transient_count += 1
                if not persist and stage != 'final_qp':
                    count = _train_block(model, optimizer, cfg, stage, pool, block * 8, namespaced_seed(int(cfg['protocol']['master_seed']), f'{stage}_minibatch', block), metrics, warnings)
                    requested_total += 8
                    effective_total += count
                    effective += count
            except Exception as exc:
                _append_jsonl(errors, {'stage': stage, 'block': block, 'error': repr(exc), 'traceback': traceback.format_exc(), 'action': 'continue_next_block'})
        if persist and stage != 'final_qp':
            for block, pool in enumerate(pools):
                count = _train_block(model, optimizer, cfg, stage, pool, block * 8, namespaced_seed(int(cfg['protocol']['master_seed']), f'{stage}_minibatch', block), metrics, warnings)
                requested_total += 8
                effective_total += count
                effective += count
        carrier_count = len(pools) if persist else transient_count
        stage_report[stage] = {'requested_updates': len(plans) * 8, 'effective_updates': effective, 'carriers': carrier_count, 'all_carriers_precomputed_before_update1': bool(persist), 'carrier_storage': 'host_memory' if persist else 'transient', 'scene_edges': health_scene_edges, 'd_reset_scene_edges': health_d_resets, 'd_reset_fraction': health_d_resets / max(health_scene_edges, 1), 'tangent_exact_eigensolve_edges': health_exact_eigensolves}
        _atomic_torch(output / f'stage_{stage}.pt', {'model': model.state_dict(), 'tree': copy.deepcopy(model.tree), 'report': stage_report[stage], 'requested_total': requested_total, 'effective_total': effective_total})
        return (pools, optimizer)
    common_plans = common_r_refresh_plan(master_seed=int(cfg['protocol']['master_seed']))
    run_stage('common_r', common_plans, 'random', False)
    scalar0_pools, _ = run_stage('scalar0', scalar_pool_plan(master_seed=int(cfg['protocol']['master_seed']), stage='scalar0'), 'random', True)
    scalar0_pools.clear()
    scalar1_pools, _ = run_stage('scalar1', scalar_pool_plan(master_seed=int(cfg['protocol']['master_seed']), stage='scalar1'), 'learned', True)
    tree_rows = _tree_rows(model, scalar1_pools, device)
    accumulator, calibration_store, fitted_report = _install_tree_and_initial_ridge(model, tree_rows, cfg)
    fixed_tree_signature = json.dumps(model.tree, sort_keys=True, separators=(',', ':'))
    _atomic_torch(output / 'fixed_tree_initial_ridge.pt', {'model': model.state_dict(), 'tree': copy.deepcopy(model.tree), 'report': fitted_report})
    scalar1_pools.clear()
    final_optimizer = _optimizer(model, cfg, 'final_qp')
    final_scene_edges = final_d_resets = final_exact_eigensolves = 0
    for refresh, plan in enumerate(final_refresh_plan(master_seed=int(cfg['protocol']['master_seed']))):
        try:
            if json.dumps(model.tree, sort_keys=True, separators=(',', ':')) != fixed_tree_signature:
                raise RuntimeError('fixed Tree changed during final-q/p')
            parts = [collector.collect(half['rf_num_steps'], source_seed=half['source_seed'], noise_seed=half['noise_seed'], route_seed=namespaced_seed(int(cfg['protocol']['master_seed']), 'final_unused_route', refresh * 2 + half['noise_slot']), parent_mode='learned') for half in plan['noise_halves']]
            pool = _concat_pools(parts)
            scene_edges = int(pool['plas_d_reset_mask'].numel())
            d_resets = int(pool['plas_d_reset_mask'].sum())
            eigensolves = int(pool['tangent_exact_eigensolve_edges'].sum())
            final_scene_edges += scene_edges
            final_d_resets += d_resets
            final_exact_eigensolves += eigensolves
            ridge_report = _update_ridge(model, accumulator, calibration_store, pool, cfg, device)
            effective = _train_block(model, final_optimizer, cfg, 'final_qp', pool, refresh * 8, plan['optimizer_seed'], metrics, warnings)
            if json.dumps(model.tree, sort_keys=True, separators=(',', ':')) != fixed_tree_signature:
                raise RuntimeError('fixed Tree changed during final-q/p')
            requested_total += 8
            effective_total += effective
            _append_jsonl(metrics, {'stage': 'final_qp_refresh', 'refresh': refresh, 'effective_updates': effective, 'ridge': ridge_report, 'accepted_fields': int(len(pool['state'])), 'scene_edges': scene_edges, 'd_reset_scene_edges': d_resets, 'tangent_exact_eigensolve_edges': eigensolves})
        except Exception as exc:
            _append_jsonl(errors, {'stage': 'final_qp', 'refresh': refresh, 'error': repr(exc), 'traceback': traceback.format_exc(), 'action': 'continue_next_refresh'})
    stage_report['final_qp'] = {'requested_updates': 200, 'effective_updates': effective_total - sum((stage_report[s]['effective_updates'] for s in ('common_r', 'scalar0', 'scalar1'))), 'refreshes': 25, 'scene_edges': final_scene_edges, 'd_reset_scene_edges': final_d_resets, 'd_reset_fraction': final_d_resets / max(final_scene_edges, 1), 'tangent_exact_eigensolve_edges': final_exact_eigensolves}
    bundle = {'schema': 'hamiformer.hamiballs2.formal_posthd_plas_semantic.main.v1', 'status': 'COMPLETE' if requested_total == 800 and effective_total == 800 else 'COMPLETE_WITH_WARNINGS', 'model_config': cfg['model'], 'model_state_dict': model.state_dict(), 'tree': copy.deepcopy(model.tree), 'contract': contract, 'stage_report': stage_report, 'requested_updates': requested_total, 'effective_updates': effective_total, 'wall_seconds': time.perf_counter() - started}
    _atomic_torch(output / 'formal_posthd_terminal.pt', bundle)
    if (output / 'RUNNING').exists():
        (output / 'RUNNING').unlink()
    (output / 'COMPLETE').write_text(bundle['status'] + '\n', encoding='utf-8')
    print(json.dumps({key: bundle[key] for key in ('status', 'requested_updates', 'effective_updates', 'wall_seconds')}), flush=True)
__all__ = ['CarrierCollector', 'execute_training']
