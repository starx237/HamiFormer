from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training import hami1_random_streams as seeds
from hamiformer.training.hamiballs1 import router_carriers as collector
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training.hamiballs1 import fixed_carrier as fixed
from hamiformer.training import base as stage_a
from hamiformer.training.hamiballs1 import scalar_tree_fit as scalar_tree_support
from hamiformer.training.hamiballs1 import leaf_router as leaf_gate_support
from hamiformer.training.hamiballs1 import ridge_training as final
from hamiformer.training.hamiballs1 import hierarchical_residual as hierarchical_residual_support
from hamiformer.training.hamiballs1 import tree_fitting as refiner_evaluation
from hamiformer.training.hamiballs1 import tree_refinement as refiner_training
from hamiformer.training.hamiballs1 import carrier_cache as carrier_cache
from hamiformer.training.hamiballs1.tree_encoder import CausalObservableTreeEncoder
from hamiformer.training.hamiballs1.tree_candidate import conditional_delta
from hamiformer.training.hamiballs1.component_tree import QPLateTreeController
from hamiformer.training.hamiballs1.stage_tree import StageBoundaryHierarchicalController
OUTPUT = ROOT / 'outputs/hami1/tree_residual_matched_model'
TREE_OUTPUT = OUTPUT / 'component_router_parent_honest_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
SEALED_OUTPUT = OUTPUT / 'HIERARCHICAL_TREE_self_contained_main.pt'
COMPONENT_ROUTER_ROOT = collector.GATE_ROOT
COMPONENT_ROUTER_GATE = COMPONENT_ROUTER_ROOT / 'qp/dual_qp_observable_disjoint_endpoint_global_recurrent_upper_semideviation_gate.pt'
TREE_SOURCE_SEED = seeds.derive_seed('g_source_permutation', 0)
TREE_SOURCE_OFFSET = seeds.LOCKED_STAGE_OFFSETS['scalar1']
TREE_SOURCES = 800
TREE_BLOCK = 64
CELLS_PER_SOURCE = 128
OBSERVABLE_DIM = 71
DESIGN_DIM = 72

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)

def _source_ledger() -> tuple[list[dict[str, object]], set[str]]:
    ledger = [json.loads(line) for line in collector.ROW_LEDGER.read_text(encoding='utf-8').splitlines() if line.strip()]
    from hamiformer.training.source_partitions import load_scene_exclusions
    excluded = load_scene_exclusions(collector.EXCLUSION)
    return (ledger, excluded)

@torch.no_grad()
def _decode_parent_carrier(encoder: CausalObservableTreeEncoder, carrier, target: torch.Tensor, attrs: torch.Tensor, source: torch.Tensor, *, cell_seed: int) -> dict[str, np.ndarray]:
    rows: dict[str, list[torch.Tensor]] = {key: [] for key in ('x', 'source', 'target', 'h_candidate', 'd_candidate')}
    names: list[str] | None = None
    for field in carrier.trace.traces:
        rollout = field.rollout
        previous_g, _ = packed._packed_previous_gate_sequence(rollout.gate)
        features = encoder._features(previous_mixed=rollout.previous_mixed, h_candidate=rollout.h_candidate, hr_candidate=rollout.hr_candidate, d_candidate=rollout.d_candidate, attrs=attrs, tau=field.tau, previous_g=previous_g, residual_hidden=rollout.residual_hidden, base_gate=rollout.gate)
        local_names = list(features)
        if names is None:
            names = local_names
        elif names != local_names:
            raise RuntimeError('HIERARCHICAL_TREE causal feature order drifted across RF fields')
        x = torch.stack([features[name] for name in names], dim=-1)
        shape = x.shape[:-1]
        rows['x'].append(x.reshape(-1, len(names)).cpu())
        rows['source'].append(source[:, None, None].expand(shape).reshape(-1).cpu())
        for key, value in (('target', target), ('h_candidate', rollout.h_candidate), ('d_candidate', rollout.d_candidate)):
            rows[key].append(value.reshape(-1, 4).cpu())
    if names is None:
        raise RuntimeError('HIERARCHICAL_TREE parent carrier contained no RF fields')
    joined = {key: torch.cat(value) for key, value in rows.items()}
    source_np = joined['source'].numpy()
    rng = np.random.default_rng(cell_seed)
    chosen = []
    for value in np.unique(source_np):
        available = np.flatnonzero(source_np == value)
        chosen.append(rng.choice(available, min(CELLS_PER_SOURCE, len(available)), replace=False))
    index = torch.from_numpy(np.sort(np.concatenate(chosen)).astype(np.int64))
    return {'x': joined['x'][index].float().numpy(), 'names': np.asarray(names), 'source': joined['source'][index].long().numpy(), 'target': joined['target'][index].float().numpy(), 'h_candidate': joined['h_candidate'][index].float().numpy(), 'd_candidate': joined['d_candidate'][index].float().numpy()}

def _collect_parent_records(*, load_models, encoder_type=CausalObservableTreeEncoder) -> tuple[dict[str, object], list[dict[str, object]]]:
    device = torch.device('cuda')
    models = load_models(parent_root=COMPONENT_ROUTER_ROOT, short_configuration=collector.RUN_SPEC, device=device)
    train = stage_a._load_train_cache(models['config'], device=device)
    ledger, excluded = _source_ledger()
    indices, positions = packed._sample_permutation_block_excluding_scenes(size=train.size, count=TREE_SOURCES, seed=TREE_SOURCE_SEED, offset=TREE_SOURCE_OFFSET, scene_ids=[str(row['scene_id']) for row in ledger], excluded_scenes=excluded)
    encoder = encoder_type().to(device).eval()
    records: list[dict[str, np.ndarray]] = []
    collection: list[dict[str, object]] = []
    started = time.perf_counter()
    block = 0
    for start in range(0, TREE_SOURCES, TREE_BLOCK):
        local = indices[start:start + TREE_BLOCK]
        midpoint = (int(local.numel()) + 1) // 2
        for noise_slot, (noise, use) in enumerate(zip(final_noise_seeds(), (local[:midpoint], local[midpoint:]))):
            if not int(use.numel()):
                continue
            generator = torch.Generator(device=device).manual_seed(int(noise) + block)
            with torch.no_grad():
                carrier, _x0, target, attrs, _time = carrier_cache._collect_shared_main_carrier(config=models['config'], contract={'frame_dt': 1.0 / 30.0}, d=models['d'], hamiltonian=models['h'], residual=models['r'], gate=models['gate'], train=train, stream=fixed._FixedStream(use), state_scale=models['state_scale'], attr_scale=models['attr_scale'], num_steps=20, batch_size=int(use.numel()), source_rng=generator, device=device)
            cell_seed = seeds.derive_seed('etrg_same_run_cell_subsample', 2 * block + noise_slot)
            record = _decode_parent_carrier(encoder, carrier, target, attrs, use, cell_seed=cell_seed)
            records.append(record)
            collection.append({'block': block, 'noise_slot': noise_slot, 'noise_seed': int(noise) + block, 'sources': int(use.numel()), 'sampled_cells': len(record['source'])})
        block += 1
        print(json.dumps({'HIERARCHICAL_TREE_parent_bootstrap': {'completed_sources': min(start + TREE_BLOCK, TREE_SOURCES), 'total_sources': TREE_SOURCES}}), flush=True)
    data = {'x': np.concatenate([row['x'] for row in records]), 'names': records[0]['names'].tolist(), 'source': np.concatenate([row['source'] for row in records]), 'target': np.concatenate([row['target'] for row in records]), 'h_candidate': np.concatenate([row['h_candidate'] for row in records]), 'd_candidate': np.concatenate([row['d_candidate'] for row in records])}
    collection.append({'wall_seconds': time.perf_counter() - started})
    return (data, collection)

def final_noise_seeds() -> tuple[int, int]:
    return (1942634267, 2035767743)

def _fit_honest_tree(data: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    source = np.asarray(data['source'], np.int64)
    fold = clean._fold(source)
    fit, select, head, safety = (fold == 0, fold == 1, fold == 2, fold == 3)
    initial_tree, _leaf, feature_names = clean._tree(data, fit | select)
    lookup = {name: index for index, name in enumerate(data['names'])}
    raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in feature_names]]
    fit_mean = raw[fit].mean(0)
    fit_scale = np.maximum(raw[fit].std(0), 1e-05)
    fit_design = np.concatenate((np.clip((raw - fit_mean) / fit_scale, -8.0, 8.0), np.ones((len(raw), 1), np.float32)), axis=-1).astype(np.float64)
    responsibility, _ = refiner_training._responsibility(data, fit)
    source_weight = refiner_training._source_equal_weight(source, fit | select)
    responsibility, _, _ = refiner_training._weighted_standardize(responsibility, source_weight, fit)
    tree = copy.deepcopy(initial_tree)
    while refiner_evaluation._renumber(tree) < refiner_evaluation.MAX_LEAVES:
        refined, row = refiner_evaluation._best_refinement(tree, data, fit_design, fit, select, feature_names, responsibility, source_weight)
        if refined is None:
            break
        tree = refined
    mean = raw[head].mean(0)
    scale = np.maximum(raw[head].std(0), 1e-05)
    design = np.concatenate((np.clip((raw - mean) / scale, -8.0, 8.0), np.ones((len(raw), 1), np.float32)), axis=-1).astype(np.float64)
    leaf = clean._assign(tree, np.asarray(data['x'], np.float32), list(data['names']))
    residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
    fitted = StageBoundaryHierarchicalController._hierarchical_solve(design, residual, leaf, head)[0]
    deployed, calibration = StageBoundaryHierarchicalController._global_calibrate(fitted, design, residual, leaf, source, safety)
    payload = {'schema': 'hamiformer.hamiballs.model_tree.v1', 'status': 'COMPLETE', 'sources': int(len(np.unique(source))), 'sampled_cells': int(len(source)), 'tree': copy.deepcopy(tree), 'leaf_count': refiner_evaluation._renumber(tree), 'feature_names': feature_names, 'feature_mean': torch.from_numpy(mean), 'feature_scale': torch.from_numpy(scale), 'residual_weight': torch.from_numpy(deployed).float(), 'component_calibration': calibration, 'split_admission': 'q/p pooled nonharm AND cell/source majority on disjoint fold1', 'source_folds': {'fit': 0, 'select': 1, 'head': 2, 'safety': 3}, 'wall_seconds': time.perf_counter() - started}
    torch.save(payload, TREE_OUTPUT)
    return payload

class MatchedHierarchicalResidual(hierarchical_residual_support.RoutedHierarchicalResidual):

    def __init__(self) -> None:
        super().__init__()
        payload = torch.load(TREE_OUTPUT, map_location='cpu', weights_only=False)
        calibration = payload['component_calibration']
        self.alpha_qp = (float(calibration['q']['alpha']), float(calibration['p']['alpha']))

    def _candidate_blend_alpha(self):
        return self.alpha_qp

    def _install_optional_gate_bias(self, candidate):
        del candidate
        return None

    def _install_optional_gate_adjuster(self, candidate):
        device = next(candidate.parameters()).device
        candidate.register_parameter('etrg_leaf_gate_linear', torch.nn.Parameter(torch.zeros(8, 2, DESIGN_DIM, device=device)))
        candidate._declared_extra_trainable_parameters = 8 * 2 * DESIGN_DIM
        alpha = self.alpha_qp

        def adjust(observable: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
            design = torch.cat((observable, torch.ones_like(observable[..., :1])), dim=-1)
            local = candidate.etrg_leaf_gate_linear[leaf.long()]
            return observable.new_tensor(alpha) * torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
        return adjust

    def _initialize(self, candidate, collector_module) -> None:
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        super()._initialize(candidate, collector_module)
        candidate._declared_trainable_parameters_override = 8 * 2 * DESIGN_DIM
        alpha_qp = self.alpha_qp

        def blended_parent_candidates(module, *, edge, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, previous_g, residual_hidden, base_gate, **_unused):
            delta, _leaf = conditional_delta(self.encoder, module._integrated_ridge_weight, edge=int(edge), previous_mixed=previous_mixed, h_candidate=h_candidate, base_hr_candidate=hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_g, residual_hidden=residual_hidden)
            strong = h_candidate + delta
            alpha = strong.new_tensor(alpha_qp).repeat_interleave(2)
            return (hr_candidate + alpha * (strong - hr_candidate), base_gate)
        collector_module.refine_candidates = types.MethodType(blended_parent_candidates, collector_module)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload.update({'schema': 'hamiformer.hamiballs.hierarchical_ridge.v1', 'component_alpha': list(self.alpha_qp), 'parent_gate_frozen': True, 'trainable_gate_parameters': 8 * 2 * DESIGN_DIM})
        torch.save(payload, RIDGE_OUTPUT)
