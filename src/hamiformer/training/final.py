from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
import json
import time
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import gate_training
from hamiformer.training.hamiballs1 import tree_heads
from hamiformer.training.hamiballs1 import training_partitions as disjoint_training
from hamiformer.training.hamiballs1 import tree_training as hierarchical_tree
from hamiformer.training import router as component_router
from hamiformer.training.hamiballs1 import refresh as gate_stages
from hamiformer.training.hamiballs1.tree_encoder import CausalObservableTreeEncoder
OUTPUT = ROOT / 'outputs/hami1/hami1_final'
TREE_OUTPUT = OUTPUT / 'scalar1_parent_ordinary_tree.pt'
PARENT_ROWS = OUTPUT / 'scalar1_parent_rows.npz'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
MODEL_OUTPUT = OUTPUT / 'routed_expert.pt'
SCALAR1_GATE = hierarchical_tree.COMPONENT_ROUTER_ROOT / 'scalar1/global_sequence_projection_regret_with_source_object_qp_no_harm_gate.pt'
GATE_REGISTRATION = ROOT / 'configs/hamiballs1/gate.json'
R_REGISTRATION = ROOT / 'configs/hamiballs1/residual.json'

def _scalar1_models(*, parent_root: Path, short_configuration: Path, device):
    component_router.configure_runner()
    canonical = gate_stages.canonical
    canonical.PREPARED = parent_root / 'prepared'
    canonical._install(r_registration=R_REGISTRATION, gate_registration=GATE_REGISTRATION)
    registration = gate_training.carrier_cache._load_registration(GATE_REGISTRATION)
    parent, _contract, d, h, residual, _digests = gate_training.gate_training_gate._load_gate_training_models(registration, device=device)
    models = {'config': parent['config'], 'state_scale': parent['state_scale'].to(device), 'attr_scale': parent['attr_scale'].to(device), 'd': d, 'h': h, 'r': residual}
    gate = gate_training.gate_tools._build_gate(models['config'], registration, device=device)
    payload = torch.load(SCALAR1_GATE, map_location='cpu', weights_only=False)
    gate.load_state_dict(payload['gate_state_dict'], strict=True)
    models['gate'] = gate.to(device).eval()
    return models

class ScalarParentTreeEncoder(CausalObservableTreeEncoder):

    def _features(self, **kwargs):
        previous_g = kwargs['previous_g']
        attrs = kwargs['attrs']
        if previous_g.ndim == 2:
            previous_g = previous_g[:, :, None, None].expand(previous_g.shape[0], previous_g.shape[1], attrs.shape[1], 2)
        elif previous_g.ndim == 3:
            previous_g = previous_g[..., None].expand(*previous_g.shape, 2)
        kwargs['previous_g'] = previous_g
        return super()._features(**kwargs)

def _collect_scalar1_parent_records():
    return hierarchical_tree._collect_parent_records(load_models=_scalar1_models, encoder_type=ScalarParentTreeEncoder)

def _save_rows(data: dict[str, object]) -> None:
    np.savez_compressed(PARENT_ROWS, x=np.asarray(data['x']), names=np.asarray(data['names']), source=np.asarray(data['source']), target=np.asarray(data['target']), h_candidate=np.asarray(data['h_candidate']), d_candidate=np.asarray(data['d_candidate']))

def _fit_tree(data: dict[str, object]) -> dict[str, object]:
    previous_output = hierarchical_tree.TREE_OUTPUT
    hierarchical_tree.TREE_OUTPUT = TREE_OUTPUT
    try:
        payload = hierarchical_tree._fit_honest_tree(data)
    finally:
        hierarchical_tree.TREE_OUTPUT = previous_output
    payload['schema'] = 'hamiformer.hamiballs.model_tree.v1'
    payload['tree_fixed_during_final_qp'] = True
    torch.save(payload, TREE_OUTPUT)
    return payload

class ConsolidatedHierarchicalFinal(tree_heads.MatchedObliqueHierarchicalResidual):

    def _initialize(self, candidate, collector_module) -> None:
        shared_qp_trainable = {name for name, parameter in candidate.named_parameters() if parameter.requires_grad}
        super()._initialize(candidate, collector_module)
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(name in shared_qp_trainable or name == 'etrg_leaf_gate_linear')
        candidate._declared_trainable_parameters_override = sum((parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad))

def _run_final() -> None:
    tree_heads.TREE_OUTPUT = TREE_OUTPUT
    tree_heads.RIDGE_OUTPUT = RIDGE_OUTPUT
    hierarchical_tree.OUTPUT = OUTPUT
    hierarchical_tree.TREE_OUTPUT = TREE_OUTPUT
    hierarchical_tree.FINAL_OUTPUT = FINAL_OUTPUT
    hierarchical_tree.RIDGE_OUTPUT = RIDGE_OUTPUT
    hierarchical_tree.scalar_tree_support.TREE_OUTPUT = TREE_OUTPUT
    hierarchical_tree.scalar_tree_support.RIDGE_OUTPUT = RIDGE_OUTPUT
    hierarchical_tree.leaf_gate_support.RIDGE_OUTPUT = RIDGE_OUTPUT
    hierarchical_tree.hierarchical_residual_support.RIDGE_OUTPUT = RIDGE_OUTPUT
    final = hierarchical_tree.final
    final.COMPONENT_ROUTER_ROOT = hierarchical_tree.COMPONENT_ROUTER_ROOT
    final.OUTPUT = FINAL_OUTPUT
    final.QP_OUTPUT = FINAL_OUTPUT / 'qp'
    final.RIDGE_OUTPUT = RIDGE_OUTPUT
    final.FINAL_GATE_PARENT_OVERRIDE = SCALAR1_GATE
    final.FINAL_GATE_RESUME = False
    final.FINAL_FIT_SAMPLES = 1600
    final.FINAL_HOLDOUT_SAMPLES = 64
    final.FINAL_COLLECTION_BATCH = 64
    final.FINAL_REFRESH_SOURCES = 64
    final.QP_MAXIMUM_LR = 0.003
    final.QP_MINIMUM_LR = 0.0001
    gate_training.DISJOINT_PAIRED_NOISE_SOURCES = True
    gate_training.BATCH_PAIRED_NOISE_COLLECTION = True
    gate_training.NONFATAL_ONLINE_REPLAY_AUDIT = True
    gate_training.NONFATAL_NONFINITE_OPTIMIZER_STEP = True
    gate_training.NO_HARM_FINAL_QP_NO_HARM_WEIGHT = disjoint_training.NO_HARM_WEIGHT
    torch.backends.cudnn.enabled = False
    try:
        final.main(controller_factory=ConsolidatedHierarchicalFinal)
    finally:
        final.FINAL_GATE_PARENT_OVERRIDE = None
        final.FINAL_GATE_RESUME = False
        gate_training.DISJOINT_PAIRED_NOISE_SOURCES = False
        gate_training.BATCH_PAIRED_NOISE_COLLECTION = False
        gate_training.NO_HARM_FINAL_QP_NO_HARM_WEIGHT = 0.0

def _seal(tree_payload: dict[str, object], collection: list[dict[str, object]]):
    gate_path = next((FINAL_OUTPUT / 'qp').glob('*.pt'))
    gate = torch.load(gate_path, map_location='cpu', weights_only=False)
    ridge = torch.load(RIDGE_OUTPUT, map_location='cpu', weights_only=False)
    calibration = tree_payload['component_calibration']
    alpha = (float(calibration['q']['alpha']), float(calibration['p']['alpha']))
    bundle = {'schema': 'hamiformer.hamiballs.routed_expert.v1', 'status': 'COMPLETE', 'gate_state_dict': gate['gate_state_dict'], 'tree': copy.deepcopy(ridge['tree']), 'feature_names': list(ridge['feature_names']), 'feature_mean': ridge['feature_mean'], 'feature_scale': ridge['feature_scale'], 'ridge_weight': ridge['ridge_weight'], 'component_alpha': alpha, 'training': {'updates': 800, 'common_r': 200, 'scalar0': 200, 'scalar1': 200, 'final_qp': 200, 'tree_fixed_during_final_qp': True, 'refresh_every': 8, 'paired_noise_seeds': list(hierarchical_tree.final_noise_seeds())}}
    torch.save(bundle, MODEL_OUTPUT)
    _export_runtime(bundle)
    return {'schema': 'hamiformer.hamiballs.final_training.v1', 'status': 'COMPLETE', 'tree': str(TREE_OUTPUT), 'gate': str(gate_path), 'ridge': str(RIDGE_OUTPUT), 'model': str(MODEL_OUTPUT), 'component_alpha': list(alpha), 'tree_leaf_count': int(tree_payload['leaf_count']), 'tree_collection': collection, 'updates': 800}

def _export_runtime(bundle):
    from hamiformer.models.hamiballs_gate_dual_channel import HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate
    from hamiformer.training.hamiballs1.parent_models import _build_wide_d
    models = _scalar1_models(parent_root=hierarchical_tree.COMPONENT_ROUTER_ROOT, short_configuration=None, device=torch.device('cpu'))
    scalar = models.pop('gate')
    gate = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate.from_scalar(scalar, delta_width=component_router.QP_WIDTH)
    state = bundle['gate_state_dict']
    gate.register_parameter('etrg_leaf_gate_linear', torch.nn.Parameter(torch.zeros_like(state['etrg_leaf_gate_linear'])))
    gate.load_state_dict(state, strict=True)
    registration = json.loads(GATE_REGISTRATION.read_text())
    wide = torch.load(registration['parents']['wide_checkpoint'], map_location='cpu', weights_only=False)
    models['wide_d'] = _build_wide_d(wide, device=torch.device('cpu'))
    models['base_gate'] = gate.eval().requires_grad_(False)
    models['gate_payload'] = {key: bundle[key] for key in ('tree', 'feature_names', 'feature_mean', 'feature_scale', 'ridge_weight', 'component_alpha')}
    models['config'] = copy.deepcopy(models['config'])
    models['config']['dataset']['root'] = 'data/hamiballs_canonical_v2/train_adapter48'
    for key in ('d', 'h', 'r', 'wide_d'):
        models[key].cpu().eval().requires_grad_(False)
    torch.save(models, OUTPUT / 'runtime.pt')

def main() -> None:
    global OUTPUT, TREE_OUTPUT, PARENT_ROWS, FINAL_OUTPUT, RIDGE_OUTPUT, MODEL_OUTPUT
    global SCALAR1_GATE, GATE_REGISTRATION, R_REGISTRATION
    parser = argparse.ArgumentParser(description='Fit the HamiBalls-1 model tree and final routed residual')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--scalar-gate', type=Path, required=True)
    parser.add_argument('--r-config', type=Path, required=True)
    parser.add_argument('--gate-config', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--row-ledger', type=Path, required=True)
    parser.add_argument('--source-exclusions', type=Path)
    args = parser.parse_args()
    for path in (args.scalar_gate, args.r_config, args.gate_config, args.row_ledger):
        if not path.is_file():
            parser.error(f'required training input not found: {path}')
    OUTPUT = args.output.resolve()
    TREE_OUTPUT = OUTPUT / 'tree.pt'
    PARENT_ROWS = OUTPUT / 'tree_training_rows.npz'
    FINAL_OUTPUT = OUTPUT / 'final'
    RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
    MODEL_OUTPUT = OUTPUT / 'routed_expert.pt'
    SCALAR1_GATE = args.scalar_gate.resolve()
    GATE_REGISTRATION = args.gate_config.resolve()
    R_REGISTRATION = args.r_config.resolve()
    hierarchical_tree.COMPONENT_ROUTER_ROOT = args.parent_root.resolve()
    hierarchical_tree.collector.ROW_LEDGER = args.row_ledger.resolve()
    hierarchical_tree.clean.LEDGER = args.row_ledger.resolve()
    exclusions = args.source_exclusions.resolve() if args.source_exclusions is not None else None
    if exclusions is not None and (not exclusions.is_file()):
        parser.error(f'source exclusions not found: {exclusions}')
    hierarchical_tree.collector.EXCLUSION = exclusions
    gate_stages.EXCLUSION_CONFIGURATION = exclusions
    gate_stages.canonical.DATASET_ROOT = args.dataset_root.resolve()
    hierarchical_tree.final.R_REGISTRATION_OVERRIDE = args.r_config.resolve()
    hierarchical_tree.final.GATE_REGISTRATION_OVERRIDE = GATE_REGISTRATION
    if (OUTPUT / 'COMPLETE').is_file():
        raise FileExistsError(OUTPUT)
    if not SCALAR1_GATE.is_file():
        raise FileNotFoundError('the scalar1 parent gate is required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if PARENT_ROWS.is_file():
        with np.load(PARENT_ROWS, allow_pickle=False) as cached:
            data = {name: cached[name] for name in cached.files}
        data['names'] = [str(value) for value in data['names'].tolist()]
        collection = [{'reused_exact_persisted_parent_rows': True}]
    else:
        data, collection = _collect_scalar1_parent_records()
        _save_rows(data)
    tree_payload = torch.load(TREE_OUTPUT, map_location='cpu', weights_only=False) if TREE_OUTPUT.is_file() else _fit_tree(data)
    del data
    print(json.dumps({'model_tree': {'leaf_count': tree_payload['leaf_count'], 'calibration': tree_payload['component_calibration']}}, default=hierarchical_tree._json_default), flush=True)
    _run_final()
    report = _seal(tree_payload, collection)
    report['wall_seconds'] = time.perf_counter() - started
    (OUTPUT / 'experiment_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUTPUT / 'COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True), flush=True)
if __name__ == '__main__':
    main()
