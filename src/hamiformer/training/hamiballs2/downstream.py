from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np
from hamiformer.data_generation3d.signature import dataset_signature
import torch
import yaml
ROOT = project_root()
from hamiformer.data.hamiballs2 import HamiBalls2CropPackDataset
from hamiformer.models.hamiballs2_dual_expert import hamiballs2_node_graph_features
from hamiformer.models.hamiballs2_hamiltonian import HamiBalls2ContinuousHamiltonian
from hamiformer.models.hamiballs2_posthd_formal import HamiBalls2FormalPostHD, parameter_count
from hamiformer.baselines.physiformer import HamiBalls2WideD
from hamiformer.training.hamiballs2_posthd_protocol import DATASET_SIGNATURE, common_r_refresh_plan, final_refresh_plan, formal_contract, namespaced_seed, scalar_pool_plan, sha256_file, validate_formal_contract

def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value

def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_torch(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)

def _static_statistics(dataset: HamiBalls2CropPackDataset, chunk: int=512) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(6, dtype=np.float64)
    square = np.zeros(6, dtype=np.float64)
    count = 0
    for start in range(0, len(dataset), chunk):
        stop = min(start + chunk, len(dataset))
        attrs = torch.from_numpy(np.array(dataset.arrays['attrs'][start:stop], copy=True)).float()
        mask = torch.from_numpy(np.array(dataset.arrays['object_mask'][start:stop], copy=True)).bool()
        node = hamiballs2_node_graph_features(mask, torch.from_numpy(np.array(dataset.arrays['spring_mask'][start:stop], copy=True)), torch.from_numpy(np.array(dataset.arrays['spring_k'][start:stop], copy=True)).float(), torch.from_numpy(np.array(dataset.arrays['spring_rest_length'][start:stop], copy=True)).float())
        values = torch.cat((attrs, node), -1)[mask].double().numpy()
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += len(values)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return (mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count)

def _load_models(cfg: dict[str, Any], device: torch.device):
    wide_cfg = yaml.safe_load(_resolve(cfg['wide_d']['config']).read_text(encoding='utf-8'))
    wide = HamiBalls2WideD(**wide_cfg['model']).to(device)
    wide_payload = torch.load(Path(cfg['wide_d']['checkpoint']), map_location='cpu', weights_only=False)
    wide.load_state_dict(wide_payload[cfg['wide_d']['state_key']], strict=True)
    wide.eval().requires_grad_(False)
    h_cfg = yaml.safe_load(_resolve(cfg['hamiltonian']['config']).read_text(encoding='utf-8'))
    phase_scale = cfg['model']['phase_scale']
    h = HamiBalls2ContinuousHamiltonian(**h_cfg['model'], q_scale=tuple(phase_scale[:3]), p_scale=tuple(phase_scale[3:])).to(device)
    h_payload = torch.load(Path(cfg['hamiltonian']['checkpoint']), map_location='cpu', weights_only=False)
    h.load_state_dict(h_payload[cfg['hamiltonian']['state_key']], strict=True)
    h.eval().requires_grad_(False)
    return (wide, h, wide_cfg, h_cfg)

def _validate_config(cfg: dict[str, Any]) -> dict[str, object]:
    if dataset_signature(cfg) != DATASET_SIGNATURE:
        raise ValueError('training and generator configurations disagree')
    p = cfg['protocol']
    if int(p.get('master_seed', -1)) != 42:
        raise ValueError('formal post-HD random tapes must derive from master seed 42')
    contract = formal_contract(int(p['master_seed']))
    validate_formal_contract(contract)
    observed_updates = {'common_r': int(p['common_r_updates']), 'scalar0': int(p['scalar0_updates']), 'scalar1': int(p['scalar1_updates']), 'final_qp': int(p['final_qp_updates'])}
    if observed_updates != contract['stage_updates']:
        raise ValueError(f'stage schedule differs from formal contract: {observed_updates}')
    exact = {'tree_fit_count': 1, 'tree_fixed_during_final_qp': True, 'cumulative_ridge_across_final_refreshes': True, 'final_refreshes': 25, 'updates_per_final_refresh': 8}
    drift = {name: (p.get(name), expected) for name, expected in exact.items() if p.get(name) != expected}
    if drift:
        raise ValueError(f'scientific protocol drift: {drift}')
    if list(p['q_p_training_scales']) != [1.0, 1.0]:
        raise ValueError('formal q/p component admission must begin at [1,1]')
    declared = {'source_block_size': 64, 'scalar_pool_blocks': 25, 'final_noise_halves': 2, 'common_r_window_edges': 12, 'scalar1_no_harm_weight': 0.25, 'final_qp_no_harm_weight': 0.25, 'ridge_relative': 0.01, 'ridge_intercept_multiplier': 0.01}
    drift = {name: (p.get(name), expected) for name, expected in declared.items() if p.get(name) != expected}
    scalar_counts = {int(key): int(value) for key, value in p['scalar_mixed_rf_counts_per_B64'].items()}
    final_counts = {int(key): int(value) for key, value in p['final_mixed_rf_counts_per_noise_B32'].items()}
    if scalar_counts != {8: 32, 12: 20, 20: 12}:
        drift['scalar_mixed_rf_counts_per_B64'] = (scalar_counts, {8: 32, 12: 20, 20: 12})
    if final_counts != {8: 16, 12: 10, 20: 6}:
        drift['final_mixed_rf_counts_per_noise_B32'] = (final_counts, {8: 16, 12: 10, 20: 6})
    if drift:
        raise ValueError(f'scientific protocol drift: {drift}')
    operations = cfg['operations']
    if operations.get('nonfinite_update_policy') != 'warn_skip_continue':
        raise ValueError('unattended nonfinite updates must warn, skip and continue')
    if int(cfg['hamiltonian'].get('plas_substeps_per_frame', -1)) != 1:
        raise ValueError('formal HamiBalls-2 H/PLAS interface uses one frame step with local numerical-health fallback')
    if cfg['hamiltonian'].get('plas_unhealthy_edge_policy') != 'd_reset':
        raise ValueError('unhealthy PLAS scene-edges must use the registered local D reset')
    if cfg['hamiltonian'].get('carrier_tangent_diagnostic') != 'exact_threshold_certified_gram':
        raise ValueError('carrier tangent health must use certified bounds plus an exact Gram eigensolve on ambiguous scene-edges')
    return contract

def prepare(cfg: dict[str, Any], config_path: Path, *, runtime_smoke: bool=False) -> dict[str, object]:
    contract = _validate_config(cfg)
    dataset_root = Path(cfg['dataset_root'])
    metadata = json.loads((dataset_root / 'metadata/dataset_config.json').read_text(encoding='utf-8'))
    audit = json.loads((dataset_root / 'audit/full_audit.json').read_text(encoding='utf-8'))
    if metadata.get('semantic_hash') != DATASET_SIGNATURE or audit.get('full_validation') is not True:
        raise RuntimeError('dataset provenance/full audit does not satisfy the formal contract')
    for section in ('wide_d', 'hamiltonian'):
        for key in ('config', 'checkpoint'):
            path = _resolve(cfg[section][key]) if key == 'config' else Path(cfg[section][key])
            if not path.is_file():
                raise FileNotFoundError(path)
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wide, h, wide_cfg, h_cfg = _load_models(cfg, device)
    initialization_seed = namespaced_seed(int(cfg['protocol']['master_seed']), 'model_initialization')
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        model = HamiBalls2FormalPostHD(**cfg['model']).to(device)
    train = HamiBalls2CropPackDataset(cfg['train_pack'])
    static_mean, static_scale, static_rows = _static_statistics(train)
    model.static_feature_mean.copy_(torch.from_numpy(static_mean).to(device))
    model.static_feature_scale.copy_(torch.from_numpy(static_scale).to(device))
    post_neural = parameter_count(model)
    ridge_coefficients = int(model.ridge_weight.numel())
    h_parameters = sum((value.numel() for value in h.parameters()))
    d_parameters = sum((value.numel() for value in wide.parameters()))
    if d_parameters != int(cfg['wide_d']['parameters']):
        raise RuntimeError(f'wide-D parameter contract drift: {d_parameters}')
    if h_parameters != int(cfg['hamiltonian']['parameters']):
        raise RuntimeError(f'H parameter contract drift: {h_parameters}')
    ratio = (h_parameters + post_neural + ridge_coefficients) / d_parameters
    if ratio > 0.0166:
        raise RuntimeError(f'H+post-HD budget {100 * ratio:.4f}% exceeds declared Hami-1 ratio')
    stage_trainable = {stage: model.set_training_stage(stage) for stage in ('common_r', 'scalar0', 'scalar1', 'final_qp', 'frozen')}
    model.set_training_stage('frozen')
    smoke = copy.deepcopy(model)
    one_leaf = {'feature': [-2], 'threshold': [-2.0], 'children_left': [-1], 'children_right': [-1], 'node_to_leaf': [0], 'depth': 0}
    smoke.install_fitted_state(tree=one_leaf, feature_mean=torch.zeros(smoke.observable_dim, device=device), feature_scale=torch.ones(smoke.observable_dim, device=device), ridge_weight=torch.zeros_like(smoke.ridge_weight), component_alpha=torch.zeros(2, device=device), static_mean=torch.from_numpy(static_mean).to(device), static_scale=torch.from_numpy(static_scale).to(device))
    with torch.no_grad():
        shape = (2, 3, 4, 6)
        synthetic = torch.zeros(shape, device=device)
        smoke_result = smoke(synthetic, synthetic, synthetic, torch.zeros(2, 3, 4, cfg['model']['d_token_dim'], device=device), x0=torch.zeros(2, 4, 6, device=device), attrs=torch.ones(2, 4, 3, device=device), node_graph=torch.zeros(2, 4, 3, device=device), physical_time=torch.arange(1, 4, device=device).float()[None].expand(2, -1) / 30, tau=torch.full((2,), 0.5, device=device), object_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], device=device, dtype=torch.bool))
    if not all((bool(torch.isfinite(value).all()) for value in smoke_result.values() if value.is_floating_point())):
        raise FloatingPointError('formal deployment smoke produced a non-finite tensor')
    real_smoke: dict[str, object] | None = None
    if runtime_smoke:
        from hamiformer.training.hamiballs2.carriers import CarrierCollector, _common_window, _forward, _local_common_candidates, _next_same_source, _subset, _validate_common_hull_pair
        collector = CarrierCollector(cfg=cfg, dataset=train, wide=wide, hamiltonian=h, model=model, device=device)
        pool = collector.collect([20], source_seed=42, noise_seed=43, route_seed=44, parent_mode='random')
        indices = np.arange(len(pool['state']), dtype=np.int64)
        hull_indices = _next_same_source(pool)[indices]
        batch = _common_window(_subset(pool, indices, device), update=1, length=int(cfg['protocol']['common_r_window_edges']))
        hull_batch = _common_window(_subset(pool, hull_indices, device), update=1, length=int(cfg['protocol']['common_r_window_edges']))
        _validate_common_hull_pair(batch, hull_batch, indices, hull_indices)
        model.set_training_stage('common_r')
        carrier_output = _forward(model, batch, force_random=True)
        hull_output = _forward(model, hull_batch, force_random=True)
        local_hr = _local_common_candidates(model, batch)
        from hamiformer.training.hamiballs2_posthd_formal import common_residual_objective
        loss = common_residual_objective(local_h=batch['local_h'] / model.phase_scale, local_hr=local_hr / model.phase_scale, recovery_h=carrier_output['h_candidate'] / model.phase_scale, recovery_hr=carrier_output['hr_candidate'] / model.phase_scale, hull_hr=hull_output['hr_candidate'] / model.phase_scale, hull_d=hull_batch['d_candidate'] / model.phase_scale, target=batch['target'] / model.phase_scale, object_mask=batch['object_mask'], qp_scales=torch.tensor(cfg['protocol']['q_p_training_scales'], device=device)).total
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not bool(torch.isfinite(loss)) or not gradients or (not all((gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients))):
            raise FloatingPointError('real carrier forward/backward smoke failed')
        real_smoke = {'rf_num_steps': 20, 'source_count': 1, 'accepted_source_fields': int(len(pool['state'])), 'common_r_loss': float(loss.detach().cpu()), 'maximum_tangent_diagnostic': None, 'tangent_health_gate_exact': True, 'tangent_exact_eigensolve_edges': int(pool['tangent_exact_eigensolve_edges'].sum()), 'plas_d_reset_scene_edges': int(pool['plas_d_reset_edges'].sum()), 'optimizer_created': False, 'optimizer_step_called': False}
        model.zero_grad(set_to_none=True)
        model.set_training_stage('frozen')
    config_hash = sha256_file(config_path)
    provenance = {'formal_config_sha256': config_hash, 'wide_config_sha256': sha256_file(_resolve(cfg['wide_d']['config'])), 'wide_checkpoint_sha256': sha256_file(cfg['wide_d']['checkpoint']), 'h_config_sha256': sha256_file(_resolve(cfg['hamiltonian']['config'])), 'h_checkpoint_sha256': sha256_file(cfg['hamiltonian']['checkpoint']), 'formal_source_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in (ROOT / 'src/hamiformer/models/hamiballs2_posthd_formal.py', ROOT / 'src/hamiformer/baselines/physiformer.py', ROOT / 'src/hamiformer/physics/generic_type2.py', ROOT / 'src/hamiformer/physics/generic_gfjp.py', ROOT / 'src/hamiformer/training/hamiballs2_posthd_formal.py', ROOT / 'src/hamiformer/training/hamiballs2_posthd_fitting.py', ROOT / 'src/hamiformer/training/hamiballs2_posthd_protocol.py', ROOT / 'src/hamiformer/training/hamiballs2/downstream.py', ROOT / 'src/hamiformer/training/hamiballs2/carriers.py')}}
    contact_arguments = [name for name in inspect.signature(HamiBalls2FormalPostHD.forward).parameters if 'contact' in name.lower() or 'event' in name.lower()]
    if contact_arguments:
        raise RuntimeError('formal runtime unexpectedly exposes contact/event inputs')
    manifest: dict[str, object] = {
        'schema': 'hamiformer.hamiballs2.formal_posthd_preparation.v1',
        'status': 'PREPARED_NOT_STARTED',
        'optimizer_updates': 0,
        'full_carrier_fields_generated': 0,
        'config': str(config_path),
        'contract': contract,
        'provenance': provenance,
        'parameters': {
            'wide_d': d_parameters,
            'hamiltonian': h_parameters,
            'post_hd_neural': post_neural,
            'ridge_coefficients': ridge_coefficients,
            'h_plus_post_over_d_fraction': ratio,
            'stage_trainable': stage_trainable,
        },
        'model_initialization': {
            'namespace': 'model_initialization',
            'seed': initialization_seed,
            'derived_from_master_seed': int(cfg['protocol']['master_seed']),
        },
        'static_statistics': {
            'mean': static_mean.tolist(),
            'scale': static_scale.tolist(),
            'valid_objects': static_rows,
            'source': 'training split',
        },
        'common_r_refresh_plan': common_r_refresh_plan(master_seed=int(cfg['protocol']['master_seed'])),
        'scalar0_pool_plan': scalar_pool_plan(master_seed=int(cfg['protocol']['master_seed']), stage='scalar0'),
        'scalar1_pool_plan': scalar_pool_plan(master_seed=int(cfg['protocol']['master_seed']), stage='scalar1'),
        'final_refresh_plan': final_refresh_plan(master_seed=int(cfg['protocol']['master_seed'])),
        'smoke': {name: list(value.shape) for name, value in smoke_result.items()},
        'real_carrier_forward_backward_smoke': real_smoke,
    }
    _atomic_torch(output / 'formal_posthd_initialization.pt', {'schema': manifest['schema'], 'status': manifest['status'], 'model_config': cfg['model'], 'model_state_dict': model.state_dict(), 'tree': None, 'ridge_fitted': False, 'optimizer_updates': 0, 'model_initialization_seed': initialization_seed, 'contract': contract, 'provenance': provenance})
    _atomic_json(output / 'preparation_manifest.json', manifest)
    (output / 'PREPARED_NOT_STARTED').write_text('optimizer_updates=0\n', encoding='utf-8')
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--execute-training', action='store_true')
    parser.add_argument('--runtime-smoke', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.execute_training:
        from hamiformer.training.hamiballs2.carriers import execute_training
        execute_training(cfg, args.config)
        return
    manifest = prepare(cfg, args.config, runtime_smoke=args.runtime_smoke)
    print(json.dumps({'status': manifest['status'], 'optimizer_updates': 0, 'parameters': manifest['parameters'], 'output': cfg['output_dir']}, sort_keys=True), flush=True)
if __name__ == '__main__':
    main()
