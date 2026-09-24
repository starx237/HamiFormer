from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import torch
ROOT = project_root()
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate
from hamiformer.models.hamiballs_observable_sidecar_physical_features_observable_sidecar_r import HamiBallsPhysicalFeaturesObservableSidecarResidual
from hamiformer.training.training_schedule import ExplicitEpochBatchStream
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.data import packing as support
from hamiformer.training.hamiballs1 import observable_statistics as feedback_carrier
from hamiformer.training.hamiballs1.router_initialization import initialise_standard_readout
SCHEMA = 'hamiformer.hamiballs.canonical_v2.gate_preparation.v1'
STATISTICS_SCHEMA = 'hamiformer.hamiballs.canonical_v2.observable71_statistics.v1'
METRIC_GATE_SCHEMA = 'hamiformer.hamiballs.canonical_v2.fresh_metric_gate.v1'
FIT_SOURCES = 64

def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _sidecar_from_core(core, parent: dict[str, Any], *, device: torch.device):
    config = parent['config']
    sidecar = HamiBallsPhysicalFeaturesObservableSidecarResidual(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim'])).to(device=device, dtype=torch.float32)
    state = sidecar.state_dict()
    core_state = core.state_dict()
    network_names = {name for name in state if name.startswith('network.')}
    if set(core_state) != network_names:
        raise ValueError('canonical R0 and observable sidecar core keys differ')
    for name, value in core_state.items():
        state[name] = value.detach().to(state[name]).clone()
    sidecar.load_state_dict(state, strict=True)
    sidecar.per_object_previous_g = bool(core.per_object_previous_g)
    sidecar.eval()
    return sidecar

def _build_metric_gate(parent: dict[str, Any], gate_registration: dict[str, Any]):
    config = parent['config']
    seed = support.namespaced_seed('g_architecture_init')
    torch.manual_seed(seed)
    gate = HamiBallsPerObjectCompactCommittedGate(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim']), residual_hidden_dim=71, rank=int(gate_registration['gate']['rank'])).to(device='cpu', dtype=torch.float32)
    neutral_digest = module_digest(gate)
    readout_seed = support.namespaced_seed('g_readout_init')
    state = initialise_standard_readout(gate.state_dict(), seed=readout_seed)
    gate.load_state_dict(state, strict=True)
    return (gate, state, neutral_digest, readout_seed)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r-registration', type=Path, required=True)
    parser.add_argument('--gate-registration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    gate_registration_path = args.gate_registration.expanduser().resolve()
    gate_registration = json.loads(gate_registration_path.read_text(encoding='utf-8'))
    r_terminal = gate_registration['r_terminal']
    loaded = support.load_canonical_r(registration_path=args.r_registration, terminal_path=Path(r_terminal['path']), expected_terminal_sha256=r_terminal['sha256'], device=torch.device(args.device))
    if loaded['r_registration_sha256'] != r_terminal['training_registration_sha256'] or gate_registration['parents'] != loaded['registration']['parents']:
        raise ValueError('gate and canonical R0 registrations are not matched')
    parent = loaded['parent']
    config = parent['config']
    from hamiformer.training.hamiballs1 import router_models as gate_training
    stage_a = gate_training.stage_a
    sidecar = _sidecar_from_core(loaded['r'], parent, device=torch.device(args.device))
    train = stage_a._load_train_cache(config, device=torch.device(args.device))
    state_scale = parent['state_scale'].to(device=args.device, dtype=torch.float32)
    attr_scale = stage_a._attribute_scale(train.attrs).to(device=args.device, dtype=torch.float32)
    stream = ExplicitEpochBatchStream(train.size, FIT_SOURCES, seed=support.namespaced_seed('g_source_permutation'))
    rngs = {name: torch.Generator(device=args.device).manual_seed(support.namespaced_seed('g_stats_carrier', counter)) for counter, name in enumerate(('source', 'reset_tau', 'reset'))}
    first_indices = stream.permutation[:FIT_SOURCES].clone()
    carrier, x0, _target, attrs, physical_time = gate_training._collect_per_object_external_carrier(config=config, contract=stage_a._train_contract(config), d=loaded['d'], hamiltonian=loaded['h'], residual=sidecar, train=train, stream=stream, state_scale=state_scale, attr_scale=attr_scale, num_steps=20, batch_size=FIT_SOURCES, source_rng=rngs['source'], reset_tau_rng=rngs['reset_tau'], reset_rng=rngs['reset'], device=torch.device(args.device))
    fitted_rows = feedback_carrier._fit_observable_statistics(sidecar, carrier, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale)
    set_trainable(sidecar, False)
    sidecar.eval()
    sidecar_digest = module_digest(sidecar)
    output.mkdir(parents=True)
    statistics_path = output / 'observable71_statistics.pt'
    _atomic_torch(statistics_path, {'schema': STATISTICS_SCHEMA, 'role': 'train-only-gate-observable-statistics', 'r_state_dict': sidecar.state_dict(), 'r_core_digest': loaded['r_digest'], 'sidecar_digest': sidecar_digest, 'fitted_rows': fitted_rows, 'fit_source_indices': first_indices, 'source_permutation_seed': support.namespaced_seed('g_source_permutation'), 'source_noise_seed': support.namespaced_seed('g_stats_carrier', 0), 'training_partition_only': True})
    gate, gate_state, neutral_digest, readout_seed = _build_metric_gate(parent, gate_registration)
    metric_gate_path = output / 'metric_gate.pt'
    _atomic_torch(metric_gate_path, {'schema': METRIC_GATE_SCHEMA, 'role': 'canonical-v2-observable71-fresh-standard-readout-scalar', 'gate_state_dict': gate_state, 'trained_updates': 0, 'architecture_seed': support.namespaced_seed('g_architecture_init'), 'readout_seed': readout_seed, 'residual_hidden_dim': 71, 'parameter_count': sum((p.numel() for p in gate.parameters())), 'neutral_gate_digest': neutral_digest, 'metric_gate_digest': module_digest(gate)})
    summary = {'schema': SCHEMA, 'status': 'COMPLETE', 'r_terminal_sha256': loaded['r_terminal_sha256'], 'r_digest': loaded['r_digest'], 'sidecar_digest': sidecar_digest, 'fitted_rows': fitted_rows, 'fit_sources': FIT_SOURCES, 'source_permutation_seed': support.namespaced_seed('g_source_permutation'), 'fit_source_indices_sha256': support.json_digest(first_indices.tolist()), 'statistics_path': str(statistics_path), 'statistics_sha256': sha256_file(statistics_path), 'metric_gate_path': str(metric_gate_path), 'metric_gate_sha256': sha256_file(metric_gate_path), 'gate_registration_sha256': sha256_file(gate_registration_path), 'training_partition_only': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (output / 'COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))
