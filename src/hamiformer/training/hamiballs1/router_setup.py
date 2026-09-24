from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import torch
ROOT = project_root()
SOURCE_EXCLUSIONS = None
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.data import packing as support
from hamiformer.training import base as stage_a
from hamiformer.training.hamiballs1 import router_preparation as prepare
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training.hamiballs1 import carrier_cache as carrier_cache
DATASET_ROOT = Path('./data/hamiballs_canonical_v2/train_adapter48')
ARTIFACT_ROOT = Path('./artifacts/hamiballs_canonical_v2_gate_seed42')
R_ARTIFACT_NAME = 'hamiballs_canonical_v2_r0_seed42'
GATE_PURPOSE = 'hami1_observable_gate_v1'
PREPARED = ARTIFACT_ROOT / 'prepared'
ROUND_OUTPUTS = (ARTIFACT_ROOT / 'scalar_r0', ARTIFACT_ROOT / 'scalar_r1')
FIT_SAMPLES = 2048
HOLDOUT_SAMPLES = 64
SELF_POLICY_REFRESH_EVERY = 8
SELF_POLICY_REFRESH_SOURCES = 64
SELF_POLICY_REFRESH_FIT_SAMPLES = 1600
SOURCE_OFFSETS = {'scalar0': 64, 'scalar1': 2176, 'risk': 4288}

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding='utf-8'))

def _validate_gate_registration(path: Path, *, r_registration_path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = _read_json(path)
    if raw.get('schema') != carrier_cache.SCHEMA or raw.get('purpose') != GATE_PURPOSE or (raw.get('training', {}).get('sampling_seed') != support.namespaced_seed('g_model_registration')):
        raise ValueError('canonical gate registration identity changed')
    r_registration = support.load_r_registration(r_registration_path)
    terminal = raw.get('r_terminal', {})
    if raw.get('parents') != r_registration.get('parents') or terminal.get('path') != str(ARTIFACT_ROOT.parent / R_ARTIFACT_NAME / 'r_terminal.pt') or terminal.get('training_registration_sha256') != sha256_file(r_registration_path.expanduser().resolve()) or (terminal.get('expected_updates') != {'total': 200, 'n8': 50, 'n12': 50, 'n20': 100}) or (terminal.get('pipeline_component_updates') != 200) or (terminal.get('pipeline_critical_path_rounds') != 200):
        raise ValueError('canonical gate R0 parent contract changed')
    terminal_path = Path(terminal['path'])
    if not terminal_path.is_file() or sha256_file(terminal_path) != terminal['sha256']:
        raise ValueError('Residual state is missing or mismatched')
    if raw['gate'].get('architecture') != 'per_object_field_gru_compact' or raw['gate'].get('rank') != 12:
        raise ValueError('Unsupported gate configuration')
    return raw

def _observable_sidecar(*, core, parent: dict[str, Any], statistics_path: Path, expected_statistics_sha256: str, device: torch.device):
    if sha256_file(statistics_path) != expected_statistics_sha256:
        raise ValueError('canonical observable statistics changed')
    payload = torch.load(statistics_path, map_location='cpu', weights_only=False)
    statistics_state = payload.get('r_state_dict')
    if payload.get('schema') != prepare.STATISTICS_SCHEMA or not isinstance(statistics_state, dict) or payload.get('r_core_digest') != module_digest(core) or (payload.get('training_partition_only') is not True):
        raise ValueError('canonical observable statistics contract changed')
    sidecar = prepare._sidecar_from_core(core, parent, device=device)
    state = sidecar.state_dict()
    for name in state:
        if not name.startswith('network.'):
            state[name] = statistics_state[name].detach().to(state[name]).clone()
    sidecar.load_state_dict(state, strict=True)
    sidecar.per_object_previous_g = bool(core.per_object_previous_g)
    set_trainable(sidecar, False)
    sidecar.eval()
    if module_digest(sidecar) != payload.get('sidecar_digest'):
        raise ValueError('canonical observable sidecar digest changed')
    return sidecar

def _install(*, r_registration: Path, gate_registration: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _validate_gate_registration(gate_registration, r_registration_path=r_registration)
    prepared_summary = _read_json(PREPARED / 'summary.json')
    if prepared_summary.get('status') != 'COMPLETE' or not (PREPARED / 'COMPLETE').is_file() or prepared_summary.get('gate_registration_sha256') != sha256_file(gate_registration.expanduser().resolve()):
        raise ValueError('canonical gate preparation is incomplete or mismatched')

    def load_registration(path: Path):
        if path.expanduser().resolve() != gate_registration.expanduser().resolve():
            raise ValueError('canonical gate registration path changed')
        return _validate_gate_registration(path, r_registration_path=r_registration)

    def load_models(registration, *, device: torch.device):
        terminal = registration['r_terminal']
        loaded = support.load_canonical_r(registration_path=r_registration, terminal_path=Path(terminal['path']), expected_terminal_sha256=terminal['sha256'], device=device)
        residual = _observable_sidecar(core=loaded['r'], parent=loaded['parent'], statistics_path=Path(prepared_summary['statistics_path']), expected_statistics_sha256=prepared_summary['statistics_sha256'], device=device)
        frozen = {'d': support.freeze(loaded['d'], name='D'), 'h': 'identity_affine_map' if loaded['h'] is None else support.freeze(loaded['h'], name='H'), 'r': support.freeze(residual, name='r-sidecar')}
        contract = {**loaded['parent_contract'], 'train_contract': stage_a._train_contract(loaded['parent']['config'])}
        return (loaded['parent'], contract, loaded['d'], loaded['h'], residual, frozen)

    def build_gate(config, registration, *, device: torch.device):
        seed = support.namespaced_seed('g_architecture_init')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gate = HamiBallsPerObjectCompactCommittedGate(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim']), residual_hidden_dim=71, rank=int(registration['gate']['rank'])).to(device=device, dtype=torch.float32)
        gate.temporal.flatten_parameters()
        return gate
    packed.carrier_cache._load_registration = load_registration
    packed.gate_training_gate._load_gate_training_models = load_models
    packed.gate_tools._build_gate = build_gate
    return (raw, prepared_summary)
