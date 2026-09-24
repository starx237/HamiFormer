from __future__ import annotations
from hamiformer.utils.paths import project_root
import json
from pathlib import Path
from typing import Any
import torch
from hamiformer.training.hamiballs_recovery import module_digest
from hamiformer.utils import sha256_file
from hamiformer.data import packing as support
from hamiformer.training.hamiballs1 import refresh as base
from hamiformer.training.hamiballs1 import router_setup as canonical
from hamiformer.training.hamiballs1 import frozen_residual as residual_training
from hamiformer.training.hamiballs1 import gate_schedule as gate_schedule
from hamiformer.training.hamiballs1 import carrier_cache as carrier_cache
from hamiformer.training.hamiballs1 import continuous_carrier as adapter
from hamiformer.training import residual as r_tool
GATE_PURPOSE = 'physical_features_refresh8_continuous_integrator_wide_gate_support_gate_v1'
SCHEMA = 'hamiformer.hamiballs.physical_features_refresh8.continuous_gate_run.v1'
STAGE_SCHEMA = 'hamiformer.hamiballs.physical_features_refresh8.continuous_gate_stage.v1'
EXPECTED = {'total': 200, 'n8': 50, 'n12': 50, 'n20': 100}
_ORIGINAL_INSTALL = canonical._install

def _load_r(*, registration_path: Path, terminal_path: Path, expected_terminal_sha256: str, device: torch.device) -> dict[str, Any]:
    registration_path = registration_path.expanduser().resolve()
    terminal_path = terminal_path.expanduser().resolve()
    r_tool.configure_runner()
    registration = r_tool._load_registration(registration_path)
    if sha256_file(terminal_path) != expected_terminal_sha256:
        raise ValueError('continuous PhysicalFeatures r provenance changed')
    parent, _wide, parent_contract = residual_training._load_parents(registration)
    d, h, residual, expected = r_tool.base._build_continuous_experts(parent, device=device)
    payload = torch.load(terminal_path, map_location='cpu', weights_only=False)
    terminal = payload.get('terminal', {})
    if payload.get('schema') != r_tool.SCHEMA or payload.get('role') != r_tool.ROLE or payload.get('config') != parent['config'] or (terminal.get('status') != 'PASS') or (terminal.get('counts') != EXPECTED) or (terminal.get('frozen_digests') != expected):
        raise ValueError('continuous PhysicalFeatures r terminal contract changed')
    residual.load_state_dict(payload['r_state_dict'], strict=True)
    digest = module_digest(residual)
    if digest != terminal.get('r_terminal_digest'):
        raise ValueError('continuous PhysicalFeatures r digest changed')
    adapter._ACTIVE_METHOD = registration['integrator']
    adapter._ACTIVE_H = h
    adapter._ACTIVE_ATTR_SCALE = parent['attr_scale'].to(device=device, dtype=torch.float32)
    adapter._ACTIVE_FRAME_DT = float(r_tool.base.stage_a._train_contract(parent['config'])['frame_dt'])
    return {'registration': registration, 'parent': parent, 'parent_contract': parent_contract, 'd': d, 'h': h, 'r': residual, 'r_digest': digest, 'r_terminal_sha256': sha256_file(terminal_path), 'r_registration_sha256': sha256_file(registration_path)}

def _validate_gate(path: Path, *, r_registration_path: Path) -> dict[str, Any]:
    raw = json.loads(path.expanduser().resolve().read_text(encoding='utf-8'))
    registration = r_tool._load_registration(r_registration_path.expanduser().resolve())
    terminal = raw.get('r_terminal', {})
    terminal_path = Path(str(terminal.get('path', ''))).expanduser().resolve()
    if raw.get('schema') != carrier_cache.SCHEMA or raw.get('purpose') != GATE_PURPOSE or (raw.get('parents') != registration['parents']) or (raw.get('training', {}).get('sampling_seed') != support.namespaced_seed('g_model_registration')) or (terminal.get('training_registration_sha256') != sha256_file(r_registration_path)) or (terminal.get('expected_updates') != EXPECTED) or (not terminal_path.is_file()) or (sha256_file(terminal_path) != terminal.get('sha256')) or (raw.get('gate', {}).get('rank') != 12) or (raw.get('objective', {}).get('event_or_contact_labels') != 'forbidden'):
        raise ValueError('continuous PhysicalFeatures WideGate gate registration changed')
    return raw

def _install(*, r_registration: Path, gate_registration: Path):
    method = r_tool._load_registration(r_registration)['integrator']
    raw, prepared = _ORIGINAL_INSTALL(r_registration=r_registration, gate_registration=gate_registration)
    carrier_cache._collect_shared_main_carrier = adapter._collect_continuous_main_carrier
    if not method.startswith('gfjp_'):
        base.packed._field_tensors = adapter._continuous_field_tensors
        base.packed._online_forward = adapter._continuous_online_forward
    return (raw, prepared)

def configure_runner() -> None:
    gate_schedule.configure_runner()
    r_tool.configure_runner()
    support.load_canonical_r = _load_r
    canonical._validate_gate_registration = _validate_gate
    canonical._install = _install
    base.SCHEMA = SCHEMA
    base.STAGE_SCHEMA = STAGE_SCHEMA
    base.RUNNER_PATH = Path(__file__).resolve()
