from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = project_root()
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import frozen_residual as residual_training
from hamiformer.training.hamiballs1 import router_models as gate_training
from hamiformer.training.hamiballs1 import residual_models as gate
from hamiformer.training import base as stage_a
R_SCHEMA = gate_training.SCHEMA
GATE_SCHEMA = 'hamiformer.hamiballs.gate_training.per_object_gate.registration.v1'

def _load_gate_training_models(registration, *, device):
    main, _wide, base_contract = residual_training._load_parents({'parents': registration['parents']})
    config = copy.deepcopy(main['config'])
    stage_a._validate_runtime_config(main['config'], smoke=False)
    contract = {**base_contract, 'train_contract': stage_a._train_contract(main['config'])}
    path = Path(registration['r_terminal']['path'])
    payload = torch.load(path, map_location='cpu', weights_only=False)
    terminal = payload.get('terminal', {})
    if payload.get('schema') != R_SCHEMA or terminal.get('status') != 'PASS' or terminal.get('counts') != registration['r_terminal']['expected_updates']:
        raise ValueError('GateTraining r terminal provenance mismatch')
    d, h, residual, _expected = gate_training._build_per_object_residual(main, device=device)
    residual.load_state_dict(payload['r_state_dict'], strict=True)
    frozen = {'d': gate._freeze(d, name='D'), 'h': gate._freeze(h, name='H'), 'r': gate._freeze(residual, name='r')}
    if frozen['r'] != terminal['r_terminal_digest']:
        raise ValueError('GateTraining r digest mismatch')
    return (main, contract, d, h, residual, frozen)
