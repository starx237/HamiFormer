from __future__ import annotations
from hamiformer.utils.paths import project_root
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import torch
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import frozen_residual as residual_training
ROOT = project_root()
RUN_SEED = 42

def namespaced_seed(namespace: str, counter: int=0) -> int:
    payload = f'hamiballs-canonical-v2|run-seed={RUN_SEED}|{namespace}|counter={counter}'.encode('utf-8')
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big') & 2147483647
    return value or 1

def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()

def emitted_registration_matches(path: Path, registration: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        emitted = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return emitted == registration and json_digest(emitted) == json_digest(registration)

def load_r_registration(path: Path):
    from hamiformer.training.residual import _load_registration
    return _load_registration(path)

def load_parents(registration: dict[str, Any]):
    return residual_training._load_parents({'parents': registration['parents']})

def load_canonical_r(*, registration_path, terminal_path, expected_terminal_sha256, device):
    from hamiformer.training.hamiballs1.gate_runtime import _load_r
    return _load_r(registration_path=registration_path, terminal_path=terminal_path, expected_terminal_sha256=expected_terminal_sha256, device=device)

def freeze(module: torch.nn.Module, *, name: str) -> str:
    set_trainable(module, False)
    module.eval()
    digest = module_digest(module)
    if any((parameter.requires_grad for parameter in module.parameters())):
        raise AssertionError(f'{name} retained trainable parameters')
    return digest
__all__ = ['RUN_SEED', 'freeze', 'json_digest', 'load_canonical_r', 'load_parents', 'load_r_registration', 'namespaced_seed']
