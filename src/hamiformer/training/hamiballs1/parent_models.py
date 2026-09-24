from hamiformer.utils.paths import project_root
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import torch
from torch import nn
ROOT = project_root()
from hamiformer.training.hamiballs_recovery import set_trainable
from hamiformer.training.hamiballs_d import build_hamiballs_d
def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def _build_wide_d(wide_parent: dict[str, Any], *, device: torch.device) -> nn.Module:
    config = wide_parent.get('config')
    architecture = wide_parent.get('architecture')
    if not isinstance(config, dict) or not isinstance(architecture, dict):
        raise ValueError('wide-D parent lacks config/architecture ledger')
    if wide_parent.get('architecture_sha256') != _canonical_hash(architecture):
        raise ValueError('wide-D architecture ledger hash differs from checkpoint')
    matched = architecture.get('matched_wide')
    nested = architecture.get('nested_initialization')
    if architecture.get('role') != 'wide-d' or not isinstance(matched, dict) or (not isinstance(nested, dict)) or (nested.get('function_preserving') is not True) or (architecture.get('initial_forward_equivalence', {}).get('within_tolerance') is not True):
        raise ValueError('wide-D parent lacks function-preserving evidence')
    wide_inner = int(matched.get('wide_inner_dim', -1))
    if wide_inner < 8:
        raise ValueError('wide-D parent has invalid inner dimension')
    wide = build_hamiballs_d(config['model'], q_dim=int(config['dataset']['q_dim']), attr_dim=int(config['dataset']['attr_dim']), seed=int(config['seed']), device=device, mlp_inner_dim=wide_inner)
    state = wide_parent.get('d_state_dict')
    if not isinstance(state, dict):
        raise ValueError('wide-D parent lacks terminal D state')
    wide.load_state_dict(state, strict=True)
    set_trainable(wide, False)
    wide.eval()
    return wide
