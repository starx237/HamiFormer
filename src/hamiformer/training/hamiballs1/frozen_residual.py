from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any
import numpy as np
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual
from hamiformer.training.training_schedule import ExplicitEpochBatchStream, generator_state_dict, load_generator_state_dict, set_optimizer_learning_rate
from hamiformer.training.hamiballs_recovery import clear_frozen_gradients, frozen_gradients_absent, module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.training import base as stage_a
SCHEMA = 'hamiformer.hami1.residual.checkpoint.v1'
ROLE = 'hami1-residual'

def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def _set_marker(output: Path, name: str, content: str) -> None:
    for other in ('RUNNING', 'PAUSED', 'COMPLETE', 'FAILED'):
        if other != name:
            (output / other).unlink(missing_ok=True)
    (output / name).write_text(content, encoding='utf-8')

def _terminal_summary(path: Path, *, role: str) -> dict[str, Any]:
    summary_path = path.parent / 'terminal_summary.json'
    if not summary_path.is_file() or not (path.parent / 'COMPLETE').is_file():
        raise FileNotFoundError(f'missing COMPLETE terminal evidence for {path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('status') != 'PASS' or summary.get('role') != role or summary.get('complete_training_run') is not True or (summary.get('training_data_only') is not True):
        raise ValueError(f'{role} terminal is not a completed train-only artifact')
    return summary

def _load_parents(registration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parents = registration['parents']
    main_path = Path(parents['main_checkpoint'])
    wide_path = Path(parents['wide_checkpoint'])
    if not main_path.is_file() or sha256_file(main_path) != parents['main_checkpoint_sha256']:
        raise ValueError('registered Main 50k checkpoint is missing or changed')
    if not wide_path.is_file() or sha256_file(wide_path) != parents['wide_checkpoint_sha256']:
        raise ValueError('registered matched wide-D checkpoint is missing or changed')
    main = torch.load(main_path, map_location='cpu', weights_only=False)
    wide = torch.load(wide_path, map_location='cpu', weights_only=False)
    terminal_step = int(parents['terminal_step'])
    if main.get('schema') != stage_a.CHECKPOINT_SCHEMA or main.get('role') != 'main' or int(main.get('step', -1)) != terminal_step or (not isinstance(main.get('config'), dict)) or (main.get('config_sha256') != _canonical_hash(main['config'])):
        raise ValueError('registered Main parent is not the expected immutable 50k terminal')
    if wide.get('schema') != stage_a.WIDE_CHECKPOINT_SCHEMA or wide.get('role') != 'wide-d' or int(wide.get('step', -1)) != terminal_step or (not isinstance(wide.get('config'), dict)) or (wide.get('config_sha256') != _canonical_hash(wide['config'])):
        raise ValueError('registered wide parent is not the expected immutable 50k terminal')
    if main.get('joint_training') is not True:
        raise ValueError('Main parent must come from joint D/H training')
    _terminal_summary(wide_path, role='wide-d')
    required_states = ('d_state_dict', 'h_state_dict')
    for name in required_states:
        if not isinstance(main.get(name), dict):
            raise ValueError(f'Main terminal lacks {name}')
        if any((not bool(torch.isfinite(value).all()) for value in main[name].values() if torch.is_tensor(value))):
            raise ValueError(f'Main terminal contains nonfinite {name}')
    if not isinstance(wide.get('d_state_dict'), dict):
        raise ValueError('wide-D terminal lacks D state')
    if main['config'] != wide['config'] or main.get('dataset_contract') != wide.get('dataset_contract') or (not torch.equal(main['state_scale'], wide['state_scale'])) or (not torch.equal(main['attr_scale'], wide['attr_scale'])):
        raise ValueError('Main and wide-D 50k terminal contracts differ')
    if main.get('dataset_contract', {}).get('train_only') is not True:
        raise ValueError('Main parent is not train-only')
    h_digest = stage_a.module_digest_from_state(main['h_state_dict'])
    if h_digest != main.get('h_freeze_digest'):
        raise ValueError('Main H state does not equal its recorded frozen H digest')
    architecture = wide.get('architecture')
    if not isinstance(architecture, dict) or wide.get('architecture_sha256') != _canonical_hash(architecture) or architecture.get('role') != 'wide-d' or (architecture.get('nested_initialization', {}).get('function_preserving') is not True) or (architecture.get('initial_forward_equivalence', {}).get('within_tolerance') is not True):
        raise ValueError('wide-D parent lacks function-preserving initialization evidence')
    contract = {'main_checkpoint': str(main_path), 'main_checkpoint_sha256': parents['main_checkpoint_sha256'], 'wide_checkpoint': str(wide_path), 'wide_checkpoint_sha256': parents['wide_checkpoint_sha256'], 'main_config_sha256': main['config_sha256'], 'dataset_contract': main['dataset_contract'], 'terminal_source_manifest': main['source_manifest'], 'main_d_digest': stage_a.module_digest_from_state(main['d_state_dict']), 'main_h_digest': h_digest, 'wide_d_digest': stage_a.module_digest_from_state(wide['d_state_dict'])}
    return (main, wide, contract)

def _build_frozen_experts_and_zero_r(parent: dict[str, Any], *, device: torch.device) -> tuple[nn.Module, nn.Module, HamiBallsDTokenResidual, dict[str, str]]:
    config = parent['config']
    if config.get('residual', {}).get('parameterization') != HAMIBALLS_RESIDUAL_UNBOUNDED_V1 or float(config.get('residual', {}).get('output_scale', math.nan)) != 1.0:
        raise ValueError('ResidualTraining requires the canonical unbounded scale-one residual')
    state_scale = parent['state_scale'].to(device=device, dtype=torch.float32)
    d = stage_a.build_hamiballs_d(config['model'], q_dim=int(config['dataset']['q_dim']), attr_dim=int(config['dataset']['attr_dim']), seed=int(config['seed']), device=device)
    torch.manual_seed(int(config['seed']) + 151)
    torch.cuda.manual_seed_all(int(config['seed']) + 151)
    if parent.get('hamiltonian_kind') == 'continuous':
        from hamiformer.training.hamiballs1.hamiltonian import _build_continuous_h
        hamiltonian = _build_continuous_h(config, state_scale, device=device)
    else:
        hamiltonian = stage_a._build_h(config, state_scale, device=device)
    torch.manual_seed(int(config['seed']) + 211)
    torch.cuda.manual_seed_all(int(config['seed']) + 211)
    residual = HamiBallsDTokenResidual(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim']), hidden_size=int(config['residual']['hidden_size']), parameterization=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g=bool(config['residual'].get('per_object_previous_g', False))).to(device=device, dtype=torch.float32)
    residual.set_output_scale(1.0)
    if not isinstance(residual.network, nn.Sequential) or not isinstance(residual.network[-1], nn.Linear):
        raise ValueError('ResidualTraining residual output-head schema drifted')
    with torch.no_grad():
        residual.network[-1].weight.zero_()
        residual.network[-1].bias.zero_()
    if torch.count_nonzero(residual.network[-1].weight).item() or torch.count_nonzero(residual.network[-1].bias).item():
        raise AssertionError('ResidualTraining zero-residual initialization failed')
    d.load_state_dict(parent['d_state_dict'], strict=True)
    hamiltonian.load_state_dict(parent['h_state_dict'], strict=True)
    for module, name in ((d, 'D'), (hamiltonian, 'H')):
        set_trainable(module, False)
        module.eval()
        if any((parameter.requires_grad for parameter in module.parameters())):
            raise AssertionError(f'frozen {name} retained a trainable parameter')
    set_trainable(residual, True)
    residual.train()
    frozen = {'d': module_digest(d), 'h': module_digest(hamiltonian)}
    expected = {'d': stage_a.module_digest_from_state(parent['d_state_dict']), 'h': stage_a.module_digest_from_state(parent['h_state_dict'])}
    if frozen != expected:
        raise AssertionError('loaded ResidualTraining frozen expert digest differs from Main terminal')
    return (d, hamiltonian, residual, frozen)

def _new_sampling(*, train_size: int, registration: dict[str, Any], device: torch.device) -> tuple[ExplicitEpochBatchStream, dict[str, torch.Generator]]:
    seed = int(registration['training']['sampling_seed'])
    return (ExplicitEpochBatchStream(train_size, int(registration['training']['carrier_batch_size']), seed=seed + 1), {'source': torch.Generator(device=device).manual_seed(seed + 2), 'reset_tau': torch.Generator(device=device).manual_seed(seed + 3), 'reset': torch.Generator(device=device).manual_seed(seed + 4)})

def _sampling_digest(stream: ExplicitEpochBatchStream, rngs: dict[str, torch.Generator]) -> dict[str, str]:

    def digest(generator: torch.Generator) -> str:
        raw = generator.get_state().detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    return {'stream': stage_a._state_digest(stream.state_dict()), **{name: digest(generator) for name, generator in sorted(rngs.items())}}

def _checkpoint(*, output: Path, total: int, config: dict[str, Any], config_sha256: str, parent_contract: dict[str, Any], source_manifest: dict[str, str], residual: HamiBallsDTokenResidual, optimizer: torch.optim.Optimizer, stream: ExplicitEpochBatchStream, rngs: dict[str, torch.Generator], counts: dict[str, int], frozen_digests: dict[str, str], history_rows: int) -> Path:
    path = output / f'checkpoint_{total:06d}.pt'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite immutable ResidualTraining checkpoint: {path}')
    payload = {'schema': SCHEMA, 'role': ROLE, 'index': total, 'config': config, 'config_sha256': config_sha256, 'parent_contract': parent_contract, 'source_manifest': source_manifest, 'r_state_dict': residual.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'stream': stream.state_dict(), 'rng_states': generator_state_dict(rngs), 'torch_cpu_rng_state': torch.get_rng_state(), 'torch_cuda_rng_states': torch.cuda.get_rng_state_all(), 'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(), 'counts': dict(counts), 'frozen_digests': dict(frozen_digests), 'r_digest': module_digest(residual), 'history_rows': int(history_rows), 'argv': list(sys.argv)}
    stage_a._atomic_torch(path, payload)
    return path

def _expected_counts(registration: dict[str, Any]) -> dict[str, int]:
    updates = registration['training']['updates_by_n']
    return {'total': int(sum(updates.values())), 'n8': int(updates['8']), 'n12': int(updates['12']), 'n20': int(updates['20'])}
