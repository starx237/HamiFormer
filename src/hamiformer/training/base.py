from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from hamiformer.data import PhaseWindowDataset, collate_phase_windows, load_phase_scales
from hamiformer.flow.pf_rf_v1 import make_pf_rf_v1_pair, sample_pf_rf_v1_tau
from hamiformer.models import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual, HamiBallsPerObjectCompactCommittedGate
from hamiformer.physics.generic_type2 import TokenConditionalTypeIIGenerator
from hamiformer.training.training_schedule import ExplicitEpochBatchStream, cosine_learning_rate, generator_state_dict, load_generator_state_dict, set_optimizer_learning_rate
from hamiformer.training.hamiballs_d import build_hamiballs_d, d_clean_and_tokens, matched_wide_inner_dim, nested_wide_from_narrow, parameter_count, update_hamiballs_d
from hamiformer.training.hamiballs_recovery import module_digest
from hamiformer.utils import sha256_file
CONFIG_SCHEMA = 'hamiformer.hami1.training.config.v1'
CHECKPOINT_SCHEMA = 'hamiformer.hami1.joint.checkpoint.v1'
WIDE_CHECKPOINT_SCHEMA = 'hamiformer.hami1.physiformer.checkpoint.v1'
TRAINING_STEPS = 50000
@dataclass(frozen=True)
class TensorCache:
    raw_x0: torch.Tensor
    raw_future: torch.Tensor
    attrs: torch.Tensor
    physical_time: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.raw_x0.shape[0])

def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _set_marker(output: Path, name: str, content: str) -> None:
    for other in ('RUNNING', 'PAUSED', 'COMPLETE', 'HEALTH_FAILED', 'FAILED'):
        if other != name:
            (output / other).unlink(missing_ok=True)
    (output / name).write_text(content, encoding='utf-8')

def _require_exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f'{label} keys mismatch: missing={sorted(keys - set(value))}, extra={sorted(set(value) - keys)}')

def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or int(value) < 1:
        raise ValueError(f'{label} must be a positive integer')
    return int(value)

def _finite_positive(value: object, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f'{label} must be finite and positive')
    return parsed

def _validate_config(raw: dict[str, Any], *, production_batches: bool) -> None:
    if not isinstance(raw, dict):
        raise ValueError('H1 training config must be a JSON object')
    if raw.get('schema') != CONFIG_SCHEMA:
        raise ValueError('wrong H1 training config schema')
    _require_exact(raw, {'schema', 'seed', 'dataset', 'model', 'hamiltonian', 'residual', 'gate', 'optimization', 'rectified_flow'}, 'H1 training config')
    if type(raw['seed']) is not int:
        raise ValueError('H1 training seed must be an integer')
    dataset = raw['dataset']
    if not isinstance(dataset, dict):
        raise ValueError('dataset must be an object')
    _require_exact(dataset, {'root', 'train_split', 'num_objects', 'future_steps', 'q_dim', 'attr_dim'}, 'dataset')
    if (str(dataset['train_split']), int(dataset['num_objects']), int(dataset['future_steps']), int(dataset['q_dim']), int(dataset['attr_dim'])) != ('train', 5, 48, 2, 3):
        raise ValueError('H1 training requires the natural K5/48/q2/attr3 train contract')
    model = raw['model']
    if not isinstance(model, dict):
        raise ValueError('model must be an object')
    _require_exact(model, {'state_dim', 'hidden_size', 'depth', 'num_heads', 'mlp_ratio', 'mlp_inner_dim', 'num_register_tokens', 'dropout', 'qk_norm', 'block_attn_pattern', 'temporal_rope_mode'}, 'model')
    if int(model['state_dim']) != 4 or float(model['dropout']) != 0.0:
        raise ValueError('H1 model must be the deterministic state4 D architecture')
    hamiltonian = raw['hamiltonian']
    if not isinstance(hamiltonian, dict):
        raise ValueError('hamiltonian must be an object')
    _require_exact(hamiltonian, {'hidden_size', 'depth', 'heads', 'expansion', 'spatial_attention_mode', 'object_attention_mode', 'student_t_dof', 'gfjp_iterations', 'rollout_horizon', 'rollout_weight', 'health_weight', 'mixed_singular_floor', 'mixed_condition_limit', 'tangent_spectral_norm_limit', 'health_safety_margin'}, 'hamiltonian')
    if (int(hamiltonian['hidden_size']), int(hamiltonian['gfjp_iterations']), int(hamiltonian['rollout_horizon']), float(hamiltonian['tangent_spectral_norm_limit'])) != (16, 2, 4, 5.0):
        raise ValueError('H1 Hamiltonian configuration drift')
    residual = raw['residual']
    if not isinstance(residual, dict):
        raise ValueError('residual must be an object')
    _require_exact(residual, {'hidden_size', 'parameterization', 'output_scale'}, 'residual')
    if (str(residual['parameterization']), float(residual['output_scale'])) != (HAMIBALLS_RESIDUAL_UNBOUNDED_V1, 1.0):
        raise ValueError('H1 residual must use the unbounded scale-one parameterization')
    gate = raw['gate']
    if not isinstance(gate, dict):
        raise ValueError('gate must be an object')
    _require_exact(gate, {'architecture', 'rank', 'regret_auxiliary_weight'}, 'gate')
    if (str(gate['architecture']), int(gate['rank']), float(gate['regret_auxiliary_weight'])) != ('per_object_field_gru_compact', 12, 0.1):
        raise ValueError('H1 gate must use the registered per-object rank-12 architecture')
    optimization = raw['optimization']
    if not isinstance(optimization, dict):
        raise ValueError('optimization must be an object')
    _require_exact(optimization, {'batch_size', 'cache_batch_size', 'num_workers', 'weight_decay', 'grad_clip'}, 'optimization')
    batch_size = _positive_int(optimization['batch_size'], 'optimization.batch_size')
    cache_batch_size = _positive_int(optimization['cache_batch_size'], 'optimization.cache_batch_size')
    if production_batches and (batch_size, cache_batch_size) != (64, 256):
        raise ValueError('H1 D/H batch configuration drift')
    if type(optimization['num_workers']) is not int or int(optimization['num_workers']) < 0:
        raise ValueError('optimization.num_workers must be a nonnegative integer')
    _finite_positive(optimization['weight_decay'], 'optimization.weight_decay')
    _finite_positive(optimization['grad_clip'], 'optimization.grad_clip')
    flow = raw['rectified_flow']
    if not isinstance(flow, dict):
        raise ValueError('rectified_flow must be an object')
    _require_exact(flow, {'train_tau_p_mean', 'train_tau_p_std', 't_eps', 'phase_noise_scale'}, 'rectified_flow')
    if tuple((float(flow[key]) for key in ('train_tau_p_mean', 'train_tau_p_std', 't_eps', 'phase_noise_scale'))) != (-0.8, 0.8, 0.05, 1.0):
        raise ValueError('H1 rectified-flow configuration drift')

def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    _validate_config(raw, production_batches=True)
    return raw

def _train_contract(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(config['dataset']['root'])).expanduser().resolve()
    metadata_path = root / 'metadata' / 'dataset_meta.json'
    stats_path = root / 'stats' / 'phase_scales.json'
    train_manifest = root / 'manifests' / 'train.jsonl'
    for path in (metadata_path, stats_path, train_manifest):
        if not path.is_file():
            raise FileNotFoundError(f'Recovery train-only contract component is missing: {path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError('dataset metadata is not an object')
    train_sha, stats_sha = (sha256_file(train_manifest), sha256_file(stats_path))
    if metadata.get('manifest_sha256', {}).get('train') != train_sha or metadata.get('stats_sha256') != stats_sha:
        raise ValueError('train manifest/stats hash mismatch')
    semantic = metadata.get('semantic_config', {})
    sampling = semantic.get('sampling', {})
    physics = semantic.get('physics', {})
    if (sampling.get('collision_mode'), int(sampling.get('num_objects', -1)), int(sampling.get('short_steps', -1))) != ('natural', 5, 48):
        raise ValueError('Recovery refuses a non-natural or incompatible train root')
    integrator = physics.get('smooth_integrator', physics.get('integrator'))
    if integrator != 'pre_kick_symplectic_euler':
        raise ValueError('Recovery requires the registered pre-kick symplectic generator')
    frame_dt = float(physics.get('frame_dt', math.nan))
    if not math.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError('train root has invalid frame_dt')
    return {'root': str(root), 'metadata_sha256': sha256_file(metadata_path), 'semantic_hash': metadata.get('semantic_hash'), 'implementation_fingerprint': metadata.get('implementation_fingerprint'), 'train_manifest_sha256': train_sha, 'stats_sha256': stats_sha, 'frame_dt': frame_dt, 'train_only': True}

def _make_dataset(config: dict[str, Any]) -> PhaseWindowDataset:
    dataset = config['dataset']
    return PhaseWindowDataset(Path(str(dataset['root'])) / 'manifests' / 'train.jsonl', num_objects=int(dataset['num_objects']), future_steps=int(dataset['future_steps']), q_dim=int(dataset['q_dim']), attr_dim=int(dataset['attr_dim']), verify_content_hash=False)

def _load_train_cache(config: dict[str, Any], *, device: torch.device, limit: int | None=None) -> TensorCache:
    dataset = _make_dataset(config)
    if limit is not None and int(limit) < 1:
        raise ValueError('train cache limit must be positive')
    loader = DataLoader(dataset, batch_size=int(config['optimization']['cache_batch_size']), shuffle=False, num_workers=int(config['optimization']['num_workers']), pin_memory=True, collate_fn=collate_phase_windows, drop_last=False)
    rows: dict[str, list[torch.Tensor]] = {'x0': [], 'future': [], 'attrs': [], 'time': []}
    loaded = 0
    for batch in loader:
        take = int(batch.x0.shape[0])
        if limit is not None:
            take = min(take, int(limit) - loaded)
        if take <= 0:
            break
        rows['x0'].append(batch.x0[:take])
        rows['future'].append(batch.future[:take])
        rows['attrs'].append(batch.attrs[:take])
        rows['time'].append(batch.time[:take])
        loaded += take
        if limit is not None and loaded >= int(limit):
            break
    if not rows['x0']:
        raise RuntimeError('Recovery train cache is empty')
    return TensorCache(raw_x0=torch.cat(rows['x0']).to(device=device, dtype=torch.float32), raw_future=torch.cat(rows['future']).to(device=device, dtype=torch.float32), attrs=torch.cat(rows['attrs']).to(device=device, dtype=torch.float32), physical_time=torch.cat(rows['time']).to(device=device, dtype=torch.float32))

def _attribute_scale(attrs: torch.Tensor) -> torch.Tensor:
    scale = attrs.to(dtype=torch.float64).square().mean(dim=(0, 1)).sqrt().float()
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
        raise ValueError('train attribute RMS scale is invalid')
    return scale

def _build_h(config: dict[str, Any], state_scale: torch.Tensor, *, device: torch.device) -> TokenConditionalTypeIIGenerator:
    dataset, hamiltonian = (config['dataset'], config['hamiltonian'])
    q_dim = int(dataset['q_dim'])
    return TokenConditionalTypeIIGenerator(num_objects=int(dataset['num_objects']), spatial_tokens=1, coordinate_dim=q_dim, token_context_dim=int(dataset['attr_dim']), hidden_size=int(hamiltonian['hidden_size']), depth=int(hamiltonian['depth']), heads=int(hamiltonian['heads']), expansion=float(hamiltonian['expansion']), q_scale=tuple((float(value) for value in state_scale[:q_dim])), p_scale=tuple((float(value) for value in state_scale[q_dim:])), spatial_attention_mode=str(hamiltonian['spatial_attention_mode']), object_attention_mode=str(hamiltonian['object_attention_mode'])).to(device=device, dtype=torch.float32)

def _build_main_modules(config: dict[str, Any], state_scale: torch.Tensor, *, device: torch.device) -> tuple[nn.Module, HamiBallsDTokenResidual, HamiBallsPerObjectCompactCommittedGate]:
    seed = int(config['seed'])
    torch.manual_seed(seed + 151)
    torch.cuda.manual_seed_all(seed + 151)
    hamiltonian = _build_h(config, state_scale, device=device)
    torch.manual_seed(seed + 211)
    torch.cuda.manual_seed_all(seed + 211)
    residual = HamiBallsDTokenResidual(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim']), hidden_size=int(config['residual']['hidden_size']), parameterization=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g=bool(config['residual'].get('per_object_previous_g', False))).to(device=device, dtype=torch.float32)
    residual.set_output_scale(1.0)
    torch.manual_seed(seed + 251)
    torch.cuda.manual_seed_all(seed + 251)
    gate = HamiBallsPerObjectCompactCommittedGate(token_dim=int(config['model']['hidden_size']), state_dim=int(config['model']['state_dim']), attr_dim=int(config['dataset']['attr_dim']), residual_hidden_dim=int(config['residual']['hidden_size']), rank=int(config['gate']['rank'])).to(device=device, dtype=torch.float32)
    gate.temporal.flatten_parameters()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return (hamiltonian, residual, gate)

def _random_source(target: torch.Tensor, *, generator: torch.Generator, noise_scale: float) -> torch.Tensor:
    return float(noise_scale) * torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)

def _normalised_batch(train: TensorCache, indices: torch.Tensor, state_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_x0, raw_future = (train.raw_x0[indices], train.raw_future[indices])
    x0 = raw_x0 / state_scale.reshape(1, 1, -1)
    target = raw_future / state_scale.reshape(1, 1, 1, -1)
    return (raw_x0, raw_future, x0, target, train.attrs[indices], train.physical_time[indices])

def _smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result['optimization']['batch_size'] = 8
    result['optimization']['cache_batch_size'] = 16
    result['optimization']['num_workers'] = 0
    return result

def _validate_runtime_config(config: dict[str, Any], *, smoke: bool) -> None:
    _validate_config(config, production_batches=not smoke)

def _lr(plan: dict[str, Any], *, update: int, total: int) -> float:
    values = plan['learning_rate']
    return cosine_learning_rate(update, total_steps=total, warmup_steps=int(values['warmup_updates']), maximum=float(values['maximum']), minimum=float(values['minimum']))

def _source_manifest() -> dict[str, str]:
    root = project_root()
    relative = ('src/hamiformer/training/base.py', 'src/hamiformer/models/hamiballs_committed.py', 'src/hamiformer/models/__init__.py', 'src/hamiformer/evaluation/hamiballs_formal.py', 'src/hamiformer/training/hamiballs_formal.py', 'src/hamiformer/training/hamiballs_trajectory.py', 'src/hamiformer/training/hamiballs_recovery.py', 'src/hamiformer/training/hamiballs_d.py', 'src/hamiformer/physics/hamiballs_type2.py', 'src/hamiformer/flow/stateful_pf_rf_v1.py')
    result: dict[str, str] = {}
    for name in relative:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Recovery source manifest member missing: {path}')
        result[name] = sha256_file(path)
    return result

def _carrier_batch(train: TensorCache, stream: ExplicitEpochBatchStream, *, state_scale: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = stream.next_indices().to(device=device)
    _raw_x0, _raw_future, x0, target, attrs, physical_time = _normalised_batch(train, indices, state_scale)
    return (x0, target, attrs, physical_time)

def _log_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    handle.flush()

def _initial_forward_equivalence(*, narrow_d: nn.Module, wide_d: nn.Module, train: TensorCache, state_scale: torch.Tensor, t_eps: float) -> dict[str, Any]:
    batch = min(8, train.size)
    if batch < 1:
        raise ValueError('nested wide-D needs a nonempty train batch')
    device = state_scale.device
    indices = torch.arange(batch, device=device, dtype=torch.long)
    _raw_x0, _raw_future, x0, clean, attrs, physical_time = _normalised_batch(train, indices, state_scale)
    tau = torch.linspace(float(t_eps), 1.0 - float(t_eps), batch, device=device)
    narrow_d.eval()
    wide_d.eval()
    with torch.inference_mode():
        narrow_clean, _ = d_clean_and_tokens(narrow_d, clean, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        wide_clean, _ = d_clean_and_tokens(wide_d, clean, tau, x0=x0, attrs=attrs, physical_time=physical_time)
    difference = (wide_clean - narrow_clean).abs()
    maximum = float(difference.max().cpu())
    mean = float(difference.mean().cpu())
    tolerance = 2e-06
    if not math.isfinite(maximum) or maximum > tolerance:
        raise AssertionError(f'nested wide-D pre-update drift {maximum:.9g} exceeds {tolerance:.9g}')
    return {'batch': batch, 'tau_min': float(tau.min().cpu()), 'tau_max': float(tau.max().cpu()), 'max_abs': maximum, 'mean_abs': mean, 'tolerance': tolerance, 'within_tolerance': True}

def _wide_checkpoint_payload(*, config: dict[str, Any], config_sha256: str, contract: dict[str, Any], source_manifest: dict[str, str], step: int, d: nn.Module, optimizer: torch.optim.Optimizer, stream: ExplicitEpochBatchStream, rngs: dict[str, torch.Generator], state_scale: torch.Tensor, attr_scale: torch.Tensor, architecture: dict[str, Any], history_rows: int) -> dict[str, Any]:
    return {'schema': WIDE_CHECKPOINT_SCHEMA, 'role': 'wide-d', 'config': config, 'config_sha256': config_sha256, 'dataset_contract': contract, 'source_manifest': source_manifest, 'step': step, 'd_state_dict': d.state_dict(), 'optimizer_state_dicts': {'d': optimizer.state_dict()}, 'stream': stream.state_dict(), 'rng_states': generator_state_dict(rngs), 'torch_cpu_rng_state': torch.get_rng_state(), 'torch_cuda_rng_states': torch.cuda.get_rng_state_all(), 'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(), 'state_scale': state_scale.detach().cpu(), 'attr_scale': attr_scale.detach().cpu(), 'architecture': architecture, 'architecture_sha256': _canonical_hash(architecture), 'history_rows': history_rows, 'script_sha256': sha256_file(Path(__file__).resolve()), 'argv': list(os.sys.argv)}

def _run_wide(*, args: argparse.Namespace, config: dict[str, Any], contract: dict[str, Any], source_manifest: dict[str, str], device: torch.device, smoke: bool) -> None:
    output = args.output_dir.expanduser().resolve()
    config_sha = _canonical_hash(config)
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(str(contract['root']))
    state_scale = load_phase_scales(root / 'stats' / 'phase_scales.json', int(config['dataset']['q_dim']), expected_train_manifest_sha256=str(contract['train_manifest_sha256'])).state(device=device, dtype=torch.float32)
    train = _load_train_cache(config, device=device, limit=None if not smoke else int(args.smoke_cache_samples))
    if train.size < int(config['optimization']['batch_size']):
        raise ValueError('train cache is smaller than a Recovery D batch')
    attr_scale = _attribute_scale(train.attrs).to(device=device)
    hamiltonian, residual, gate = _build_main_modules(config, state_scale, device=device)
    deployed_extra = parameter_count(hamiltonian) + parameter_count(residual) + parameter_count(gate)
    del hamiltonian, residual, gate
    torch.cuda.empty_cache()
    narrow = build_hamiballs_d(config['model'], q_dim=int(config['dataset']['q_dim']), attr_dim=int(config['dataset']['attr_dim']), seed=seed, device=device)
    narrow_parameters = parameter_count(narrow)
    width_match = matched_wide_inner_dim(narrow, deployed_extra_parameters=deployed_extra)
    wide, nested_metadata = nested_wide_from_narrow(narrow, config['model'], q_dim=int(config['dataset']['q_dim']), attr_dim=int(config['dataset']['attr_dim']), seed=seed, device=device, wide_inner_dim=int(width_match['wide_inner_dim']))
    equivalence = _initial_forward_equivalence(narrow_d=narrow, wide_d=wide, train=train, state_scale=state_scale, t_eps=float(config['rectified_flow']['t_eps']))
    del narrow
    torch.cuda.empty_cache()
    wide_parameters = parameter_count(wide)
    if wide_parameters - narrow_parameters != int(width_match['wide_added_parameters']):
        raise AssertionError('wide-D parameter count disagrees with nested width ledger')
    architecture = {'role': 'wide-d', 'narrow_d_parameters': narrow_parameters, 'wide_d_parameters': wide_parameters, 'deployed_main_extra_parameters': deployed_extra, 'matched_wide': width_match, 'nested_initialization': nested_metadata, 'initial_forward_equivalence': equivalence, 'train_only': True, 'complete_training_run': not smoke}
    optimizer = torch.optim.AdamW(wide.parameters(), lr=0.0003, weight_decay=float(config['optimization']['weight_decay']))
    stream = ExplicitEpochBatchStream(train.size, int(config['optimization']['batch_size']), seed=seed + 17)
    rngs = {'d_tau': torch.Generator(device=device).manual_seed(seed + 23), 'd_source': torch.Generator(device=device).manual_seed(seed + 29)}
    start_step, history_rows = (1, 0)
    resume = None if args.resume is None else args.resume.expanduser().resolve()
    if resume is None:
        if output.exists():
            raise FileExistsError('refusing to overwrite a wide-D artifact')
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / 'config_resolved.json', config)
        _atomic_json(output / 'train_contract.json', contract)
        _atomic_json(output / 'source_manifest.json', source_manifest)
        _atomic_json(output / 'architecture.json', architecture)
        history = output / 'history.jsonl'
    else:
        if not output.is_dir() or resume.parent != output or (not resume.is_file()):
            raise ValueError('wide-D resume must name an immutable checkpoint in its output')
        payload = torch.load(resume, map_location='cpu', weights_only=False)
        if payload.get('schema') != WIDE_CHECKPOINT_SCHEMA or payload.get('role') != 'wide-d' or payload.get('config_sha256') != config_sha or (payload.get('dataset_contract') != contract) or (payload.get('source_manifest') != source_manifest) or (payload.get('architecture_sha256') != _canonical_hash(architecture)):
            raise ValueError('wide-D resume contract mismatch')
        stages = sorted(output.glob('checkpoint_*.pt'))
        if not stages or resume != stages[-1]:
            raise ValueError('wide-D resume must use the latest immutable stage')
        wide.load_state_dict(payload['d_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dicts']['d'])
        stream.load_state_dict(payload['stream'])
        load_generator_state_dict(rngs, payload['rng_states'])
        torch.set_rng_state(payload['torch_cpu_rng_state'])
        torch.cuda.set_rng_state_all(payload['torch_cuda_rng_states'])
        np.random.set_state(payload['numpy_rng_state'])
        random.setstate(payload['python_rng_state'])
        start_step = int(payload['step']) + 1
        history_rows = int(payload['history_rows'])
        history = output / f'history_resume_from_{start_step - 1:06d}.jsonl'
        if history.exists():
            raise FileExistsError('wide-D resume history segment already exists')
    terminal = 48 if smoke else int(args.max_steps)
    stages = set(range(int(args.save_every), terminal + 1, int(args.save_every)))
    stages.add(terminal)
    if start_step > terminal:
        raise ValueError('wide-D resume checkpoint is already terminal')
    _set_marker(output, 'RUNNING', 'running\n')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with history.open('x', encoding='utf-8') as handle:
            for step in range(start_step, terminal + 1):
                step_started = time.perf_counter()
                d_lr = cosine_learning_rate(step, total_steps=terminal, warmup_steps=min(1000, terminal - 1), maximum=0.0003, minimum=3e-07)
                set_optimizer_learning_rate(optimizer, d_lr)
                indices = stream.next_indices().to(device=device)
                _raw_x0, _raw_future, x0, clean, attrs, physical_time = _normalised_batch(train, indices, state_scale)
                tau = sample_pf_rf_v1_tau(x0.shape[0], mean=float(config['rectified_flow']['train_tau_p_mean']), std=float(config['rectified_flow']['train_tau_p_std']), device=device, dtype=torch.float32, generator=rngs['d_tau'])
                pair = make_pf_rf_v1_pair(clean, tau, noise_scale=float(config['rectified_flow']['phase_noise_scale']), t_eps=float(config['rectified_flow']['t_eps']), generator=rngs['d_source'])
                wide.train()
                update = update_hamiballs_d(wide, optimizer, None, clean=clean, noisy=pair.noisy, tau=pair.tau, target_velocity=pair.target_velocity, x0=x0, attrs=attrs, physical_time=physical_time, t_eps=float(config['rectified_flow']['t_eps']), grad_clip=float(config['optimization']['grad_clip']))
                row: dict[str, Any] = {'step': step, 'role': 'wide-d', 'd_velocity_loss': update.velocity_loss, 'd_clean_loss': update.clean_loss, 'd_grad': update.gradient_norm_preclip, 'd_lr': d_lr}
                if step in stages:
                    checkpoint = _wide_checkpoint_payload(config=config, config_sha256=config_sha, contract=contract, source_manifest=source_manifest, step=step, d=wide, optimizer=optimizer, stream=stream, rngs=rngs, state_scale=state_scale, attr_scale=attr_scale, architecture=architecture, history_rows=history_rows + 1)
                    stage_path = output / f'checkpoint_{step:06d}.pt'
                    if stage_path.exists():
                        raise FileExistsError('refusing to overwrite an immutable wide-D stage')
                    _atomic_torch(stage_path, checkpoint)
                    row['stage_checkpoint'] = stage_path.name
                    if args.pause_after_stage == step:
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        row['wall_seconds'] = time.perf_counter() - step_started
                        _log_line(handle, row)
                        _set_marker(output, 'PAUSED', f'paused after immutable stage {step}\n')
                        return
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                row['wall_seconds'] = time.perf_counter() - step_started
                _log_line(handle, row)
                history_rows += 1
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _atomic_json(output / 'terminal_summary.json', {'schema': WIDE_CHECKPOINT_SCHEMA, 'status': 'PASS', 'role': 'wide-d', 'complete_training_run': not smoke, 'wall_seconds': time.perf_counter() - started, 'd_terminal_digest': module_digest(wide), 'architecture_sha256': _canonical_hash(architecture), 'training_data_only': True})
        _set_marker(output, 'COMPLETE', 'train-only matched wide-D terminal complete\n')
    except BaseException as error:
        _set_marker(output, 'FAILED', f'{type(error).__name__}: {error}\n')
        raise

def _state_digest(state: Any) -> str:

    def cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple((cpu(item) for item in value))
        return value
    buffer = io.BytesIO()
    torch.save(cpu(state), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

def module_digest_from_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='Train the capacity-matched HamiBalls-1 diffusion baseline')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--role', choices=('wide-d',), default='wide-d')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pause-after-stage', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-cache-samples', type=int, default=512)
    parser.add_argument('--max-steps', type=int, default=TRAINING_STEPS)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not 1 <= args.max_steps <= TRAINING_STEPS or args.save_every < 1:
        parser.error('invalid step or checkpoint interval')
    production = _load_config(args.config.expanduser().resolve())
    config = _smoke_config(production) if args.smoke else production
    _validate_runtime_config(config, smoke=args.smoke)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires an available CUDA device')
    _run_wide(args=args, config=config, contract=_train_contract(config), source_manifest=_source_manifest(), device=device, smoke=args.smoke)
if __name__ == '__main__':
    main()
