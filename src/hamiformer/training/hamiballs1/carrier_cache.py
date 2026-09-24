from __future__ import annotations
from hamiformer.utils.paths import project_root
import json
from pathlib import Path
import sys
from typing import Any
import torch
ROOT = project_root()
from hamiformer.training import hamiballs_recovery as recovery
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import residual_models as gate
from hamiformer.training import base as stage_a
SCHEMA = 'hamiformer.hamiballs.carrier_cache.soft_projection_gate.registration.v1'

def _collect_shared_main_carrier(*, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module, residual: torch.nn.Module, gate_model: torch.nn.Module | None=None, gate: torch.nn.Module | None=None, train: Any, stream: Any, state_scale: torch.Tensor, attr_scale: torch.Tensor, num_steps: int, batch_size: int, source_rng: torch.Generator, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_gate = gate if gate is not None else gate_model
    if selected_gate is None:
        raise ValueError('shared CarrierCache main carrier requires a gate')
    if stream.batch_size != batch_size:
        raise AssertionError('main carrier stream has an incorrect batch size')
    x0, target, attrs, physical_time = stage_a._carrier_batch(train, stream, state_scale=state_scale, device=device)
    source = stage_a._random_source(target, generator=source_rng, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
    if source.device.type == 'cuda':
        torch.cuda.synchronize(source.device)
    carrier = recovery.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=selected_gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']))
    if source.device.type == 'cuda':
        torch.cuda.synchronize(source.device)
    return (carrier, x0, target, attrs, physical_time)

def _load_registration(path):
    raw = json.loads(path.expanduser().resolve().read_text())
    if raw.get('schema') != SCHEMA:
        raise ValueError('Invalid gate configuration')
    terminal = raw['r_terminal']
    target = Path(terminal['path'])
    if not target.is_file() or sha256_file(target) != terminal['sha256']:
        raise ValueError('Invalid residual input')
    return raw
