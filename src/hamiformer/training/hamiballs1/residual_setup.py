from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
from pathlib import Path
import torch
from hamiformer.models.hamiballs_committed import sample_hamiballs_per_object_rf_reset_schedule
from hamiformer.training import hamiballs_ordered_recovery_ordered_r as ordered
from hamiformer.training.hamiballs_COMPONENT_HULL_component_hull_r import COMPONENT_HULL_component_hull_residual_update
from hamiformer.training.hamiballs_step_scale_shared_step_scale_r import fit_reference_step_scales
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.training import base as stage_a
from hamiformer.training.hamiballs1 import hamiltonian as hami1_hamiltonian
from hamiformer.training.hamiballs1 import router_models as core
from hamiformer.training.hamiballs1 import router_state as runner
ROOT = project_root()
SCHEMA = 'hamiformer.hami1.residual.checkpoint.v1'
REGISTRATION_SCHEMA = 'hamiformer.hami1.residual.registration.v1'
ROLE = 'hami1-residual'
PURPOSE = 'hami1_component_residual_training'
H_SCHEMA = 'hamiformer.hami1.continuous_h.relation_training.v1'
H_ROLE = 'hami1-continuous-h-relation-training'
_ACTIVE_REGISTRATION = None
_CONFIGURATION = json.loads((ROOT / 'configs/hamiballs1/residual.json').read_text())
OBJECTIVE = _CONFIGURATION['objective']
PROHIBITIONS = set(_CONFIGURATION.get('prohibitions', ()))

def _calibrate(_carrier, *, x0, target, state_scale, q_dim):
    return (fit_reference_step_scales(x0, target, q_dim=q_dim), 1)

def _source_manifest():
    members = ('src/hamiformer/training/residual.py', 'src/hamiformer/training/hamiballs1/residual_setup.py', 'src/hamiformer/training/hamiballs1/router_models.py', 'src/hamiformer/training/hamiballs1/router_state.py', 'src/hamiformer/training/hamiballs1/frozen_residual.py', 'src/hamiformer/training/hamiballs_COMPONENT_HULL_component_hull_r.py', 'src/hamiformer/training/hamiballs_step_scale_shared_step_scale_r.py')
    return {member: sha256_file(ROOT / member) for member in members}

def _load_registration(path):
    global _ACTIVE_REGISTRATION
    raw = json.loads(Path(path).expanduser().resolve().read_text())
    for key in ('schema', 'purpose', 'integrator', 'hull_weight', 'objective'):
        if raw.get(key) != _CONFIGURATION.get(key):
            raise ValueError(f'Invalid residual training configuration: {key}')
    if raw.get('training') != _CONFIGURATION['training']:
        raise ValueError('Residual training schedule does not match the configuration')
    parents = raw['parents']
    if parents['terminal_step'] != 50000:
        raise ValueError('Residual training requires completed joint experts')
    for name in ('main_checkpoint', 'wide_checkpoint'):
        target = Path(parents[name]).expanduser().resolve()
        if not target.is_file() or sha256_file(target) != parents[name + '_sha256']:
            raise ValueError(f'Invalid parent model: {name}')
    h = raw['continuous_h']
    hp = Path(h['path']).expanduser().resolve()
    if h['expected_updates'] != 50000 or h['expected_schema'] != H_SCHEMA or h['expected_role'] != H_ROLE or (not hp.is_file()) or (sha256_file(hp) != h['sha256']):
        raise ValueError('Invalid continuous Hamiltonian input')
    _ACTIVE_REGISTRATION = raw
    return raw

def _build_continuous_experts(parent, *, device):
    if _ACTIVE_REGISTRATION is None:
        raise RuntimeError('Load the residual training configuration first')
    derived = copy.deepcopy(parent)
    derived['config'].setdefault('residual', {})['per_object_previous_g'] = True
    d, _, residual, _ = core._RESIDUAL_TRAINING_BUILD(derived, device=device)
    contract = _ACTIVE_REGISTRATION['continuous_h']
    payload = torch.load(contract['path'], map_location='cpu', weights_only=False)
    config = copy.deepcopy(parent['config'])
    if payload.get('schema') != H_SCHEMA or payload.get('role') != H_ROLE or payload.get('status') != 'COMPLETE' or (payload.get('updates') != contract['expected_updates']) or (payload.get('config') != config) or (payload.get('parent_checkpoint_sha256') != _ACTIVE_REGISTRATION['parents']['main_checkpoint_sha256']) or (not torch.equal(payload['state_scale'], parent['state_scale'])) or (not torch.equal(payload['attr_scale'], parent['attr_scale'])):
        raise ValueError('Hamiltonian and diffusion inputs are incompatible')
    h = hami1_hamiltonian._build_continuous_h(config, parent['state_scale'].to(device=device, dtype=torch.float32), device=device)
    h.load_state_dict(payload['h_state_dict'], strict=True)
    h.eval()
    set_trainable(h, False)
    if module_digest(h) != contract['h_digest']:
        raise ValueError('Hamiltonian state mismatch')
    return (d, h, residual, {'d': module_digest(d), 'h': module_digest(h)})

def _collect_external_carrier(*, config, contract, d, hamiltonian, residual, train, stream, state_scale, attr_scale, num_steps, batch_size, source_rng, reset_tau_rng, reset_rng, device):
    if stream.batch_size != batch_size:
        raise ValueError('Carrier batch size mismatch')
    x0, target, attrs, physical_time = stage_a._carrier_batch(train, stream, state_scale=state_scale, device=device)
    source = stage_a._random_source(target, generator=source_rng, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
    reset_tau = torch.rand(source.shape[0], device=device, dtype=source.dtype, generator=reset_tau_rng)
    reset_gate, _, reset_rf_tau = sample_hamiballs_per_object_rf_reset_schedule(reset_tau, edges=int(config['dataset']['future_steps']), objects=int(source.shape[2]), num_steps=num_steps, generator=reset_rng, pure_probability=0.5, min_tau=0.1, max_tau=0.6, reset_probability=0.15)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    carrier = stage_a.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=None, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='external', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']), external_gate=reset_gate, external_gate_tau=reset_rf_tau, continuous_integrator_method=None)
    if any((field.jets is None for field in carrier.trace.traces)):
        raise RuntimeError('PLAS carrier is missing affine maps')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return (carrier, x0, target, attrs, physical_time)

def configure_runner():
    ordered.reset_ordered_scale_state()
    ordered.calibrate_ordered_scales = _calibrate
    core.SCHEMA, core.REGISTRATION_SCHEMA = (SCHEMA, REGISTRATION_SCHEMA)
    core.ROLE, core.PURPOSE = (ROLE, PURPOSE)
    core.OBJECTIVE, core.PROHIBITIONS = (OBJECTIVE, PROHIBITIONS)
    core._source_manifest = _source_manifest
    core._build_per_object_residual = _build_continuous_experts
    core._collect_per_object_external_carrier = _collect_external_carrier
    core.joint_residual_joint_residual_update = COMPONENT_HULL_component_hull_residual_update
    runner._load_registration = _load_registration
    runner.REGISTERED_RESIDUAL_LR = copy.deepcopy(OBJECTIVE['schedule']['learning_rate'])
