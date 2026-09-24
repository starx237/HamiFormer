from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
from pathlib import Path
import sys
from typing import Any
import torch
ROOT = project_root()
from hamiformer.models.hamiballs_committed import sample_hamiballs_per_object_rf_reset_schedule
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import frozen_residual as residual_training
from hamiformer.training import base as stage_a
_RESIDUAL_TRAINING_BUILD = residual_training._build_frozen_experts_and_zero_r
_RESIDUAL_TRAINING_SET_MARKER = residual_training._set_marker
SCHEMA = 'hamiformer.hamiballs.gate_training.per_object_previous_g.checkpoint.v1'
REGISTRATION_SCHEMA = 'hamiformer.hamiballs.gate_training.per_object_previous_g.registration.v1'
ROLE = 'frozen-50k-per-object-previous-g-joint-r'
PURPOSE = 'freeze_50k_main_d_h_train_fresh_zero_per_object_previous_g_joint_r_only'
OBJECTIVE = {'clean_teacher_forcing': 'smooth_l1_direct_plus_componentwise_qp_no_regret_v1', 'mixed_ar': 'train_carrier_calibrated_per_object_pseudohuber_direct_plus_detached_hull_v1', 'history': 'clean_gt_predecessor_and_shared_accepted_mixed_per_object_previous_g_v1', 'carrier': 'trajectory_shared_rf_node_per_object_physical_reset_schedule_v1', 'weighting': 'fixed_half_clean_half_mixed_v1', 'backward': 'ordinary_complete_shared_residual_mlp_v1', 'deployment': 'unbounded_r_output_scale_1_no_mask_no_inference_scale_v1', 'initialization': 'fresh_hidden_layers_zero_output_head_r_equals_zero_v1'}
PROHIBITIONS = {'no_d_or_h_parameter_update', 'no_gate_construction_forward_or_training', 'training_split_inputs_only', 'fixed_training_configuration', 'no_reuse_of_terminal_r_or_gate_weights', 'no_oracle_history_or_full_oracle_trace'}

def _load_registration(path):
    from hamiformer.training.residual import _load_registration as load
    return load(path)

def _source_manifest() -> dict[str, str]:
    members = ('src/hamiformer/training/hamiballs1/router_models.py', 'src/hamiformer/training/hamiballs1/frozen_residual.py', 'src/hamiformer/training/base.py', 'src/hamiformer/models/hamiballs_committed.py', 'src/hamiformer/training/hamiballs_recovery.py', 'src/hamiformer/training/hamiballs_recovery_full_no_regret.py', 'src/hamiformer/training/hamiballs_trajectory.py', 'src/hamiformer/evaluation/hamiballs_formal.py')
    return {member: sha256_file(ROOT / member) for member in members}

def _set_marker(output: Path, name: str, content: str) -> None:
    _RESIDUAL_TRAINING_SET_MARKER(output, name, content.replace('ResidualTraining', 'GateTraining'))

def _build_per_object_residual(parent: dict[str, Any], *, device: torch.device):
    derived = copy.deepcopy(parent)
    derived_config = copy.deepcopy(parent['config'])
    derived_config.setdefault('residual', {})['per_object_previous_g'] = True
    derived['config'] = derived_config
    return _RESIDUAL_TRAINING_BUILD(derived, device=device)

def _collect_per_object_external_carrier(*, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module | None, residual: torch.nn.Module, train: Any, stream: Any, state_scale: torch.Tensor, attr_scale: torch.Tensor, num_steps: int, batch_size: int, source_rng: torch.Generator, reset_tau_rng: torch.Generator, reset_rng: torch.Generator, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stream.batch_size != batch_size:
        raise AssertionError('external carrier stream has an incorrect batch size')
    x0, target, attrs, physical_time = stage_a._carrier_batch(train, stream, state_scale=state_scale, device=device)
    source = stage_a._random_source(target, generator=source_rng, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
    reset_tau = torch.rand(source.shape[0], device=device, dtype=source.dtype, generator=reset_tau_rng)
    reset_gate, _pure, reset_rf_tau = sample_hamiballs_per_object_rf_reset_schedule(reset_tau, edges=int(config['dataset']['future_steps']), objects=int(source.shape[2]), num_steps=num_steps, generator=reset_rng, pure_probability=0.5, min_tau=0.1, max_tau=0.6, reset_probability=0.15)
    if source.device.type == 'cuda':
        torch.cuda.synchronize(source.device)
    carrier = stage_a.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=None, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='external', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']), external_gate=reset_gate, external_gate_tau=reset_rf_tau)
    if source.device.type == 'cuda':
        torch.cuda.synchronize(source.device)
    return (carrier, x0, target, attrs, physical_time)
