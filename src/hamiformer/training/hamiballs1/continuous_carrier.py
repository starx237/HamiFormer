from __future__ import annotations
from hamiformer.utils.paths import project_root
from typing import Any
import torch
from hamiformer.models.hamiballs_committed import rollout_hamiballs_committed_edges
from hamiformer.physics.continuous_hamiltonian import TokenConditionalContinuousHamiltonian, normalized_continuous_hamiltonian_step
from hamiformer.training.hamiballs_formal import previous_gate_sequence
from hamiformer.training.hamiballs_trajectory import per_object_convex_projection_gate_loss
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training import base as stage_a
_ACTIVE_METHOD: str | None = None
_ACTIVE_H: TokenConditionalContinuousHamiltonian | None = None
_ACTIVE_ATTR_SCALE: torch.Tensor | None = None
_ACTIVE_FRAME_DT: float | None = None

def _collect_continuous_main_carrier(*, config: dict[str, Any], contract: dict[str, Any], d: torch.nn.Module, hamiltonian: torch.nn.Module, residual: torch.nn.Module, gate_model: torch.nn.Module | None=None, gate: torch.nn.Module | None=None, train: Any, stream: Any, state_scale: torch.Tensor, attr_scale: torch.Tensor, num_steps: int, batch_size: int, source_rng: torch.Generator, device: torch.device):
    selected_gate = gate if gate is not None else gate_model
    if selected_gate is None or _ACTIVE_METHOD is None:
        raise ValueError('continuous main carrier requires gate and method')
    if stream.batch_size != batch_size:
        raise AssertionError('continuous main carrier stream has wrong batch size')
    x0, target, attrs, physical_time = stage_a._carrier_batch(train, stream, state_scale=state_scale, device=device)
    source = stage_a._random_source(target, generator=source_rng, noise_scale=float(config['rectified_flow']['phase_noise_scale']))
    carrier_method = {'gfjp_symplectic_euler': None, 'gfjp_leapfrog': 'gfjp_leapfrog'}.get(_ACTIVE_METHOD, _ACTIVE_METHOD)
    carrier = stage_a.collect_recovery_carrier(d=d, hamiltonian=hamiltonian, residual=residual, gate=selected_gate, source=source, x0=x0, attrs=attrs, physical_time=physical_time, state_scale=state_scale, attr_scale=attr_scale, q_dim=int(config['dataset']['q_dim']), frame_dt=float(contract['frame_dt']), t_eps=float(config['rectified_flow']['t_eps']), num_steps=num_steps, mode='main', mixed_singular_floor=float(config['hamiltonian']['mixed_singular_floor']), mixed_condition_limit=float(config['hamiltonian']['mixed_condition_limit']), tangent_spectral_norm_limit=float(config['hamiltonian']['tangent_spectral_norm_limit']), continuous_integrator_method=carrier_method)
    return (carrier, x0, target, attrs, physical_time)

def _continuous_field_tensors(field: Any, *, x0: torch.Tensor, target: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> packed.TensorCache:
    rollout = field.rollout
    if rollout is None or field.d_tokens is None or field.jets is not None:
        raise RuntimeError('continuous gate cache received an invalid field')
    incoming = previous_gate_sequence(rollout.gate.detach())
    if incoming.ndim == 2:
        incoming = incoming[:, :, None].expand_as(rollout.gate)
    _unused, teacher = per_object_convex_projection_gate_loss(rollout.gate, rollout.hr_candidate, rollout.d_candidate, target)
    previous_heun_defect = getattr(field, 'previous_heun_defect', None)
    if previous_heun_defect is None:
        previous_heun_defect = torch.zeros_like(field.state)
    return {'d_tokens': field.d_tokens.detach(), 'noisy': field.state.detach(), 'x0': x0.detach(), 'previous_mixed': rollout.previous_mixed.detach(), 'h_candidate': rollout.h_candidate.detach(), 'hr_candidate': rollout.hr_candidate.detach(), 'd_candidate': rollout.d_candidate.detach(), 'mixed': rollout.mixed.detach(), 'attrs': attrs.detach(), 'tau': field.tau.detach(), 'physical_time': physical_time[:, 1:].detach(), 'previous_g': incoming.detach(), 'residual_hidden': rollout.residual_hidden.detach(), 'previous_heun_defect': previous_heun_defect.detach(), 'teacher': teacher.detach(), 'target': target.detach(), 'field_start_previous_mixed': rollout.previous_mixed[:, 0].detach(), 'field_start_previous_g': incoming[:, 0].detach()}

def _continuous_online_forward(gate: torch.nn.Module, residual: torch.nn.Module, cache: packed.TensorCache, *, state_scale: torch.Tensor, q_dim: int):
    del q_dim
    if _ACTIVE_H is None or _ACTIVE_METHOD is None or _ACTIVE_ATTR_SCALE is None or (_ACTIVE_FRAME_DT is None):
        raise RuntimeError('continuous gate expert was not initialized')
    create_graph = torch.is_grad_enabled()

    def h_builder(_edge: int, previous: torch.Tensor) -> torch.Tensor:
        return normalized_continuous_hamiltonian_step(_ACTIVE_H, previous, cache['attrs'], state_scale=state_scale, attr_scale=_ACTIVE_ATTR_SCALE, step_size=_ACTIVE_FRAME_DT, method=_ACTIVE_METHOD, create_graph=create_graph)
    replay = rollout_hamiballs_committed_edges(h_builder=h_builder, residual=residual, gate=gate, d_tokens=cache['d_tokens'], noisy=cache['noisy'], x0=cache['x0'], d_candidate=cache['d_candidate'], attrs=cache['attrs'], tau=cache['tau'], physical_time=cache['physical_time'], initial_previous_mixed=cache['field_start_previous_mixed'], initial_previous_g=cache['field_start_previous_g'])
    _unused, teacher = per_object_convex_projection_gate_loss(replay.gate, replay.hr_candidate, replay.d_candidate, cache['target'])
    effective = dict(cache)
    effective.update({'previous_mixed': replay.previous_mixed, 'h_candidate': replay.h_candidate, 'hr_candidate': replay.hr_candidate, 'mixed': replay.mixed, 'previous_g': previous_gate_sequence(replay.gate, initial=cache['field_start_previous_g']), 'residual_hidden': replay.residual_hidden, 'teacher': teacher})
    return (replay.gate, effective)
