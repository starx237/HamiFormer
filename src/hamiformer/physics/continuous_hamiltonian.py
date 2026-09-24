from __future__ import annotations
from typing import Literal
import torch
from .generic_type2 import TokenConditionalTypeIIGenerator

class TokenConditionalContinuousHamiltonian(TokenConditionalTypeIIGenerator):
    architecture = 'token_conditional_continuous_hamiltonian'

def canonical_vector_field(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, context: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    if state.ndim < 3 or state.shape[-1] != 2 * hamiltonian.coordinate_dim:
        raise ValueError('continuous state must end in [objects, 2 * q_dim]')
    if state.shape[-2] != hamiltonian.num_objects:
        raise ValueError('continuous state object count differs from Hamiltonian')
    q_dim = hamiltonian.coordinate_dim
    q = state[..., :q_dim].reshape(*state.shape[:-2], -1)
    p = state[..., q_dim:].reshape(*state.shape[:-2], -1)
    if not q.requires_grad:
        q = q.requires_grad_(True)
    if not p.requires_grad:
        p = p.requires_grad_(True)
    energy = hamiltonian(q, p, context)
    grad_q, grad_p = torch.autograd.grad(energy.sum(), (q, p), create_graph=create_graph, retain_graph=create_graph)
    shape = (*state.shape[:-2], hamiltonian.num_objects, q_dim)
    return torch.cat((grad_p.reshape(shape), -grad_q.reshape(shape)), dim=-1)

def explicit_hamiltonian_step(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, context: torch.Tensor, *, step_size: float, method: Literal['euler', 'rk4'], create_graph: bool) -> torch.Tensor:
    if not step_size > 0.0:
        raise ValueError('step_size must be positive')
    if method == 'euler':
        return state + step_size * canonical_vector_field(hamiltonian, state, context, create_graph=create_graph)
    if method != 'rk4':
        raise ValueError('method must be euler or rk4')
    k1 = canonical_vector_field(hamiltonian, state, context, create_graph=create_graph)
    k2 = canonical_vector_field(hamiltonian, state + 0.5 * step_size * k1, context, create_graph=create_graph)
    k3 = canonical_vector_field(hamiltonian, state + 0.5 * step_size * k2, context, create_graph=create_graph)
    k4 = canonical_vector_field(hamiltonian, state + step_size * k3, context, create_graph=create_graph)
    return state + step_size / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def generalized_symplectic_hamiltonian_step(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, context: torch.Tensor, *, step_size: float, method: Literal['symplectic_euler', 'leapfrog'], fixed_point_iterations: int=3, fixed_point_damping: float=1.0, create_graph: bool) -> torch.Tensor:
    if not step_size > 0.0:
        raise ValueError('step_size must be positive')
    if fixed_point_iterations < 1:
        raise ValueError('fixed_point_iterations must be positive')
    if not 0.0 < fixed_point_damping <= 1.0:
        raise ValueError('fixed_point_damping must lie in (0, 1]')
    if method not in {'symplectic_euler', 'leapfrog'}:
        raise ValueError('method must be symplectic_euler or leapfrog')
    q_dim = hamiltonian.coordinate_dim
    q0, p0 = (state[..., :q_dim], state[..., q_dim:])
    h = float(step_size)
    damping = float(fixed_point_damping)

    def merge(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat((q, p), dim=-1)
    coefficient = h if method == 'symplectic_euler' else 0.5 * h
    p_implicit = p0
    for _ in range(fixed_point_iterations):
        field = canonical_vector_field(hamiltonian, merge(q0, p_implicit), context, create_graph=create_graph)
        proposal = p0 + coefficient * field[..., q_dim:]
        p_implicit = (1.0 - damping) * p_implicit + damping * proposal
    left_field = canonical_vector_field(hamiltonian, merge(q0, p_implicit), context, create_graph=create_graph)
    if method == 'symplectic_euler':
        return merge(q0 + h * left_field[..., :q_dim], p_implicit)
    q_implicit = q0 + h * left_field[..., :q_dim]
    for _ in range(fixed_point_iterations):
        right_field = canonical_vector_field(hamiltonian, merge(q_implicit, p_implicit), context, create_graph=create_graph)
        proposal = q0 + 0.5 * h * (left_field[..., :q_dim] + right_field[..., :q_dim])
        q_implicit = (1.0 - damping) * q_implicit + damping * proposal
    right_field = canonical_vector_field(hamiltonian, merge(q_implicit, p_implicit), context, create_graph=create_graph)
    p_next = p_implicit + 0.5 * h * right_field[..., q_dim:]
    return merge(q_implicit, p_next)

def continuous_hamiltonian_step(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, context: torch.Tensor, *, step_size: float, method: Literal['euler', 'rk4', 'explicit_euler', 'symplectic_euler', 'leapfrog'], create_graph: bool, fixed_point_iterations: int=3, fixed_point_damping: float=1.0) -> torch.Tensor:
    if method in {'euler', 'explicit_euler', 'rk4'}:
        return explicit_hamiltonian_step(hamiltonian, state, context, step_size=step_size, method='euler' if method == 'explicit_euler' else method, create_graph=create_graph)
    return generalized_symplectic_hamiltonian_step(hamiltonian, state, context, step_size=step_size, method=method, fixed_point_iterations=fixed_point_iterations, fixed_point_damping=fixed_point_damping, create_graph=create_graph)

def normalized_explicit_hamiltonian_step(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, step_size: float, method: Literal['euler', 'rk4'], create_graph: bool) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError('normalized continuous state must be [B,K,state]')
    if attrs.ndim != 3 or attrs.shape[:2] != state.shape[:2]:
        raise ValueError('continuous attrs must align with [B,K,*]')
    if state_scale.ndim != 1 or state_scale.shape[0] != state.shape[-1]:
        raise ValueError('continuous state scale must match the state dimension')
    if attr_scale.ndim != 1 or attr_scale.shape[0] != attrs.shape[-1]:
        raise ValueError('continuous attribute scale must match attrs')
    object_scale = state_scale.reshape(1, 1, -1).to(device=state.device, dtype=state.dtype)
    context = (attrs / attr_scale.reshape(1, 1, -1).to(device=attrs.device, dtype=attrs.dtype)).unsqueeze(-2)
    with torch.enable_grad():
        raw = state * object_scale
        advanced = explicit_hamiltonian_step(hamiltonian, raw, context, step_size=step_size, method=method, create_graph=create_graph)
        normalized = advanced / object_scale
    return normalized if create_graph else normalized.detach()

def normalized_continuous_hamiltonian_step(hamiltonian: TokenConditionalContinuousHamiltonian, state: torch.Tensor, attrs: torch.Tensor, *, state_scale: torch.Tensor, attr_scale: torch.Tensor, step_size: float, method: Literal['explicit_euler', 'symplectic_euler', 'leapfrog'], create_graph: bool, fixed_point_iterations: int=3, fixed_point_damping: float=1.0) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError('normalized continuous state must be [B,K,state]')
    if attrs.ndim != 3 or attrs.shape[:2] != state.shape[:2]:
        raise ValueError('continuous attrs must align with [B,K,*]')
    object_scale = state_scale.reshape(1, 1, -1).to(device=state.device, dtype=state.dtype)
    context = (attrs / attr_scale.reshape(1, 1, -1).to(device=attrs.device, dtype=attrs.dtype)).unsqueeze(-2)
    with torch.enable_grad():
        advanced = continuous_hamiltonian_step(hamiltonian, state * object_scale, context, step_size=step_size, method=method, create_graph=create_graph, fixed_point_iterations=fixed_point_iterations, fixed_point_damping=fixed_point_damping)
        normalized = advanced / object_scale
    return normalized if create_graph else normalized.detach()

def continuous_rollout(hamiltonian: TokenConditionalContinuousHamiltonian, initial: torch.Tensor, context: torch.Tensor, *, edges: int, step_size: float, method: Literal['euler', 'rk4'], create_graph: bool) -> torch.Tensor:
    if edges < 1:
        raise ValueError('edges must be positive')
    state = initial
    trajectory = []
    for _ in range(edges):
        state = explicit_hamiltonian_step(hamiltonian, state, context, step_size=step_size, method=method, create_graph=create_graph)
        trajectory.append(state)
        if not create_graph:
            state = state.detach()
    return torch.stack(trajectory, dim=-3)

def generalized_continuous_rollout(hamiltonian: TokenConditionalContinuousHamiltonian, initial: torch.Tensor, context: torch.Tensor, *, edges: int, step_size: float, method: Literal['explicit_euler', 'symplectic_euler', 'leapfrog'], create_graph: bool, fixed_point_iterations: int=3, fixed_point_damping: float=1.0) -> torch.Tensor:
    if edges < 1:
        raise ValueError('edges must be positive')
    state = initial
    trajectory = []
    for _ in range(edges):
        state = continuous_hamiltonian_step(hamiltonian, state, context, step_size=step_size, method=method, create_graph=create_graph, fixed_point_iterations=fixed_point_iterations, fixed_point_damping=fixed_point_damping)
        trajectory.append(state)
        if not create_graph:
            state = state.detach()
    return torch.stack(trajectory, dim=-3)

def student_t_location(residual: torch.Tensor, *, dof: float) -> torch.Tensor:
    if not dof > 0.0:
        raise ValueError('Student-t dof must be positive')
    return (0.5 * (dof + 1.0) * torch.log1p(residual.square() / dof)).mean()
