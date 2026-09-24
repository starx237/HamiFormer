from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
GF_JET_PROPAGATOR_SCAN_SOLVER_FAMILY = 'gfjp_scan.type2_local_jet.affine_symplectic.associative_prefix.v3'
GF_JET_PROPAGATOR_SERIAL_SOLVER_FAMILY = 'gfjp_serial.type2_local_jet.affine_symplectic.mixed_writeback.v1'
REFERENCE_GF_TANGENT_SCAN_SOLVER_FAMILY = 'gf_tangent_scan.mixed_coordinate_affine_symplectic.associative_prefix.v2'
REFERENCE_LGF_SCAN_SOLVER_FAMILY = 'lgf_scan.direct_ms_newton.associative_scan.v1'
SUPPORTED_GF_SCAN_SOLVER_FAMILIES = frozenset({GF_JET_PROPAGATOR_SCAN_SOLVER_FAMILY, GF_JET_PROPAGATOR_SERIAL_SOLVER_FAMILY, REFERENCE_GF_TANGENT_SCAN_SOLVER_FAMILY, REFERENCE_LGF_SCAN_SOLVER_FAMILY})
SUPPORTED_INTEGRATOR_SOLVER_FAMILIES = frozenset({'integrator.explicit_euler.serial_mixed.v1', 'integrator.symplectic_euler.serial_mixed.v1', 'integrator.leapfrog.serial_mixed.v1'})
SUPPORTED_H_SOLVER_FAMILIES = SUPPORTED_GF_SCAN_SOLVER_FAMILIES | SUPPORTED_INTEGRATOR_SOLVER_FAMILIES
GF_TANGENT_SCAN_SOLVER_FAMILY = GF_JET_PROPAGATOR_SCAN_SOLVER_FAMILY
LGF_SCAN_SOLVER_FAMILY = GF_JET_PROPAGATOR_SCAN_SOLVER_FAMILY

@dataclass(frozen=True)
class AffinePrefix:
    matrix: torch.Tensor
    offset: torch.Tensor

def _validate_affine(matrix: torch.Tensor, offset: torch.Tensor) -> None:
    if matrix.ndim != 4:
        raise ValueError('matrix 必须为 [B,T,D,D]')
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('matrix 的最后两维必须为方阵')
    if matrix.shape[1] < 1:
        raise ValueError('affine recurrence 至少需要一条物理边')
    if offset.shape != matrix.shape[:-1]:
        raise ValueError('offset 必须为 [B,T,D]')
    if matrix.device != offset.device or matrix.dtype != offset.dtype:
        raise ValueError('matrix 与 offset 的 device/dtype 必须一致')

def compose_affine(left_matrix: torch.Tensor, left_offset: torch.Tensor, right_matrix: torch.Tensor, right_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = left_matrix @ right_matrix
    offset = torch.matmul(left_matrix, right_offset.unsqueeze(-1)).squeeze(-1)
    offset = offset + left_offset
    return (matrix, offset)

def serial_prefix(matrix: torch.Tensor, offset: torch.Tensor) -> AffinePrefix:
    _validate_affine(matrix, offset)
    batch, edges, state_dim, _ = matrix.shape
    running_matrix = torch.eye(state_dim, device=matrix.device, dtype=matrix.dtype).expand(batch, state_dim, state_dim)
    running_offset = torch.zeros(batch, state_dim, device=offset.device, dtype=offset.dtype)
    matrices: list[torch.Tensor] = []
    offsets: list[torch.Tensor] = []
    for edge in range(edges):
        running_matrix, running_offset = compose_affine(matrix[:, edge], offset[:, edge], running_matrix, running_offset)
        matrices.append(running_matrix)
        offsets.append(running_offset)
    return AffinePrefix(matrix=torch.stack(matrices, dim=1), offset=torch.stack(offsets, dim=1))

def parallel_doubling_prefix(matrix: torch.Tensor, offset: torch.Tensor) -> AffinePrefix:
    _validate_affine(matrix, offset)
    edges = matrix.shape[1]
    current_matrix = matrix
    current_offset = offset
    stride = 1
    while stride < edges:
        previous_matrix = current_matrix
        previous_offset = current_offset
        next_matrix = previous_matrix.clone()
        next_offset = previous_offset.clone()
        composed_matrix, composed_offset = compose_affine(previous_matrix[:, stride:], previous_offset[:, stride:], previous_matrix[:, :-stride], previous_offset[:, :-stride])
        next_matrix[:, stride:] = composed_matrix
        next_offset[:, stride:] = composed_offset
        current_matrix = next_matrix
        current_offset = next_offset
        stride *= 2
    return AffinePrefix(matrix=current_matrix, offset=current_offset)

def apply_prefix(prefix: AffinePrefix, initial_state: torch.Tensor) -> torch.Tensor:
    _validate_affine(prefix.matrix, prefix.offset)
    if initial_state.shape != (prefix.matrix.shape[0], prefix.matrix.shape[-1]):
        raise ValueError('initial_state 必须为 [B,D]')
    result = torch.matmul(prefix.matrix, initial_state[:, None, :, None]).squeeze(-1)
    return result + prefix.offset

def validate_convex_responsibility(responsibility: torch.Tensor, *, expected_shape: tuple[int, ...]) -> None:
    if responsibility.shape != expected_shape:
        raise ValueError(f'responsibility 必须为 {expected_shape}')
    if not bool(torch.isfinite(responsibility).all().item()):
        raise ValueError('responsibility 必须为 finite')
    if bool(((responsibility < 0.0) | (responsibility > 1.0)).any().item()):
        raise ValueError('responsibility 必须位于 [0,1]')

def add_edge_innovation(offset: torch.Tensor, innovation: torch.Tensor | None) -> torch.Tensor:
    if innovation is None:
        return offset
    if innovation.shape != offset.shape:
        raise ValueError('innovation 必须与 offset 同形状 [B,T,D]')
    return offset + innovation

def _validate_phase_sources(source_phase: torch.Tensor) -> None:
    if source_phase.ndim != 3 or source_phase.shape[-1] != 2:
        raise ValueError('source_phase 必须为 [B,T,2]，最后一维按 (q,p) 排列')

def harmonic_symplectic_euler_jets(source_phase: torch.Tensor, *, step_size: float, omega: float=1.0) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_phase_sources(source_phase)
    h = torch.as_tensor(step_size, device=source_phase.device, dtype=source_phase.dtype)
    omega_sq = torch.as_tensor(omega ** 2, device=source_phase.device, dtype=source_phase.dtype)
    one = torch.ones_like(h)
    matrix_single = torch.stack([torch.stack([one - h.square() * omega_sq, h]), torch.stack([-h * omega_sq, one])])
    matrix = matrix_single.expand(source_phase.shape[0], source_phase.shape[1], 2, 2).clone()
    offset = torch.zeros_like(source_phase)
    return (matrix, offset)

def pendulum_symplectic_euler_jets(source_phase: torch.Tensor, *, step_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_phase_sources(source_phase)
    h = torch.as_tensor(step_size, device=source_phase.device, dtype=source_phase.dtype)
    anchor_q = source_phase[..., 0]
    cosine = torch.cos(anchor_q)
    sine = torch.sin(anchor_q)
    one = torch.ones_like(anchor_q)
    matrix = torch.stack([torch.stack([one - h.square() * cosine, h * one], dim=-1), torch.stack([-h * cosine, one], dim=-1)], dim=-2)
    affine_term = anchor_q * cosine - sine
    offset = torch.stack([h.square() * affine_term, h * affine_term], dim=-1)
    return (matrix, offset)

def make_edge_sources(noisy_future: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
    if noisy_future.ndim != 3:
        raise ValueError('noisy_future 必须为 [B,T,D]')
    if initial_state.shape != (noisy_future.shape[0], noisy_future.shape[-1]):
        raise ValueError('initial_state 必须为 [B,D]')
    if noisy_future.shape[1] < 1:
        raise ValueError('窗口至少需要一个未来状态')
    return torch.cat([initial_state[:, None], noisy_future[:, :-1]], dim=1)

def pgf_clean_estimate(noisy_future: torch.Tensor, initial_state: torch.Tensor, *, system: Literal['harmonic', 'pendulum'], step_size: float, omega: float=1.0, innovation: torch.Tensor | None=None, prefix_method: Literal['serial', 'parallel']='parallel') -> tuple[torch.Tensor, AffinePrefix]:
    sources = make_edge_sources(noisy_future, initial_state)
    if system == 'harmonic':
        matrix, offset = harmonic_symplectic_euler_jets(sources, step_size=step_size, omega=omega)
    elif system == 'pendulum':
        matrix, offset = pendulum_symplectic_euler_jets(sources, step_size=step_size)
    else:
        raise ValueError(f'未知 system: {system}')
    forced_offset = add_edge_innovation(offset, innovation)
    if prefix_method == 'serial':
        prefix = serial_prefix(matrix, forced_offset)
    elif prefix_method == 'parallel':
        prefix = parallel_doubling_prefix(matrix, forced_offset)
    else:
        raise ValueError(f'未知 prefix_method: {prefix_method}')
    return (apply_prefix(prefix, initial_state), prefix)

def symplectic_defect(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError('当前诊断只支持最后两维为 2x2 的单自由度映射')
    j_matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=matrix.device, dtype=matrix.dtype)
    residual = matrix.transpose(-1, -2) @ j_matrix @ matrix - j_matrix
    return residual.square().sum(dim=(-1, -2)).sqrt()

def symplectic_euler_rollout(initial_state: torch.Tensor, *, steps: int, step_size: float, system: Literal['harmonic', 'pendulum'], omega: float=1.0) -> torch.Tensor:
    if initial_state.ndim != 2 or initial_state.shape[-1] != 2:
        raise ValueError('initial_state 必须为 [B,2]')
    if steps < 1:
        raise ValueError('steps 必须为正')
    h = torch.as_tensor(step_size, device=initial_state.device, dtype=initial_state.dtype)
    q = initial_state[..., 0]
    p = initial_state[..., 1]
    states: list[torch.Tensor] = []
    for _ in range(steps):
        if system == 'harmonic':
            force = omega ** 2 * q
        elif system == 'pendulum':
            force = torch.sin(q)
        else:
            raise ValueError(f'未知 system: {system}')
        p = p - h * force
        q = q + h * p
        states.append(torch.stack([q, p], dim=-1))
    return torch.stack(states, dim=1)
