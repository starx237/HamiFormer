from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from hamiformer.physics.pgf_scan import AffinePrefix, add_edge_innovation, apply_prefix, parallel_doubling_prefix, serial_prefix

@dataclass(frozen=True)
class TypeIIJet:
    matrix: torch.Tensor
    offset: torch.Tensor
    source_graph: torch.Tensor
    target_graph: torch.Tensor
    mixed_hessian: torch.Tensor
    unsafe_fraction: torch.Tensor

@dataclass(frozen=True)
class TypeIINewtonResult:
    state: torch.Tensor
    residual_max: torch.Tensor
    iterations: int
    converged: bool

class TypeIIPGFGenerator(nn.Module):

    def __init__(self, *, theta_dim: int, hidden_size: int, depth: int, q_scale: float, p_scale: float, theta_scale: tuple[float, ...], periodic_q_embedding: bool=False, fourier_features: tuple[int, ...]=()) -> None:
        super().__init__()
        if theta_dim < 1:
            raise ValueError('theta_dim 必须为正')
        if hidden_size < 4 or depth < 1:
            raise ValueError('hidden_size 至少为 4，depth 必须为正')
        if q_scale <= 0.0 or p_scale <= 0.0:
            raise ValueError('q_scale/p_scale 必须为正')
        if len(theta_scale) != theta_dim or any((value <= 0.0 for value in theta_scale)):
            raise ValueError('theta_scale 必须含 theta_dim 个正数')
        self.theta_dim = int(theta_dim)
        self.periodic_q_embedding = bool(periodic_q_embedding)
        if any((int(value) < 1 for value in fourier_features)):
            raise ValueError('Fourier frequencies must be positive integers')
        self.fourier_features = tuple(sorted({int(value) for value in fourier_features}))
        self.register_buffer('q_scale', torch.tensor(float(q_scale)))
        self.register_buffer('p_scale', torch.tensor(float(p_scale)))
        self.register_buffer('theta_scale', torch.tensor(theta_scale, dtype=torch.float32))
        layers: list[nn.Module] = []
        base_input_dim = 2 + theta_dim
        input_dim = base_input_dim + 2 * len(self.fourier_features) * base_input_dim
        if self.periodic_q_embedding:
            input_dim += 2
        for layer_index in range(depth):
            in_features = input_dim if layer_index == 0 else hidden_size
            layers.extend([nn.Linear(in_features, hidden_size), nn.SiLU()])
        final = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(final.weight, mean=0.0, std=0.001)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if q.shape != p_next.shape:
            raise ValueError('q 与 p_next 必须同形状')
        if theta.shape != (*q.shape, self.theta_dim):
            raise ValueError('theta 必须为 [*q.shape, theta_dim]')
        q_normalized = q / self.q_scale.to(q)
        p_normalized = p_next / self.p_scale.to(p_next)
        normalized = torch.cat([q_normalized.unsqueeze(-1), p_normalized.unsqueeze(-1), theta / self.theta_scale.to(theta)], dim=-1)
        q_features = [normalized]
        if self.periodic_q_embedding:
            angle = math.pi * q_normalized
            q_features.extend([torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)])
        for frequency in self.fourier_features:
            phase = math.pi * float(frequency) * normalized
            q_features.extend([torch.sin(phase), torch.cos(phase)])
        inputs = torch.cat(q_features, dim=-1)
        return self.network(inputs).squeeze(-1)

class SpectralTypeIIGenerator(nn.Module):

    def __init__(self, *, theta_dim: int, q_modes: int, p_degree: int, theta_degree: int, q_scale: float, p_scale: float, theta_center: float, theta_radius: float) -> None:
        super().__init__()
        if theta_dim != 1:
            raise ValueError('the first spectral Type-II generator supports scalar theta only')
        if q_modes < 0 or p_degree < 0 or theta_degree < 0:
            raise ValueError('spectral orders must be nonnegative')
        if q_scale <= 0.0 or p_scale <= 0.0 or theta_radius <= 0.0:
            raise ValueError('spectral scales must be positive')
        self.theta_dim = 1
        self.q_modes = int(q_modes)
        self.p_degree = int(p_degree)
        self.theta_degree = int(theta_degree)
        self.register_buffer('q_scale', torch.tensor(float(q_scale)))
        self.register_buffer('p_scale', torch.tensor(float(p_scale)))
        self.register_buffer('theta_center', torch.tensor(float(theta_center)))
        self.register_buffer('theta_radius', torch.tensor(float(theta_radius)))
        self.coefficients = nn.Parameter(torch.zeros(1 + 2 * self.q_modes, self.p_degree + 1, self.theta_degree + 1))

    @staticmethod
    def _legendre(value: torch.Tensor, degree: int) -> torch.Tensor:
        values = [torch.ones_like(value)]
        if degree >= 1:
            values.append(value)
        for index in range(2, degree + 1):
            values.append(((2 * index - 1) * value * values[-1] - (index - 1) * values[-2]) / index)
        return torch.stack(values, dim=-1)

    @staticmethod
    def _legendre_with_derivative(value: torch.Tensor, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = [torch.ones_like(value)]
        derivatives = [torch.zeros_like(value)]
        if degree >= 1:
            values.append(value)
            derivatives.append(torch.ones_like(value))
        for index in range(2, degree + 1):
            values.append(((2 * index - 1) * value * values[-1] - (index - 1) * values[-2]) / index)
            derivatives.append(((2 * index - 1) * (values[-2] + value * derivatives[-1]) - (index - 1) * derivatives[-2]) / index)
        return (torch.stack(values, dim=-1), torch.stack(derivatives, dim=-1))

    def basis(self, q: torch.Tensor, p_next: torch.Tensor, theta: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if q.shape != p_next.shape or theta.shape != (*q.shape, 1):
            raise ValueError('spectral Type-II input shapes are inconsistent')
        angle = math.pi * q / self.q_scale.to(q)
        q_values = [torch.ones_like(q)]
        q_derivatives = [torch.zeros_like(q)]
        angle_scale = math.pi / self.q_scale.to(q)
        for mode in range(1, self.q_modes + 1):
            q_values.extend([torch.cos(float(mode) * angle), torch.sin(float(mode) * angle)])
            q_derivatives.extend([-float(mode) * angle_scale * torch.sin(float(mode) * angle), float(mode) * angle_scale * torch.cos(float(mode) * angle)])
        p_values, p_derivative_normalized = self._legendre_with_derivative(p_next / self.p_scale.to(p_next), self.p_degree)
        theta_values = self._legendre((theta[..., 0] - self.theta_center.to(theta)) / self.theta_radius.to(theta), self.theta_degree)
        return ((torch.stack(q_values, dim=-1), torch.stack(q_derivatives, dim=-1), p_values, p_derivative_normalized / self.p_scale.to(p_next)), theta_values)

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (q_values, _q_derivatives, p_values, _p_derivatives), theta_values = self.basis(q, p_next, theta)
        return torch.einsum('...i,...j,...k,ijk->...', q_values, p_values, theta_values, self.coefficients)

def _mixed_derivatives(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, theta: torch.Tensor, *, step_size: float, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError('step_size 必须为 finite 正数')
    if q_anchor.shape != p_next_anchor.shape:
        raise ValueError('q_anchor 与 p_next_anchor 必须同形状')
    q = q_anchor.clone().requires_grad_(True)
    p_next = p_next_anchor.clone().requires_grad_(True)
    with torch.enable_grad():
        residual = generator(q, p_next, theta)
        if residual.shape != q.shape:
            raise ValueError('generator 输出必须与 q_anchor 同形状')
        h = torch.as_tensor(step_size, device=q.device, dtype=q.dtype)
        generating_value = q * p_next + h * residual
        grad_q, grad_p = torch.autograd.grad(generating_value.sum(), (q, p_next), create_graph=True, retain_graph=True)
        hessian_qq = torch.autograd.grad(grad_q.sum(), q, create_graph=create_graph, retain_graph=True)[0]
        hessian_qp = torch.autograd.grad(grad_q.sum(), p_next, create_graph=create_graph, retain_graph=True)[0]
        hessian_pp = torch.autograd.grad(grad_p.sum(), p_next, create_graph=create_graph, retain_graph=create_graph)[0]
    return (grad_q, grad_p, hessian_qq, hessian_qp, hessian_pp)

def type2_generator_eom(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, theta: torch.Tensor, *, step_size: float, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError('step_size 必须为 finite 正数')
    q = q_anchor.clone().requires_grad_(True)
    p_next = p_next_anchor.clone().requires_grad_(True)
    with torch.enable_grad():
        h = torch.as_tensor(step_size, device=q.device, dtype=q.dtype)
        generating_value = q * p_next + h * generator(q, p_next, theta)
        grad_q, grad_p = torch.autograd.grad(generating_value.sum(), (q, p_next), create_graph=create_graph, retain_graph=create_graph)
    return (grad_q, grad_p)

def type2_generator_jets(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, theta: torch.Tensor, *, step_size: float, mixed_hessian_floor: float, create_graph: bool) -> TypeIIJet:
    if mixed_hessian_floor <= 0.0:
        raise ValueError('mixed_hessian_floor 必须为正')
    grad_q, grad_p, hessian_qq, hessian_qp, hessian_pp = _mixed_derivatives(generator, q_anchor, p_next_anchor, theta, step_size=step_size, create_graph=create_graph)
    floor = torch.as_tensor(mixed_hessian_floor, device=hessian_qp.device, dtype=hessian_qp.dtype)
    sign = torch.where(hessian_qp >= 0.0, 1.0, -1.0)
    safe_mixed = sign * hessian_qp.abs().clamp_min(floor)
    unsafe = hessian_qp.abs() < floor
    m_qq = safe_mixed - hessian_pp * hessian_qq / safe_mixed
    m_qp = hessian_pp / safe_mixed
    m_pq = -hessian_qq / safe_mixed
    m_pp = 1.0 / safe_mixed
    matrix = torch.stack([torch.stack([m_qq, m_qp], dim=-1), torch.stack([m_pq, m_pp], dim=-1)], dim=-2)
    source_graph = torch.stack([q_anchor, grad_q], dim=-1)
    target_graph = torch.stack([grad_p, p_next_anchor], dim=-1)
    offset = target_graph - torch.matmul(matrix, source_graph.unsqueeze(-1)).squeeze(-1)
    return TypeIIJet(matrix=matrix, offset=offset, source_graph=source_graph, target_graph=target_graph, mixed_hessian=hessian_qp, unsafe_fraction=unsafe.to(matrix.dtype).mean())

def solve_type2_step_newton(generator: nn.Module, state: torch.Tensor, theta: torch.Tensor, *, step_size: float, mixed_hessian_floor: float, max_iterations: int=12, tolerance: float=1e-07, damping: float=1.0) -> TypeIINewtonResult:
    if state.ndim < 2 or state.shape[-1] != 2:
        raise ValueError('state 必须为 [...,2]')
    if theta.shape[:-1] != state.shape[:-1]:
        raise ValueError('theta 前导维必须与 state 一致')
    if max_iterations < 1:
        raise ValueError('max_iterations 必须为正')
    if tolerance <= 0.0 or mixed_hessian_floor <= 0.0:
        raise ValueError('tolerance/mixed_hessian_floor 必须为正')
    if not 0.0 < damping <= 1.0:
        raise ValueError('damping 必须位于 (0,1]')
    q = state[..., 0].detach()
    p = state[..., 1].detach()
    fixed_theta = theta.detach()
    p_next = p.clone()
    converged = False
    residual_max = torch.full((), math.inf, device=state.device, dtype=state.dtype)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        grad_q, _, _, mixed_hessian, _ = _mixed_derivatives(generator, q, p_next, fixed_theta, step_size=step_size, create_graph=False)
        residual = grad_q.detach() - p
        residual_max = residual.abs().amax()
        if bool((residual_max <= tolerance).item()):
            converged = True
            break
        sign = torch.where(mixed_hessian.detach() >= 0.0, 1.0, -1.0)
        safe_mixed = sign * mixed_hessian.detach().abs().clamp_min(torch.as_tensor(mixed_hessian_floor, device=state.device, dtype=state.dtype))
        p_next = (p_next - damping * residual / safe_mixed).detach()
    predicted_p, predicted_q = type2_generator_eom(generator, q, p_next, fixed_theta, step_size=step_size, create_graph=False)
    residual_max = (predicted_p.detach() - p).abs().amax()
    converged = bool((residual_max <= tolerance).item())
    return TypeIINewtonResult(state=torch.stack([predicted_q.detach(), p_next.detach()], dim=-1), residual_max=residual_max.detach(), iterations=iterations, converged=converged)

def solve_type2_step_unrolled(generator: nn.Module, state: torch.Tensor, theta: torch.Tensor, *, step_size: float, mixed_hessian_floor: float, iterations: int=4, create_graph: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.ndim < 2 or state.shape[-1] != 2:
        raise ValueError('state must be [...,2]')
    if theta.shape[:-1] != state.shape[:-1] or theta.shape[-1] < 1:
        raise ValueError('theta must align with state')
    if iterations < 1 or mixed_hessian_floor <= 0.0:
        raise ValueError('iterations and mixed_hessian_floor must be positive')
    with torch.enable_grad():
        q = state[..., 0]
        p = state[..., 1]
        p_next = p.clone().requires_grad_(True)
        floor = state.new_tensor(mixed_hessian_floor)
        for _ in range(iterations):
            predicted_p, _ = type2_generator_eom(generator, q, p_next, theta, step_size=step_size, create_graph=True)
            mixed = torch.autograd.grad(predicted_p.sum(), p_next, create_graph=True, retain_graph=True)[0]
            safe_mixed = torch.where(mixed >= 0.0, 1.0, -1.0) * mixed.abs().clamp_min(floor)
            p_next = p_next - (predicted_p - p) / safe_mixed
        predicted_p, predicted_q = type2_generator_eom(generator, q, p_next, theta, step_size=step_size, create_graph=True)
        residual = predicted_p - p
        mixed = torch.autograd.grad(predicted_p.sum(), p_next, create_graph=create_graph, retain_graph=create_graph)[0]
    if not create_graph:
        return (torch.stack([predicted_q.detach(), p_next.detach()], dim=-1), residual.detach(), mixed.detach())
    return (torch.stack([predicted_q, p_next], dim=-1), residual, mixed)

def learned_pgf_clean_estimate(generator: nn.Module, noisy_future: torch.Tensor, initial_state: torch.Tensor, theta_sys: torch.Tensor, *, step_size: float, mixed_hessian_floor: float, create_graph: bool, innovation: torch.Tensor | None=None, prefix_method: str='parallel') -> tuple[torch.Tensor, AffinePrefix, TypeIIJet]:
    jet = learned_pgf_edge_jets(generator, noisy_future, initial_state, theta_sys, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, create_graph=create_graph)
    forced_offset = add_edge_innovation(jet.offset, innovation)
    if prefix_method == 'parallel':
        prefix = parallel_doubling_prefix(jet.matrix, forced_offset)
    elif prefix_method == 'serial':
        prefix = serial_prefix(jet.matrix, forced_offset)
    else:
        raise ValueError(f'未知 prefix_method: {prefix_method}')
    estimate = apply_prefix(prefix, initial_state)
    if not create_graph:
        estimate = estimate.detach()
        prefix = AffinePrefix(matrix=prefix.matrix.detach(), offset=prefix.offset.detach())
    return (estimate, prefix, jet)

def learned_pgf_edge_jets(generator: nn.Module, noisy_future: torch.Tensor, initial_state: torch.Tensor, theta_sys: torch.Tensor, *, step_size: float, mixed_hessian_floor: float, create_graph: bool) -> TypeIIJet:
    if noisy_future.ndim != 3 or noisy_future.shape[-1] != 2:
        raise ValueError('noisy_future 必须为 [B,T,2]')
    batch, edges, _ = noisy_future.shape
    if edges < 1:
        raise ValueError('noisy_future 至少需要一条物理边')
    if initial_state.shape != (batch, 2):
        raise ValueError('initial_state 必须为 [B,2]')
    if theta_sys.ndim == 2:
        if theta_sys.shape[0] != batch:
            raise ValueError('trajectory context must be [B,theta_dim]')
        theta = theta_sys[:, None, :].expand(batch, edges, theta_sys.shape[-1])
    elif theta_sys.ndim == 3:
        if theta_sys.shape[:2] != (batch, edges):
            raise ValueError('edge context must be [B,T,theta_dim]')
        theta = theta_sys
    else:
        raise ValueError('context must be [B,theta_dim] or [B,T,theta_dim]')
    source_q = torch.cat([initial_state[:, None, 0], noisy_future[:, :-1, 0]], dim=1)
    target_p = noisy_future[..., 1]
    return type2_generator_jets(generator, source_q, target_p, theta, step_size=step_size, mixed_hessian_floor=mixed_hessian_floor, create_graph=create_graph)
