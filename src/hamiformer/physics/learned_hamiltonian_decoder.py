from __future__ import annotations
import torch
from torch import nn

def _mlp(input_dim: int, output_dim: int, *, hidden_size: int, depth: int) -> nn.Sequential:
    if min(input_dim, output_dim, hidden_size, depth) <= 0:
        raise ValueError('MLP 维度、宽度与深度必须为正')
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_size), nn.SiLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
    layers.append(nn.Linear(hidden_size, output_dim))
    return nn.Sequential(*layers)

class DeepSetHamiltonian(nn.Module):

    def __init__(self, *, state_dim: int, theta_dim: int, state_scale: torch.Tensor, hidden_size: int=96, depth: int=3) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim % 2 != 0 or theta_dim <= 0:
            raise ValueError('state_dim 必须为正偶数，theta_dim 必须为正')
        if state_scale.shape != (state_dim,):
            raise ValueError('state_scale 必须为 [state_dim]')
        if not bool(torch.isfinite(state_scale).all().item()) or not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须为有限正数')
        self.state_dim = int(state_dim)
        self.theta_dim = int(theta_dim)
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.object_encoder = _mlp(state_dim + theta_dim, hidden_size, hidden_size=hidden_size, depth=depth)
        self.global_energy = _mlp(hidden_size, 1, hidden_size=hidden_size, depth=depth)

    def _raw_energy(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        phase_normalized = phase / self.state_scale.to(phase).view(1, 1, -1)
        token = torch.cat([phase_normalized, theta.to(phase)], dim=-1)
        pooled = self.object_encoder(token).sum(dim=1)
        return self.global_energy(pooled).squeeze(-1)

    def forward(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 3 or phase.shape[-1] != self.state_dim:
            raise ValueError('phase 必须为 [B,K,state_dim]')
        if theta.shape != (*phase.shape[:2], self.theta_dim):
            raise ValueError('theta 必须为 [B,K,theta_dim]')
        energy = self._raw_energy(phase, theta)
        gauge = self._raw_energy(torch.zeros_like(phase), theta)
        return energy - gauge

class ImplicitMidpointHamiltonianDecoder(nn.Module):

    def __init__(self, *, hamiltonian: DeepSetHamiltonian, q_dim: int, fixed_point_iterations: int=6) -> None:
        super().__init__()
        if q_dim <= 0 or fixed_point_iterations <= 0:
            raise ValueError('q_dim/fixed_point_iterations 必须为正')
        if hamiltonian.state_dim != 2 * q_dim:
            raise ValueError('Hamiltonian state_dim 必须等于 2*q_dim')
        self.hamiltonian = hamiltonian
        self.q_dim = int(q_dim)
        self.fixed_point_iterations = int(fixed_point_iterations)

    def vector_field(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        build_parameter_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if build_parameter_graph:
                phase_for_grad = phase
                if not phase_for_grad.requires_grad:
                    phase_for_grad = phase_for_grad.detach().requires_grad_(True)
            else:
                phase_for_grad = phase.detach().requires_grad_(True)
            energy = self.hamiltonian(phase_for_grad, theta)
            gradient = torch.autograd.grad(energy.sum(), phase_for_grad, create_graph=build_parameter_graph, retain_graph=build_parameter_graph)[0]
        q_gradient = gradient[..., :self.q_dim]
        p_gradient = gradient[..., self.q_dim:]
        field = torch.cat([p_gradient, -q_gradient], dim=-1)
        return field if build_parameter_graph else field.detach()

    def _midpoint_step(self, phase: torch.Tensor, theta: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        if delta_time.shape != (phase.shape[0],):
            raise ValueError('delta_time 必须为 [B]')
        step = delta_time.view(-1, 1, 1)
        start = phase
        estimate = start
        for _ in range(self.fixed_point_iterations):
            midpoint = 0.5 * (start + estimate)
            estimate = start + step * self.vector_field(midpoint, theta)
        return estimate

    def midpoint_residual(self, start: torch.Tensor, end: torch.Tensor, theta: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        if start.shape != end.shape:
            raise ValueError('start/end shape 必须一致')
        step = delta_time.view(-1, 1, 1)
        midpoint = 0.5 * (start + end)
        return end - start - step * self.vector_field(midpoint, theta)

    def forward(self, reference_phase: torch.Tensor, theta: torch.Tensor, physical_time: torch.Tensor, *, reference_index: int=0) -> torch.Tensor:
        if reference_phase.ndim != 3 or reference_phase.shape[-1] != 2 * self.q_dim:
            raise ValueError('reference_phase 必须为 [B,K,2*q_dim]')
        batch, objects, _ = reference_phase.shape
        if theta.shape[:2] != (batch, objects):
            raise ValueError('theta batch/object 维必须与 reference_phase 一致')
        if theta.shape[-1] != self.hamiltonian.theta_dim:
            raise ValueError('theta 最后一维与 Hamiltonian theta_dim 不一致')
        if physical_time.ndim != 2 or physical_time.shape[0] != batch:
            raise ValueError('physical_time 必须为 [B,T]')
        frames = physical_time.shape[1]
        if frames < 1 or not 0 <= reference_index < frames:
            raise ValueError('reference_index 超出物理时间范围')
        if frames > 1 and (not bool((physical_time[:, 1:] > physical_time[:, :-1]).all().item())):
            raise ValueError('physical_time 必须严格递增')
        states: list[torch.Tensor | None] = [None] * frames
        states[reference_index] = reference_phase
        phase = reference_phase
        for index in range(reference_index + 1, frames):
            delta = physical_time[:, index] - physical_time[:, index - 1]
            phase = self._midpoint_step(phase, theta, delta)
            states[index] = phase
        phase = reference_phase
        for index in range(reference_index - 1, -1, -1):
            delta = physical_time[:, index] - physical_time[:, index + 1]
            phase = self._midpoint_step(phase, theta, delta)
            states[index] = phase
        if any((state is None for state in states)):
            raise RuntimeError('内部错误：Hamiltonian decoder 未填满全部物理时刻')
        return torch.stack([state for state in states if state is not None], dim=1)

    def energy(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 4 or phase.shape[-1] != 2 * self.q_dim:
            raise ValueError('phase 必须为 [B,T,K,2*q_dim]')
        batch, frames, objects, state_dim = phase.shape
        if theta.shape != (batch, objects, self.hamiltonian.theta_dim):
            raise ValueError('theta shape 错误')
        theta_time = theta[:, None].expand(batch, frames, objects, self.hamiltonian.theta_dim)
        return self.hamiltonian(phase.reshape(batch * frames, objects, state_dim), theta_time.reshape(batch * frames, objects, self.hamiltonian.theta_dim)).reshape(batch, frames)
