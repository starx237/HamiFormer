from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

class HamiltonianVectorField(nn.Module):

    def __init__(self, hamiltonian: nn.Module, *, q_dim: int) -> None:
        super().__init__()
        state_dim = int(getattr(hamiltonian, 'state_dim', -1))
        theta_dim = int(getattr(hamiltonian, 'theta_dim', -1))
        if q_dim <= 0 or state_dim != 2 * q_dim or theta_dim <= 0:
            raise ValueError('Hamiltonian 必须满足 state_dim=2*q_dim 且 theta_dim>0')
        self.hamiltonian = hamiltonian
        self.q_dim = int(q_dim)
        self.state_dim = state_dim
        self.theta_dim = theta_dim

    def forward(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 3 or phase.shape[-1] != self.state_dim:
            raise ValueError('phase 必须为 [B,K,state_dim]')
        if theta.shape != (*phase.shape[:2], self.theta_dim):
            raise ValueError('theta 必须为 [B,K,theta_dim]')
        build_parameter_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if build_parameter_graph:
                phase_for_grad = phase
                if not phase_for_grad.requires_grad:
                    phase_for_grad = phase.detach().requires_grad_(True)
            else:
                phase_for_grad = phase.detach().requires_grad_(True)
            energy = self.hamiltonian(phase_for_grad, theta)
            if energy.shape != (phase.shape[0],):
                raise ValueError('Hamiltonian 必须为每个 batch 返回一个标量')
            gradient = torch.autograd.grad(energy.sum(), phase_for_grad, create_graph=build_parameter_graph, retain_graph=build_parameter_graph)[0]
        grad_q = gradient[..., :self.q_dim]
        grad_p = gradient[..., self.q_dim:]
        field = torch.cat([grad_p, -grad_q], dim=-1)
        return field if build_parameter_graph else field.detach()

class EquivariantDirectVectorField(nn.Module):

    def __init__(self, *, state_dim: int, theta_dim: int, state_scale: torch.Tensor, hidden_size: int=96, depth: int=3) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim % 2 != 0 or theta_dim <= 0:
            raise ValueError('state_dim 必须为正偶数，theta_dim 必须为正')
        if state_scale.shape != (state_dim,) or not bool(torch.isfinite(state_scale).all().item()):
            raise ValueError('state_scale 必须为有限的 [state_dim]')
        if not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须严格为正')
        if hidden_size <= 0 or depth <= 0:
            raise ValueError('hidden_size/depth 必须为正')
        self.state_dim = int(state_dim)
        self.theta_dim = int(theta_dim)
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.input = nn.Linear(state_dim + theta_dim, hidden_size)
        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks.append(nn.Sequential(nn.LayerNorm(2 * hidden_size), nn.Linear(2 * hidden_size, 2 * hidden_size), nn.SiLU(), nn.Linear(2 * hidden_size, hidden_size)))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, state_dim))

    def forward(self, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 3 or phase.shape[-1] != self.state_dim:
            raise ValueError('phase 必须为 [B,K,state_dim]')
        if theta.shape != (*phase.shape[:2], self.theta_dim):
            raise ValueError('theta 必须为 [B,K,theta_dim]')
        scale = self.state_scale.to(phase).view(1, 1, -1)
        hidden = self.input(torch.cat([phase / scale, theta.to(phase)], dim=-1))
        for block in self.blocks:
            global_message = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            hidden = hidden + block(torch.cat([hidden, global_message], dim=-1))
        return self.output(hidden) * scale

class ImplicitMidpointVectorIntegrator(nn.Module):

    def __init__(self, vector_field: nn.Module, *, fixed_point_iterations: int=6) -> None:
        super().__init__()
        state_dim = int(getattr(vector_field, 'state_dim', -1))
        theta_dim = int(getattr(vector_field, 'theta_dim', -1))
        if state_dim <= 0 or theta_dim <= 0 or fixed_point_iterations <= 0:
            raise ValueError('vector field 合同或 fixed_point_iterations 无效')
        self.vector_field = vector_field
        self.state_dim = state_dim
        self.theta_dim = theta_dim
        self.fixed_point_iterations = int(fixed_point_iterations)

    def step(self, phase: torch.Tensor, theta: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 3 or phase.shape[-1] != self.state_dim:
            raise ValueError('phase 必须为 [B,K,state_dim]')
        if theta.shape != (*phase.shape[:2], self.theta_dim):
            raise ValueError('theta shape 与 phase 不一致')
        if delta_time.shape != (phase.shape[0],):
            raise ValueError('delta_time 必须为 [B]')
        step = delta_time.view(-1, 1, 1)
        start = phase
        estimate = start
        for _ in range(self.fixed_point_iterations):
            midpoint = 0.5 * (start + estimate)
            estimate = start + step * self.vector_field(midpoint, theta)
        return estimate

    def residual(self, start: torch.Tensor, end: torch.Tensor, theta: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        if start.shape != end.shape:
            raise ValueError('start/end shape 必须一致')
        if theta.shape != (*start.shape[:2], self.theta_dim):
            raise ValueError('theta shape 与 start/end 不一致')
        if delta_time.shape != (start.shape[0],):
            raise ValueError('delta_time 必须为 [B]')
        midpoint = 0.5 * (start + end)
        step = delta_time.view(-1, 1, 1)
        return end - start - step * self.vector_field(midpoint, theta)

@dataclass
class ShootingDecodeOutput:
    trajectory: torch.Tensor
    segments: torch.Tensor
    continuity_gaps: torch.Tensor
    midpoint_residuals: torch.Tensor

class BatchedShootingDecoder(nn.Module):

    def __init__(self, integrator: ImplicitMidpointVectorIntegrator, *, stride: int) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError('stride 必须为正')
        self.integrator = integrator
        self.stride = int(stride)
        self.state_dim = integrator.state_dim
        self.theta_dim = integrator.theta_dim

    def forward(self, anchors: torch.Tensor, theta: torch.Tensor, physical_time: torch.Tensor) -> ShootingDecodeOutput:
        if anchors.ndim != 4 or anchors.shape[-1] != self.state_dim:
            raise ValueError('anchors 必须为 [B,S,K,state_dim]')
        batch, segments_count, objects, _ = anchors.shape
        if theta.shape != (batch, objects, self.theta_dim):
            raise ValueError('theta 必须为 [B,K,theta_dim]')
        if physical_time.ndim != 2 or physical_time.shape[0] != batch:
            raise ValueError('physical_time 必须为 [B,F+1]')
        future_frames = physical_time.shape[1] - 1
        if future_frames <= 0 or future_frames % self.stride != 0:
            raise ValueError('G1 最小实现要求 future_frames 可被 stride 整除')
        if segments_count != future_frames // self.stride:
            raise ValueError('anchor 数必须等于 future_frames/stride')
        if not bool((physical_time[:, 1:] > physical_time[:, :-1]).all().item()):
            raise ValueError('physical_time 必须严格递增')
        local_delta = physical_time[:, 1:] - physical_time[:, :-1]
        local_delta = local_delta.reshape(batch, segments_count, self.stride)
        flat_delta = local_delta.reshape(batch * segments_count, self.stride)
        flat_theta = theta[:, None].expand(batch, segments_count, objects, self.theta_dim)
        flat_theta = flat_theta.reshape(batch * segments_count, objects, self.theta_dim)
        phase = anchors.reshape(batch * segments_count, objects, self.state_dim)
        states = [phase]
        residuals = []
        for local_index in range(self.stride):
            start = phase
            phase = self.integrator.step(start, flat_theta, flat_delta[:, local_index])
            residuals.append(self.integrator.residual(start, phase, flat_theta, flat_delta[:, local_index]))
            states.append(phase)
        decoded = torch.stack(states, dim=1).reshape(batch, segments_count, self.stride + 1, objects, self.state_dim)
        midpoint_residuals = torch.stack(residuals, dim=1).reshape(batch, segments_count, self.stride, objects, self.state_dim)
        if segments_count > 1:
            continuity_gaps = anchors[:, 1:] - decoded[:, :-1, -1]
        else:
            continuity_gaps = anchors.new_empty(batch, 0, objects, self.state_dim)
        frame_index = torch.arange(future_frames + 1, device=anchors.device, dtype=torch.long)
        segment_index = torch.div(frame_index, self.stride, rounding_mode='floor').clamp(max=segments_count - 1)
        within_segment = frame_index - segment_index * self.stride
        trajectory = decoded[:, segment_index, within_segment]
        return ShootingDecodeOutput(trajectory=trajectory, segments=decoded, continuity_gaps=continuity_gaps, midpoint_residuals=midpoint_residuals)
