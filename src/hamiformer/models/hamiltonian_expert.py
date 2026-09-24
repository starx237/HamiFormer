from __future__ import annotations
from contextlib import nullcontext
import torch
from torch import nn
from hamiformer.types import HamiltonianOccurrences, HamiltonianOutput
from .edge_transformer import ScalarGeneratingNetwork

class HamiltonianExpert(nn.Module):

    def __init__(self, *, q_dim: int, state_dim: int, attr_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float=0.0, force_float32: bool=True) -> None:
        super().__init__()
        if state_dim != 2 * q_dim:
            raise ValueError('HamiltonianExpert 要求 state_dim=2*q_dim')
        self.q_dim = int(q_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.force_float32 = bool(force_float32)
        network_kwargs = dict(q_dim=q_dim, object_context_dim=state_dim + attr_dim, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.r_plus = ScalarGeneratingNetwork(**network_kwargs)
        self.r_minus = ScalarGeneratingNetwork(**network_kwargs)

    def _autocast_context(self, reference: torch.Tensor):
        if reference.device.type in {'cuda', 'cpu'}:
            return torch.autocast(device_type=reference.device.type, enabled=False)
        return nullcontext()

    def _object_context(self, attrs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if attrs.ndim != 3 or attrs.shape[-1] != self.attr_dim:
            raise ValueError('attrs 必须为 [B,K,attr_dim]')
        zeros = torch.zeros(*attrs.shape[:-1], self.state_dim, device=attrs.device, dtype=dtype)
        return torch.cat([zeros, attrs.detach().to(dtype=dtype)], dim=-1)

    @staticmethod
    def _stationary_edge_time(physical_time: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if physical_time.ndim != 2 or physical_time.shape[1] != 2:
            raise ValueError('单条 edge 的 physical_time 必须为 [B,2]')
        delta = physical_time[:, 1] - physical_time[:, 0]
        if bool((delta <= 0).any().item()):
            raise ValueError('物理时间必须严格递增')
        return torch.stack([torch.zeros_like(delta), delta], dim=-1).to(dtype=dtype)

    def _right_update(self, q_left: torch.Tensor, p_right: torch.Tensor, *, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32 if self.force_float32 else q_left.dtype
        context = self._object_context(attrs, dtype=dtype)
        edge_time = self._stationary_edge_time(physical_time, dtype=dtype)
        with torch.enable_grad(), self._autocast_context(q_left):
            q_input = q_left.detach().to(dtype=dtype).requires_grad_(True)
            p_input = p_right.detach().to(dtype=dtype).requires_grad_(True)
            residual = self.r_plus(q_input, p_input, object_context=context, tau=tau.detach().to(dtype=dtype), edge_time=edge_time)
            grad_q, grad_p = torch.autograd.grad(residual.sum(), (q_input, p_input), create_graph=create_graph)
            q_right = q_input + grad_p
            p_left = p_input + grad_q
        return (q_right, p_left)

    def _left_update(self, q_right: torch.Tensor, p_left: torch.Tensor, *, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32 if self.force_float32 else q_right.dtype
        context = self._object_context(attrs, dtype=dtype)
        edge_time = self._stationary_edge_time(physical_time, dtype=dtype)
        with torch.enable_grad(), self._autocast_context(q_right):
            q_input = q_right.detach().to(dtype=dtype).requires_grad_(True)
            p_input = p_left.detach().to(dtype=dtype).requires_grad_(True)
            residual = self.r_minus(q_input, p_input, object_context=context, tau=tau.detach().to(dtype=dtype), edge_time=edge_time)
            grad_q, grad_p = torch.autograd.grad(residual.sum(), (q_input, p_input), create_graph=create_graph)
            q_left = q_input - grad_p
            p_right = p_input - grad_q
        return (q_left, p_right)

    def local_clean_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> HamiltonianOccurrences:
        if clean_future.ndim != 4 or clean_future.shape[-1] != self.state_dim:
            raise ValueError('clean_future 必须为 [B,F,K,state_dim]')
        batch, future_steps, objects, _ = clean_future.shape
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 shape 错误')
        if physical_time.shape != (batch, future_steps + 1):
            raise ValueError('physical_time 必须为 [B,F+1]')
        full = torch.cat([x0[:, None], clean_future], dim=1)
        q, p = (full[..., :self.q_dim], full[..., self.q_dim:])
        q_left = q[:, :-1].reshape(batch * future_steps, objects, self.q_dim)
        p_left = p[:, :-1].reshape(batch * future_steps, objects, self.q_dim)
        q_right = q[:, 1:].reshape(batch * future_steps, objects, self.q_dim)
        p_right = p[:, 1:].reshape(batch * future_steps, objects, self.q_dim)
        attrs_edges = attrs[:, None].expand(batch, future_steps, objects, self.attr_dim)
        attrs_edges = attrs_edges.reshape(batch * future_steps, objects, self.attr_dim)
        edge_times = torch.stack([physical_time[:, :-1], physical_time[:, 1:]], dim=-1).reshape(batch * future_steps, 2)
        tau_clean = torch.ones(batch * future_steps, device=clean_future.device, dtype=clean_future.dtype)
        q_plus, p_plus = self._right_update(q_left, p_right, attrs=attrs_edges, tau=tau_clean, physical_time=edge_times, create_graph=create_graph)
        q_minus, p_minus = self._left_update(q_right, p_left, attrs=attrs_edges, tau=tau_clean, physical_time=edge_times, create_graph=create_graph)
        shape = (batch, future_steps, objects, self.q_dim)
        return HamiltonianOccurrences(q_plus=q_plus.reshape(shape), p_plus=p_plus.reshape(shape), q_minus=q_minus.reshape(shape), p_minus=p_minus.reshape(shape))

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, q_scale: torch.Tensor | None=None, p_scale: torch.Tensor | None=None, create_graph: bool | None=None) -> HamiltonianOutput:
        if noisy_future.ndim != 4 or noisy_future.shape[-1] != self.state_dim:
            raise ValueError('noisy_future 必须为 [B,F,K,state_dim]')
        batch, future_steps, objects, _ = noisy_future.shape
        if future_steps < 1:
            raise ValueError('future_steps 必须为正')
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 shape 错误')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs shape 错误')
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if physical_time.shape != (batch, future_steps + 1):
            raise ValueError('physical_time 必须为 [B,F+1]')
        if create_graph is None:
            create_graph = self.training
        noisy_p = noisy_future[..., self.q_dim:]
        current_q = x0[..., :self.q_dim]
        current_p = x0[..., self.q_dim:]
        q_plus_list: list[torch.Tensor] = []
        p_plus_list: list[torch.Tensor] = []
        q_minus_list: list[torch.Tensor] = []
        p_minus_list: list[torch.Tensor] = []
        for edge_index in range(future_steps):
            edge_time = physical_time[:, edge_index:edge_index + 2]
            next_q, reconstructed_p_left = self._right_update(current_q, noisy_p[:, edge_index], attrs=attrs, tau=tau, physical_time=edge_time, create_graph=create_graph)
            reconstructed_q_left, next_p = self._left_update(next_q, current_p, attrs=attrs, tau=tau, physical_time=edge_time, create_graph=create_graph)
            q_plus_list.append(next_q)
            p_plus_list.append(reconstructed_p_left)
            q_minus_list.append(reconstructed_q_left)
            p_minus_list.append(next_p)
            current_q, current_p = (next_q, next_p)
        q_plus = torch.stack(q_plus_list, dim=1)
        p_plus = torch.stack(p_plus_list, dim=1)
        q_minus = torch.stack(q_minus_list, dim=1)
        p_minus = torch.stack(p_minus_list, dim=1)
        clean = torch.cat([q_plus, p_minus], dim=-1)
        full_clean = torch.cat([x0[:, None], clean], dim=1)
        expected_q_left = full_clean[:, :-1, ..., :self.q_dim]
        expected_p_left = full_clean[:, :-1, ..., self.q_dim:]
        q_den = 1.0 if q_scale is None else q_scale.to(q_minus).view(1, 1, 1, -1)
        p_den = 1.0 if p_scale is None else p_scale.to(p_plus).view(1, 1, 1, -1)
        left_residual = torch.cat([(q_minus - expected_q_left) / q_den, (p_plus - expected_p_left) / p_den], dim=-1)
        disagreement = left_residual.pow(2).mean(dim=-1).sqrt().flatten(1).amax(dim=1)
        if not create_graph:
            clean = clean.detach()
            disagreement = disagreement.detach()
            q_plus = q_plus.detach()
            p_plus = p_plus.detach()
            q_minus = q_minus.detach()
            p_minus = p_minus.detach()
        return HamiltonianOutput(clean=clean, disagreement=disagreement, occurrences=HamiltonianOccurrences(q_plus=q_plus, p_plus=p_plus, q_minus=q_minus, p_minus=p_minus))
