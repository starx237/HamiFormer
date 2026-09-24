from __future__ import annotations
import torch
from torch import nn
from hamiformer.types import HamiltonianOccurrences, HamiltonianOutput
from .edge_transformer import DirectGeneratingNetwork

class DirectVectorExpert(nn.Module):

    def __init__(self, *, q_dim: int, state_dim: int, attr_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float=0.0) -> None:
        super().__init__()
        if state_dim != 2 * q_dim:
            raise ValueError('DirectVectorExpert 要求 state_dim=2*q_dim')
        self.q_dim = int(q_dim)
        self.state_dim = int(state_dim)
        kwargs = dict(q_dim=q_dim, object_context_dim=state_dim + attr_dim, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.s_plus = DirectGeneratingNetwork(**kwargs)
        self.s_minus = DirectGeneratingNetwork(**kwargs)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, q_scale: torch.Tensor | None=None, p_scale: torch.Tensor | None=None) -> HamiltonianOutput:
        batch, future_steps, objects, _ = noisy_future.shape
        full = torch.cat([x0[:, None], noisy_future], dim=1)
        q, p = (full[..., :self.q_dim], full[..., self.q_dim:])
        q_left, q_right = (q[:, :-1], q[:, 1:])
        p_left, p_right = (p[:, :-1], p[:, 1:])
        context = torch.cat([x0.detach(), attrs.detach()], dim=-1)
        context = context[:, None].expand(batch, future_steps, objects, -1)
        relative_time = physical_time - physical_time[:, :1]
        edge_time = torch.stack([relative_time[:, :-1], relative_time[:, 1:]], dim=-1)

        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch * future_steps, objects, -1)
        common = {'object_context': flatten(context), 'tau': tau[:, None].expand(batch, future_steps).reshape(-1), 'edge_time': edge_time.reshape(batch * future_steps, 2)}
        delta_q_plus, delta_p_plus = self.s_plus(flatten(q_left), flatten(p_right), **common)
        delta_q_minus, delta_p_minus = self.s_minus(flatten(q_right), flatten(p_left), **common)
        shape = (batch, future_steps, objects, self.q_dim)
        q_plus = q_left + delta_q_plus.reshape(shape)
        p_plus = p_right + delta_p_plus.reshape(shape)
        q_minus = q_right - delta_q_minus.reshape(shape)
        p_minus = p_left - delta_p_minus.reshape(shape)
        q_clean = q_plus.clone()
        p_clean = p_minus.clone()
        q_clean[:, :-1] = 0.5 * (q_plus[:, :-1] + q_minus[:, 1:])
        p_clean[:, :-1] = 0.5 * (p_minus[:, :-1] + p_plus[:, 1:])
        q_den = 1.0 if q_scale is None else q_scale.to(q_clean).view(1, 1, 1, -1)
        p_den = 1.0 if p_scale is None else p_scale.to(p_clean).view(1, 1, 1, -1)
        difference = torch.cat([(q_plus[:, :-1] - q_minus[:, 1:]) / q_den, (p_minus[:, :-1] - p_plus[:, 1:]) / p_den], dim=-1)
        disagreement = difference.pow(2).mean(dim=-1).sqrt().flatten(1).amax(dim=1)
        return HamiltonianOutput(clean=torch.cat([q_clean, p_clean], dim=-1), disagreement=disagreement, occurrences=HamiltonianOccurrences(q_plus, p_plus, q_minus, p_minus))
