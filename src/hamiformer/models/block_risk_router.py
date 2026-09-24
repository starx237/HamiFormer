from __future__ import annotations
import torch
from torch import nn
from .physical_posterior_h import PosteriorCondition

class BlockRiskRouter(nn.Module):

    def __init__(self, *, context_dim: int, hidden_size: int=64, depth: int=2) -> None:
        super().__init__()
        if min(context_dim, hidden_size, depth) <= 0:
            raise ValueError('router context/hidden/depth 必须为正')
        layers: list[nn.Module] = [nn.Linear(context_dim, hidden_size), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, condition: PosteriorCondition) -> torch.Tensor:
        context = condition.global_context
        if context.ndim != 2:
            raise ValueError('condition.global_context 必须为 [B,D]')
        return self.network(context).squeeze(-1)

    @staticmethod
    def h_soft_preference(predicted_difference: torch.Tensor, *, temperature: float, cost_gap: float=0.0) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError('router temperature 必须为正')
        return torch.sigmoid(-(predicted_difference + cost_gap) / temperature)
