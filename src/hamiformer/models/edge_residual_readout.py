from __future__ import annotations
import torch
from torch import nn

class EdgeResidualReadout(nn.Module):

    def __init__(self, *, token_dim: int, state_dim: int, theta_dim: int, hidden_size: int, depth: int=2, direct_state_paths: bool=False) -> None:
        super().__init__()
        if min(token_dim, state_dim, theta_dim, hidden_size, depth) < 1:
            raise ValueError('所有维度与 depth 必须为正')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.theta_dim = int(theta_dim)
        self.direct_state_paths = bool(direct_state_paths)
        input_dim = token_dim + 2 * state_dim + theta_dim + 1
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.append(nn.Linear(input_dim if index == 0 else hidden_size, hidden_size))
            layers.append(nn.SiLU())
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, state_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.direct_state_paths:
            self.token_projection = nn.Linear(token_dim, state_dim, bias=False)
            self.source_projection = nn.Linear(state_dim, state_dim, bias=False)
            self.candidate_projection = nn.Linear(state_dim, state_dim, bias=False)
            for layer in (self.token_projection, self.source_projection, self.candidate_projection):
                nn.init.zeros_(layer.weight)

    def forward(self, token: torch.Tensor, mixed_source: torch.Tensor, h_candidate: torch.Tensor, tau: torch.Tensor, theta_sys: torch.Tensor) -> torch.Tensor:
        if token.ndim != 2 or token.shape[-1] != self.token_dim:
            raise ValueError('token 必须为 [B,token_dim]')
        batch = token.shape[0]
        expected_state = (batch, self.state_dim)
        if mixed_source.shape != expected_state or h_candidate.shape != expected_state:
            raise ValueError('mixed_source/h_candidate 必须为 [B,state_dim]')
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if theta_sys.shape != (batch, self.theta_dim):
            raise ValueError('theta_sys 必须为 [B,theta_dim]')
        value = torch.cat([token, mixed_source, h_candidate, tau[:, None], theta_sys], dim=-1)
        residual = self.output(self.body(value))
        if self.direct_state_paths:
            residual = residual + self.token_projection(token) + self.source_projection(mixed_source) + self.candidate_projection(h_candidate)
        return residual
__all__ = ['EdgeResidualReadout']
