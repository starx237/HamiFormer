from __future__ import annotations
import torch
from torch import nn
from .common import ManualSelfAttention, RMSNorm, SinusoidalMLP, SwiGLU

class EdgeTransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(hidden_size)
        self.attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=True)
        self.norm_mlp = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))

class EdgeTokenTransformer(nn.Module):

    def __init__(self, *, q_dim: int, object_context_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.q_projection = nn.Linear(q_dim, hidden_size)
        self.p_projection = nn.Linear(q_dim, hidden_size)
        self.context_projection = nn.Sequential(nn.Linear(object_context_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.q_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.p_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.rf_time = SinusoidalMLP(hidden_size)
        self.physical_time = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([EdgeTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, q_tokens: torch.Tensor, p_tokens: torch.Tensor, *, object_context: torch.Tensor, tau: torch.Tensor, edge_time: torch.Tensor) -> torch.Tensor:
        if q_tokens.shape != p_tokens.shape or q_tokens.ndim != 3:
            raise ValueError('q_tokens/p_tokens 必须同形且为 [E,K,d]')
        edges, objects, _ = q_tokens.shape
        if object_context.shape[:2] != (edges, objects):
            raise ValueError('object_context 必须为 [E,K,C]')
        if tau.shape != (edges,) or edge_time.shape != (edges, 2):
            raise ValueError('tau/edge_time 必须为 [E] 与 [E,2]')
        context = self.context_projection(object_context)
        q_hidden = self.q_projection(q_tokens) + context + self.q_type
        p_hidden = self.p_projection(p_tokens) + context + self.p_type
        query_condition = self.rf_time(tau) + self.physical_time(edge_time[:, 0]) + self.physical_time(edge_time[:, 1])
        query = self.query.unsqueeze(0) + query_condition
        sequence = torch.cat([query[:, None, :], q_hidden, p_hidden], dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        return self.final_norm(sequence)

class ScalarGeneratingNetwork(nn.Module):

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__()
        self.backbone = EdgeTokenTransformer(**kwargs)
        hidden_size = int(kwargs['hidden_size'])
        self.scalar_head = nn.Linear(hidden_size, 1)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        sequence = self.backbone(*args, **kwargs)
        return self.scalar_head(sequence[:, 0]).squeeze(-1)

class DirectGeneratingNetwork(nn.Module):

    def __init__(self, *, q_dim: int, **kwargs: int | float) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.backbone = EdgeTokenTransformer(q_dim=q_dim, **kwargs)
        hidden_size = int(kwargs['hidden_size'])
        self.vector_head = nn.Linear(2 * hidden_size, 2 * q_dim)

    def forward(self, q_tokens: torch.Tensor, p_tokens: torch.Tensor, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.backbone(q_tokens, p_tokens, **kwargs)
        objects = q_tokens.shape[1]
        q_hidden = sequence[:, 1:1 + objects]
        p_hidden = sequence[:, 1 + objects:1 + 2 * objects]
        delta_q, delta_p = self.vector_head(torch.cat([q_hidden, p_hidden], dim=-1)).chunk(2, dim=-1)
        return (delta_q, delta_p)
