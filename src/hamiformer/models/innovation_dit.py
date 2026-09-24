from __future__ import annotations
import torch
from torch import nn
from .common import RMSNorm, SinusoidalMLP, modulate
from .phase_dit import FactorizedPhaseBlock

class InnovationPhaseDiT(nn.Module):

    def __init__(self, *, innovation_dim: int, phase_dim: int, q_dim: int, attr_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, num_register_tokens: int, dropout: float=0.0, qk_norm: bool=True) -> None:
        super().__init__()
        if innovation_dim <= 0 or phase_dim != 2 * q_dim:
            raise ValueError('innovation_dim 必须为正，phase_dim 必须等于 2*q_dim')
        if min(hidden_size, depth, num_heads) <= 0 or hidden_size % num_heads != 0:
            raise ValueError('hidden/depth/heads 配置无效')
        self.innovation_dim = int(innovation_dim)
        self.phase_dim = int(phase_dim)
        self.q_dim = int(q_dim)
        self.token_projection = nn.Linear(innovation_dim + phase_dim, hidden_size)
        self.condition_projection = nn.Sequential(nn.Linear(phase_dim + attr_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.rf_time_embedding = SinusoidalMLP(hidden_size)
        self.physical_time_embedding = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([FactorizedPhaseBlock(hidden_size, num_heads, mlp_ratio, num_register_tokens, q_dim, dropout, qk_norm) for _ in range(depth)])
        self.num_register_tokens = int(num_register_tokens)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size) * 0.02)
        else:
            self.register_tokens = None
        self.final_norm = RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output = nn.Linear(hidden_size, innovation_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_innovation: torch.Tensor, decoded_phase: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> torch.Tensor:
        if noisy_innovation.ndim != 4 or noisy_innovation.shape[-1] != self.innovation_dim:
            raise ValueError('noisy_innovation 必须为 [B,T,K,innovation_dim]')
        batch, frames, objects, _ = noisy_innovation.shape
        if decoded_phase.shape != (batch, frames, objects, self.phase_dim):
            raise ValueError('decoded_phase shape 错误')
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if x0.shape != (batch, objects, self.phase_dim):
            raise ValueError('x0 必须为 [B,K,phase_dim]')
        if attrs.ndim != 3 or attrs.shape[:2] != (batch, objects):
            raise ValueError('attrs 必须为 [B,K,attr_dim]')
        if physical_time.shape != (batch, frames + 1):
            raise ValueError('physical_time 必须含初始时刻，为 [B,T+1]')
        rf_condition = self.rf_time_embedding(tau)
        condition = self.condition_projection(torch.cat([x0, attrs], dim=-1))
        condition = condition[:, None].expand(batch, frames, objects, -1)
        relative_time = physical_time[:, 1:] - physical_time[:, :1]
        time_embedding = self.physical_time_embedding(relative_time.reshape(-1))
        time_embedding = time_embedding.reshape(batch, frames, 1, -1)
        tokens = self.token_projection(torch.cat([noisy_innovation, decoded_phase], dim=-1))
        tokens = tokens + condition + time_embedding
        q_coords = decoded_phase[..., :self.q_dim]
        if self.register_tokens is None:
            registers = tokens.new_empty(batch, 0, tokens.shape[-1])
        else:
            registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            tokens, registers = block(tokens, registers, rf_condition, q_coords, relative_time)
        shift, scale = self.final_modulation(rf_condition).chunk(2, dim=-1)
        flat = tokens.reshape(batch, frames * objects, -1)
        flat = modulate(self.final_norm(flat), shift, scale)
        return self.output(flat).reshape(batch, frames, objects, self.innovation_dim)
