from __future__ import annotations
import torch
from torch import nn
from hamiformer.types import PhaseObservation
from .common import RMSNorm, SinusoidalMLP, modulate
from .phase_dit import FactorizedPhaseBlock
from .physical_posterior_h import MaskedPhaseObservationEncoder, PosteriorCondition

class MaskedPhaseDiTCore(nn.Module):

    def __init__(self, *, state_dim: int, q_dim: int, state_scale: torch.Tensor, condition_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, num_register_tokens: int, dropout: float=0.0, qk_norm: bool=True) -> None:
        super().__init__()
        if state_dim != 2 * q_dim:
            raise ValueError('MaskedPhaseDiTCore 要求 state_dim=2*q_dim')
        if state_scale.shape != (state_dim,) or not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须为 [state_dim] 有限正数')
        if min(condition_dim, hidden_size, depth, num_heads) <= 0:
            raise ValueError('condition/model 维度必须为正')
        if hidden_size % num_heads != 0:
            raise ValueError('hidden_size 必须能被 num_heads 整除')
        if num_register_tokens < 0:
            raise ValueError('num_register_tokens 不能为负')
        self.state_dim = int(state_dim)
        self.q_dim = int(q_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.object_condition_projection = nn.Linear(condition_dim, hidden_size)
        self.global_condition_projection = nn.Linear(condition_dim, hidden_size)
        self.rf_time_embedding = SinusoidalMLP(hidden_size)
        self.physical_time_embedding = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([FactorizedPhaseBlock(hidden_size, num_heads, mlp_ratio, num_register_tokens, q_dim, dropout, qk_norm) for _ in range(depth)])
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size) * 0.02)
        else:
            self.register_tokens = None
        self.final_norm = RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output = nn.Linear(hidden_size, state_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_trajectory: torch.Tensor, tau: torch.Tensor, *, condition: PosteriorCondition, physical_time: torch.Tensor) -> torch.Tensor:
        if noisy_trajectory.ndim != 4 or noisy_trajectory.shape[-1] != self.state_dim:
            raise ValueError('noisy_trajectory 必须为 [B,T,K,state_dim]')
        batch, frames, objects, _ = noisy_trajectory.shape
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if physical_time.shape != (batch, frames):
            raise ValueError('physical_time 必须为 [B,T]')
        if condition.object_context.shape[:2] != (batch, objects):
            raise ValueError('condition.object_context 的 batch/object 维不一致')
        condition_dim = condition.object_context.shape[-1]
        if condition.global_context.shape != (batch, condition_dim):
            raise ValueError('condition.global_context shape 错误')
        scale = self.state_scale.to(noisy_trajectory).view(1, 1, 1, -1)
        normalized_noisy = noisy_trajectory / scale
        relative_time = physical_time - physical_time[:, :1]
        time_embedding = self.physical_time_embedding(relative_time.reshape(-1))
        time_embedding = time_embedding.reshape(batch, frames, 1, -1)
        object_condition = self.object_condition_projection(condition.object_context)
        global_condition = self.global_condition_projection(condition.global_context)
        tokens = self.state_projection(normalized_noisy)
        tokens = tokens + object_condition[:, None]
        tokens = tokens + global_condition[:, None, None]
        tokens = tokens + time_embedding
        q_coords = noisy_trajectory[..., :self.q_dim]
        rf_condition = self.rf_time_embedding(tau)
        if self.register_tokens is None:
            registers = tokens.new_empty(batch, 0, tokens.shape[-1])
        else:
            registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            tokens, registers = block(tokens, registers, rf_condition, q_coords, relative_time)
        shift, modulation_scale = self.final_modulation(rf_condition).chunk(2, dim=-1)
        flat = tokens.reshape(batch, frames * objects, -1)
        flat = modulate(self.final_norm(flat), shift, modulation_scale)
        normalized_clean = self.output(flat).reshape(batch, frames, objects, self.state_dim)
        return normalized_clean * scale

class MaskedTrajectoryDExpert(nn.Module):

    def __init__(self, *, state_dim: int, q_dim: int, attr_dim: int, state_scale: torch.Tensor, encoder_hidden_size: int=128, encoder_depth: int=4, encoder_heads: int=4, encoder_mlp_ratio: float=2.0, dit_hidden_size: int=128, dit_depth: int=4, dit_heads: int=4, dit_mlp_ratio: float=2.0, num_register_tokens: int=2, dropout: float=0.0, qk_norm: bool=True) -> None:
        super().__init__()
        self.encoder = MaskedPhaseObservationEncoder(state_dim=state_dim, attr_dim=attr_dim, state_scale=state_scale, hidden_size=encoder_hidden_size, depth=encoder_depth, num_heads=encoder_heads, mlp_ratio=encoder_mlp_ratio, dropout=dropout, qk_norm=qk_norm)
        self.core = MaskedPhaseDiTCore(state_dim=state_dim, q_dim=q_dim, state_scale=state_scale, condition_dim=encoder_hidden_size, hidden_size=dit_hidden_size, depth=dit_depth, num_heads=dit_heads, mlp_ratio=dit_mlp_ratio, num_register_tokens=num_register_tokens, dropout=dropout, qk_norm=qk_norm)

    def encode(self, observation: PhaseObservation) -> PosteriorCondition:
        return self.encoder(observation)

    def forward(self, noisy_trajectory: torch.Tensor, tau: torch.Tensor, *, condition: PosteriorCondition, physical_time: torch.Tensor) -> torch.Tensor:
        return self.core(noisy_trajectory, tau, condition=condition, physical_time=physical_time)
