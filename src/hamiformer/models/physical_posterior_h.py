from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from hamiformer.types import PhaseObservation
from .common import ContinuousRoPE, ManualSelfAttention, RMSNorm, SinusoidalMLP, SwiGLU

@dataclass
class PosteriorCondition:
    object_context: torch.Tensor
    global_context: torch.Tensor

class _MaskedObservationBlock(nn.Module):

    def __init__(self, *, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float, qk_norm: bool) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.object_norm = RMSNorm(hidden_size)
        self.temporal_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.object_attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm, use_sdpa=True)
        self.temporal_attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm, use_sdpa=True)
        self.temporal_rope = ContinuousRoPE(hidden_size // num_heads, coord_dim=1)
        self.mlp = SwiGLU(hidden_size, mlp_ratio, dropout)

    def forward(self, tokens: torch.Tensor, relative_time: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError('encoder tokens 必须为 [B,T,K,D]')
        batch, frames, objects, hidden = tokens.shape
        if hidden != self.hidden_size:
            raise ValueError('encoder hidden size 不一致')
        if relative_time.shape != (batch, frames):
            raise ValueError('relative_time 必须为 [B,T]')
        object_tokens = tokens.reshape(batch * frames, objects, hidden)
        object_delta = self.object_attention(self.object_norm(object_tokens))
        tokens = tokens + object_delta.reshape(batch, frames, objects, hidden)
        temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(batch * objects, frames, hidden)
        time_coords = relative_time[:, None, :, None].expand(batch, objects, frames, 1)
        time_coords = time_coords.reshape(batch * objects, frames, 1)
        temporal_delta = self.temporal_attention(self.temporal_norm(temporal_tokens), rope=self.temporal_rope, coords=time_coords)
        temporal_delta = temporal_delta.reshape(batch, objects, frames, hidden).permute(0, 2, 1, 3)
        tokens = tokens + temporal_delta
        flat = tokens.reshape(batch, frames * objects, hidden)
        flat = flat + self.mlp(self.mlp_norm(flat))
        return flat.reshape(batch, frames, objects, hidden)

class MaskedPhaseObservationEncoder(nn.Module):

    def __init__(self, *, state_dim: int, attr_dim: int, state_scale: torch.Tensor, hidden_size: int=128, depth: int=4, num_heads: int=4, mlp_ratio: float=2.0, dropout: float=0.0, qk_norm: bool=True) -> None:
        super().__init__()
        if state_dim <= 0 or attr_dim <= 0 or hidden_size <= 0 or (depth <= 0):
            raise ValueError('encoder 维度与深度必须为正')
        if state_scale.shape != (state_dim,) or not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须为 [state_dim] 有限正数')
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.hidden_size = int(hidden_size)
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.input_projection = nn.Sequential(nn.Linear(2 * state_dim + 2 * attr_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.time_embedding = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([_MaskedObservationBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, qk_norm=qk_norm) for _ in range(depth)])
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, observation: PhaseObservation) -> PosteriorCondition:
        phase = observation.phase
        phase_mask = observation.phase_mask
        attrs = observation.attrs
        attr_mask = observation.attr_mask
        time = observation.time
        if phase.ndim != 4 or phase.shape[-1] != self.state_dim:
            raise ValueError('observation.phase 必须为 [B,T,K,state_dim]')
        batch, frames, objects, _ = phase.shape
        if phase_mask.shape != phase.shape or phase_mask.dtype != torch.bool:
            raise ValueError('phase_mask 必须是与 phase 同形的 bool tensor')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('observation.attrs shape 错误')
        if attr_mask.shape != attrs.shape or attr_mask.dtype != torch.bool:
            raise ValueError('attr_mask 必须是与 attrs 同形的 bool tensor')
        if time.shape != (batch, frames):
            raise ValueError('observation.time 必须为 [B,T]')
        phase_normalized = phase / self.state_scale.to(phase).view(1, 1, 1, -1)
        attrs_time = attrs[:, None].expand(batch, frames, objects, self.attr_dim)
        attr_mask_time = attr_mask[:, None].expand(batch, frames, objects, self.attr_dim)
        features = torch.cat([phase_normalized, phase_mask.to(dtype=phase.dtype), attrs_time, attr_mask_time.to(dtype=phase.dtype)], dim=-1)
        relative_time = time - time[:, :1]
        time_tokens = self.time_embedding(relative_time.reshape(-1))
        time_tokens = time_tokens.reshape(batch, frames, 1, self.hidden_size)
        tokens = self.input_projection(features) + time_tokens
        for block in self.blocks:
            tokens = block(tokens, relative_time)
        tokens = self.final_norm(tokens)
        observed = phase_mask.any(dim=-1).to(dtype=tokens.dtype)
        weighted_sum = (tokens * observed.unsqueeze(-1)).sum(dim=1)
        observed_count = observed.sum(dim=1, keepdim=False).unsqueeze(-1)
        observed_pool = weighted_sum / observed_count.clamp_min(1.0)
        fallback_pool = tokens.mean(dim=1)
        object_context = torch.where(observed_count > 0, observed_pool, fallback_pool)
        global_context = object_context.mean(dim=1)
        return PosteriorCondition(object_context=object_context, global_context=global_context)

class _EquivariantResidualBlock(nn.Module):

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner = max(hidden_size, int(hidden_size * mlp_ratio))
        self.norm = nn.LayerNorm(hidden_size)
        self.update = nn.Sequential(nn.Linear(2 * hidden_size, inner), nn.SiLU(), nn.Linear(inner, hidden_size))

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(objects)
        mean_message = normalized.mean(dim=1, keepdim=True).expand_as(normalized)
        return objects + self.update(torch.cat([normalized, mean_message], dim=-1))

class PhysicalPosteriorVectorField(nn.Module):

    def __init__(self, *, num_objects: int, object_latent_dim: int, condition_dim: int, hidden_size: int=256, depth: int=4, mlp_ratio: float=2.0) -> None:
        super().__init__()
        if min(num_objects, object_latent_dim, condition_dim, hidden_size, depth) <= 0:
            raise ValueError('posterior field 的维度与深度必须为正')
        self.num_objects = int(num_objects)
        self.object_latent_dim = int(object_latent_dim)
        self.latent_dim = self.num_objects * self.object_latent_dim
        self.latent_projection = nn.Linear(object_latent_dim, hidden_size)
        self.object_condition_projection = nn.Linear(condition_dim, hidden_size)
        self.global_condition_projection = nn.Linear(condition_dim, hidden_size)
        self.rf_time_embedding = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([_EquivariantResidualBlock(hidden_size, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, object_latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor, tau: torch.Tensor, *, condition: PosteriorCondition) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError('latent 必须为 [B,latent_dim]')
        batch = latent.shape[0]
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if condition.object_context.shape[:2] != (batch, self.num_objects):
            raise ValueError('object_context batch/object 维不一致')
        if condition.global_context.shape != (batch, condition.object_context.shape[-1]):
            raise ValueError('global_context shape 错误')
        objects = latent.reshape(batch, self.num_objects, self.object_latent_dim)
        hidden = self.latent_projection(objects)
        hidden = hidden + self.object_condition_projection(condition.object_context)
        hidden = hidden + self.global_condition_projection(condition.global_context)[:, None]
        hidden = hidden + self.rf_time_embedding(tau)[:, None]
        for block in self.blocks:
            hidden = block(hidden)
        velocity = self.output(self.final_norm(hidden))
        return velocity.reshape(batch, self.latent_dim)

class PhysicalPosteriorHExpert(nn.Module):

    def __init__(self, *, num_objects: int, state_dim: int, attr_dim: int, state_scale: torch.Tensor, encoder_hidden_size: int=128, encoder_depth: int=4, encoder_heads: int=4, encoder_mlp_ratio: float=2.0, field_hidden_size: int=256, field_depth: int=4, field_mlp_ratio: float=2.0, dropout: float=0.0) -> None:
        super().__init__()
        object_latent_dim = state_dim + 1
        self.encoder = MaskedPhaseObservationEncoder(state_dim=state_dim, attr_dim=attr_dim, state_scale=state_scale, hidden_size=encoder_hidden_size, depth=encoder_depth, num_heads=encoder_heads, mlp_ratio=encoder_mlp_ratio, dropout=dropout)
        self.field = PhysicalPosteriorVectorField(num_objects=num_objects, object_latent_dim=object_latent_dim, condition_dim=encoder_hidden_size, hidden_size=field_hidden_size, depth=field_depth, mlp_ratio=field_mlp_ratio)

    @property
    def latent_dim(self) -> int:
        return self.field.latent_dim

    def encode(self, observation: PhaseObservation) -> PosteriorCondition:
        return self.encoder(observation)

    def forward(self, latent: torch.Tensor, tau: torch.Tensor, *, condition: PosteriorCondition) -> torch.Tensor:
        return self.field(latent, tau, condition=condition)
