from __future__ import annotations
import math
import torch
from torch import nn

class CausalPhaseBurninContextEncoder(nn.Module):
    architecture = 'causal_phase_burnin_context_transformer'

    def __init__(self, *, num_objects: int, coordinate_dim: int, context_dim: int, hidden_size: int, depth: int, heads: int, max_burnin_frames: int, q_scale: tuple[float, ...], p_scale: tuple[float, ...]) -> None:
        super().__init__()
        if min(num_objects, coordinate_dim, context_dim, hidden_size, depth, heads, max_burnin_frames) < 1:
            raise ValueError('causal phase context dimensions must be positive')
        if hidden_size % heads != 0:
            raise ValueError('causal phase context hidden_size must be divisible by heads')
        q_values, p_values = (tuple((float(value) for value in q_scale)), tuple((float(value) for value in p_scale)))
        if len(q_values) != coordinate_dim or len(p_values) != coordinate_dim:
            raise ValueError('causal phase context scales must have coordinate_dim entries')
        if any((not math.isfinite(value) or value <= 0.0 for value in q_values + p_values)):
            raise ValueError('causal phase context scales must be finite positive')
        self.num_objects = int(num_objects)
        self.coordinate_dim = int(coordinate_dim)
        self.context_dim = int(context_dim)
        self.hidden_size = int(hidden_size)
        self.max_burnin_frames = int(max_burnin_frames)
        self.register_buffer('q_scale', torch.tensor(q_values))
        self.register_buffer('p_scale', torch.tensor(p_values))
        self.phase_projection = nn.Linear(2 * self.coordinate_dim, self.hidden_size)
        self.frame_position = nn.Parameter(torch.zeros(self.max_burnin_frames, self.hidden_size))
        object_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=heads, dim_feedforward=2 * self.hidden_size, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        temporal_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=heads, dim_feedforward=2 * self.hidden_size, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.object_encoder = nn.TransformerEncoder(object_layer, num_layers=depth, enable_nested_tensor=False)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=depth, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.context_dim)
        nn.init.normal_(self.frame_position, mean=0.0, std=0.02)

    def forward(self, burnin: torch.Tensor) -> torch.Tensor:
        if burnin.ndim != 4 or burnin.shape[2:] != (self.num_objects, 2 * self.coordinate_dim):
            raise ValueError('burnin must be [batch,past_frames,num_objects,2*coordinate_dim]')
        batch, frames = burnin.shape[:2]
        if not 1 <= frames <= self.max_burnin_frames:
            raise ValueError('burnin frame count is outside the configured causal context range')
        if not bool(torch.isfinite(burnin).all()):
            raise ValueError('burnin phase must be finite')
        q = burnin[..., :self.coordinate_dim] / self.q_scale.to(burnin)
        p = burnin[..., self.coordinate_dim:] / self.p_scale.to(burnin)
        token = self.phase_projection(torch.cat([q, p], dim=-1))
        object_tokens = self.object_encoder(token.reshape(batch * frames, self.num_objects, self.hidden_size))
        frame_tokens = object_tokens.reshape(batch, frames, self.num_objects, self.hidden_size).mean(dim=2)
        ordered = frame_tokens + self.frame_position[:frames].to(frame_tokens)
        temporal = self.temporal_encoder(ordered)
        return self.output_projection(self.output_norm(temporal[:, -1]))

def broadcast_phase_context(context: torch.Tensor, *, objects: int) -> torch.Tensor:
    if context.ndim != 2 or objects < 1 or (not bool(torch.isfinite(context).all())):
        raise ValueError('phase context must be finite [batch,context_dim]')
    return context[:, None, None, :].expand(-1, objects, 1, -1)
