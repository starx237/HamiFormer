from __future__ import annotations
import torch
from torch import nn
from hamiformer.flow.rectified_flow import clean_to_velocity
from hamiformer.physics.shooting import BatchedShootingDecoder, EquivariantDirectVectorField, HamiltonianVectorField, ShootingDecodeOutput
from hamiformer.types import FieldOutput
from .common import RMSNorm, SinusoidalMLP, modulate
from .phase_dit import FactorizedPhaseBlock

class FullWindowShootingEncoder(nn.Module):

    def __init__(self, *, state_dim: int, q_dim: int, attr_dim: int, state_scale: torch.Tensor, stride: int, hidden_size: int=96, depth: int=3, num_heads: int=4, mlp_ratio: float=2.0, num_register_tokens: int=1, dropout: float=0.0, qk_norm: bool=True) -> None:
        super().__init__()
        if state_dim != 2 * q_dim:
            raise ValueError('state_dim 必须等于 2*q_dim')
        if attr_dim <= 0 or stride <= 0:
            raise ValueError('attr_dim/stride 必须为正')
        if state_scale.shape != (state_dim,) or not bool(torch.isfinite(state_scale).all().item()):
            raise ValueError('state_scale 必须为有限的 [state_dim]')
        if not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须严格为正')
        if min(hidden_size, depth, num_heads) <= 0 or hidden_size % num_heads != 0:
            raise ValueError('Transformer 维度无效')
        if num_register_tokens < 0:
            raise ValueError('num_register_tokens 不能为负')
        self.state_dim = int(state_dim)
        self.q_dim = int(q_dim)
        self.attr_dim = int(attr_dim)
        self.stride = int(stride)
        self.hidden_size = int(hidden_size)
        self.num_register_tokens = int(num_register_tokens)
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.input_projection = nn.Linear(3 * state_dim, hidden_size)
        self.object_condition_projection = nn.Linear(state_dim + attr_dim, hidden_size)
        self.rf_time_embedding = SinusoidalMLP(hidden_size)
        self.physical_time_embedding = SinusoidalMLP(hidden_size)
        self.blocks = nn.ModuleList([FactorizedPhaseBlock(hidden_size, num_heads, mlp_ratio, num_register_tokens, q_dim, dropout, qk_norm) for _ in range(depth)])
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size) * 0.02)
        else:
            self.register_tokens = None
        self.final_norm = RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.anchor_head = nn.Linear(hidden_size, state_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.zeros_(self.anchor_head.bias)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, known_values: torch.Tensor | None=None, known_mask: torch.Tensor | None=None) -> torch.Tensor:
        if noisy_future.ndim != 4 or noisy_future.shape[-1] != self.state_dim:
            raise ValueError('noisy_future 必须为 [B,F,K,state_dim]')
        batch, frames, objects, _ = noisy_future.shape
        if frames <= 0 or frames % self.stride != 0:
            raise ValueError('G1 最小实现要求 F 可被 stride 整除')
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 必须为 [B,K,state_dim]')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs 必须为 [B,K,attr_dim]')
        if physical_time.shape != (batch, frames + 1):
            raise ValueError('physical_time 必须为 [B,F+1]')
        if not bool((physical_time[:, 1:] > physical_time[:, :-1]).all().item()):
            raise ValueError('physical_time 必须严格递增')
        if known_values is None:
            known_values = torch.zeros_like(noisy_future)
        if known_mask is None:
            known_mask = torch.zeros_like(noisy_future, dtype=torch.bool)
        if known_values.shape != noisy_future.shape:
            raise ValueError('known_values shape 必须与 noisy_future 一致')
        if known_mask.shape != noisy_future.shape or known_mask.dtype != torch.bool:
            raise ValueError('known_mask 必须是与 noisy_future 同形的 bool tensor')
        scale = self.state_scale.to(noisy_future).view(1, 1, 1, -1)
        token_features = torch.cat([noisy_future / scale, known_values.to(noisy_future) / scale, known_mask.to(dtype=noisy_future.dtype)], dim=-1)
        object_condition = self.object_condition_projection(torch.cat([x0 / scale[:, 0], attrs.to(x0)], dim=-1))
        relative_time = physical_time[:, 1:] - physical_time[:, :1]
        time_embedding = self.physical_time_embedding(relative_time.reshape(-1))
        time_embedding = time_embedding.reshape(batch, frames, 1, self.hidden_size)
        tokens = self.input_projection(token_features)
        tokens = tokens + object_condition[:, None]
        tokens = tokens + time_embedding
        q_coords = noisy_future[..., :self.q_dim]
        rf_condition = self.rf_time_embedding(tau)
        if self.register_tokens is None:
            registers = tokens.new_empty(batch, 0, self.hidden_size)
        else:
            registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            tokens, registers = block(tokens, registers, rf_condition, q_coords, relative_time)
        shooting_indices = torch.arange(self.stride - 1, frames - 1, self.stride, device=noisy_future.device)
        if shooting_indices.numel() == 0:
            return noisy_future.new_empty(batch, 0, objects, self.state_dim)
        selected = tokens[:, shooting_indices]
        shift, modulation_scale = self.final_modulation(rf_condition).chunk(2, dim=-1)
        flat = selected.reshape(batch, -1, self.hidden_size)
        flat = modulate(self.final_norm(flat), shift, modulation_scale)
        normalized_anchor = self.anchor_head(flat).reshape(batch, shooting_indices.numel(), objects, self.state_dim)
        return normalized_anchor * scale

class _ShootingRectifiedFlowBase(nn.Module):

    def __init__(self, *, encoder: FullWindowShootingEncoder, decoder: BatchedShootingDecoder, t_eps: float=0.05) -> None:
        super().__init__()
        if encoder.state_dim != decoder.state_dim:
            raise ValueError('encoder/decoder state_dim 不一致')
        if encoder.attr_dim != decoder.theta_dim:
            raise ValueError('attrs 必须完整对应固定 theta_sys')
        if encoder.stride != decoder.stride:
            raise ValueError('encoder/decoder stride 不一致')
        if not 0.0 < t_eps <= 1.0:
            raise ValueError('t_eps 必须位于 (0,1]')
        self.encoder = encoder
        self.decoder = decoder
        self.t_eps = float(t_eps)

    def _decode(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, known_values: torch.Tensor | None, known_mask: torch.Tensor | None) -> tuple[torch.Tensor, ShootingDecodeOutput, torch.Tensor]:
        predicted_anchors = self.encoder(noisy_future, tau, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
        anchors = torch.cat([x0[:, None], predicted_anchors], dim=1)
        decoded = self.decoder(anchors, attrs, physical_time)
        return (decoded.trajectory[:, 1:], decoded, anchors)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, known_values: torch.Tensor | None=None, known_mask: torch.Tensor | None=None) -> FieldOutput:
        clean, decoded, anchors = self._decode(noisy_future, tau, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
        velocity = clean_to_velocity(clean, noisy_future, tau, t_eps=self.t_eps)
        scale = self.encoder.state_scale.to(clean).view(1, 1, 1, -1)
        if decoded.continuity_gaps.shape[1] == 0:
            disagreement = clean.new_zeros(clean.shape[0])
        else:
            disagreement = (decoded.continuity_gaps / scale).square().flatten(1).mean(dim=1).sqrt()
        diagnostics = {'anchors': anchors, 'continuity_gaps': decoded.continuity_gaps, 'midpoint_residuals': decoded.midpoint_residuals, 'segments': decoded.segments}
        return FieldOutput(clean=clean, velocity=velocity, disagreement=disagreement, diagnostics=diagnostics)

class HamiltonianShootingRFExpert(_ShootingRectifiedFlowBase):

    def __init__(self, *, encoder: FullWindowShootingEncoder, decoder: BatchedShootingDecoder, t_eps: float=0.05) -> None:
        if not isinstance(decoder.integrator.vector_field, HamiltonianVectorField):
            raise TypeError('H expert 必须使用 HamiltonianVectorField，禁止 direct-vector 旁路')
        super().__init__(encoder=encoder, decoder=decoder, t_eps=t_eps)

class DirectShootingRFControl(_ShootingRectifiedFlowBase):

    def __init__(self, *, encoder: FullWindowShootingEncoder, decoder: BatchedShootingDecoder, t_eps: float=0.05) -> None:
        if not isinstance(decoder.integrator.vector_field, EquivariantDirectVectorField):
            raise TypeError('direct control 必须使用 EquivariantDirectVectorField')
        super().__init__(encoder=encoder, decoder=decoder, t_eps=t_eps)
