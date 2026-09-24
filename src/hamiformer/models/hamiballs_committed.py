from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
import torch
from torch import nn
HAMIBALLS_RESIDUAL_UNBOUNDED_V1 = 'unbounded_v1'
HAMIBALLS_RESIDUAL_RMS_BOUNDED_GAIN_V1 = 'rms_bounded_gain_v1'
HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_V1 = 'disagreement_radius_v1'
HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_UNIT_JACOBIAN_V1 = 'disagreement_radius_unit_jacobian_v1'
HAMIBALLS_RESIDUAL_H_STEP_RADIUS_UNIT_JACOBIAN_V1 = 'h_step_radius_unit_jacobian_v1'
_HAMIBALLS_RESIDUAL_PARAMETERIZATIONS = {HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HAMIBALLS_RESIDUAL_RMS_BOUNDED_GAIN_V1, HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_V1, HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_UNIT_JACOBIAN_V1}

def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f'{name} contains NaN/Inf')

class HamiBallsDTokenResidual(nn.Module):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=32, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=False) -> None:
        super().__init__()
        if min(token_dim, state_dim, attr_dim, hidden_size) < 1:
            raise ValueError('residual dimensions must be positive')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.hidden_size = int(hidden_size)
        self.parameterization = str(parameterization)
        self.per_object_previous_g = bool(per_object_previous_g)
        if self.parameterization not in _HAMIBALLS_RESIDUAL_PARAMETERIZATIONS:
            raise ValueError(f'unknown HamiBalls residual parameterization: {self.parameterization!r}')
        self.output_scale = 1.0
        feature_dim = self.token_dim + 5 * self.state_dim + self.attr_dim + 3
        self.network = nn.Sequential(nn.Linear(feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        if self.parameterization == HAMIBALLS_RESIDUAL_RMS_BOUNDED_GAIN_V1:
            self.gain_head: nn.Linear | None = nn.Linear(self.hidden_size, 1)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
        else:
            self.gain_head = None

    def set_output_scale(self, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('residual output scale must lie in [0,1]')
        if self.parameterization != HAMIBALLS_RESIDUAL_UNBOUNDED_V1 and value != 1.0:
            raise ValueError('canonical bounded residual parameterization forbids train/deployment output scaling')
        self.output_scale = float(value)

    @staticmethod
    def _rms_bounded_direction(direction: torch.Tensor) -> torch.Tensor:
        scale = direction.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        scaled = direction / scale
        denominator = torch.sqrt(scale.reciprocal().square() + scaled.square().mean(dim=-1, keepdim=True))
        bounded = scaled / denominator
        boundary_value = bounded.detach()
        bounded_scale = boundary_value.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        bounded_rms = bounded_scale * (boundary_value / bounded_scale).square().mean(dim=-1, keepdim=True).sqrt()
        inward = 1.0 - 4.0 * torch.finfo(bounded.dtype).eps
        correction = torch.where(bounded_rms >= 1.0, bounded_rms.new_tensor(inward) / bounded_rms.clamp_min(inward), torch.ones_like(bounded_rms))
        return bounded * correction

    @staticmethod
    def _stable_rms(value: torch.Tensor) -> torch.Tensor:
        scale = value.abs().amax(dim=-1, keepdim=True)
        safe_scale = scale.clamp_min(1.0)
        rms = safe_scale * (value / safe_scale).square().mean(dim=-1, keepdim=True).sqrt()
        return rms.squeeze(-1)

    @classmethod
    def _unit_jacobian_disagreement_radius(cls, direction: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        if radius.shape != direction.shape[:-1]:
            raise ValueError('disagreement radius must align with residual direction')
        radius_column = radius[..., None]
        scale = torch.maximum(radius_column, direction.abs().amax(dim=-1, keepdim=True))
        safe_scale = scale.clamp_min(torch.finfo(direction.dtype).tiny)
        scaled_direction = direction / safe_scale
        scaled_radius = radius_column / safe_scale
        denominator = torch.sqrt(scaled_radius.square() + scaled_direction.square().mean(dim=-1, keepdim=True)).clamp_min(torch.finfo(direction.dtype).tiny)
        innovation = radius_column * scaled_direction / denominator
        output_rms = cls._stable_rms(innovation.detach())
        inward = 1.0 - 4.0 * torch.finfo(direction.dtype).eps
        positive = radius > 0.0
        correction = torch.where(positive & (output_rms >= radius), radius.new_tensor(inward) * radius / output_rms.clamp_min(torch.finfo(output_rms.dtype).tiny), torch.ones_like(radius))
        correction = torch.where(positive, correction, torch.zeros_like(correction))
        return innovation * correction[..., None]

    def _parameterize_direction(self, direction: torch.Tensor, hidden: torch.Tensor, *, disagreement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direction_rms = self._stable_rms(direction)
        if self.parameterization == HAMIBALLS_RESIDUAL_UNBOUNDED_V1:
            innovation = direction * self.output_scale
            gain = direction_rms.new_full(direction_rms.shape, self.output_scale)
        elif self.parameterization == HAMIBALLS_RESIDUAL_RMS_BOUNDED_GAIN_V1:
            if self.gain_head is None:
                raise RuntimeError('bounded residual lacks its gain head')
            bounded_direction = self._rms_bounded_direction(direction)
            gain = torch.sigmoid(self.gain_head(hidden).squeeze(-1))
            innovation = gain[..., None] * bounded_direction
        elif self.parameterization == HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_V1:
            if disagreement.shape != direction.shape:
                raise ValueError('D-H disagreement must align with residual direction')
            bounded_direction = self._rms_bounded_direction(direction)
            gain = self._stable_rms(disagreement.detach())
            innovation = gain[..., None] * bounded_direction
        else:
            if disagreement.shape != direction.shape:
                raise ValueError('D-H disagreement must align with residual direction')
            gain = self._stable_rms(disagreement.detach())
            innovation = self._unit_jacobian_disagreement_radius(direction, gain)
        innovation_rms = self._stable_rms(innovation)
        _finite('HamiBalls residual direction RMS', direction_rms)
        _finite('HamiBalls residual gain', gain)
        _finite('HamiBalls residual innovation RMS', innovation_rms)
        _finite('HamiBalls residual innovation', innovation)
        return (innovation, gain, direction_rms, innovation_rms)

    def _step_features(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'{name} must be [B,K,state_dim]')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs must be [B,K,attr_dim]')
        if tau.shape != (batch,) or physical_time.shape != (batch,):
            raise ValueError('tau/physical_time must be [B]')
        if self.per_object_previous_g:
            if previous_g.shape != (batch, objects):
                raise ValueError('per-object previous_g must be [B,K]')
            scalar = torch.stack([tau[:, None].expand(batch, objects), physical_time[:, None].expand(batch, objects), 1.0 - previous_g], dim=-1)
        else:
            if previous_g.shape != (batch,):
                raise ValueError('previous_g must be [B]')
            scalar = torch.stack([tau, physical_time, 1.0 - previous_g], dim=-1)
            scalar = scalar[:, None, :].expand(batch, objects, 3)
        return torch.cat([d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, scalar], dim=-1)

    def forward_step_with_hidden(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        innovation, _, _, _ = self._parameterize_direction(direction, hidden, disagreement=d_candidate - h_candidate)
        return (innovation, hidden)

    def forward_step_with_diagnostics(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        innovation, gain, direction_rms, innovation_rms = self._parameterize_direction(direction, hidden, disagreement=d_candidate - h_candidate)
        return (innovation, hidden, gain, direction_rms, innovation_rms)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        innovation, _ = self.forward_step_with_hidden(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        return innovation

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_tokens.ndim != 4:
            raise ValueError('d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        expected_state = (batch, frames, objects, self.state_dim)
        for name, value in (('noisy', noisy), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('d_candidate', d_candidate)):
            if value.shape != expected_state:
                raise ValueError(f'{name} must be [B,F,K,state_dim]')
        if physical_time.shape != (batch, frames):
            raise ValueError('physical_time must be [B,F]')
        if self.per_object_previous_g:
            if previous_g.shape != (batch, frames, objects):
                raise ValueError('per-object previous_g must be [B,F,K]')
        elif previous_g.shape != (batch, frames):
            raise ValueError('previous_g must be [B,F]')
        x0_sequence = x0[:, None].expand(batch, frames, objects, self.state_dim)
        attrs_sequence = attrs[:, None].expand(batch, frames, objects, self.attr_dim)
        if self.per_object_previous_g:
            scalar = torch.stack([tau[:, None, None].expand(batch, frames, objects), physical_time[:, :, None].expand(batch, frames, objects), 1.0 - previous_g], dim=-1)
        else:
            scalar = torch.stack([tau[:, None].expand(batch, frames), physical_time, 1.0 - previous_g], dim=-1)[:, :, None].expand(batch, frames, objects, 3)
        features = torch.cat([d_tokens, noisy, x0_sequence, previous_mixed, h_candidate, d_candidate, attrs_sequence, scalar], dim=-1)
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        innovation, _, _, _ = self._parameterize_direction(direction, hidden, disagreement=d_candidate - h_candidate)
        return innovation

class HamiBallsHStepResidual(HamiBallsDTokenResidual):
    feature_dim = 15

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=66, per_object_previous_g: bool=True, trust_ratio: float=1.0) -> None:
        nn.Module.__init__(self)
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-step residual currently requires state_dim=4, attr_dim=3')
        if min(token_dim, hidden_size) < 1:
            raise ValueError('H-step residual dimensions must be positive')
        if not math.isfinite(trust_ratio) or not 0.0 < trust_ratio <= 1.0:
            raise ValueError('H-step trust ratio must lie in (0,1]')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.hidden_size = int(hidden_size)
        self.parameterization = HAMIBALLS_RESIDUAL_H_STEP_RADIUS_UNIT_JACOBIAN_V1
        self.per_object_previous_g = bool(per_object_previous_g)
        self.output_scale = 1.0
        self.gain_head = None
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim))
        self.register_buffer('feature_std', torch.ones(self.feature_dim))
        self.register_buffer('trust_ratio', torch.tensor(float(trust_ratio)))

    def set_output_scale(self, value: float) -> None:
        if value != 1.0:
            raise ValueError('H-step residual has a canonical fixed output scale')
        self.output_scale = 1.0

    def set_feature_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.feature_dim,) or std.shape != (self.feature_dim,):
            raise ValueError('H-step feature normalization has the wrong shape')
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError('H-step feature normalization must be finite')
        if bool((std <= 0.0).any()):
            raise ValueError('H-step feature std must be positive')
        with torch.no_grad():
            self.feature_mean.copy_(mean.to(self.feature_mean))
            self.feature_std.copy_(std.to(self.feature_std))

    def _h_step_features(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('H-step d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'H-step {name} has the wrong shape')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('H-step attrs has the wrong shape')
        if tau.shape != (batch,) or physical_time.shape != (batch,):
            raise ValueError('H-step tau/time has the wrong shape')
        if previous_g.shape not in ((batch,), (batch, objects)):
            raise ValueError('H-step previous_g must be [B] or [B,K]')
        step = h_candidate - previous_mixed
        norms = torch.cat([step[..., :2].norm(dim=-1, keepdim=True), step[..., 2:].norm(dim=-1, keepdim=True)], dim=-1)
        scalar = torch.stack([tau[:, None].expand(batch, objects), physical_time[:, None].expand(batch, objects)], dim=-1)
        features = torch.cat([step, step.abs(), norms, attrs, scalar], dim=-1)
        normalized = (features - self.feature_mean) / self.feature_std
        return (normalized, step)

    def _readout(self, features: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        q_radius = self._stable_rms(step[..., :2]).detach() * self.trust_ratio
        p_radius = self._stable_rms(step[..., 2:]).detach() * self.trust_ratio
        innovation = torch.cat([self._unit_jacobian_disagreement_radius(direction[..., :2], q_radius), self._unit_jacobian_disagreement_radius(direction[..., 2:], p_radius)], dim=-1)
        _finite('HamiBalls H-step residual', innovation)
        return (innovation, hidden)

    def forward_step_with_hidden(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, step = self._h_step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        return self._readout(features, step)

    def forward_step_with_diagnostics(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, step = self._h_step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        innovation, hidden = self._readout(features, step)
        gain = self._stable_rms(step).detach()
        direction_rms = self._stable_rms(self.network[4](hidden))
        innovation_rms = self._stable_rms(innovation)
        return (innovation, hidden, gain, direction_rms, innovation_rms)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        return self.forward_step_with_hidden(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)[0]

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_tokens.ndim != 4:
            raise ValueError('H-step d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        x0_sequence = x0[:, None].expand(batch, frames, objects, self.state_dim)
        attrs_sequence = attrs[:, None].expand(batch, frames, objects, self.attr_dim)
        tau_sequence = tau[:, None].expand(batch, frames).reshape(-1)
        if previous_g.shape == (batch, frames):
            flat_previous_g = previous_g.reshape(batch * frames)
        elif previous_g.shape == (batch, frames, objects):
            flat_previous_g = previous_g.reshape(batch * frames, objects)
        else:
            raise ValueError('H-step previous_g must be [B,F] or [B,F,K]')
        features, step = self._h_step_features(d_tokens.reshape(batch * frames, objects, self.token_dim), noisy.reshape(batch * frames, objects, self.state_dim), x0_sequence.reshape(batch * frames, objects, self.state_dim), previous_mixed.reshape(batch * frames, objects, self.state_dim), h_candidate.reshape(batch * frames, objects, self.state_dim), d_candidate.reshape(batch * frames, objects, self.state_dim), attrs_sequence.reshape(batch * frames, objects, self.attr_dim), tau_sequence, physical_time.reshape(-1), flat_previous_g)
        innovation, _hidden = self._readout(features, step)
        return innovation.reshape(batch, frames, objects, self.state_dim)

class HamiBallsHPrivateResidual(HamiBallsDTokenResidual):
    feature_dim = 14

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=69, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-private residual currently requires state_dim=4, attr_dim=3')
        if hidden_size != 69:
            raise ValueError('H-private residual width is fixed by parameter matching')
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _step_features(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('H-private d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'H-private {name} has the wrong shape')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('H-private attrs has the wrong shape')
        if tau.shape != (batch,) or physical_time.shape != (batch,):
            raise ValueError('H-private tau/time must be [B]')
        if self.per_object_previous_g:
            if previous_g.shape == (batch, objects):
                route = 1.0 - previous_g
            elif previous_g.shape == (batch,):
                route = (1.0 - previous_g)[:, None].expand(batch, objects)
            else:
                raise ValueError('H-private per-object previous_g must be [B,K] or [B]')
        else:
            if previous_g.shape != (batch,):
                raise ValueError('H-private previous_g must be [B]')
            route = (1.0 - previous_g)[:, None].expand(batch, objects)
        scalar = torch.stack([tau[:, None].expand(batch, objects), physical_time[:, None].expand(batch, objects), route], dim=-1)
        return torch.cat([previous_mixed, h_candidate, attrs, scalar], dim=-1)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_tokens.ndim != 4:
            raise ValueError('H-private d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        x0_sequence = x0[:, None].expand(batch, frames, objects, self.state_dim)
        attrs_sequence = attrs[:, None].expand(batch, frames, objects, self.attr_dim)
        tau_sequence = tau[:, None].expand(batch, frames).reshape(-1)
        if self.per_object_previous_g:
            if previous_g.shape == (batch, frames, objects):
                flat_previous_g = previous_g.reshape(batch * frames, objects)
            elif previous_g.shape == (batch, frames):
                flat_previous_g = previous_g.reshape(batch * frames)
            else:
                raise ValueError('H-private per-object previous_g must be [B,F,K] or [B,F]')
        else:
            if previous_g.shape != (batch, frames):
                raise ValueError('H-private previous_g must be [B,F]')
            flat_previous_g = previous_g.reshape(batch * frames)
        innovation = self.forward_step(d_tokens.reshape(batch * frames, objects, self.token_dim), noisy.reshape(batch * frames, objects, self.state_dim), x0_sequence.reshape(batch * frames, objects, self.state_dim), previous_mixed.reshape(batch * frames, objects, self.state_dim), h_candidate.reshape(batch * frames, objects, self.state_dim), d_candidate.reshape(batch * frames, objects, self.state_dim), attrs_sequence.reshape(batch * frames, objects, self.attr_dim), tau_sequence, physical_time.reshape(-1), flat_previous_g)
        return innovation.reshape(batch, frames, objects, self.state_dim)

class HamiBallsHAffineResidual(HamiBallsDTokenResidual):
    feature_dim = 14

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=67, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-affine residual currently requires state_dim=4, attr_dim=3')
        if hidden_size != 67:
            raise ValueError('H-affine residual width is fixed by parameter matching')
        if parameterization != HAMIBALLS_RESIDUAL_UNBOUNDED_V1:
            raise ValueError('H-affine residual is canonically unbounded')
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _step_features(self, *args, **kwargs) -> torch.Tensor:
        return HamiBallsHPrivateResidual._step_features(self, *args, **kwargs)

    def _readout(self, features: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        raw = self.network[4](hidden)
        additive, multiplier = raw.split(self.state_dim, dim=-1)
        innovation = (additive + multiplier * step) * self.output_scale
        _finite('HamiBalls H-affine residual', innovation)
        return (innovation, hidden, raw)

    def forward_step_with_hidden(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        innovation, hidden, _raw = self._readout(features, h_candidate - previous_mixed)
        return (innovation, hidden)

    def forward_step_with_diagnostics(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        step = h_candidate - previous_mixed
        innovation, hidden, raw = self._readout(features, step)
        return (innovation, hidden, self._stable_rms(step).detach(), self._stable_rms(raw), self._stable_rms(innovation))

    def forward_step(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> torch.Tensor:
        return self.forward_step_with_hidden(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)[0]

    def forward(self, d_tokens, noisy, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> torch.Tensor:
        if d_tokens.ndim != 4:
            raise ValueError('H-affine d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        x0_sequence = x0[:, None].expand(batch, frames, objects, self.state_dim)
        attrs_sequence = attrs[:, None].expand(batch, frames, objects, self.attr_dim)
        tau_sequence = tau[:, None].expand(batch, frames).reshape(-1)
        if self.per_object_previous_g:
            if previous_g.shape == (batch, frames, objects):
                flat_previous_g = previous_g.reshape(batch * frames, objects)
            elif previous_g.shape == (batch, frames):
                flat_previous_g = previous_g.reshape(batch * frames)
            else:
                raise ValueError('H-affine previous_g must be [B,F,K] or [B,F]')
        else:
            if previous_g.shape != (batch, frames):
                raise ValueError('H-affine previous_g must be [B,F]')
            flat_previous_g = previous_g.reshape(batch * frames)
        innovation = self.forward_step(d_tokens.reshape(batch * frames, objects, self.token_dim), noisy.reshape(batch * frames, objects, self.state_dim), x0_sequence.reshape(batch * frames, objects, self.state_dim), previous_mixed.reshape(batch * frames, objects, self.state_dim), h_candidate.reshape(batch * frames, objects, self.state_dim), d_candidate.reshape(batch * frames, objects, self.state_dim), attrs_sequence.reshape(batch * frames, objects, self.attr_dim), tau_sequence, physical_time.reshape(-1), flat_previous_g)
        return innovation.reshape(batch, frames, objects, self.state_dim)

class HamiBallsHMatrixResidual(HamiBallsHAffineResidual):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=62, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-matrix residual requires state_dim=4, attr_dim=3')
        if hidden_size != 62:
            raise ValueError('H-matrix residual width is fixed by parameter matching')
        HamiBallsDTokenResidual.__init__(self, token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        output_dim = self.state_dim + self.state_dim * self.state_dim
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, output_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _readout(self, features: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        raw = self.network[4](hidden)
        additive = raw[..., :self.state_dim]
        matrix = raw[..., self.state_dim:].reshape(*raw.shape[:-1], self.state_dim, self.state_dim)
        innovation = (additive + torch.einsum('...ij,...j->...i', matrix, step)) * self.output_scale
        _finite('HamiBalls H-matrix residual', innovation)
        return (innovation, hidden, raw)

class HamiBallsHRecoveryAffineResidual(HamiBallsHAffineResidual):
    feature_dim = 17

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=66, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-recovery affine residual requires 4D state and 3D attrs')
        if hidden_size != 66:
            raise ValueError('H-recovery affine width is fixed by parameter matching')
        HamiBallsDTokenResidual.__init__(self, token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> torch.Tensor:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('H-recovery d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'H-recovery {name} has the wrong shape')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('H-recovery attrs has the wrong shape')
        if tau.shape != (batch,) or physical_time.shape != (batch,):
            raise ValueError('H-recovery tau/time has the wrong shape')
        if previous_g.shape == (batch, objects):
            route = 1.0 - previous_g
        elif previous_g.shape == (batch,):
            route = (1.0 - previous_g)[:, None].expand(batch, objects)
        else:
            raise ValueError('H-recovery previous_g must be [B,K] or [B]')
        scalar = torch.stack([tau[:, None].expand(batch, objects), route], dim=-1)
        reset_disagreement = route[..., None] * (d_candidate - h_candidate)
        return torch.cat([previous_mixed, h_candidate, attrs, scalar, reset_disagreement], dim=-1)

class HamiBallsHRecoveryMatrixResidual(HamiBallsHRecoveryAffineResidual):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=61, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('H-recovery matrix residual requires 4D state and 3D attrs')
        if hidden_size != 61:
            raise ValueError('H-recovery matrix width is fixed by parameter matching')
        HamiBallsDTokenResidual.__init__(self, token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        output_dim = self.state_dim + self.state_dim * self.state_dim
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, output_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _readout(self, features: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        raw = self.network[4](hidden)
        additive = raw[..., :self.state_dim]
        matrix = raw[..., self.state_dim:].reshape(*raw.shape[:-1], self.state_dim, self.state_dim)
        innovation = (additive + torch.einsum('...ij,...j->...i', matrix, step)) * self.output_scale
        _finite('HamiBalls H-recovery matrix residual', innovation)
        return (innovation, hidden, raw)

class HamiBallsHRecoveryTokenAffineResidual(HamiBallsHRecoveryAffineResidual):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=32, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3 or token_dim != 128:
            raise ValueError('H-recovery-token affine requires token/state/attr 128/4/3')
        if hidden_size != 32:
            raise ValueError('H-recovery-token affine width is fixed by parameter matching')
        HamiBallsDTokenResidual.__init__(self, token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        self.feature_dim = 17 + self.token_dim
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> torch.Tensor:
        base = HamiBallsHRecoveryAffineResidual._step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        batch, objects = d_token.shape[:2]
        if previous_g.shape == (batch, objects):
            route = 1.0 - previous_g
        elif previous_g.shape == (batch,):
            route = (1.0 - previous_g)[:, None].expand(batch, objects)
        else:
            raise ValueError('H-recovery-token previous_g must be [B,K] or [B]')
        return torch.cat([base, route[..., None] * d_token], dim=-1)

class HamiBallsHConditionalRecoveryAffineResidual(HamiBallsHRecoveryTokenAffineResidual):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 3 * self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _readout(self, features: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        raw = self.network[4](hidden)
        additive, multiplier, recovery = raw.split(self.state_dim, dim=-1)
        route = features[..., 12:13]
        if bool(getattr(self, 'recovery_readout_only_training', False)):
            output = self.network[4]
            assert isinstance(output, nn.Linear)
            recovery = torch.nn.functional.linear(hidden.detach(), output.weight[8:12], output.bias[8:12])
        if bool(getattr(self, 'recovery_branch_only_training', False)):
            additive = additive.detach()
            multiplier = multiplier.detach()
        innovation = (additive + multiplier * step + route * recovery) * self.output_scale
        _finite('HamiBalls conditional-recovery affine residual', innovation)
        return (innovation, hidden, raw)

class HamiBallsHConditionalRecoveryLiteResidual(HamiBallsHRecoveryAffineResidual):
    feature_dim = 17

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=64, parameterization: str=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g: bool=True) -> None:
        if state_dim != 4 or attr_dim != 3 or hidden_size != 64:
            raise ValueError('conditional recovery lite requires state/attr/h 4/3/64')
        HamiBallsDTokenResidual.__init__(self, token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=hidden_size, parameterization=parameterization, per_object_previous_g=per_object_previous_g)
        self.network = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 3 * self.state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _step_features(self, *args, **kwargs) -> torch.Tensor:
        return HamiBallsHRecoveryAffineResidual._step_features(self, *args, **kwargs)
    _readout = HamiBallsHConditionalRecoveryAffineResidual._readout

class HamiBallsHStickyRecoveryAffineResidual(HamiBallsHAffineResidual):

    def _blend_recovery(self, learned: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        batch, objects = learned.shape[:2]
        if previous_g.shape == (batch, objects):
            keep = previous_g
        elif previous_g.shape == (batch,):
            keep = previous_g[:, None].expand(batch, objects)
        else:
            raise ValueError('sticky recovery previous_g must be [B,K] or [B]')
        return keep[..., None] * learned + (1.0 - keep[..., None]) * (d_candidate - h_candidate)

    def forward_step_with_hidden(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g) -> tuple[torch.Tensor, torch.Tensor]:
        learned, hidden = super().forward_step_with_hidden(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        return (self._blend_recovery(learned, h_candidate, d_candidate, previous_g), hidden)

    def forward_step_with_diagnostics(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g):
        features = self._step_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        step = h_candidate - previous_mixed
        learned, hidden, raw = self._readout(features, step)
        innovation = self._blend_recovery(learned, h_candidate, d_candidate, previous_g)
        return (innovation, hidden, self._stable_rms(step).detach(), self._stable_rms(raw), self._stable_rms(innovation))

class HamiBallsCommittedGate(nn.Module):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, hidden_size: int=32) -> None:
        super().__init__()
        if min(token_dim, state_dim, attr_dim, hidden_size) < 1:
            raise ValueError('gate dimensions must be positive')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.hidden_size = int(hidden_size)
        feature_dim = self.token_dim + 8 * self.state_dim + self.attr_dim + 3
        self.object_encoder = nn.Sequential(nn.Linear(feature_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU())
        self.temporal = nn.GRU(input_size=2 * self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_objects(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        batch, objects = d_token.shape[:2]
        disagreement = hr_candidate - d_candidate
        scalar = torch.stack([tau, physical_time, previous_g], dim=-1)
        scalar = scalar[:, None].expand(batch, objects, 3)
        features = torch.cat([d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, disagreement, disagreement.abs(), attrs, scalar], dim=-1)
        encoded = self.object_encoder(features)
        return torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('hr_candidate', hr_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'{name} must be [B,K,state_dim]')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs must be [B,K,attr_dim]')
        if any((value.shape != (batch,) for value in (tau, physical_time, previous_g))):
            raise ValueError('gate scalar inputs must be [B]')
        pooled = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        temporal, next_hidden = self.temporal(pooled[:, None], hidden)
        gate = torch.sigmoid(self.output(temporal[:, 0, :]).squeeze(-1))
        _finite('HamiBalls committed gate', gate)
        return (gate, next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if d_tokens.ndim != 4 or d_tokens.shape[-1] != self.token_dim:
            raise ValueError('d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 must be [B,K,state_dim]')
        pooled = []
        for edge in range(frames):
            pooled.append(self._encode_objects(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], previous_g[:, edge]))
        temporal, next_hidden = self.temporal(torch.stack(pooled, dim=1), hidden)
        gate = torch.sigmoid(self.output(temporal).squeeze(-1))
        _finite('HamiBalls committed gate', gate)
        return (gate, next_hidden)

class HamiBallsCompactCommittedGate(nn.Module):

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, residual_hidden_dim: int, rank: int=12) -> None:
        super().__init__()
        if min(token_dim, state_dim, attr_dim, residual_hidden_dim, rank) < 1:
            raise ValueError('compact gate dimensions must be positive')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.hidden_size = int(rank)
        self.rank = int(rank)
        geometry_dim = 8 * self.state_dim + self.attr_dim + 3
        self.token_adapter = nn.Linear(self.token_dim, self.rank)
        self.geometry_adapter = nn.Linear(geometry_dim, self.rank)
        self.residual_adapter = nn.Linear(self.residual_hidden_dim, self.rank)
        self.temporal = nn.GRU(input_size=2 * self.rank, hidden_size=self.rank, num_layers=1, batch_first=True)
        self.output = nn.Linear(self.rank, 1)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_objects(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        batch, objects = d_token.shape[:2]
        if residual_hidden.shape != (batch, objects, self.residual_hidden_dim):
            raise ValueError('residual_hidden must be [B,K,residual_hidden_dim]')
        disagreement = hr_candidate - d_candidate
        scalar = torch.stack([tau, physical_time, previous_g], dim=-1)
        scalar = scalar[:, None].expand(batch, objects, 3)
        geometry = torch.cat([noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, disagreement, disagreement.abs(), attrs, scalar], dim=-1)
        encoded = torch.nn.functional.silu(self.token_adapter(d_token) + self.geometry_adapter(geometry) + self.residual_adapter(residual_hidden))
        return torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('hr_candidate', hr_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'{name} must be [B,K,state_dim]')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs must be [B,K,attr_dim]')
        if any((value.shape != (batch,) for value in (tau, physical_time, previous_g))):
            raise ValueError('gate scalar inputs must be [B]')
        if residual_hidden is None:
            raise ValueError('compact gate requires residual_hidden')
        pooled = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        temporal, next_hidden = self.temporal(pooled[:, None], hidden)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(self.output(temporal[:, 0]).squeeze(-1) / temperature)
        _finite('HamiBalls compact committed gate', gate)
        return (gate, next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if d_tokens.ndim != 4 or d_tokens.shape[-1] != self.token_dim:
            raise ValueError('d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 must be [B,K,state_dim]')
        if residual_hidden is None or residual_hidden.shape != (batch, frames, objects, self.residual_hidden_dim):
            raise ValueError('compact gate residual_hidden must be [B,F,K,residual_hidden_dim]')
        pooled = [self._encode_objects(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], previous_g[:, edge], residual_hidden[:, edge]) for edge in range(frames)]
        temporal, next_hidden = self.temporal(torch.stack(pooled, dim=1), hidden)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(self.output(temporal).squeeze(-1) / temperature)
        _finite('HamiBalls compact committed gate', gate)
        return (gate, next_hidden)

class HamiBallsPerObjectCompactCommittedGate(nn.Module):
    resets_each_rf_field = True

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int, residual_hidden_dim: int, rank: int=12, candidate_step_observables: bool=False, function_preserving_candidate_step_observables: bool=False, function_preserving_pairwise_relations: bool=False) -> None:
        super().__init__()
        if min(token_dim, state_dim, attr_dim, residual_hidden_dim, rank) < 1:
            raise ValueError('per-object compact gate dimensions must be positive')
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.attr_dim = int(attr_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.hidden_size = int(rank)
        self.rank = int(rank)
        self.candidate_step_observables = bool(candidate_step_observables)
        self.function_preserving_candidate_step_observables = bool(function_preserving_candidate_step_observables)
        self.function_preserving_pairwise_relations = bool(function_preserving_pairwise_relations)
        if self.candidate_step_observables and self.function_preserving_candidate_step_observables:
            raise ValueError('candidate step observable paths are mutually exclusive')
        if (self.candidate_step_observables or self.function_preserving_candidate_step_observables) and self.state_dim % 2:
            raise ValueError('candidate q/p step observables require even state_dim')
        geometry_dim = 8 * self.state_dim + self.attr_dim + 3
        if self.candidate_step_observables:
            geometry_dim += 2 * self.state_dim + 6
        self.token_adapter = nn.Linear(self.token_dim, self.rank)
        self.geometry_adapter = nn.Linear(geometry_dim, self.rank)
        if self.function_preserving_candidate_step_observables:
            self.step_observable_adapter: nn.Linear | None = nn.Linear(2 * self.state_dim + 6, self.rank, bias=False)
            nn.init.zeros_(self.step_observable_adapter.weight)
        else:
            self.step_observable_adapter = None
        if self.function_preserving_pairwise_relations:
            pair_dim = 7 * self.state_dim + 3 * self.attr_dim
            self.pairwise_message_input: nn.Linear | None = nn.Linear(pair_dim, self.rank)
            self.pairwise_message_output: nn.Linear | None = nn.Linear(self.rank, self.rank, bias=False)
            nn.init.zeros_(self.pairwise_message_output.weight)
        else:
            self.pairwise_message_input = None
            self.pairwise_message_output = None
        self.residual_adapter = nn.Linear(self.residual_hidden_dim, self.rank)
        self.global_adapter = nn.Linear(2 * self.rank, self.rank)
        self.temporal = nn.GRU(input_size=self.rank, hidden_size=self.rank, num_layers=1, batch_first=True)
        self.output = nn.Linear(self.rank, 1)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _pairwise_relational_message(self, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        batch, objects = previous_mixed.shape[:2]
        if self.pairwise_message_input is None or self.pairwise_message_output is None:
            return previous_mixed.new_zeros(batch, objects, self.rank)
        disagreement = hr_candidate - d_candidate
        relative = [value[:, :, None] - value[:, None, :] for value in (noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, disagreement)]
        attr_i = attrs[:, :, None].expand(batch, objects, objects, self.attr_dim)
        attr_j = attrs[:, None, :].expand(batch, objects, objects, self.attr_dim)
        pair = torch.cat([*relative, attr_i, attr_j, attr_i - attr_j], dim=-1)
        message = self.pairwise_message_output(torch.nn.functional.silu(self.pairwise_message_input(pair)))
        if objects == 1:
            return message[:, :, 0] * 0.0
        mask = ~torch.eye(objects, dtype=torch.bool, device=message.device)
        return (message * mask[None, :, :, None]).sum(dim=2) / float(objects - 1)

    def _absolute_disagreement_features(self, disagreement: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        del attrs
        return disagreement.abs()

    def _encode_objects(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        batch, objects = d_token.shape[:2]
        if previous_g.shape != (batch, objects):
            raise ValueError('per-object previous_g must be [B,K]')
        if residual_hidden.shape != (batch, objects, self.residual_hidden_dim):
            raise ValueError('residual_hidden must be [B,K,residual_hidden_dim]')
        disagreement = hr_candidate - d_candidate
        scalar = torch.cat([tau[:, None, None].expand(batch, objects, 1), physical_time[:, None, None].expand(batch, objects, 1), previous_g[:, :, None]], dim=-1)
        geometry_parts = [noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, disagreement, self._absolute_disagreement_features(disagreement, attrs), attrs, scalar]
        step_observables: torch.Tensor | None = None
        if self.candidate_step_observables or self.function_preserving_candidate_step_observables:
            q_dim = self.state_dim // 2
            d_step = d_candidate - previous_mixed
            hr_step = hr_candidate - previous_mixed
            summaries = torch.cat([d_step[..., :q_dim].norm(dim=-1, keepdim=True), d_step[..., q_dim:].norm(dim=-1, keepdim=True), hr_step[..., :q_dim].norm(dim=-1, keepdim=True), hr_step[..., q_dim:].norm(dim=-1, keepdim=True), disagreement[..., :q_dim].norm(dim=-1, keepdim=True), disagreement[..., q_dim:].norm(dim=-1, keepdim=True)], dim=-1)
            step_observables = torch.cat([d_step.abs(), hr_step.abs(), summaries], dim=-1)
            if self.candidate_step_observables:
                geometry_parts.append(step_observables)
        geometry = torch.cat(geometry_parts, dim=-1)
        local_pre_activation = self.token_adapter(d_token) + self.geometry_adapter(geometry) + self.residual_adapter(residual_hidden)
        if self.step_observable_adapter is not None:
            assert step_observables is not None
            local_pre_activation = local_pre_activation + self.step_observable_adapter(step_observables)
        if self.pairwise_message_output is not None:
            local_pre_activation = local_pre_activation + self._pairwise_relational_message(noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs)
        local = torch.nn.functional.silu(local_pre_activation)
        pooled = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        return torch.nn.functional.silu(local + self.global_adapter(pooled)[:, None])

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if d_token.ndim != 3 or d_token.shape[-1] != self.token_dim:
            raise ValueError('d_token must be [B,K,token_dim]')
        batch, objects = d_token.shape[:2]
        state_shape = (batch, objects, self.state_dim)
        for name, value in (('noisy_state', noisy_state), ('x0', x0), ('previous_mixed', previous_mixed), ('h_candidate', h_candidate), ('hr_candidate', hr_candidate), ('d_candidate', d_candidate)):
            if value.shape != state_shape:
                raise ValueError(f'{name} must be [B,K,state_dim]')
        if attrs.shape != (batch, objects, self.attr_dim):
            raise ValueError('attrs must be [B,K,attr_dim]')
        if tau.shape != (batch,) or physical_time.shape != (batch,):
            raise ValueError('tau and physical_time must be [B]')
        if residual_hidden is None:
            raise ValueError('per-object compact gate requires residual_hidden')
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('per-object gate hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(self.output(temporal[:, 0]).reshape(batch, objects) / temperature)
        _finite('HamiBalls per-object compact committed gate', gate)
        return (gate, next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d_tokens.ndim != 4 or d_tokens.shape[-1] != self.token_dim:
            raise ValueError('d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 must be [B,K,state_dim]')
        if physical_time.shape != (batch, frames):
            raise ValueError('physical_time must be [B,F]')
        if residual_hidden.shape != (batch, frames, objects, self.residual_hidden_dim):
            raise ValueError('per-object residual_hidden must be [B,F,K,residual_hidden_dim]')
        supplied_previous_sequence: torch.Tensor | None = None
        if previous_g is None:
            running_previous_g = d_tokens.new_ones(batch, objects)
        elif previous_g.shape == (batch, objects):
            running_previous_g = previous_g
        elif previous_g.shape == (batch, frames, objects):
            running_previous_g = previous_g[:, 0]
            supplied_previous_sequence = previous_g
        else:
            raise ValueError('previous_g must be [B,K] or [B,F,K]')
        hidden: torch.Tensor | None = None
        rows: list[torch.Tensor] = []
        for edge in range(frames):
            gate, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running_previous_g if supplied_previous_sequence is None else supplied_previous_sequence[:, edge], hidden=hidden, residual_hidden=residual_hidden[:, edge])
            rows.append(gate)
            running_previous_g = gate
        assert hidden is not None
        return (torch.stack(rows, dim=1), hidden)

def sample_hamiballs_reset_gate(tau: torch.Tensor, *, edges: int, generator: torch.Generator, pure_probability: float=0.5, min_tau: float=0.1, max_tau: float=0.6, reset_probability: float=0.15) -> tuple[torch.Tensor, torch.Tensor]:
    if tau.ndim != 1 or edges < 1:
        raise ValueError('tau must be [B] and edges must be positive')
    if not all((math.isfinite(value) and 0.0 <= value <= 1.0 for value in (pure_probability, min_tau, max_tau, reset_probability))) or min_tau > max_tau:
        raise ValueError('invalid HamiBalls reset probabilities/tau band')
    batch = tau.shape[0]
    pure = torch.rand(batch, device=tau.device, generator=generator, dtype=tau.dtype) < pure_probability
    active = ~pure & (tau >= min_tau) & (tau <= max_tau)
    reset = (torch.rand(batch, edges, device=tau.device, generator=generator, dtype=tau.dtype) < reset_probability) & active[:, None]
    active_rows = torch.nonzero(active, as_tuple=False).flatten()
    missing_rows = active_rows[~reset[active_rows].any(dim=1)]
    if missing_rows.numel() > 0:
        forced = torch.randint(edges, (missing_rows.numel(),), device=tau.device, generator=generator)
        reset[missing_rows, forced] = True
    zero_choice = torch.rand(batch, edges, device=tau.device, generator=generator, dtype=tau.dtype) < 0.5
    continuous = torch.rand(batch, edges, device=tau.device, generator=generator, dtype=tau.dtype)
    reset_value = torch.where(zero_choice, torch.zeros_like(continuous), continuous)
    gate = torch.where(reset, reset_value, torch.ones_like(continuous))
    return (gate, pure)

def sample_hamiballs_rf_reset_schedule(tau: torch.Tensor, *, edges: int, num_steps: int, generator: torch.Generator, pure_probability: float=0.5, min_tau: float=0.1, max_tau: float=0.6, reset_probability: float=0.15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(num_steps) is not int or num_steps < 2:
        raise ValueError('num_steps must be an integer >= 2')
    gate, pure = sample_hamiballs_reset_gate(tau, edges=edges, generator=generator, pure_probability=pure_probability, min_tau=min_tau, max_tau=max_tau, reset_probability=reset_probability)
    reset_rf_tau = torch.round(tau * num_steps) / float(num_steps)
    reset_rf_tau = reset_rf_tau.clamp(0.0, 1.0)
    return (gate, pure, reset_rf_tau)

def sample_hamiballs_per_object_rf_reset_schedule(tau: torch.Tensor, *, edges: int, objects: int, num_steps: int, generator: torch.Generator, pure_probability: float=0.5, min_tau: float=0.1, max_tau: float=0.6, reset_probability: float=0.15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tau.ndim != 1 or edges < 1 or objects < 1:
        raise ValueError('tau must be [B], edges/objects must be positive')
    if type(num_steps) is not int or num_steps < 2:
        raise ValueError('num_steps must be an integer >= 2')
    if not all((math.isfinite(value) and 0.0 <= value <= 1.0 for value in (pure_probability, min_tau, max_tau, reset_probability))) or min_tau > max_tau:
        raise ValueError('invalid HamiBalls reset probabilities/tau band')
    batch = tau.shape[0]
    pure = torch.rand(batch, device=tau.device, generator=generator, dtype=tau.dtype) < pure_probability
    active = ~pure & (tau >= min_tau) & (tau <= max_tau)
    reset = (torch.rand(batch, edges, objects, device=tau.device, generator=generator, dtype=tau.dtype) < reset_probability) & active[:, None, None]
    active_rows = torch.nonzero(active, as_tuple=False).flatten()
    if active_rows.numel() > 0:
        missing = ~reset[active_rows].any(dim=1)
        missing_rows, missing_objects = torch.nonzero(missing, as_tuple=True)
        if missing_rows.numel() > 0:
            forced = torch.randint(edges, (missing_rows.numel(),), device=tau.device, generator=generator)
            reset[active_rows[missing_rows], forced, missing_objects] = True
    zero_choice = torch.rand(batch, edges, objects, device=tau.device, generator=generator, dtype=tau.dtype) < 0.5
    continuous = torch.rand(batch, edges, objects, device=tau.device, generator=generator, dtype=tau.dtype)
    reset_value = torch.where(zero_choice, torch.zeros_like(continuous), continuous)
    gate = torch.where(reset, reset_value, torch.ones_like(continuous))
    reset_rf_tau = torch.round(tau * num_steps) / float(num_steps)
    return (gate, pure, reset_rf_tau.clamp(0.0, 1.0))

@dataclass(frozen=True)
class HamiBallsCommittedRollout:
    h_candidate: torch.Tensor
    innovation: torch.Tensor
    residual_gain: torch.Tensor
    residual_direction_rms: torch.Tensor
    innovation_rms: torch.Tensor
    residual_hidden: torch.Tensor
    hr_candidate: torch.Tensor
    d_candidate: torch.Tensor
    gate: torch.Tensor
    mixed: torch.Tensor
    previous_mixed: torch.Tensor
    final_state: torch.Tensor
    final_previous_g: torch.Tensor
    gate_hidden: torch.Tensor | None
    final_residual_context: object | None = None
    base_hr_candidate: torch.Tensor | None = None
    base_gate: torch.Tensor | None = None
HamiBallsHCandidateBuilder = Callable[[int, torch.Tensor], torch.Tensor]
HamiBallsGatePolicy = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def rollout_hamiballs_committed_edges(*, h_builder: HamiBallsHCandidateBuilder, residual: HamiBallsDTokenResidual, gate: HamiBallsCommittedGate | HamiBallsCompactCommittedGate | HamiBallsPerObjectCompactCommittedGate | None, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, initial_previous_mixed: torch.Tensor | None=None, initial_previous_g: torch.Tensor | None=None, initial_gate_hidden: torch.Tensor | None=None, exogenous_gate: torch.Tensor | None=None, gate_policy: HamiBallsGatePolicy | None=None, per_object_gate_policy: bool=False, component_gate_policy: bool=False, initial_residual_context: object | None=None) -> HamiBallsCommittedRollout:
    if d_tokens.ndim != 4 or noisy.ndim != 4 or d_candidate.ndim != 4:
        raise ValueError('committed inputs must be [B,F,K,*]')
    batch, frames, objects = d_candidate.shape[:3]
    if d_candidate.shape[-1] != residual.state_dim:
        raise ValueError('D candidate state dimension differs from residual')
    if exogenous_gate is not None:
        valid_exogenous_shapes = {(batch, frames), (batch, frames, objects), (batch, frames, objects, 2)}
        if tuple(exogenous_gate.shape) not in valid_exogenous_shapes:
            raise ValueError('exogenous_gate must be [B,F], [B,F,K], or [B,F,K,2]')
    if type(per_object_gate_policy) is not bool:
        raise TypeError('per_object_gate_policy must be bool')
    if type(component_gate_policy) is not bool:
        raise TypeError('component_gate_policy must be bool')
    if component_gate_policy and (not per_object_gate_policy):
        raise ValueError('component callback requires per-object gate policy')
    supplied_policies = sum((value is not None for value in (gate, exogenous_gate, gate_policy)))
    if supplied_policies > 1:
        raise ValueError('learned, exogenous and callback gates are mutually exclusive')
    if per_object_gate_policy and gate_policy is None:
        raise ValueError('per-object callback policy requires gate_policy')
    if supplied_policies == 0:
        exogenous_gate = d_candidate.new_ones(batch, frames)
    previous_mixed = x0 if initial_previous_mixed is None else initial_previous_mixed
    if previous_mixed.shape != (batch, objects, residual.state_dim):
        raise ValueError('initial_previous_mixed must be [B,K,state_dim]')
    per_object_gate = isinstance(gate, HamiBallsPerObjectCompactCommittedGate) or bool(getattr(gate, 'per_object_gate', False)) or per_object_gate_policy or (exogenous_gate is not None and exogenous_gate.ndim in {3, 4})
    component_gate = bool(getattr(gate, 'component_gate', False)) or component_gate_policy or (exogenous_gate is not None and exogenous_gate.ndim == 4)
    if component_gate and residual.state_dim % 2:
        raise ValueError('component gate requires an even residual state dimension')
    component_history_gate = bool(component_gate and getattr(gate, 'component_history_gate', False))
    candidate_refiner = None if gate is None else getattr(gate, 'refine_candidates', None)
    begin_committed_rollout = None if gate is None else getattr(gate, 'begin_committed_rollout', None)
    if begin_committed_rollout is not None:
        begin_committed_rollout()
    previous_component_g: torch.Tensor | None = None
    if per_object_gate:
        if initial_gate_hidden is not None:
            raise ValueError('per-object reset gate does not support gate hidden')
        if component_history_gate and initial_previous_g is not None and (initial_previous_g.shape == (batch, objects, 2)):
            previous_component_g = initial_previous_g
            previous_g = previous_component_g.mean(dim=-1)
        else:
            previous_g = d_candidate.new_ones(batch, objects) if initial_previous_g is None else initial_previous_g
        if previous_g.shape != (batch, objects):
            raise ValueError('per-object initial_previous_g must be [B,K]')
        if component_history_gate and previous_component_g is None:
            previous_component_g = previous_g[..., None].expand(-1, -1, 2)
        refiner_previous_component_g = previous_g[..., None].expand(-1, -1, 2) if component_gate and candidate_refiner is not None else None
        hidden: torch.Tensor | None = None
    else:
        previous_g = d_candidate.new_ones(batch) if initial_previous_g is None else initial_previous_g
        if previous_g.shape != (batch,):
            raise ValueError('initial_previous_g must be [B]')
        hidden = initial_gate_hidden
        refiner_previous_component_g = None
    h_rows: list[torch.Tensor] = []
    r_rows: list[torch.Tensor] = []
    r_gain_rows: list[torch.Tensor] = []
    r_direction_rms_rows: list[torch.Tensor] = []
    r_rms_rows: list[torch.Tensor] = []
    r_hidden_rows: list[torch.Tensor] = []
    hr_rows: list[torch.Tensor] = []
    g_rows: list[torch.Tensor] = []
    mixed_rows: list[torch.Tensor] = []
    previous_rows: list[torch.Tensor] = []
    base_hr_rows: list[torch.Tensor] | None = [] if candidate_refiner is not None else None
    base_gate_rows: list[torch.Tensor] | None = [] if candidate_refiner is not None else None
    residual_context = initial_residual_context
    for edge in range(frames):
        previous_rows.append(previous_mixed)
        expert_previous = previous_mixed
        h_value = h_builder(edge, expert_previous)
        if h_value.shape != (batch, objects, residual.state_dim):
            raise ValueError('h_builder must return [B,K,state_dim]')
        residual_args = (d_tokens[:, edge], noisy[:, edge], x0, expert_previous, h_value, d_candidate[:, edge], attrs, tau, physical_time[:, edge], previous_g if per_object_gate and getattr(residual, 'per_object_previous_g', False) else previous_g.mean(dim=1) if per_object_gate else previous_g)
        contextual_forward = getattr(residual, 'forward_step_with_context_diagnostics', None)
        if contextual_forward is None:
            innovation, residual_hidden, residual_gain, residual_direction_rms, innovation_rms = residual.forward_step_with_diagnostics(*residual_args)
        else:
            innovation, residual_hidden, residual_gain, residual_direction_rms, innovation_rms, residual_context = contextual_forward(*residual_args, context=residual_context)
        hr_value = h_value + innovation
        if gate_policy is not None:
            g_value = gate_policy(edge, previous_mixed, h_value, hr_value, d_candidate[:, edge], previous_g)
            expected_gate_shape = (batch, objects, 2) if component_gate else (batch, objects) if per_object_gate else (batch,)
            if g_value.shape != expected_gate_shape:
                expected_name = '[B,K,2]' if component_gate else '[B,K]' if per_object_gate else '[B]'
                raise ValueError(f'gate_policy must return {expected_name}')
        elif exogenous_gate is None:
            assert gate is not None
            gate_previous_g = previous_component_g if component_history_gate else previous_g
            if gate_previous_g is None:
                raise AssertionError('dual-state gate history is missing')
            gate_args = (d_tokens[:, edge], noisy[:, edge], x0, previous_mixed, h_value, hr_value, d_candidate[:, edge], attrs, tau, physical_time[:, edge], gate_previous_g, hidden)
            if isinstance(gate, (HamiBallsCompactCommittedGate, HamiBallsPerObjectCompactCommittedGate)) or bool(getattr(gate, 'requires_residual_hidden', False)):
                g_value, hidden = gate.forward_step(*gate_args, residual_hidden=residual_hidden)
            else:
                g_value, hidden = gate.forward_step(*gate_args)
        else:
            g_value = exogenous_gate[:, edge]
        if candidate_refiner is not None:
            assert base_hr_rows is not None and base_gate_rows is not None
            base_hr_rows.append(hr_value)
            base_gate_rows.append(g_value)
            hr_value, g_value = candidate_refiner(edge=edge, d_tokens=d_tokens[:, edge], noisy=noisy[:, edge], x0=x0, previous_mixed=previous_mixed, h_candidate=h_value, hr_candidate=hr_value, d_candidate=d_candidate[:, edge], attrs=attrs, tau=tau, physical_time=physical_time[:, edge], previous_g=refiner_previous_component_g if refiner_previous_component_g is not None else gate_previous_g, residual_hidden=residual_hidden, base_gate=g_value)
            if hr_value.shape != h_value.shape:
                raise ValueError('candidate refiner changed H+r shape')
            innovation = hr_value - h_value
        if per_object_gate:
            if component_gate:
                if g_value.shape != (batch, objects, 2):
                    raise ValueError('dual q/p gate must return [B,K,2]')
                q_dim = residual.state_dim // 2
                mixed_weight = torch.cat([g_value[..., 0, None].expand(batch, objects, q_dim), g_value[..., 1, None].expand(batch, objects, residual.state_dim - q_dim)], dim=-1)
            else:
                if g_value.shape != (batch, objects):
                    raise ValueError('per-object gate must return [B,K]')
                mixed_weight = g_value[:, :, None]
        else:
            if g_value.shape != (batch,):
                raise ValueError('system gate must return [B]')
            mixed_weight = g_value[:, None, None]
        mixed = d_candidate[:, edge] + mixed_weight * (hr_value - d_candidate[:, edge])
        h_rows.append(h_value)
        r_rows.append(innovation)
        r_gain_rows.append(residual_gain)
        r_direction_rms_rows.append(residual_direction_rms)
        r_rms_rows.append(innovation_rms)
        r_hidden_rows.append(residual_hidden)
        hr_rows.append(hr_value)
        g_rows.append(g_value)
        mixed_rows.append(mixed)
        previous_mixed = mixed
        if component_history_gate:
            previous_component_g = g_value
        if refiner_previous_component_g is not None:
            refiner_previous_component_g = g_value
        previous_g = g_value.mean(dim=-1) if component_gate else g_value
    return HamiBallsCommittedRollout(h_candidate=torch.stack(h_rows, dim=1), innovation=torch.stack(r_rows, dim=1), residual_gain=torch.stack(r_gain_rows, dim=1), residual_direction_rms=torch.stack(r_direction_rms_rows, dim=1), innovation_rms=torch.stack(r_rms_rows, dim=1), residual_hidden=torch.stack(r_hidden_rows, dim=1), hr_candidate=torch.stack(hr_rows, dim=1), d_candidate=d_candidate, gate=torch.stack(g_rows, dim=1), mixed=torch.stack(mixed_rows, dim=1), previous_mixed=torch.stack(previous_rows, dim=1), final_state=previous_mixed, final_previous_g=previous_g, gate_hidden=None if per_object_gate else hidden, final_residual_context=residual_context, base_hr_candidate=torch.stack(base_hr_rows, dim=1) if base_hr_rows is not None else None, base_gate=torch.stack(base_gate_rows, dim=1) if base_gate_rows is not None else None)
__all__ = ['HAMIBALLS_RESIDUAL_DISAGREEMENT_RADIUS_V1', 'HAMIBALLS_RESIDUAL_RMS_BOUNDED_GAIN_V1', 'HAMIBALLS_RESIDUAL_UNBOUNDED_V1', 'HamiBallsCommittedGate', 'HamiBallsCompactCommittedGate', 'HamiBallsPerObjectCompactCommittedGate', 'HamiBallsGatePolicy', 'HamiBallsCommittedRollout', 'HamiBallsDTokenResidual', 'rollout_hamiballs_committed_edges', 'sample_hamiballs_reset_gate', 'sample_hamiballs_rf_reset_schedule', 'sample_hamiballs_per_object_rf_reset_schedule']
