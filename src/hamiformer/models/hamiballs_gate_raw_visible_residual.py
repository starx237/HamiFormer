from __future__ import annotations
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate

class HamiBallsRawVisibleResidualGate(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True

    def __init__(self, base_gate: HamiBallsPerObjectCompactCommittedGate, *, hidden_dim: int=8) -> None:
        super().__init__()
        if type(base_gate) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('raw-visible residual gate requires the exact compact scalar base')
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError('raw-visible residual hidden_dim must be a positive integer')
        self.base_gate = base_gate
        self.base_gate.requires_grad_(False)
        self.base_gate.eval()
        self.token_dim = int(base_gate.token_dim)
        self.state_dim = int(base_gate.state_dim)
        self.attr_dim = int(base_gate.attr_dim)
        self.residual_hidden_dim = int(base_gate.residual_hidden_dim)
        self.rank = int(base_gate.rank)
        self.hidden_size = int(base_gate.hidden_size)
        self.residual_rank = int(hidden_dim)
        geometry_dim = 8 * self.state_dim + self.attr_dim + 3
        self.raw_dim = self.token_dim + self.residual_hidden_dim + 3 * geometry_dim
        self.raw_input = nn.Linear(self.raw_dim, self.residual_rank)
        self.raw_output = nn.Linear(self.residual_rank, 1)
        nn.init.zeros_(self.raw_output.weight)
        nn.init.zeros_(self.raw_output.bias)
        self.register_buffer('raw_mean', torch.zeros(self.raw_dim))
        self.register_buffer('raw_std', torch.ones(self.raw_dim))
        self.register_buffer('normalization_fitted', torch.tensor(False))
        reference = next(self.base_gate.parameters())
        self.raw_input.to(device=reference.device, dtype=reference.dtype)
        self.raw_output.to(device=reference.device, dtype=reference.dtype)
        self.raw_mean = self.raw_mean.to(device=reference.device, dtype=reference.dtype)
        self.raw_std = self.raw_std.to(device=reference.device, dtype=reference.dtype)
        self.normalization_fitted = self.normalization_fitted.to(device=reference.device)

    def train(self, mode: bool=True) -> 'HamiBallsRawVisibleResidualGate':
        super().train(mode)
        self.base_gate.eval()
        return self

    def _geometry(self, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        batch, objects = previous_mixed.shape[:2]
        disagreement = hr_candidate - d_candidate
        scalar = torch.cat([tau[:, None, None].expand(batch, objects, 1), physical_time[:, None, None].expand(batch, objects, 1), previous_g[:, :, None]], dim=-1)
        return torch.cat([noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, disagreement, disagreement.abs(), attrs, scalar], dim=-1)

    def raw_visible(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        geometry = self._geometry(noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        system = torch.cat([geometry.mean(dim=1, keepdim=True).expand_as(geometry), geometry.amax(dim=1, keepdim=True).expand_as(geometry)], dim=-1)
        raw = torch.cat([d_token, residual_hidden, geometry, system], dim=-1)
        if raw.shape[-1] != self.raw_dim or not bool(torch.isfinite(raw).all()):
            raise ValueError('raw-visible residual feature contract changed')
        return raw

    @torch.no_grad()
    def fit_normalization(self, raw: torch.Tensor) -> None:
        if raw.ndim < 2 or raw.shape[-1] != self.raw_dim:
            raise ValueError('normalization input must end in raw_dim')
        flat = raw.reshape(-1, self.raw_dim).to(dtype=torch.float64)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(1e-06)
        self.raw_mean.copy_(mean.to(device=self.raw_mean.device, dtype=self.raw_mean.dtype))
        self.raw_std.copy_(std.to(device=self.raw_std.device, dtype=self.raw_std.dtype))
        self.normalization_fitted.fill_(True)

    @staticmethod
    def value_from_correction(base_value: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        adjusted = torch.sigmoid(torch.logit(base_value) + correction)
        straight_through_zero = adjusted - adjusted.detach()
        value = torch.where(correction == 0, base_value + straight_through_zero, adjusted)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('raw-visible residual gate became non-finite')
        return value

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('raw-visible residual gate requires residual_hidden')
        if not bool(self.normalization_fitted):
            raise ValueError('raw-visible residual normalization is not fitted')
        base_value, next_hidden = self.base_gate.forward_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden=hidden, residual_hidden=residual_hidden)
        raw = self.raw_visible(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        normalized = ((raw - self.raw_mean) / self.raw_std).clamp(-12.0, 12.0)
        correction = self.raw_output(torch.nn.functional.silu(self.raw_input(normalized))).squeeze(-1)
        return (self.value_from_correction(base_value, correction), next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects = d_tokens.shape[:3]
        supplied = None
        if previous_g is None:
            running = d_tokens.new_ones(batch, objects)
        elif previous_g.shape == (batch, objects):
            running = previous_g
        elif previous_g.shape == (batch, frames, objects):
            running = previous_g[:, 0]
            supplied = previous_g
        else:
            raise ValueError('previous_g must be [B,K] or [B,F,K]')
        hidden: torch.Tensor | None = None
        values = []
        for edge in range(frames):
            value, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running if supplied is None else supplied[:, edge], hidden=hidden, residual_hidden=residual_hidden[:, edge])
            values.append(value)
            if supplied is None:
                running = value
        assert hidden is not None
        return (torch.stack(values, dim=1), hidden)
__all__ = ['HamiBallsRawVisibleResidualGate']
