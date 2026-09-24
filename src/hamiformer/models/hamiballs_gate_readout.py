from __future__ import annotations
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate

class HamiBallsFrozenTrunkResidualReadout(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True

    def __init__(self, base_gate: HamiBallsPerObjectCompactCommittedGate, *, feature_mean: torch.Tensor, feature_whitener: torch.Tensor) -> None:
        super().__init__()
        if not isinstance(base_gate, HamiBallsPerObjectCompactCommittedGate):
            raise TypeError('frozen readout requires the compact per-object gate')
        rank = int(base_gate.rank)
        if feature_mean.shape != (rank,):
            raise ValueError('feature_mean must be [rank]')
        if feature_whitener.shape != (rank, rank):
            raise ValueError('feature_whitener must be [rank,rank]')
        if not bool(torch.isfinite(feature_mean).all()) or not bool(torch.isfinite(feature_whitener).all()):
            raise FloatingPointError('readout whitening statistics must be finite')
        self.base_gate = base_gate
        self.base_gate.requires_grad_(False)
        self.base_gate.eval()
        self.token_dim = int(base_gate.token_dim)
        self.state_dim = int(base_gate.state_dim)
        self.attr_dim = int(base_gate.attr_dim)
        self.residual_hidden_dim = int(base_gate.residual_hidden_dim)
        self.hidden_size = rank
        self.rank = rank
        self.register_buffer('feature_mean', feature_mean.detach().clone())
        self.register_buffer('feature_whitener', feature_whitener.detach().clone())
        self.residual_readout = nn.Linear(rank, 1)
        reference = next(self.base_gate.parameters())
        self.residual_readout.to(device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(self.residual_readout.weight)
        nn.init.zeros_(self.residual_readout.bias)

    def train(self, mode: bool=True) -> 'HamiBallsFrozenTrunkResidualReadout':
        super().train(mode)
        self.base_gate.eval()
        return self

    def transformed_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.rank:
            raise ValueError('temporal hidden width differs from readout rank')
        return torch.matmul(hidden - self.feature_mean, self.feature_whitener)

    def value_from_temporal(self, temporal: torch.Tensor) -> torch.Tensor:
        temperature = self.base_gate.log_temperature.clamp(-6.0, 6.0).exp()
        leading = temporal.shape[:-1]
        flat = temporal.reshape(-1, self.rank)
        base_logit = self.base_gate.output(flat).reshape(leading) / temperature
        residual_logit = self.residual_readout(self.transformed_hidden(flat)).reshape(leading)
        value = torch.sigmoid(base_logit + residual_logit)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('frozen-trunk residual gate became non-finite')
        return value

    def temporal_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('frozen readout requires residual_hidden')
        encoded = self.base_gate._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        batch, objects = d_token.shape[:2]
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('per-object gate hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.base_gate.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        temporal = temporal[:, 0].reshape(batch, objects, self.rank)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        return (temporal, next_hidden)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        temporal, next_hidden = self.temporal_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden, residual_hidden)
        return (self.value_from_temporal(temporal), next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d_tokens.ndim != 4 or d_tokens.shape[-1] != self.token_dim:
            raise ValueError('d_tokens must be [B,F,K,token_dim]')
        batch, frames, objects = d_tokens.shape[:3]
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
            value, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running_previous_g if supplied_previous_sequence is None else supplied_previous_sequence[:, edge], hidden, residual_hidden[:, edge])
            rows.append(value)
            if supplied_previous_sequence is None:
                running_previous_g = value
        assert hidden is not None
        return (torch.stack(rows, dim=1), hidden)
