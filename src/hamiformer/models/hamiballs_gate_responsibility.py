from __future__ import annotations
from typing import TypeAlias
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate
EvidenceHidden: TypeAlias = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]

class _ResponsibilityEvidenceTower(nn.Module):

    def __init__(self, *, token_dim: int | None, geometry_dim: int, residual_hidden_dim: int | None, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.token_adapter = None if token_dim is None else nn.Linear(int(token_dim), self.rank)
        self.geometry_adapter = nn.Linear(int(geometry_dim), self.rank)
        self.residual_adapter = None if residual_hidden_dim is None else nn.Linear(int(residual_hidden_dim), self.rank)
        self.global_adapter = nn.Linear(2 * self.rank, self.rank)
        self.temporal = nn.GRU(input_size=self.rank, hidden_size=self.rank, num_layers=1, batch_first=True)
        self.output = nn.Linear(self.rank, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward_step(self, *, geometry: torch.Tensor, token: torch.Tensor | None, residual_hidden: torch.Tensor | None, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, objects = geometry.shape[:2]
        encoded = self.geometry_adapter(geometry)
        if self.token_adapter is not None:
            if token is None:
                raise ValueError('D evidence tower requires a token')
            encoded = encoded + self.token_adapter(token)
        elif token is not None:
            raise ValueError('H evidence tower does not accept a D token')
        if self.residual_adapter is not None:
            if residual_hidden is None:
                raise ValueError('H evidence tower requires residual hidden')
            encoded = encoded + self.residual_adapter(residual_hidden)
        elif residual_hidden is not None:
            raise ValueError('D evidence tower does not accept residual hidden')
        local = torch.nn.functional.silu(encoded)
        pooled = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        encoded = torch.nn.functional.silu(local + self.global_adapter(pooled)[:, None])
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('evidence hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        score = self.output(temporal[:, 0]).reshape(batch, objects)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        return (score, next_hidden)

class HamiBallsResponsibilitySeparatedGate(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True

    def __init__(self, base_gate: HamiBallsPerObjectCompactCommittedGate, *, evidence_rank: int=6) -> None:
        super().__init__()
        if not isinstance(base_gate, HamiBallsPerObjectCompactCommittedGate):
            raise TypeError('responsibility gate requires a compact per-object base')
        if type(evidence_rank) is not int or evidence_rank < 1:
            raise ValueError('evidence_rank must be a positive integer')
        self.base_gate = base_gate
        self.base_gate.requires_grad_(False)
        self.base_gate.eval()
        self.token_dim = int(base_gate.token_dim)
        self.state_dim = int(base_gate.state_dim)
        self.attr_dim = int(base_gate.attr_dim)
        self.residual_hidden_dim = int(base_gate.residual_hidden_dim)
        self.rank = int(base_gate.rank)
        self.hidden_size = int(base_gate.hidden_size)
        self.evidence_rank = int(evidence_rank)
        common_scalar = self.attr_dim + 3
        self.h_evidence = _ResponsibilityEvidenceTower(token_dim=None, geometry_dim=6 * self.state_dim + common_scalar, residual_hidden_dim=self.residual_hidden_dim, rank=self.evidence_rank)
        self.d_evidence = _ResponsibilityEvidenceTower(token_dim=self.token_dim, geometry_dim=5 * self.state_dim + common_scalar, residual_hidden_dim=None, rank=self.evidence_rank)
        reference = next(self.base_gate.parameters())
        self.h_evidence.to(device=reference.device, dtype=reference.dtype)
        self.d_evidence.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode: bool=True) -> 'HamiBallsResponsibilitySeparatedGate':
        super().train(mode)
        self.base_gate.eval()
        return self

    def _geometries(self, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, objects = previous_mixed.shape[:2]
        scalar = torch.cat([tau[:, None, None].expand(batch, objects, 1), physical_time[:, None, None].expand(batch, objects, 1), previous_g[:, :, None]], dim=-1)
        h_geometry = torch.cat([noisy_state, x0, previous_mixed, h_candidate, hr_candidate, (hr_candidate - previous_mixed).abs(), attrs, scalar], dim=-1)
        d_geometry = torch.cat([noisy_state, x0, previous_mixed, d_candidate, (d_candidate - previous_mixed).abs(), attrs, scalar], dim=-1)
        return (h_geometry, d_geometry)

    @staticmethod
    def value_from_evidence(base_value: torch.Tensor, h_score: torch.Tensor, d_score: torch.Tensor) -> torch.Tensor:
        if base_value.shape != h_score.shape or h_score.shape != d_score.shape:
            raise ValueError('base gate and responsibility evidence must align')
        evidence = h_score - d_score
        adjusted = torch.sigmoid(torch.logit(base_value) + evidence)
        straight_through_zero = adjusted - adjusted.detach()
        value = torch.where(evidence == 0, base_value + straight_through_zero, adjusted)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('responsibility-separated gate became non-finite')
        return value

    def forward_step_with_evidence(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: EvidenceHidden | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, EvidenceHidden, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('responsibility gate requires residual hidden')
        base_hidden, h_hidden, d_hidden = (None, None, None) if hidden is None else hidden
        base_value, next_base = self.base_gate.forward_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden=base_hidden, residual_hidden=residual_hidden)
        h_geometry, d_geometry = self._geometries(noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        h_score, next_h = self.h_evidence.forward_step(geometry=h_geometry, token=None, residual_hidden=residual_hidden, hidden=h_hidden)
        d_score, next_d = self.d_evidence.forward_step(geometry=d_geometry, token=d_token, residual_hidden=None, hidden=d_hidden)
        value = self.value_from_evidence(base_value, h_score, d_score)
        return (value, (next_base, next_h, next_d), base_value, h_score, d_score)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: EvidenceHidden | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, EvidenceHidden]:
        value, next_hidden, _base, _h, _d = self.forward_step_with_evidence(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden, residual_hidden)
        return (value, next_hidden)

    def forward_with_evidence(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, EvidenceHidden, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        hidden: EvidenceHidden | None = None
        values: list[torch.Tensor] = []
        bases: list[torch.Tensor] = []
        h_scores: list[torch.Tensor] = []
        d_scores: list[torch.Tensor] = []
        for edge in range(frames):
            value, hidden, base, h_score, d_score = self.forward_step_with_evidence(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running_previous_g if supplied_previous_sequence is None else supplied_previous_sequence[:, edge], hidden, residual_hidden[:, edge])
            values.append(value)
            bases.append(base)
            h_scores.append(h_score)
            d_scores.append(d_score)
            if supplied_previous_sequence is None:
                running_previous_g = value
        assert hidden is not None
        return (torch.stack(values, dim=1), hidden, torch.stack(bases, dim=1), torch.stack(h_scores, dim=1), torch.stack(d_scores, dim=1))

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, EvidenceHidden]:
        value, hidden, _base, _h, _d = self.forward_with_evidence(d_tokens, noisy, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden=residual_hidden)
        return (value, hidden)
__all__ = ['HamiBallsResponsibilitySeparatedGate']
