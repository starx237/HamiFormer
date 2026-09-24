from __future__ import annotations
import math
from typing import TypeAlias
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate
FEATURE_DIM = 63
VetoHidden: TypeAlias = torch.Tensor | None

class HamiBallsComponentRiskVetoGate(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True

    def __init__(self, base_gate: HamiBallsPerObjectCompactCommittedGate, *, feature_mean: torch.Tensor, feature_std: torch.Tensor, initial_risk: float=0.01, unit_jacobian_logit: bool=False) -> None:
        super().__init__()
        if not isinstance(base_gate, HamiBallsPerObjectCompactCommittedGate):
            raise TypeError('component-risk veto requires a compact per-object base')
        if feature_mean.shape != (FEATURE_DIM,) or feature_std.shape != (FEATURE_DIM,):
            raise ValueError('component-risk veto feature statistics must be [63]')
        if not 0.0 < initial_risk < 0.5:
            raise ValueError('component-risk veto initial risk must lie in (0, 0.5)')
        if not bool(torch.isfinite(feature_mean).all()) or not bool(torch.isfinite(feature_std).all()) or bool((feature_std <= 0.0).any()):
            raise ValueError('component-risk veto feature statistics are invalid')
        self.base_gate = base_gate
        self.base_gate.requires_grad_(False)
        self.base_gate.eval()
        self.token_dim = int(base_gate.token_dim)
        self.state_dim = int(base_gate.state_dim)
        self.attr_dim = int(base_gate.attr_dim)
        self.residual_hidden_dim = int(base_gate.residual_hidden_dim)
        self.rank = int(base_gate.rank)
        self.hidden_size = int(base_gate.hidden_size)
        self.register_buffer('feature_mean', feature_mean.detach().float().clone())
        self.register_buffer('feature_std', feature_std.detach().float().clone())
        self.unit_jacobian_logit = bool(unit_jacobian_logit)
        self.initial_risk = float(initial_risk)
        self.risk_readout = nn.Linear(FEATURE_DIM, 1)
        nn.init.zeros_(self.risk_readout.weight)
        if self.unit_jacobian_logit:
            nn.init.zeros_(self.risk_readout.bias)
        else:
            nn.init.constant_(self.risk_readout.bias, math.log(initial_risk / (1.0 - initial_risk)))
        reference = next(self.base_gate.parameters())
        self.risk_readout.to(device=reference.device, dtype=reference.dtype)
        self.feature_mean.data = self.feature_mean.to(device=reference.device, dtype=reference.dtype)
        self.feature_std.data = self.feature_std.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode: bool=True) -> 'HamiBallsComponentRiskVetoGate':
        super().train(mode)
        self.base_gate.eval()
        return self

    def _risk(self, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> torch.Tensor:
        from hamiformer.training.hamiballs1.residual_objective import observable_features_step
        features = observable_features_step(h_candidate, d_candidate, hr_candidate - h_candidate, attrs, physical_time)
        normalised = (features - self.feature_mean.to(features)) / self.feature_std.to(features)
        raw = self.risk_readout(normalised).squeeze(-1)
        if self.unit_jacobian_logit:
            initial_logit = math.log(self.initial_risk / (1.0 - self.initial_risk))
            inverse_jacobian = 1.0 / (self.initial_risk * (1.0 - self.initial_risk))
            raw = initial_logit + inverse_jacobian * raw
        risk = torch.sigmoid(raw)
        if self.unit_jacobian_logit:
            epsilon = torch.finfo(risk.dtype).eps
            risk = epsilon + (1.0 - 2.0 * epsilon) * risk
        if not bool(torch.isfinite(risk).all()):
            raise FloatingPointError('component-risk veto became non-finite')
        return risk

    @staticmethod
    def value_from_risk(base_value: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        if base_value.shape != risk.shape:
            raise ValueError('component-risk veto and base gate must align')
        value = base_value * (1.0 - risk)
        if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()) or bool((value > base_value).any()):
            raise FloatingPointError('component-risk veto violated monotonicity')
        return value

    def forward_step_with_risk(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: VetoHidden=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, VetoHidden, torch.Tensor, torch.Tensor]:
        base_value, next_hidden = self.base_gate.forward_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden=hidden, residual_hidden=residual_hidden)
        risk = self._risk(h_candidate, hr_candidate, d_candidate, attrs, physical_time)
        return (self.value_from_risk(base_value, risk), next_hidden, base_value, risk)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: VetoHidden=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, VetoHidden]:
        value, next_hidden, _base, _risk = self.forward_step_with_risk(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden, residual_hidden)
        return (value, next_hidden)

    def forward_with_risk(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, VetoHidden, torch.Tensor, torch.Tensor]:
        base_value, next_hidden = self.base_gate(d_tokens, noisy, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden=residual_hidden)
        batch, frames, objects = h_candidate.shape[:3]
        expanded_attrs = attrs[:, None].expand(batch, frames, objects, attrs.shape[-1])
        risk = self._risk(h_candidate.reshape(batch * frames, objects, self.state_dim), hr_candidate.reshape(batch * frames, objects, self.state_dim), d_candidate.reshape(batch * frames, objects, self.state_dim), expanded_attrs.reshape(batch * frames, objects, attrs.shape[-1]), physical_time.reshape(batch * frames)).reshape(batch, frames, objects)
        return (self.value_from_risk(base_value, risk), next_hidden, base_value, risk)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, VetoHidden]:
        value, hidden, _base, _risk = self.forward_with_risk(d_tokens, noisy, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden=residual_hidden)
        return (value, hidden)

class HamiBallsThresholdedComponentRiskVetoGate(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True

    def __init__(self, veto_gate: HamiBallsComponentRiskVetoGate, *, threshold: float) -> None:
        super().__init__()
        if not isinstance(veto_gate, HamiBallsComponentRiskVetoGate):
            raise TypeError('thresholded veto requires a trained component-risk gate')
        if not 0.0 < threshold < 1.0:
            raise ValueError('component-risk veto threshold must lie in (0,1)')
        self.veto_gate = veto_gate
        self.veto_gate.requires_grad_(False)
        self.veto_gate.eval()
        self.threshold = float(threshold)
        for name in ('token_dim', 'state_dim', 'attr_dim', 'residual_hidden_dim', 'rank', 'hidden_size'):
            setattr(self, name, int(getattr(veto_gate, name)))

    def train(self, mode: bool=True) -> 'HamiBallsThresholdedComponentRiskVetoGate':
        super().train(mode)
        self.veto_gate.eval()
        return self

    def _decision(self, base_value: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        decision = risk >= self.threshold
        value = torch.where(decision, torch.zeros_like(base_value), base_value)
        if not bool(torch.isfinite(value).all()) or bool((value > base_value).any()):
            raise FloatingPointError('thresholded component-risk veto is not monotone')
        return value

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: VetoHidden=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, VetoHidden]:
        _soft, next_hidden, base, risk = self.veto_gate.forward_step_with_risk(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden, residual_hidden)
        return (self._decision(base, risk), next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, VetoHidden]:
        _soft, hidden, base, risk = self.veto_gate.forward_with_risk(d_tokens, noisy, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden=residual_hidden)
        return (self._decision(base, risk), hidden)
__all__ = ['HamiBallsComponentRiskVetoGate', 'HamiBallsThresholdedComponentRiskVetoGate']
