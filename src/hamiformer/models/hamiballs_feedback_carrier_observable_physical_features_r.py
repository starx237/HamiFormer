from __future__ import annotations
from typing import Any
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual, _finite
from hamiformer.models.hamiballs_robust_statistics_tau_monotone_geometry_r import HamiBallsTauMonotoneGeometryResidual, TauMonotoneRecoveryContext

class HamiBallsObservablePhysicalFeaturesResidual(HamiBallsTauMonotoneGeometryResidual):
    architecture = 'feedback_carrier_observable71_physical_features_unbounded_v1'
    base_full_feature_dim = 154
    observable_dim = 71
    feature_dim = base_full_feature_dim + observable_dim
    residual_hidden_dim = 32 + observable_dim

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int) -> None:
        if token_dim + 5 * state_dim + attr_dim + 3 != self.base_full_feature_dim:
            raise ValueError('FeedbackCarrier requires the canonical 128/4/3 dimensions')
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim)
        for name in ('network', 'router_network', 'router_readout', 'q_low', 'q_high', 'p_low', 'p_high', 'quality_q', 'quality_p'):
            delattr(self, name)
        del self.calibration_log_scale
        del self.calibration_bias
        self.network = nn.Sequential(nn.Linear(self.feature_dim, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.parameterization = HAMIBALLS_RESIDUAL_UNBOUNDED_V1
        self.per_object_previous_g = True
        self.gain_head = None
        self.output_scale = 1.0
        self.stop_gradient_through_recurrent_features = True
        if sum((parameter.numel() for parameter in self.parameters())) != 8420:
            raise AssertionError('FeedbackCarrier residual parameter budget drift')

    def parameter_domains(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {'residual': tuple(self.network.parameters())}

    def set_observable_statistics(self, *, state_scale: torch.Tensor, feature_mean: torch.Tensor, feature_scale: torch.Tensor, quality_q_mean: torch.Tensor, quality_q_scale: torch.Tensor, quality_p_mean: torch.Tensor, quality_p_scale: torch.Tensor) -> None:
        values = {'state_scale': state_scale, 'feature_mean': feature_mean, 'feature_scale': feature_scale, 'quality_q_mean': quality_q_mean, 'quality_q_scale': quality_q_scale, 'quality_p_mean': quality_p_mean, 'quality_p_scale': quality_p_scale}
        expected = {'state_scale': (4,), 'feature_mean': (44,), 'feature_scale': (44,), 'quality_q_mean': (13,), 'quality_q_scale': (13,), 'quality_p_mean': (13,), 'quality_p_scale': (13,)}
        for name, value in values.items():
            if value.shape != expected[name] or not bool(torch.isfinite(value).all()):
                raise ValueError(f'invalid FeedbackCarrier observable statistic {name}')
            if name.endswith('scale') and bool((value <= 0.0).any()):
                raise ValueError(f'FeedbackCarrier observable scale {name} must be positive')
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name).copy_(value.to(getattr(self, name)))
            self.statistics_fitted.fill_(1)

    def _full_features(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        return HamiBallsDTokenResidual._step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)

    def forward_step_with_context_diagnostics(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, *, context: TauMonotoneRecoveryContext | None):
        raw, next_context = self.raw_step_features_with_context(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, context=context)
        observable = self.normalized_features(raw)
        full = self._full_features(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        features = torch.cat((full, observable), dim=-1)
        if features.shape[-1] != self.feature_dim:
            raise AssertionError('FeedbackCarrier residual feature width drift')
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        innovation, gain, direction_rms, innovation_rms = self._parameterize_direction(direction, hidden, disagreement=d_candidate - h_candidate)
        gate_visible = torch.cat((hidden, observable), dim=-1)
        if gate_visible.shape[-1] != self.residual_hidden_dim:
            raise AssertionError('FeedbackCarrier gate-visible width drift')
        _finite('FeedbackCarrier observable residual hidden', gate_visible)
        return (innovation, gate_visible, gain, direction_rms, innovation_rms, next_context)

    def forward_step_with_diagnostics(self, *args: Any, **kwargs: Any):
        return self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)[:5]

    def forward_step_with_hidden(self, *args: Any, **kwargs: Any):
        value = self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)
        return (value[0], value[1])

    def forward_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)[0]

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor) -> torch.Tensor:
        if d_tokens.ndim != 4:
            raise ValueError('FeedbackCarrier d_tokens must be [B,F,K,T]')
        batch, frames, objects = d_tokens.shape[:3]
        if previous_g.shape == (batch, frames):
            gate_rows = previous_g[..., None].expand(batch, frames, objects)
        elif previous_g.shape == (batch, frames, objects):
            gate_rows = previous_g
        else:
            raise ValueError('FeedbackCarrier previous_g must be [B,F] or [B,F,K]')
        context: TauMonotoneRecoveryContext | None = None
        rows: list[torch.Tensor] = []
        for edge in range(frames):
            value = self.forward_step_with_context_diagnostics(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], gate_rows[:, edge], context=context)
            rows.append(value[0])
            context = value[5]
        return torch.stack(rows, dim=1)
__all__ = ['HamiBallsObservablePhysicalFeaturesResidual']
