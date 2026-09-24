from __future__ import annotations
from typing import Any
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual, _finite
from hamiformer.models.hamiballs_feedback_carrier_observable_physical_features_r import HamiBallsObservablePhysicalFeaturesResidual
from hamiformer.models.hamiballs_robust_statistics_tau_monotone_geometry_r import TauMonotoneRecoveryContext

class HamiBallsPhysicalFeaturesObservableSidecarResidual(HamiBallsObservablePhysicalFeaturesResidual):
    architecture = 'observable_sidecar_physical_features'
    residual_hidden_dim = 71

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim)
        self.feature_dim = 154
        self.network = nn.Sequential(nn.Linear(self.feature_dim, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, state_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.parameterization = HAMIBALLS_RESIDUAL_UNBOUNDED_V1
        self.per_object_previous_g = False
        self.gain_head = None
        self.output_scale = 1.0
        self.stop_gradient_through_recurrent_features = True
        if sum((parameter.numel() for parameter in self.parameters())) != 6148:
            raise AssertionError('ObservableSidecar residual parameter budget drift')

    def forward_step_with_context_diagnostics(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, *, context: TauMonotoneRecoveryContext | None):
        raw, next_context = self.raw_step_features_with_context(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, context=context)
        observable = self.normalized_features(raw)
        features = HamiBallsDTokenResidual._step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        hidden = torch.nn.functional.silu(self.network[0](features))
        hidden = torch.nn.functional.silu(self.network[2](hidden))
        direction = self.network[4](hidden)
        innovation, gain, direction_rms, innovation_rms = self._parameterize_direction(direction, hidden, disagreement=d_candidate - h_candidate)
        _finite('ObservableSidecar gate-only observable', observable)
        return (innovation, observable, gain, direction_rms, innovation_rms, next_context)

    def forward_step_with_diagnostics(self, *args: Any, **kwargs: Any):
        return self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)[:5]

    def forward_step_with_hidden(self, *args: Any, **kwargs: Any):
        value = self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)
        return (value[0], value[1])

    def forward_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)[0]
__all__ = ['HamiBallsPhysicalFeaturesObservableSidecarResidual']
