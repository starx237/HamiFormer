from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from hamiformer.models.hamiballs_scaled_statistics_calibrated_local_recovery_r import HamiBallsCalibratedLocalRecoveryResidual
from hamiformer.models.hamiballs_committed import HamiBallsHPrivateResidual, _finite

@dataclass(frozen=True)
class TauMonotoneRecoveryContext:
    previous_h: torch.Tensor
    cumulative_reset: torch.Tensor
    reset_mass: torch.Tensor
    reset_energy_q2: torch.Tensor
    reset_energy_p2: torch.Tensor
    contact_provenance: torch.Tensor

class _MonotoneGeometryHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.log_slope = nn.Parameter(torch.zeros(()))
        self.geometry = nn.Sequential(nn.Linear(12, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 13:
            raise ValueError('RobustStatistics quality input must be 13-D')
        return self.log_slope.exp() * value[..., 0:1] + self.geometry(value[..., 1:])

class HamiBallsTauMonotoneGeometryResidual(HamiBallsCalibratedLocalRecoveryResidual):
    recovery_feature_dim = 44
    normalized_feature_dim = 71
    quality_feature_dim = 13

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim)
        self.router_network = nn.Sequential(nn.Linear(44, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU())
        self.router_readout = nn.Linear(24, 4)
        nn.init.zeros_(self.router_readout.weight)
        nn.init.zeros_(self.router_readout.bias)

        def adapter() -> nn.Sequential:
            result = nn.Sequential(nn.Linear(68, 2), nn.SiLU(), nn.Linear(2, 2))
            nn.init.zeros_(result[-1].weight)
            nn.init.zeros_(result[-1].bias)
            return result
        self.q_low = adapter()
        self.q_high = adapter()
        self.p_low = adapter()
        self.p_high = adapter()
        self.quality_q = _MonotoneGeometryHead()
        self.quality_p = _MonotoneGeometryHead()
        self.calibration_log_scale = nn.Parameter(torch.zeros(2))
        self.calibration_bias = nn.Parameter(torch.zeros(2))
        self.stop_gradient_through_recurrent_features = False
        del self.feature_mean
        del self.feature_scale
        self.register_buffer('feature_mean', torch.zeros(44))
        self.register_buffer('feature_scale', torch.ones(44))
        self.register_buffer('quality_q_mean', torch.zeros(13))
        self.register_buffer('quality_q_scale', torch.ones(13))
        self.register_buffer('quality_p_mean', torch.zeros(13))
        self.register_buffer('quality_p_scale', torch.ones(13))
        if sum((parameter.numel() for parameter in self.parameters())) != 4480:
            raise AssertionError('RobustStatistics parameter budget drift')

    def parameter_domains(self) -> dict[str, tuple[nn.Parameter, ...]]:
        domains = {'local': tuple(self.network.parameters()), 'router': tuple(self.router_network.parameters()) + tuple(self.router_readout.parameters()), 'quality_q': tuple(self.quality_q.parameters()), 'quality_p': tuple(self.quality_p.parameters()), 'calibrator': (self.calibration_log_scale, self.calibration_bias), 'recovery_q': tuple(self.q_low.parameters()) + tuple(self.q_high.parameters()), 'recovery_p': tuple(self.p_low.parameters()) + tuple(self.p_high.parameters())}
        flat = [parameter for rows in domains.values() for parameter in rows]
        if len(flat) != len({id(parameter) for parameter in flat}):
            raise AssertionError('RobustStatistics parameter domains overlap')
        if {id(parameter) for parameter in flat} != {id(parameter) for parameter in self.parameters()}:
            raise AssertionError('RobustStatistics parameter domains are not exhaustive')
        return domains

    def detach_field_context(self, context: TauMonotoneRecoveryContext) -> TauMonotoneRecoveryContext:
        if not isinstance(context, TauMonotoneRecoveryContext):
            raise TypeError('unexpected RobustStatistics context')
        return TauMonotoneRecoveryContext(*(value.detach() for value in (context.previous_h, context.cumulative_reset, context.reset_mass, context.reset_energy_q2, context.reset_energy_p2, context.contact_provenance)))

    def raw_step_features_with_context(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, *, context) -> tuple[torch.Tensor, TauMonotoneRecoveryContext]:
        base = HamiBallsHPrivateResidual._step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        batch, objects = previous_mixed.shape[:2]
        if context is None:
            last = torch.zeros_like(previous_mixed)
            cumulative = torch.zeros_like(previous_mixed)
            mass = previous_mixed.new_zeros(batch, objects)
            energy_q2 = previous_mixed.new_zeros(batch, objects)
            energy_p2 = previous_mixed.new_zeros(batch, objects)
            previous_provenance = previous_mixed.new_zeros(batch, objects)
        else:
            if not isinstance(context, TauMonotoneRecoveryContext):
                raise TypeError('unexpected RobustStatistics context')
            route = self._route(previous_g, batch, objects)
            last = route[..., None] * (previous_mixed - context.previous_h)
            cumulative = context.cumulative_reset + last
            mass = 1.0 - (1.0 - context.reset_mass) * (1.0 - route)
            energy_q2 = context.reset_energy_q2 + last[..., :2].square().mean(-1)
            energy_p2 = context.reset_energy_p2 + last[..., 2:].square().mean(-1)
            previous_provenance = context.contact_provenance
        disagreement = d_candidate - h_candidate
        sqsp = torch.stack((disagreement[..., :2].square().mean(-1).sqrt(), disagreement[..., 2:].square().mean(-1).sqrt()), dim=-1)
        energy = torch.stack((energy_q2.sqrt(), energy_p2.sqrt()), dim=-1)
        geometry = self._geometry(previous_mixed, attrs)
        gap = torch.minimum(geometry[..., 6], geometry[..., 9])
        radius = attrs[..., 1].clamp_min(0.0001)
        proximity = torch.exp(-gap.clamp_min(0.0) / radius)
        provenance = torch.maximum(previous_provenance, proximity).clamp(0.0, 1.0)
        next_context = TauMonotoneRecoveryContext(previous_h=h_candidate, cumulative_reset=cumulative, reset_mass=mass, reset_energy_q2=energy_q2, reset_energy_p2=energy_p2, contact_provenance=provenance)
        raw43 = torch.cat((base, last, cumulative, mass[..., None], mass[..., None] * disagreement, sqsp, energy, geometry), dim=-1)
        raw44 = torch.cat((raw43, provenance[..., None]), dim=-1)
        if raw44.shape[-1] != self.recovery_feature_dim:
            raise AssertionError('RobustStatistics raw feature width drift')
        if self.stop_gradient_through_recurrent_features:
            raw44 = raw44.detach()
            next_context = self.detach_field_context(next_context)
        return (raw44, next_context)

    @staticmethod
    def raw_quality_features(raw44: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        common = torch.cat((raw44[..., 22:23], raw44[..., 37:43], raw44[..., 35:37], raw44[..., 43:44], raw44[..., 11:12]), dim=-1)
        q = torch.cat((raw44[..., 27:28], raw44[..., 29:30], common), dim=-1)
        p = torch.cat((raw44[..., 28:29], raw44[..., 30:31], common), dim=-1)
        if q.shape[-1] != 13 or p.shape[-1] != 13:
            raise AssertionError('RobustStatistics quality feature width drift')
        return (q, p)

    def normalized_features(self, raw: torch.Tensor) -> torch.Tensor:
        core = (raw - self.feature_mean) / self.feature_scale
        q, p = self.raw_quality_features(raw)
        q = (q - self.quality_q_mean) / self.quality_q_scale
        p = (p - self.quality_p_mean) / self.quality_p_scale
        result = torch.cat((core, q, p, raw[..., 43:44]), dim=-1)
        if result.shape[-1] != self.normalized_feature_dim:
            raise AssertionError('RobustStatistics normalized feature width drift')
        return result

    def local_dimensionless(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.network(normalized[..., :14])

    def router(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.router_network(normalized[..., :44])
        return (hidden, self.router_readout(hidden))

    def quality_logits(self, normalized: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.quality_q(normalized[..., 44:57]), self.quality_p(normalized[..., 57:70])), dim=-1)

    def recovery_dimensionless(self, normalized: torch.Tensor, router_hidden: torch.Tensor, router_logit: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, 'disable_recovery_output', False)):
            return normalized.new_zeros((*normalized.shape[:-1], 4))
        del router_logit
        probability = torch.sigmoid(self.quality_logits(normalized) * self.calibration_log_scale.exp() + self.calibration_bias)
        provenance = normalized[..., 70:71].clamp(0.0, 1.0)
        adapter_input = torch.cat((normalized[..., :44], router_hidden), dim=-1)
        q = provenance * ((1.0 - probability[..., 0:1]) * self.q_low(adapter_input) + probability[..., 0:1] * self.q_high(adapter_input))
        p = provenance * ((1.0 - probability[..., 1:2]) * self.p_low(adapter_input) + probability[..., 1:2] * self.p_high(adapter_input))
        return torch.cat((q, p), dim=-1)

    def forward_step_with_context_diagnostics(self, *args: Any, context, **kwargs: Any):
        raw, next_context = self.raw_step_features_with_context(*args, context=context, **kwargs)
        normalized = self.normalized_features(raw)
        local = self.local_dimensionless(normalized)
        router_hidden, router_logit = self.router(normalized)
        dimensionless = local + self.recovery_dimensionless(normalized, router_hidden, router_logit)
        innovation = dimensionless * self.component_scale * self.output_scale
        _finite('RobustStatistics residual', innovation)
        direction_rms = self._stable_rms(dimensionless)
        innovation_rms = self._stable_rms(innovation)
        gain = innovation_rms.new_ones(innovation_rms.shape)
        return (innovation, normalized, gain, direction_rms, innovation_rms, next_context)
__all__ = ['HamiBallsTauMonotoneGeometryResidual', 'TauMonotoneRecoveryContext']
