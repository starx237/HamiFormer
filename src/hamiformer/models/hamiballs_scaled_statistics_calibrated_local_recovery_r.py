from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from hamiformer.models.hamiballs_committed import HAMIBALLS_RESIDUAL_UNBOUNDED_V1, HamiBallsDTokenResidual, HamiBallsHPrivateResidual, _finite

@dataclass(frozen=True)
class CalibratedRecoveryContext:
    previous_h: torch.Tensor
    cumulative_reset: torch.Tensor
    reset_mass: torch.Tensor
    reset_energy_q2: torch.Tensor
    reset_energy_p2: torch.Tensor

class HamiBallsCalibratedLocalRecoveryResidual(HamiBallsDTokenResidual):
    base_feature_dim = 14
    recovery_feature_dim = 43
    box_half_extent = 1.0
    wall_radius = 0.01

    def __init__(self, *, token_dim: int, state_dim: int, attr_dim: int) -> None:
        if state_dim != 4 or attr_dim != 3:
            raise ValueError('ScaledStatistics requires state/attr dimensions 4/3')
        super().__init__(token_dim=token_dim, state_dim=state_dim, attr_dim=attr_dim, hidden_size=32, parameterization=HAMIBALLS_RESIDUAL_UNBOUNDED_V1, per_object_previous_g=True)
        self.network = nn.Sequential(nn.Linear(14, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 4))
        self.router_network = nn.Sequential(nn.Linear(43, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU())
        self.router_readout = nn.Linear(24, 4)

        def adapter() -> nn.Sequential:
            result = nn.Sequential(nn.Linear(43 + 24, 2), nn.SiLU(), nn.Linear(2, 2))
            nn.init.zeros_(result[-1].weight)
            nn.init.zeros_(result[-1].bias)
            return result
        self.q_low = adapter()
        self.q_high = adapter()
        self.p_low = adapter()
        self.p_high = adapter()
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        nn.init.zeros_(self.router_readout.weight)
        nn.init.zeros_(self.router_readout.bias)
        self.calibration_log_scale = nn.Parameter(torch.zeros(4))
        self.calibration_bias = nn.Parameter(torch.zeros(4))
        self.register_buffer('state_scale', torch.ones(4))
        self.register_buffer('feature_mean', torch.zeros(43))
        self.register_buffer('feature_scale', torch.ones(43))
        self.register_buffer('component_scale', torch.ones(4))
        self.register_buffer('statistics_fitted', torch.zeros((), dtype=torch.uint8))
        if sum((parameter.numel() for parameter in self.parameters())) != 4000:
            raise AssertionError('ScaledStatistics parameter budget drift')

    def parameter_domains(self) -> dict[str, tuple[nn.Parameter, ...]]:
        domains = {'local': tuple(self.network.parameters()), 'router': tuple(self.router_network.parameters()) + tuple(self.router_readout.parameters()), 'calibrator': (self.calibration_log_scale, self.calibration_bias), 'recovery_q': tuple(self.q_low.parameters()) + tuple(self.q_high.parameters()), 'recovery_p': tuple(self.p_low.parameters()) + tuple(self.p_high.parameters())}
        flat = [parameter for rows in domains.values() for parameter in rows]
        if len({id(parameter) for parameter in flat}) != len(flat):
            raise AssertionError('ScaledStatistics parameter domains overlap')
        if {id(parameter) for parameter in flat} != {id(parameter) for parameter in self.parameters()}:
            raise AssertionError('ScaledStatistics parameter domains are not exhaustive')
        return domains

    def detach_field_context(self, context: CalibratedRecoveryContext) -> CalibratedRecoveryContext:
        if not isinstance(context, CalibratedRecoveryContext):
            raise TypeError('unexpected ScaledStatistics context')
        return CalibratedRecoveryContext(*(value.detach() for value in (context.previous_h, context.cumulative_reset, context.reset_mass, context.reset_energy_q2, context.reset_energy_p2)))

    @staticmethod
    def _route(previous_g: torch.Tensor, batch: int, objects: int) -> torch.Tensor:
        if previous_g.shape == (batch, objects):
            return 1.0 - previous_g
        if previous_g.shape == (batch,):
            return (1.0 - previous_g)[:, None].expand(batch, objects)
        raise ValueError('ScaledStatistics previous_g must be [B,K] or [B]')

    def _geometry(self, previous: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        q = previous[..., :2] * self.state_scale[:2]
        p = previous[..., 2:] * self.state_scale[2:]
        mass = attrs[..., 0, None]
        radius = attrs[..., 1]
        velocity = p / mass.clamp_min(torch.finfo(p.dtype).tiny)
        delta = q[..., :, None, :] - q[..., None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        diagonal = torch.eye(q.shape[-2], device=q.device, dtype=torch.bool)
        safe_distance = distance.clamp_min(torch.finfo(distance.dtype).eps)
        direction = delta / safe_distance[..., None]
        distance = distance.masked_fill(diagonal, torch.inf)
        surface = distance - radius[..., :, None] - radius[..., None, :]
        nearest = surface.argmin(dim=-1, keepdim=True)
        pair_gap = surface.gather(-1, nearest).squeeze(-1)
        relative_velocity = velocity[..., :, None, :] - velocity[..., None, :, :]
        closing_all = -(relative_velocity * direction).sum(dim=-1)
        pair_closing = closing_all.gather(-1, nearest).squeeze(-1)
        pair_valid = (surface > 0.0) & (closing_all > 1e-08)
        pair_denominator = torch.where(pair_valid, closing_all, torch.ones_like(closing_all))
        pair_numerator = torch.where(pair_valid, surface, torch.zeros_like(surface))
        pair_ttc = torch.where(pair_valid, pair_numerator / pair_denominator, surface.new_full((), 100.0)).amin(dim=-1)
        wall_surface = self.box_half_extent - self.wall_radius - radius[..., None] - q.abs()
        wall_axis = wall_surface.argmin(dim=-1, keepdim=True)
        wall_gap = wall_surface.gather(-1, wall_axis).squeeze(-1)
        wall_outward = q.sign() * velocity
        wall_closing = wall_outward.gather(-1, wall_axis).squeeze(-1)
        wall_valid = (wall_gap > 0.0) & (wall_closing > 1e-08)
        wall_denominator = torch.where(wall_valid, wall_closing, torch.ones_like(wall_closing))
        wall_numerator = torch.where(wall_valid, wall_gap, torch.zeros_like(wall_gap))
        wall_ttc = torch.where(wall_valid, wall_numerator / wall_denominator, wall_gap.new_full((), 100.0))
        return torch.stack((q[..., 0], q[..., 1], velocity[..., 0], velocity[..., 1], torch.linalg.vector_norm(velocity, dim=-1), torch.linalg.vector_norm(q, dim=-1), pair_gap, pair_closing, pair_ttc, wall_gap, wall_closing, wall_ttc), dim=-1)

    def raw_step_features_with_context(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, *, context) -> tuple[torch.Tensor, CalibratedRecoveryContext]:
        base = HamiBallsHPrivateResidual._step_features(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        batch, objects = previous_mixed.shape[:2]
        if context is None:
            last = torch.zeros_like(previous_mixed)
            cumulative = torch.zeros_like(previous_mixed)
            mass = previous_mixed.new_zeros(batch, objects)
            energy_q2 = previous_mixed.new_zeros(batch, objects)
            energy_p2 = previous_mixed.new_zeros(batch, objects)
        else:
            if not isinstance(context, CalibratedRecoveryContext):
                raise TypeError('unexpected ScaledStatistics context')
            route = self._route(previous_g, batch, objects)
            last = route[..., None] * (previous_mixed - context.previous_h)
            cumulative = context.cumulative_reset + last
            mass = 1.0 - (1.0 - context.reset_mass) * (1.0 - route)
            energy_q2 = context.reset_energy_q2 + last[..., :2].square().mean(-1)
            energy_p2 = context.reset_energy_p2 + last[..., 2:].square().mean(-1)
        next_context = CalibratedRecoveryContext(previous_h=h_candidate, cumulative_reset=cumulative, reset_mass=mass, reset_energy_q2=energy_q2, reset_energy_p2=energy_p2)
        disagreement = d_candidate - h_candidate
        sqsp = torch.stack((disagreement[..., :2].square().mean(-1).sqrt(), disagreement[..., 2:].square().mean(-1).sqrt()), dim=-1)
        energy = torch.stack((energy_q2.sqrt(), energy_p2.sqrt()), dim=-1)
        features = torch.cat((base, last, cumulative, mass[..., None], mass[..., None] * disagreement, sqsp, energy, self._geometry(previous_mixed, attrs)), dim=-1)
        if features.shape[-1] != self.recovery_feature_dim:
            raise AssertionError('ScaledStatistics feature width drift')
        return (features, next_context)

    def normalized_features(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.feature_mean) / self.feature_scale

    def local_dimensionless(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.network(normalized[..., :14])

    def router(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.router_network(normalized)
        return (hidden, self.router_readout(hidden))

    def recovery_dimensionless(self, normalized: torch.Tensor, router_hidden: torch.Tensor, router_logit: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(router_logit * self.calibration_log_scale.exp() + self.calibration_bias)
        adapter_input = torch.cat((normalized, router_hidden), dim=-1)
        q = probability[..., 0:1] * ((1.0 - probability[..., 2:3]) * self.q_low(adapter_input) + probability[..., 2:3] * self.q_high(adapter_input))
        p = probability[..., 1:2] * ((1.0 - probability[..., 3:4]) * self.p_low(adapter_input) + probability[..., 3:4] * self.p_high(adapter_input))
        return torch.cat((q, p), dim=-1)

    def dimensionless_from_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        local = self.local_dimensionless(normalized)
        router_hidden, router_logit = self.router(normalized)
        recovery = self.recovery_dimensionless(normalized, router_hidden, router_logit)
        return local + recovery

    def forward_step_with_context_diagnostics(self, d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, *, context):
        raw, next_context = self.raw_step_features_with_context(d_token, noisy_state, x0, previous_mixed, h_candidate, d_candidate, attrs, tau, physical_time, previous_g, context=context)
        normalized = self.normalized_features(raw)
        dimensionless = self.dimensionless_from_normalized(normalized)
        innovation = dimensionless * self.component_scale * self.output_scale
        _finite('ScaledStatistics residual', innovation)
        direction_rms = self._stable_rms(dimensionless)
        innovation_rms = self._stable_rms(innovation)
        gain = innovation_rms.new_ones(innovation_rms.shape)
        return (innovation, normalized, gain, direction_rms, innovation_rms, next_context)

    def forward_step_with_diagnostics(self, *args: Any, **kwargs: Any):
        return self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)[:5]

    def forward_step_with_hidden(self, *args: Any, **kwargs: Any):
        value = self.forward_step_with_context_diagnostics(*args, **kwargs, context=None)
        return (value[0], value[1])

    def forward_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward_step_with_hidden(*args, **kwargs)[0]

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        names = ('d_tokens', 'noisy', 'x0', 'previous_mixed', 'h_candidate', 'd_candidate', 'attrs', 'tau', 'physical_time', 'previous_g')
        values = dict(zip(names, args))
        values.update(kwargs)
        context = None
        rows = []
        for edge in range(int(values['d_tokens'].shape[1])):
            value = self.forward_step_with_context_diagnostics(values['d_tokens'][:, edge], values['noisy'][:, edge], values['x0'], values['previous_mixed'][:, edge], values['h_candidate'][:, edge], values['d_candidate'][:, edge], values['attrs'], values['tau'], values['physical_time'][:, edge], values['previous_g'][:, edge], context=context)
            rows.append(value[0])
            context = value[-1]
        return torch.stack(rows, dim=1)
__all__ = ['CalibratedRecoveryContext', 'HamiBallsCalibratedLocalRecoveryResidual']
