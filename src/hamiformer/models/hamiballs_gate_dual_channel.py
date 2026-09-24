from __future__ import annotations
import copy
import torch
from torch import nn
from .hamiballs_committed import HamiBallsPerObjectCompactCommittedGate, _finite

class HamiBallsDualChannelPerObjectCompactCommittedGate(HamiBallsPerObjectCompactCommittedGate):
    component_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.state_dim % 2:
            raise ValueError('dual q/p gate requires an even state_dim')
        self.output = nn.Linear(self.rank, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsDualChannelPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('dual gate expansion requires the exact scalar compact gate')
        dual = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        scalar_weight = state.pop('output.weight')
        scalar_bias = state.pop('output.bias')
        missing, unexpected = dual.load_state_dict(state, strict=False)
        if set(missing) != {'output.weight', 'output.bias'} or unexpected:
            raise ValueError('scalar-to-dual trunk state drifted')
        with torch.no_grad():
            dual.output.weight.copy_(scalar_weight.expand(2, -1))
            dual.output.bias.copy_(scalar_bias.expand(2))
        return dual

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('dual compact gate requires residual_hidden')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('dual gate hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(self.output(temporal[:, 0]).reshape(batch, objects, 2) / temperature)
        _finite('HamiBalls dual-channel compact committed gate', gate)
        return (gate, next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects = d_tokens.shape[:3]
        if previous_g is None:
            running_previous_g = d_tokens.new_ones(batch, objects)
            supplied = None
        elif previous_g.shape == (batch, objects):
            running_previous_g = previous_g
            supplied = None
        elif previous_g.shape == (batch, frames, objects):
            running_previous_g = previous_g[:, 0]
            supplied = previous_g
        elif previous_g.shape == (batch, objects, 2):
            running_previous_g = previous_g.mean(dim=-1)
            supplied = None
        elif previous_g.shape == (batch, frames, objects, 2):
            supplied = previous_g.mean(dim=-1)
            running_previous_g = supplied[:, 0]
        else:
            raise ValueError('dual previous_g must be [B,K], [B,F,K], [B,K,2] or [B,F,K,2]')
        hidden: torch.Tensor | None = None
        rows: list[torch.Tensor] = []
        for edge in range(frames):
            gate, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running_previous_g if supplied is None else supplied[:, edge], hidden=hidden, residual_hidden=residual_hidden[:, edge])
            rows.append(gate)
            running_previous_g = gate.mean(dim=-1)
        assert hidden is not None
        return (torch.stack(rows, dim=1), hidden)

class HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate(HamiBallsDualChannelPerObjectCompactCommittedGate):
    antisymmetric_component_delta_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        del self.output
        self.common_output = nn.Linear(self.rank, 1)
        self.delta_output = nn.Linear(self.rank, 1)
        nn.init.zeros_(self.common_output.weight)
        nn.init.zeros_(self.common_output.bias)
        nn.init.zeros_(self.delta_output.weight)
        nn.init.zeros_(self.delta_output.bias)

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('antisymmetric gate requires the exact scalar compact gate')
        result = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        common_weight = state.pop('output.weight')
        common_bias = state.pop('output.bias')
        missing, unexpected = result.load_state_dict(state, strict=False)
        expected_missing = {'common_output.weight', 'common_output.bias', 'delta_output.weight', 'delta_output.bias'}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('scalar-to-antisymmetric trunk state drifted')
        with torch.no_grad():
            result.common_output.weight.copy_(common_weight)
            result.common_output.bias.copy_(common_bias)
        result.freeze_common_parameters()
        return result

    def freeze_common_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.delta_output.parameters():
            parameter.requires_grad_(True)

    def _component_delta(self, temporal_feature: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        del residual_hidden
        return self.delta_output(temporal_feature)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('antisymmetric compact gate requires residual_hidden')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('antisymmetric hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        feature = temporal[:, 0]
        common = self.common_output(feature)
        delta = self._component_delta(feature, residual_hidden)
        logits = torch.cat((common + delta, common - delta), dim=-1)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(logits.reshape(batch, objects, 2) / temperature)
        _finite('HamiBalls antisymmetric-delta committed gate', gate)
        return (gate, next_hidden)

class HamiBallsObservableAntisymmetricDeltaPerObjectCompactCommittedGate(HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate):
    observable_antisymmetric_component_delta_gate = True
    delta_width = 8

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        del self.delta_output
        self.delta_input = nn.Linear(self.residual_hidden_dim, self.delta_width)
        self.delta_output = nn.Linear(self.delta_width, 1)
        nn.init.zeros_(self.delta_output.weight)
        nn.init.zeros_(self.delta_output.bias)

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsObservableAntisymmetricDeltaPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('observable antisymmetric gate requires exact scalar gate')
        result = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        common_weight = state.pop('output.weight')
        common_bias = state.pop('output.bias')
        missing, unexpected = result.load_state_dict(state, strict=False)
        expected_missing = {'common_output.weight', 'common_output.bias', 'delta_input.weight', 'delta_input.bias', 'delta_output.weight', 'delta_output.bias'}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('scalar-to-observable-antisymmetric state drifted')
        with torch.no_grad():
            result.common_output.weight.copy_(common_weight)
            result.common_output.bias.copy_(common_bias)
        result.freeze_common_parameters()
        return result

    def freeze_common_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.delta_input, self.delta_output):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def _component_delta(self, temporal_feature: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        del temporal_feature
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        return self.delta_output(torch.nn.functional.silu(self.delta_input(observable)))

class HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate(HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate):
    observable_disjoint_component_delta_gate = True
    delta_width = 4

    def __init__(self, *, delta_width: int=4, **kwargs: object) -> None:
        if delta_width < 1:
            raise ValueError('observable-disjoint delta width must be positive')
        self.delta_width = int(delta_width)
        super().__init__(**kwargs)
        del self.delta_output
        self.q_delta_input = nn.Linear(self.residual_hidden_dim, self.delta_width, bias=False)
        self.q_delta_output = nn.Linear(self.delta_width, 1)
        self.p_delta_input = nn.Linear(self.residual_hidden_dim, self.delta_width, bias=False)
        self.p_delta_output = nn.Linear(self.delta_width, 1)
        for output in (self.q_delta_output, self.p_delta_output):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate, *, delta_width: int=4) -> 'HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('observable disjoint gate requires exact scalar gate')
        result = cls(delta_width=delta_width, token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        common_weight = state.pop('output.weight')
        common_bias = state.pop('output.bias')
        missing, unexpected = result.load_state_dict(state, strict=False)
        expected_missing = {'common_output.weight', 'common_output.bias', 'q_delta_input.weight', 'q_delta_output.weight', 'q_delta_output.bias', 'p_delta_input.weight', 'p_delta_output.weight', 'p_delta_output.bias'}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('scalar-to-observable-disjoint state drifted')
        with torch.no_grad():
            result.common_output.weight.copy_(common_weight)
            result.common_output.bias.copy_(common_bias)
        result.freeze_common_parameters()
        return result

    def freeze_common_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.q_delta_input, self.q_delta_output, self.p_delta_input, self.p_delta_output):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def _component_deltas(self, residual_hidden: torch.Tensor, temporal_hidden: torch.Tensor | None=None) -> torch.Tensor:
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        q_delta = self.q_delta_output(torch.nn.functional.silu(self.q_delta_input(observable)))
        p_delta = self.p_delta_output(torch.nn.functional.silu(self.p_delta_input(observable)))
        return torch.cat((q_delta, p_delta), dim=-1)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('observable disjoint compact gate requires residual_hidden')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('observable disjoint hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        common = self.common_output(temporal[:, 0])
        logits = common + self._component_deltas(residual_hidden, temporal_hidden=temporal[:, 0])
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(logits.reshape(batch, objects, 2) / temperature)
        _finite('HamiBalls observable-disjoint committed gate', gate)
        return (gate, next_hidden)

class HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate(HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate):
    staged_frozen_local_recovery_gate = True
    recovery_width = 16

    def __init__(self, *, recovery_width: int=16, recovery_reads_temporal: bool=False, **kwargs: object) -> None:
        if recovery_width < 1:
            raise ValueError('staged recovery width must be positive')
        self.recovery_width = int(recovery_width)
        self.recovery_reads_temporal = bool(recovery_reads_temporal)
        super().__init__(**kwargs)
        recovery_input_dim = self.residual_hidden_dim + (self.rank if self.recovery_reads_temporal else 0)
        self.q_recovery_input = nn.Linear(recovery_input_dim, self.recovery_width, bias=False)
        self.q_recovery_output = nn.Linear(self.recovery_width, 1)
        self.p_recovery_input = nn.Linear(recovery_input_dim, self.recovery_width, bias=False)
        self.p_recovery_output = nn.Linear(self.recovery_width, 1)
        for output in (self.q_recovery_output, self.p_recovery_output):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @classmethod
    def from_disjoint(cls, base: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, *, recovery_width: int=16, recovery_reads_temporal: bool=False) -> 'HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate:
            raise TypeError('staged recovery requires an exact disjoint local gate')
        result = cls(delta_width=base.delta_width, recovery_width=recovery_width, recovery_reads_temporal=recovery_reads_temporal, token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.common_output.weight.device, dtype=base.common_output.weight.dtype)
        missing, unexpected = result.load_state_dict(copy.deepcopy(base.state_dict()), strict=False)
        expected_missing = {f'{component}_recovery_{layer}.{parameter}' for component in ('q', 'p') for layer, parameters in (('input', ('weight',)), ('output', ('weight', 'bias'))) for parameter in parameters}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('disjoint-to-staged-recovery state drifted')
        result.freeze_local_parameters()
        return result

    def freeze_local_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.q_recovery_input, self.q_recovery_output, self.p_recovery_input, self.p_recovery_output):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def _recovery_features(self, residual_hidden: torch.Tensor, temporal_hidden: torch.Tensor | None) -> torch.Tensor:
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        if not self.recovery_reads_temporal:
            return observable
        if temporal_hidden is None:
            raise ValueError('temporal-context recovery requires temporal hidden')
        temporal = temporal_hidden.reshape(-1, self.rank)
        if temporal.shape[0] != observable.shape[0]:
            raise ValueError('temporal-context recovery rows changed')
        return torch.cat((observable, temporal), dim=-1)

    def _component_deltas(self, residual_hidden: torch.Tensor, temporal_hidden: torch.Tensor | None=None) -> torch.Tensor:
        local = super()._component_deltas(residual_hidden, temporal_hidden=temporal_hidden)
        observable = self._recovery_features(residual_hidden, temporal_hidden)
        q_recovery = self.q_recovery_output(torch.nn.functional.silu(self.q_recovery_input(observable)))
        p_recovery = self.p_recovery_output(torch.nn.functional.silu(self.p_recovery_input(observable)))
        return local + torch.cat((q_recovery, p_recovery), dim=-1)

class HamiBallsStagedFrozenRiskPositiveConeCapPerObjectCompactCommittedGate(HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate):
    staged_frozen_risk_positive_cone_cap_gate = True

    def _component_deltas(self, residual_hidden: torch.Tensor, temporal_hidden: torch.Tensor | None=None) -> torch.Tensor:
        local = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate._component_deltas(self, residual_hidden, temporal_hidden=temporal_hidden)
        observable = self._recovery_features(residual_hidden, temporal_hidden)
        q_amount = self.q_recovery_output(torch.nn.functional.softplus(self.q_recovery_input(observable)))
        p_amount = self.p_recovery_output(torch.nn.functional.softplus(self.p_recovery_input(observable)))
        amount = torch.cat((q_amount, p_amount), dim=-1)
        return local - amount

    @torch.no_grad()
    def project_cap_parameters_(self) -> None:
        for output in (self.q_recovery_output, self.p_recovery_output):
            output.weight.clamp_(min=0.0)
            output.bias.clamp_(min=0.0)

class HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate(HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate):
    observable_regime_conditioned_gate = True
    expert_width = 4
    local_expert_width = 4
    recurrent_expert_width = 4
    router_width = 4

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        del self.delta_output

        def expert(width: int) -> tuple[nn.Linear, nn.Linear]:
            inner = nn.Linear(self.residual_hidden_dim, width, bias=False)
            output = nn.Linear(width, 1)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            return (inner, output)

        def router() -> tuple[nn.Linear, nn.Linear]:
            inner = nn.Linear(14, self.router_width, bias=False)
            output = nn.Linear(self.router_width, 1)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            return (inner, output)
        self.q_local_input, self.q_local_output = expert(self.local_expert_width)
        self.q_recurrent_input, self.q_recurrent_output = expert(self.recurrent_expert_width)
        self.q_router_input, self.q_router_output = router()
        self.p_local_input, self.p_local_output = expert(self.local_expert_width)
        self.p_recurrent_input, self.p_recurrent_output = expert(self.recurrent_expert_width)
        self.p_router_input, self.p_router_output = router()

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('regime-conditioned gate requires exact scalar gate')
        result = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        common_weight = state.pop('output.weight')
        common_bias = state.pop('output.bias')
        missing, unexpected = result.load_state_dict(state, strict=False)
        expected_missing = {'common_output.weight', 'common_output.bias', *{f'{component}_{role}_{layer}.{parameter}' for component in ('q', 'p') for role in ('local', 'recurrent', 'router') for layer, parameters in (('input', ('weight',)), ('output', ('weight', 'bias'))) for parameter in parameters}}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('scalar-to-regime-conditioned state drifted')
        with torch.no_grad():
            result.common_output.weight.copy_(common_weight)
            result.common_output.bias.copy_(common_bias)
        result.freeze_common_parameters()
        return result

    def freeze_common_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if any((name.startswith(f'{component}_{role}_') for component in ('q', 'p') for role in ('local', 'recurrent', 'router'))):
                parameter.requires_grad_(True)

    @staticmethod
    def _mlp(value: torch.Tensor, inner: nn.Linear, output: nn.Linear) -> torch.Tensor:
        return output(torch.nn.functional.silu(inner(value)))

    def _component_value(self, observable: torch.Tensor, *, component: str) -> torch.Tensor:
        if component == 'q':
            quality = torch.cat((observable[..., 44:57], observable[..., 70:71]), -1)
        elif component == 'p':
            quality = torch.cat((observable[..., 57:70], observable[..., 70:71]), -1)
        else:
            raise ValueError(f'unknown regime component {component}')
        local = self._mlp(observable, getattr(self, f'{component}_local_input'), getattr(self, f'{component}_local_output'))
        recurrent = self._mlp(observable, getattr(self, f'{component}_recurrent_input'), getattr(self, f'{component}_recurrent_output'))
        probability = torch.sigmoid(self._mlp(quality, getattr(self, f'{component}_router_input'), getattr(self, f'{component}_router_output')))
        return (1.0 - probability) * local + probability * recurrent

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None or self.residual_hidden_dim != 71:
            raise ValueError('regime-conditioned gate requires observable71')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('regime-conditioned hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        common = self.common_output(temporal[:, 0])
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        delta = torch.cat((self._component_value(observable, component='q'), self._component_value(observable, component='p')), dim=-1)
        logits = common + delta
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(logits.reshape(batch, objects, 2) / temperature)
        _finite('HamiBalls regime-conditioned committed gate', gate)
        return (gate, next_hidden)

class HamiBallsObservableAdditiveRecoveryPerObjectCompactCommittedGate(HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate):
    observable_additive_recovery_gate = True

    def _component_value(self, observable: torch.Tensor, *, component: str) -> torch.Tensor:
        if component == 'q':
            quality = torch.cat((observable[..., 44:57], observable[..., 70:71]), -1)
        elif component == 'p':
            quality = torch.cat((observable[..., 57:70], observable[..., 70:71]), -1)
        else:
            raise ValueError(f'unknown additive-recovery component {component}')
        local = self._mlp(observable, getattr(self, f'{component}_local_input'), getattr(self, f'{component}_local_output'))
        recurrent = self._mlp(observable, getattr(self, f'{component}_recurrent_input'), getattr(self, f'{component}_recurrent_output'))
        probability = torch.sigmoid(self._mlp(quality, getattr(self, f'{component}_router_input'), getattr(self, f'{component}_router_output')))
        return local + probability * recurrent

class HamiBallsObservablePolicyMixturePerObjectCompactCommittedGate(HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate):
    observable_policy_mixture_gate = True

    @classmethod
    def from_disjoint_pair(cls, local_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, recurrent_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate) -> 'HamiBallsObservablePolicyMixturePerObjectCompactCommittedGate':
        expected = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate
        if type(local_gate) is not expected or type(recurrent_gate) is not expected:
            raise TypeError('policy mixture requires two exact observable-disjoint gates')
        if local_gate.delta_width != cls.local_expert_width or recurrent_gate.delta_width != cls.recurrent_expert_width:
            raise ValueError('policy mixture expert widths changed')
        for name in ('token_dim', 'state_dim', 'attr_dim', 'residual_hidden_dim', 'rank'):
            if getattr(local_gate, name) != getattr(recurrent_gate, name):
                raise ValueError(f'policy mixture expert {name} changed')
        result = cls(token_dim=recurrent_gate.token_dim, state_dim=recurrent_gate.state_dim, attr_dim=recurrent_gate.attr_dim, residual_hidden_dim=recurrent_gate.residual_hidden_dim, rank=recurrent_gate.rank, candidate_step_observables=recurrent_gate.candidate_step_observables, function_preserving_candidate_step_observables=recurrent_gate.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=recurrent_gate.function_preserving_pairwise_relations).to(device=recurrent_gate.common_output.weight.device, dtype=recurrent_gate.common_output.weight.dtype)
        local_state = local_gate.state_dict()
        recurrent_state = recurrent_gate.state_dict()
        state = result.state_dict()
        expert_map = {f'{component}_local_{layer}.{parameter}': f'{component}_delta_{layer}.{parameter}' for component in ('q', 'p') for layer, parameters in (('input', ('weight',)), ('output', ('weight', 'bias'))) for parameter in parameters}
        expert_map.update({f'{component}_recurrent_{layer}.{parameter}': f'{component}_delta_{layer}.{parameter}' for component in ('q', 'p') for layer, parameters in (('input', ('weight',)), ('output', ('weight', 'bias'))) for parameter in parameters})
        for name in tuple(state):
            if '_router_' in name:
                continue
            source_name = expert_map.get(name, name)
            source = local_state if '_local_' in name else recurrent_state
            if source_name not in source:
                raise ValueError(f'policy mixture source state lacks {source_name}')
            if name not in expert_map:
                if source_name not in local_state or not torch.equal(local_state[source_name], recurrent_state[source_name]):
                    raise ValueError(f'policy mixture shared state differs at {source_name}')
            state[name] = source[source_name].detach().clone()
        result.load_state_dict(state, strict=True)
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        for component in ('q', 'p'):
            for role in ('input', 'output'):
                for parameter in getattr(result, f'{component}_router_{role}').parameters():
                    parameter.requires_grad_(True)
        return result

class HamiBallsFullObservablePolicyMixturePerObjectCompactCommittedGate(HamiBallsObservablePolicyMixturePerObjectCompactCommittedGate):
    full_observable_policy_mixture_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        for component in ('q', 'p'):
            inner = nn.Linear(self.residual_hidden_dim, self.router_width, bias=False).to(device=self.common_output.weight.device, dtype=self.common_output.weight.dtype)
            setattr(self, f'{component}_router_input', inner)

    def _component_value(self, observable: torch.Tensor, *, component: str) -> torch.Tensor:
        if component not in {'q', 'p'}:
            raise ValueError(f'unknown full-observable component {component}')
        local = self._mlp(observable, getattr(self, f'{component}_local_input'), getattr(self, f'{component}_local_output'))
        recurrent = self._mlp(observable, getattr(self, f'{component}_recurrent_input'), getattr(self, f'{component}_recurrent_output'))
        probability = torch.sigmoid(self._mlp(observable, getattr(self, f'{component}_router_input'), getattr(self, f'{component}_router_output')))
        return (1.0 - probability) * local + probability * recurrent

class HamiBallsAsymmetricFullObservablePolicyMixturePerObjectCompactCommittedGate(HamiBallsFullObservablePolicyMixturePerObjectCompactCommittedGate):
    asymmetric_full_observable_policy_mixture_gate = True
    recurrent_expert_width = 16

class HamiBallsProjectionSupportedRecurrentPolicyPerObjectCompactCommittedGate(HamiBallsFullObservablePolicyMixturePerObjectCompactCommittedGate):
    projection_supported_recurrent_policy_gate = True

    @staticmethod
    def _projection_supported_value(local_gate: torch.Tensor, recurrent_gate: torch.Tensor) -> torch.Tensor:
        if local_gate.shape != recurrent_gate.shape:
            raise ValueError('local and recurrent gate shapes must match')
        safety_support = (2.0 * local_gate).clamp_max(1.0)
        return recurrent_gate * safety_support

    @classmethod
    def from_disjoint_pair(cls, local_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, recurrent_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate) -> 'HamiBallsProjectionSupportedRecurrentPolicyPerObjectCompactCommittedGate':
        result = super().from_disjoint_pair(local_gate, recurrent_gate)
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None or self.residual_hidden_dim != 71:
            raise ValueError('projection-supported gate requires observable71')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('projection-supported hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        common = self.common_output(temporal[:, 0])
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        local_delta = torch.cat(tuple((self._mlp(observable, getattr(self, f'{component}_local_input'), getattr(self, f'{component}_local_output')) for component in ('q', 'p'))), dim=-1)
        recurrent_delta = torch.cat(tuple((self._mlp(observable, getattr(self, f'{component}_recurrent_input'), getattr(self, f'{component}_recurrent_output')) for component in ('q', 'p'))), dim=-1)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        local_gate = torch.sigmoid((common + local_delta).reshape(batch, objects, 2) / temperature)
        recurrent_gate = torch.sigmoid((common + recurrent_delta).reshape(batch, objects, 2) / temperature)
        gate = self._projection_supported_value(local_gate, recurrent_gate)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        _finite('HamiBalls projection-supported committed gate', gate)
        return (gate, next_hidden)

class HamiBallsProjectionSupportedRiskRecoveryPerObjectCompactCommittedGate(HamiBallsStagedFrozenLocalRecoveryPerObjectCompactCommittedGate):
    projection_supported_risk_recovery_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.q_safety_input = nn.Linear(self.residual_hidden_dim, self.delta_width, bias=False)
        self.q_safety_output = nn.Linear(self.delta_width, 1)
        self.p_safety_input = nn.Linear(self.residual_hidden_dim, self.delta_width, bias=False)
        self.p_safety_output = nn.Linear(self.delta_width, 1)

    @classmethod
    def from_disjoint_pair(cls, safety_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, risk_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, *, recovery_width: int=16) -> 'HamiBallsProjectionSupportedRiskRecoveryPerObjectCompactCommittedGate':
        expected = HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate
        if type(safety_gate) is not expected or type(risk_gate) is not expected:
            raise TypeError('projection-supported recovery requires exact disjoint gates')
        if safety_gate.delta_width != risk_gate.delta_width:
            raise ValueError('projection-supported parent widths differ')
        for name in ('token_dim', 'state_dim', 'attr_dim', 'residual_hidden_dim', 'rank'):
            if getattr(safety_gate, name) != getattr(risk_gate, name):
                raise ValueError(f'projection-supported parent {name} differs')
        result = cls(delta_width=risk_gate.delta_width, recovery_width=recovery_width, token_dim=risk_gate.token_dim, state_dim=risk_gate.state_dim, attr_dim=risk_gate.attr_dim, residual_hidden_dim=risk_gate.residual_hidden_dim, rank=risk_gate.rank, candidate_step_observables=risk_gate.candidate_step_observables, function_preserving_candidate_step_observables=risk_gate.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=risk_gate.function_preserving_pairwise_relations).to(device=risk_gate.common_output.weight.device, dtype=risk_gate.common_output.weight.dtype)
        missing, unexpected = result.load_state_dict(copy.deepcopy(risk_gate.state_dict()), strict=False)
        expected_missing = {f'{component}_{role}_{layer}.{parameter}' for component in ('q', 'p') for role, layers in (('recovery', (('input', ('weight',)), ('output', ('weight', 'bias')))), ('safety', (('input', ('weight',)), ('output', ('weight', 'bias'))))) for layer, parameters in layers for parameter in parameters}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('risk-to-projection-supported state drifted')
        with torch.no_grad():
            for component in ('q', 'p'):
                getattr(result, f'{component}_safety_input').weight.copy_(getattr(safety_gate, f'{component}_delta_input').weight)
                getattr(result, f'{component}_safety_output').weight.copy_(getattr(safety_gate, f'{component}_delta_output').weight)
                getattr(result, f'{component}_safety_output').bias.copy_(getattr(safety_gate, f'{component}_delta_output').bias)
        result.freeze_local_parameters()
        return result

    def _safety_deltas(self, residual_hidden: torch.Tensor) -> torch.Tensor:
        observable = residual_hidden.reshape(-1, self.residual_hidden_dim)
        return torch.cat(tuple((getattr(self, f'{component}_safety_output')(torch.nn.functional.silu(getattr(self, f'{component}_safety_input')(observable))) for component in ('q', 'p'))), dim=-1)

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None or self.residual_hidden_dim != 71:
            raise ValueError('projection-supported recovery requires observable71')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('projection-supported recovery hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        common = self.common_output(temporal[:, 0])
        recurrent_logits = common + self._component_deltas(residual_hidden)
        safety_logits = common + self._safety_deltas(residual_hidden)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        recurrent_gate = torch.sigmoid(recurrent_logits.reshape(batch, objects, 2) / temperature)
        safety_gate = torch.sigmoid(safety_logits.reshape(batch, objects, 2) / temperature)
        gate = HamiBallsProjectionSupportedRecurrentPolicyPerObjectCompactCommittedGate._projection_supported_value(safety_gate, recurrent_gate)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        _finite('HamiBalls projection-supported recovery gate', gate)
        return (gate, next_hidden)

class HamiBallsDualStatePerObjectCompactCommittedGate(HamiBallsDualChannelPerObjectCompactCommittedGate):
    component_history_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.component_history_adapter = nn.Linear(2, self.rank, bias=False)
        nn.init.zeros_(self.component_history_adapter.weight)

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsDualStatePerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('dual-state gate expansion requires the exact scalar compact gate')
        dual = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        scalar_weight = state.pop('output.weight')
        scalar_bias = state.pop('output.bias')
        missing, unexpected = dual.load_state_dict(state, strict=False)
        expected_missing = {'output.weight', 'output.bias', 'component_history_adapter.weight'}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('scalar-to-dual-state trunk state drifted')
        with torch.no_grad():
            dual.output.weight.copy_(scalar_weight.expand(2, -1))
            dual.output.bias.copy_(scalar_bias.expand(2))
        return dual

    def _encode_objects(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        batch, objects = d_token.shape[:2]
        if previous_g.shape != (batch, objects, 2):
            raise ValueError('dual-state previous_g must be [B,K,2]')
        mean = previous_g.mean(dim=-1)
        encoded = super()._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, mean, residual_hidden)
        centered = previous_g - mean[..., None]
        return encoded + self.component_history_adapter(centered)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects = d_tokens.shape[:3]
        supplied: torch.Tensor | None = None
        if previous_g is None:
            running = d_tokens.new_ones(batch, objects, 2)
        elif previous_g.shape == (batch, objects):
            running = previous_g[..., None].expand(-1, -1, 2)
        elif previous_g.shape == (batch, objects, 2):
            running = previous_g
        elif previous_g.shape == (batch, frames, objects):
            supplied = previous_g[..., None].expand(-1, -1, -1, 2)
            running = supplied[:, 0]
        elif previous_g.shape == (batch, frames, objects, 2):
            supplied = previous_g
            running = supplied[:, 0]
        else:
            raise ValueError('dual-state previous_g must be [B,K], [B,K,2], [B,F,K], or [B,F,K,2]')
        hidden: torch.Tensor | None = None
        rows: list[torch.Tensor] = []
        for edge in range(frames):
            gate, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running if supplied is None else supplied[:, edge], hidden=hidden, residual_hidden=residual_hidden[:, edge])
            rows.append(gate)
            running = gate
        assert hidden is not None
        return (torch.stack(rows, dim=1), hidden)

class HamiBallsPrivateTemporalPerObjectCompactCommittedGate(HamiBallsDualStatePerObjectCompactCommittedGate):
    private_component_temporal_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        shared_temporal = self.temporal
        shared_output = self.output
        self.temporal_q = copy.deepcopy(shared_temporal)
        self.temporal_p = copy.deepcopy(shared_temporal)
        self.output_q = nn.Linear(self.rank, 1)
        self.output_p = nn.Linear(self.rank, 1)
        with torch.no_grad():
            self.output_q.weight.copy_(shared_output.weight[:1])
            self.output_q.bias.copy_(shared_output.bias[:1])
            self.output_p.weight.copy_(shared_output.weight[1:2])
            self.output_p.bias.copy_(shared_output.bias[1:2])
        del self.temporal
        del self.output

    @classmethod
    def from_dual_state(cls, base: HamiBallsDualStatePerObjectCompactCommittedGate) -> 'HamiBallsPrivateTemporalPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsDualStatePerObjectCompactCommittedGate:
            raise TypeError('private temporal expansion requires the exact dual-state gate')
        expanded = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.log_temperature.device, dtype=base.log_temperature.dtype)
        common_state = copy.deepcopy(base.state_dict())
        temporal_state = {name.removeprefix('temporal.'): value for name, value in common_state.items() if name.startswith('temporal.')}
        output_weight = common_state['output.weight']
        output_bias = common_state['output.bias']
        common_state = {name: value for name, value in common_state.items() if not name.startswith('temporal.') and (not name.startswith('output.'))}
        missing, unexpected = expanded.load_state_dict(common_state, strict=False)
        private_prefixes = ('temporal_q.', 'temporal_p.', 'output_q.', 'output_p.')
        expected_missing = {name for name in expanded.state_dict() if name.startswith(private_prefixes)}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('dual-state-to-private shared state drifted')
        expanded.temporal_q.load_state_dict(copy.deepcopy(temporal_state), strict=True)
        expanded.temporal_p.load_state_dict(copy.deepcopy(temporal_state), strict=True)
        with torch.no_grad():
            expanded.output_q.weight.copy_(output_weight[:1])
            expanded.output_q.bias.copy_(output_bias[:1])
            expanded.output_p.weight.copy_(output_weight[1:2])
            expanded.output_p.bias.copy_(output_bias[1:2])
        return expanded

    def freeze_shared_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.temporal_q, self.temporal_p, self.output_q, self.output_p):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def private_named_parameters(self, component: str) -> tuple[tuple[str, nn.Parameter], ...]:
        if component not in {'q', 'p'}:
            raise ValueError('private temporal component must be q or p')
        prefixes = (f'temporal_{component}.', f'output_{component}.')
        return tuple(((name, parameter) for name, parameter in self.named_parameters() if name.startswith(prefixes)))

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('private temporal gate requires residual_hidden')
        batch, objects = d_token.shape[:2]
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden).reshape(batch * objects, 1, self.rank)
        q_hidden = None
        p_hidden = None
        if hidden is not None:
            if hidden.shape != (2, batch, objects, self.rank):
                raise ValueError('private temporal hidden must be [2,B,K,rank]')
            q_hidden = hidden[0].reshape(1, batch * objects, self.rank)
            p_hidden = hidden[1].reshape(1, batch * objects, self.rank)
        q_temporal, q_next = self.temporal_q(encoded, q_hidden)
        p_temporal, p_next = self.temporal_p(encoded, p_hidden)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        q_logit = self.output_q(q_temporal[:, 0])
        p_logit = self.output_p(p_temporal[:, 0])
        gate = torch.sigmoid(torch.cat((q_logit, p_logit), dim=-1).reshape(batch, objects, 2) / temperature)
        next_hidden = torch.cat((q_next.reshape(1, batch, objects, self.rank), p_next.reshape(1, batch, objects, self.rank)), dim=0)
        _finite('HamiBalls private-temporal committed gate', gate)
        return (gate, next_hidden)

class HamiBallsIndependentComponentPerObjectCompactCommittedGate(HamiBallsPerObjectCompactCommittedGate):
    component_gate = True
    component_history_gate = True
    independent_component_gate = True

    def __init__(self, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.gate_q = HamiBallsPerObjectCompactCommittedGate(**kwargs)
        self.gate_p = HamiBallsPerObjectCompactCommittedGate(**kwargs)
        self.token_dim = self.gate_q.token_dim
        self.state_dim = self.gate_q.state_dim
        self.attr_dim = self.gate_q.attr_dim
        self.residual_hidden_dim = self.gate_q.residual_hidden_dim
        self.hidden_size = self.gate_q.hidden_size
        self.rank = self.gate_q.rank
        self.candidate_step_observables = self.gate_q.candidate_step_observables
        self.function_preserving_candidate_step_observables = self.gate_q.function_preserving_candidate_step_observables
        self.function_preserving_pairwise_relations = self.gate_q.function_preserving_pairwise_relations

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate) -> 'HamiBallsIndependentComponentPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('independent q/p gate requires the exact scalar compact gate')
        result = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        result.gate_q.load_state_dict(state, strict=True)
        result.gate_p.load_state_dict(copy.deepcopy(state), strict=True)
        return result

    def component_named_parameters(self, component: str) -> tuple[tuple[str, nn.Parameter], ...]:
        if component not in {'q', 'p'}:
            raise ValueError('independent component must be q or p')
        module = self.gate_q if component == 'q' else self.gate_p
        return tuple(((f'gate_{component}.{name}', value) for name, value in module.named_parameters()))

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_g.ndim != 3 or previous_g.shape[-1] != 2:
            raise ValueError('independent q/p previous_g must be [B,K,2]')
        batch, objects = previous_g.shape[:2]
        q_hidden = None
        p_hidden = None
        if hidden is not None:
            if hidden.shape != (2, batch, objects, self.rank):
                raise ValueError('independent q/p hidden must be [2,B,K,rank]')
            q_hidden = hidden[0:1]
            p_hidden = hidden[1:2]
        q_value, q_next = self.gate_q.forward_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g[..., 0], hidden=q_hidden, residual_hidden=residual_hidden)
        p_value, p_next = self.gate_p.forward_step(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g[..., 1], hidden=p_hidden, residual_hidden=residual_hidden)
        value = torch.stack((q_value, p_value), dim=-1)
        next_hidden = torch.cat((q_next, p_next), dim=0)
        _finite('HamiBalls independent-component committed gate', value)
        return (value, next_hidden)

    def forward(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor | None=None, *, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects = d_tokens.shape[:3]
        supplied: torch.Tensor | None = None
        if previous_g is None:
            running = d_tokens.new_ones(batch, objects, 2)
        elif previous_g.shape == (batch, objects):
            running = previous_g[..., None].expand(-1, -1, 2)
        elif previous_g.shape == (batch, objects, 2):
            running = previous_g
        elif previous_g.shape == (batch, frames, objects):
            supplied = previous_g[..., None].expand(-1, -1, -1, 2)
            running = supplied[:, 0]
        elif previous_g.shape == (batch, frames, objects, 2):
            supplied = previous_g
            running = supplied[:, 0]
        else:
            raise ValueError('independent previous_g must be [B,K], [B,K,2], [B,F,K], or [B,F,K,2]')
        hidden: torch.Tensor | None = None
        rows: list[torch.Tensor] = []
        for edge in range(frames):
            value, hidden = self.forward_step(d_tokens[:, edge], noisy[:, edge], x0, previous_mixed[:, edge], h_candidate[:, edge], hr_candidate[:, edge], d_candidate[:, edge], attrs, tau, physical_time[:, edge], running if supplied is None else supplied[:, edge], hidden=hidden, residual_hidden=residual_hidden[:, edge])
            rows.append(value)
            running = value
        assert hidden is not None
        return (torch.stack(rows, dim=1), hidden)

class HamiBallsPhysicalNormPerObjectCompactCommittedGate(HamiBallsPerObjectCompactCommittedGate):
    physical_norm_disagreement = True

    def __init__(self, *, component: str, state_scale: torch.Tensor, frame_dt: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if component not in {'q', 'p'}:
            raise ValueError('physical-norm component must be q or p')
        if self.state_dim % 2:
            raise ValueError('physical-norm gate requires even state_dim')
        scale = torch.as_tensor(state_scale).detach().reshape(-1)
        if scale.shape != (self.state_dim,) or not bool(torch.isfinite(scale).all()):
            raise ValueError('physical-norm state_scale is invalid')
        if not bool((scale > 0).all()) or not float(frame_dt) > 0.0:
            raise ValueError('physical-norm scales must be positive')
        self.component = component
        self.frame_dt = float(frame_dt)
        self.register_buffer('physical_state_scale', scale.clone(), persistent=False)

    def _absolute_disagreement_features(self, disagreement: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        q_dim = self.state_dim // 2
        if self.component == 'q':
            physical = disagreement[..., :q_dim] * self.physical_state_scale[:q_dim]
            summary = physical.norm(dim=-1, keepdim=True)
        else:
            mass = attrs[..., 0:1]
            if not bool((mass > 0).all()):
                raise ValueError('physical-norm p gate requires positive mass')
            velocity = disagreement[..., q_dim:] * self.physical_state_scale[q_dim:] / mass
            summary = self.frame_dt * velocity.norm(dim=-1, keepdim=True)
        return torch.cat((summary, torch.zeros_like(disagreement[..., 1:])), dim=-1)

class HamiBallsIndependentPhysicalNormPerObjectCompactCommittedGate(HamiBallsIndependentComponentPerObjectCompactCommittedGate):
    independent_physical_norm_gate = True

    def __init__(self, *, state_scale: torch.Tensor, frame_dt: float, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.gate_q = HamiBallsPhysicalNormPerObjectCompactCommittedGate(component='q', state_scale=state_scale, frame_dt=frame_dt, **kwargs)
        self.gate_p = HamiBallsPhysicalNormPerObjectCompactCommittedGate(component='p', state_scale=state_scale, frame_dt=frame_dt, **kwargs)
        self.token_dim = self.gate_q.token_dim
        self.state_dim = self.gate_q.state_dim
        self.attr_dim = self.gate_q.attr_dim
        self.residual_hidden_dim = self.gate_q.residual_hidden_dim
        self.hidden_size = self.gate_q.hidden_size
        self.rank = self.gate_q.rank
        self.candidate_step_observables = self.gate_q.candidate_step_observables
        self.function_preserving_candidate_step_observables = self.gate_q.function_preserving_candidate_step_observables
        self.function_preserving_pairwise_relations = self.gate_q.function_preserving_pairwise_relations

    @classmethod
    def from_scalar(cls, base: HamiBallsPerObjectCompactCommittedGate, *, state_scale: torch.Tensor, frame_dt: float) -> 'HamiBallsIndependentPhysicalNormPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsPerObjectCompactCommittedGate:
            raise TypeError('physical-norm q/p gate requires the exact scalar compact gate')
        result = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations, state_scale=state_scale, frame_dt=frame_dt).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        state = copy.deepcopy(base.state_dict())
        result.gate_q.load_state_dict(state, strict=True)
        result.gate_p.load_state_dict(copy.deepcopy(state), strict=True)
        return result

class HamiBallsDualStateStartPerObjectCompactCommittedGate(HamiBallsDualStatePerObjectCompactCommittedGate):
    sequence_start_gate = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.start_output = nn.Linear(self.rank, 2)
        nn.init.zeros_(self.start_output.weight)
        nn.init.zeros_(self.start_output.bias)

    @classmethod
    def from_dual_state(cls, base: HamiBallsDualStatePerObjectCompactCommittedGate) -> 'HamiBallsDualStateStartPerObjectCompactCommittedGate':
        if type(base) is not HamiBallsDualStatePerObjectCompactCommittedGate:
            raise TypeError('start gate expansion requires the exact dual-state gate')
        expanded = cls(token_dim=base.token_dim, state_dim=base.state_dim, attr_dim=base.attr_dim, residual_hidden_dim=base.residual_hidden_dim, rank=base.rank, candidate_step_observables=base.candidate_step_observables, function_preserving_candidate_step_observables=base.function_preserving_candidate_step_observables, function_preserving_pairwise_relations=base.function_preserving_pairwise_relations).to(device=base.output.weight.device, dtype=base.output.weight.dtype)
        missing, unexpected = expanded.load_state_dict(copy.deepcopy(base.state_dict()), strict=False)
        if set(missing) != {'start_output.weight', 'start_output.bias'} or unexpected:
            raise ValueError('dual-state-to-start trunk state drifted')
        return expanded

    def forward_step(self, d_token: torch.Tensor, noisy_state: torch.Tensor, x0: torch.Tensor, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor, hidden: torch.Tensor | None=None, residual_hidden: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_hidden is None:
            raise ValueError('dual start gate requires residual_hidden')
        batch, objects = d_token.shape[:2]
        is_sequence_start = hidden is None
        encoded = self._encode_objects(d_token, noisy_state, x0, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, residual_hidden)
        hidden_flat = None
        if hidden is not None:
            if hidden.shape != (1, batch, objects, self.rank):
                raise ValueError('dual start gate hidden must be [1,B,K,rank]')
            hidden_flat = hidden.reshape(1, batch * objects, self.rank)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.rank), hidden_flat)
        representation = temporal[:, 0]
        logits = self.output(representation)
        if is_sequence_start:
            logits = logits + self.start_output(representation)
        next_hidden = next_flat.reshape(1, batch, objects, self.rank)
        temperature = self.log_temperature.clamp(-6.0, 6.0).exp()
        gate = torch.sigmoid(logits.reshape(batch, objects, 2) / temperature)
        _finite('HamiBalls dual-state start committed gate', gate)
        return (gate, next_hidden)
__all__ = ['HamiBallsDualChannelPerObjectCompactCommittedGate', 'HamiBallsAntisymmetricDeltaPerObjectCompactCommittedGate', 'HamiBallsObservableAntisymmetricDeltaPerObjectCompactCommittedGate', 'HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate', 'HamiBallsObservableRegimeConditionedPerObjectCompactCommittedGate', 'HamiBallsObservableAdditiveRecoveryPerObjectCompactCommittedGate', 'HamiBallsDualStatePerObjectCompactCommittedGate', 'HamiBallsPrivateTemporalPerObjectCompactCommittedGate', 'HamiBallsIndependentComponentPerObjectCompactCommittedGate', 'HamiBallsPhysicalNormPerObjectCompactCommittedGate', 'HamiBallsIndependentPhysicalNormPerObjectCompactCommittedGate', 'HamiBallsDualStateStartPerObjectCompactCommittedGate']
