from __future__ import annotations
import math
import torch
from torch import nn
from hamiformer.models.hamiballs_committed import _finite
from hamiformer.models.hamiballs_gate_dual_channel import HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate

class HamiBallsObservableDisjointRiskVetoGate(nn.Module):
    resets_each_rf_field = True
    requires_residual_hidden = True
    per_object_gate = True
    component_gate = True
    observable_disjoint_risk_veto_gate = True

    def __init__(self, base_gate: HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate, *, initial_risk: float=0.01) -> None:
        super().__init__()
        if type(base_gate) is not HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate:
            raise TypeError('observable-disjoint veto requires the exact dual parent')
        if not 0.0 < initial_risk < 0.5:
            raise ValueError('initial risk must lie in (0, 0.5)')
        self.base_gate = base_gate
        self.base_gate.requires_grad_(False)
        self.base_gate.eval()
        for name in ('token_dim', 'state_dim', 'attr_dim', 'residual_hidden_dim', 'rank', 'hidden_size'):
            setattr(self, name, int(getattr(base_gate, name)))
        self.initial_risk = float(initial_risk)
        self.inverse_initial_jacobian = 1.0 / (self.initial_risk * (1.0 - self.initial_risk))
        self.initial_logit = math.log(self.initial_risk / (1.0 - self.initial_risk))
        self.q_risk = nn.Linear(self.residual_hidden_dim, 1)
        self.p_risk = nn.Linear(self.residual_hidden_dim, 1)
        for head in (self.q_risk, self.p_risk):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        reference = next(base_gate.parameters())
        self.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode: bool=True) -> 'HamiBallsObservableDisjointRiskVetoGate':
        super().train(mode)
        self.base_gate.eval()
        return self

    def _risk(self, residual_hidden: torch.Tensor) -> torch.Tensor:
        if residual_hidden.shape[-1] != self.residual_hidden_dim:
            raise ValueError('observable-disjoint veto hidden width changed')
        flat = residual_hidden.reshape(-1, self.residual_hidden_dim)
        raw = torch.cat((self.q_risk(flat), self.p_risk(flat)), dim=-1)
        logits = self.initial_logit + self.inverse_initial_jacobian * raw
        risk = torch.sigmoid(logits).reshape(*residual_hidden.shape[:-1], 2)
        epsilon = torch.finfo(risk.dtype).eps
        risk = epsilon + (1.0 - 2.0 * epsilon) * risk
        _finite('HamiBalls observable-disjoint veto risk', risk)
        return risk

    @staticmethod
    def value_from_risk(base_value: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        if base_value.shape != risk.shape:
            raise ValueError('observable-disjoint base gate and risk must align')
        value = base_value * (1.0 - risk)
        if bool((value < 0.0).any()) or bool((value > base_value).any()):
            raise FloatingPointError('observable-disjoint veto violated monotonicity')
        _finite('HamiBalls observable-disjoint veto gate', value)
        return value

    def forward_step_with_risk(self, *args, residual_hidden: torch.Tensor, **kwargs):
        base_value, hidden = self.base_gate.forward_step(*args, residual_hidden=residual_hidden, **kwargs)
        risk = self._risk(residual_hidden)
        return (self.value_from_risk(base_value, risk), hidden, base_value, risk)

    def forward_step(self, *args, residual_hidden: torch.Tensor, **kwargs):
        value, hidden, _base, _risk = self.forward_step_with_risk(*args, residual_hidden=residual_hidden, **kwargs)
        return (value, hidden)

    def forward(self, *args, residual_hidden: torch.Tensor, **kwargs):
        base_value, hidden = self.base_gate(*args, residual_hidden=residual_hidden, **kwargs)
        risk = self._risk(residual_hidden)
        return (self.value_from_risk(base_value, risk), hidden)
__all__ = ['HamiBallsObservableDisjointRiskVetoGate']
