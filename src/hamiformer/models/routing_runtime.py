from __future__ import annotations
import types
from typing import Callable
import torch

def _componentize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[..., None].expand(*value.shape, 2)
    if value.ndim == 3 and value.shape[-1] == 2:
        return value
    raise ValueError(f'expected [B,K] or [B,K,2] previous gate, got {tuple(value.shape)}')

def conditional_delta(encoder, weight: torch.Tensor, *, edge: int, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, base_hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    previous_qp = _componentize(previous_g)
    dummy_current_gate = previous_qp.new_zeros(previous_qp.shape)
    features = encoder._features(previous_mixed=previous_mixed, h_candidate=h_candidate, hr_candidate=base_hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_qp, residual_hidden=residual_hidden, base_gate=dummy_current_gate)
    leaf = encoder._leaves(features)
    if int(edge) == 0 and (not bool(getattr(encoder, 'apply_on_edge_zero', False))):
        return (torch.zeros_like(base_hr_candidate), leaf)
    x = torch.stack([features[name] for name in encoder.feature_names], dim=-1)
    x = ((x - encoder.feature_mean) / encoder.feature_scale).clamp(-8.0, 8.0)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    selected = weight[leaf.long()]
    q_delta = torch.matmul(x.unsqueeze(-2), selected[..., 0, :, :]).squeeze(-2)
    p_delta = torch.matmul(x.unsqueeze(-2), selected[..., 1, :, :]).squeeze(-2)
    return (torch.cat([q_delta, p_delta], dim=-1), leaf)

def install_pre_gate_refiner(policy: torch.nn.Module, encoder, weight_getter: Callable[[], torch.Tensor], gate_bias_getter: Callable[[], torch.Tensor] | None=None, *, replace_base_hr: bool=False, gate_logit_adjuster: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None=None, candidate_blend_alpha: float | tuple[float, float] | None=None) -> None:
    if bool(getattr(policy, '_etrg_pre_gate_installed', False)):
        raise RuntimeError('pre-gate refiner already installed')
    if gate_bias_getter is not None and gate_logit_adjuster is not None:
        raise ValueError('choose either a constant gate bias or a logit adjuster')
    original_forward = policy.forward_step
    original_component_history = bool(getattr(policy, 'component_history_gate', False))
    policy.component_history_gate = True
    policy._etrg_pre_gate_installed = True

    def begin_committed_rollout(module):
        module._etrg_runtime_edge = 0

    def forward_step(module, d_token, noisy_state, x0, previous_mixed, h_candidate, base_hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden=None, residual_hidden=None):
        if residual_hidden is None:
            raise ValueError('ETrg conditional residual requires residual_hidden')
        previous_qp = _componentize(previous_g)
        delta, leaf = conditional_delta(encoder, weight_getter(), edge=int(getattr(module, '_etrg_runtime_edge', 0)), previous_mixed=previous_mixed, h_candidate=h_candidate, base_hr_candidate=base_hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_qp, residual_hidden=residual_hidden)
        strong_hr = h_candidate + delta if replace_base_hr else base_hr_candidate + delta
        if candidate_blend_alpha is None:
            corrected_hr = strong_hr
        else:
            if isinstance(candidate_blend_alpha, tuple):
                if len(candidate_blend_alpha) != 2:
                    raise ValueError('q/p candidate blend requires exactly two values')
                alpha = strong_hr.new_tensor(candidate_blend_alpha).repeat_interleave(2)
            else:
                alpha = float(candidate_blend_alpha)
            corrected_hr = base_hr_candidate + alpha * (strong_hr - base_hr_candidate)
        gate_previous = previous_qp if original_component_history else previous_qp.mean(dim=-1)
        value, next_hidden = original_forward(d_token, noisy_state, x0, previous_mixed, h_candidate, corrected_hr, d_candidate, attrs, tau, physical_time, gate_previous, hidden, residual_hidden=residual_hidden)
        if gate_logit_adjuster is not None:
            bias = gate_logit_adjuster(residual_hidden, leaf)
            if bias.shape != value.shape:
                raise ValueError('ETrg gate logit adjustment shape drifted')
            value = torch.sigmoid(torch.logit(value.clamp(1e-06, 1.0 - 1e-06)) + bias)
        elif gate_bias_getter is not None:
            bias = gate_bias_getter()[leaf]
            value = torch.sigmoid(torch.logit(value.clamp(1e-06, 1.0 - 1e-06)) + bias)
        module._etrg_last_corrected_hr = corrected_hr
        module._etrg_last_leaf = leaf
        module._etrg_runtime_edge = int(getattr(module, '_etrg_runtime_edge', 0)) + 1
        return (value, next_hidden)

    def refine_candidates(module, *, edge, hr_candidate, base_gate, **_kwargs):
        corrected = getattr(module, '_etrg_last_corrected_hr', None)
        if corrected is None or corrected.shape != hr_candidate.shape:
            raise RuntimeError('pre-gate residual was not evaluated before mixing')
        return (corrected, base_gate)
    policy._etrg_runtime_edge = 0
    policy.begin_committed_rollout = types.MethodType(begin_committed_rollout, policy)
    policy.forward_step = types.MethodType(forward_step, policy)
    policy.refine_candidates = types.MethodType(refine_candidates, policy)
