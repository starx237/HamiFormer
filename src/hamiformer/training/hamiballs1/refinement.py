from __future__ import annotations
from hamiformer.utils.paths import project_root
import json
from pathlib import Path
import sys
import torch
from torch import nn
ROOT = project_root()
from hamiformer.training.hamiballs1 import ridge_features as moe
CHECKPOINT = ROOT / 'outputs/hami1/moe/depth3_heads.pt'
GATE_REPORT = ROOT / 'outputs/hami1/gate/report.json'
NOISE_SEEDS = (1942634267, 2035767743)

def _quantize(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)

def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.square(value).sum(-1) + 1e-12)

def _cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(-1) / (_norm(left) * _norm(right)).clamp_min(1e-08)

class RegimeRefinedGate(nn.Module):
    per_object_gate = True
    component_gate = True
    requires_residual_hidden = True

    def __init__(self, base_gate: nn.Module, *, enable_r: bool=True, enable_gate: bool=True) -> None:
        super().__init__()
        self.base_gate = base_gate
        self.component_history_gate = bool(getattr(base_gate, 'component_history_gate', False))
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        self.tree = checkpoint['tree']
        self.feature_names = list(checkpoint['feature_names'])
        self.register_buffer('feature_mean', torch.as_tensor(checkpoint['feature_mean'], dtype=torch.float32))
        self.register_buffer('feature_scale', torch.as_tensor(checkpoint['feature_scale'], dtype=torch.float32))
        leaf_count = int(checkpoint['leaf_count'])
        width = int(checkpoint['width'])
        input_dim = int(checkpoint['input_dim'])
        self.q_heads = nn.ModuleList([moe.Head(input_dim, width) for _ in range(leaf_count)])
        self.p_heads = nn.ModuleList([moe.Head(input_dim, width) for _ in range(leaf_count)])
        self.q_heads.load_state_dict(checkpoint['q_heads'])
        self.p_heads.load_state_dict(checkpoint['p_heads'])
        gate_report = json.loads(GATE_REPORT.read_text(encoding='utf-8'))
        self.register_buffer('biases', torch.as_tensor(gate_report['biases'], dtype=torch.float32))
        self.leaf_count = leaf_count
        self.enable_r = bool(enable_r)
        self.enable_gate = bool(enable_gate)

    def forward_step(self, *args, **kwargs):
        return self.base_gate.forward_step(*args, **kwargs)

    def _features(self, *, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor, base_gate: torch.Tensor) -> dict[str, torch.Tensor]:
        previous = _quantize(previous_mixed)
        h = _quantize(h_candidate)
        hr = _quantize(hr_candidate)
        d = _quantize(d_candidate)
        obs = _quantize(residual_hidden)
        gate = _quantize(base_gate)
        previous_g = _quantize(previous_g)
        attrs = _quantize(attrs)
        while attrs.ndim < gate.ndim:
            attrs = attrs[:, None]
        attrs = attrs.expand(*gate.shape[:-1], attrs.shape[-1])
        vectors = {'gap': d - h, 'r': hr - h, 'Hstep': h - previous, 'Dstep': d - previous}
        features: dict[str, torch.Tensor] = {f'observable71_{i}': obs[..., i] for i in range(obs.shape[-1])}
        for name, value in vectors.items():
            for i in range(4):
                features[f'{name}_{i}'] = value[..., i]
            features[f'{name}_q_norm'] = _norm(value[..., :2])
            features[f'{name}_p_norm'] = _norm(value[..., 2:])
        for name, left, right in (('Hstep_Dstep', vectors['Hstep'], vectors['Dstep']), ('r_gap', vectors['r'], vectors['gap'])):
            features[f'{name}_q_cos'] = _cos(left[..., :2], right[..., :2])
            features[f'{name}_p_cos'] = _cos(left[..., 2:], right[..., 2:])
        features.update({'gate_q': gate[..., 0], 'gate_p': gate[..., 1], 'previous_g_q': previous_g[..., 0], 'previous_g_p': previous_g[..., 1], 'mass': attrs[..., 0], 'radius': attrs[..., 1], 'attr2': attrs[..., 2]})
        rf_trace = torch.round((tau - 0.1) / 0.05).clamp(0.0, 17.0) / 17.0
        while rf_trace.ndim < gate[..., 0].ndim:
            rf_trace = rf_trace[..., None]
        features['rf_trace'] = rf_trace.expand_as(gate[..., 0])
        return features

    def _leaves(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = next(iter(features.values())).shape
        result = torch.empty(shape, dtype=torch.long, device=self.biases.device)
        stack = [(self.tree, torch.ones(shape, dtype=torch.bool, device=result.device))]
        while stack:
            node, mask = stack.pop()
            if 'leaf' in node:
                result[mask] = int(node['leaf'])
                continue
            choose = features[str(node['feature'])] <= float(node['threshold'])
            stack.append((node['right'], mask & ~choose))
            stack.append((node['left'], mask & choose))
        return result

    def refine_candidates(self, *, edge: int, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor, base_gate: torch.Tensor, **_unused):
        if edge == 0:
            return (hr_candidate, base_gate)
        features = self._features(previous_mixed=previous_mixed, h_candidate=h_candidate, hr_candidate=hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_g, residual_hidden=residual_hidden, base_gate=base_gate)
        leaf = self._leaves(features)
        x = torch.stack([features[name] for name in self.feature_names], dim=-1)
        x = ((x - self.feature_mean) / self.feature_scale).clamp(-8.0, 8.0)
        delta = torch.zeros_like(hr_candidate)
        gate = base_gate.clone()
        for regime in range(self.leaf_count):
            use = leaf == regime
            if bool(use.any()):
                local_delta = torch.zeros((int(use.sum()), 4), device=x.device, dtype=x.dtype)
                local_delta[:, :2] = self.q_heads[regime](x[use])
                local_delta[:, 2:] = self.p_heads[regime](x[use])
                if self.enable_r:
                    delta[use] = local_delta
                local_gate = base_gate[use].clone()
                if self.enable_gate:
                    for component in range(2):
                        bias = self.biases[regime, component]
                        value = local_gate[:, component].clamp(1e-06, 1.0 - 1e-06)
                        local_gate[:, component] = torch.sigmoid(torch.logit(value) + bias)
                gate[use] = local_gate
        return (hr_candidate + delta, gate)
