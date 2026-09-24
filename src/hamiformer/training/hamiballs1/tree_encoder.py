from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import torch
from torch import nn

def _quantize(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)

def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.square(value).sum(-1) + 1e-12)

def _cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(-1) / (_norm(left) * _norm(right)).clamp_min(1e-08)

class CausalObservableTreeEncoder(nn.Module):

    def __init__(self, *, tree: dict[str, object] | None=None, feature_names: list[str] | None=None, feature_mean: torch.Tensor | None=None, feature_scale: torch.Tensor | None=None) -> None:
        super().__init__()
        names = list(feature_names or ['rf_trace'])
        mean = torch.zeros(len(names)) if feature_mean is None else feature_mean.float()
        scale = torch.ones(len(names)) if feature_scale is None else feature_scale.float()
        if mean.shape != (len(names),) or scale.shape != (len(names),):
            raise ValueError('ETrg encoder normalization shape drifted')
        self.tree = copy.deepcopy(tree or {'depth': 0, 'leaf': 0})
        self.feature_names = names
        self.register_buffer('feature_mean', mean.clone())
        self.register_buffer('feature_scale', scale.clone())

    def set_fitted_state(self, *, tree: dict[str, object], feature_names: list[str], feature_mean: torch.Tensor, feature_scale: torch.Tensor) -> None:
        names = list(feature_names)
        mean = feature_mean.to(self.feature_mean).float()
        scale = feature_scale.to(self.feature_scale).float()
        if mean.shape != (len(names),) or scale.shape != (len(names),):
            raise ValueError('fitted ETrg normalization shape drifted')
        self.tree = copy.deepcopy(tree)
        self.feature_names = names
        self.feature_mean = mean.clone()
        self.feature_scale = scale.clone()

    def _features(self, *, previous_mixed: torch.Tensor, h_candidate: torch.Tensor, hr_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, previous_g: torch.Tensor, residual_hidden: torch.Tensor, base_gate: torch.Tensor) -> dict[str, torch.Tensor]:
        del hr_candidate, base_gate
        previous = _quantize(previous_mixed)
        h = _quantize(h_candidate)
        d = _quantize(d_candidate)
        obs = _quantize(residual_hidden)
        incoming = _quantize(previous_g)
        attrs = _quantize(attrs)
        while attrs.ndim < incoming.ndim:
            attrs = attrs[:, None]
        attrs = attrs.expand(*incoming.shape[:-1], attrs.shape[-1])
        vectors = {'gap': d - h, 'Hstep': h - previous, 'Dstep': d - previous}
        features: dict[str, torch.Tensor] = {f'observable71_{index}': obs[..., index] for index in range(obs.shape[-1])}
        for name, value in vectors.items():
            for index in range(4):
                features[f'{name}_{index}'] = value[..., index]
            features[f'{name}_q_norm'] = _norm(value[..., :2])
            features[f'{name}_p_norm'] = _norm(value[..., 2:])
        features['Hstep_Dstep_q_cos'] = _cos(vectors['Hstep'][..., :2], vectors['Dstep'][..., :2])
        features['Hstep_Dstep_p_cos'] = _cos(vectors['Hstep'][..., 2:], vectors['Dstep'][..., 2:])
        features.update({'previous_g_q': incoming[..., 0], 'previous_g_p': incoming[..., 1], 'mass': attrs[..., 0], 'radius': attrs[..., 1], 'attr2': attrs[..., 2]})
        rf_trace = torch.round((tau - 0.1) / 0.05).clamp(0.0, 17.0) / 17.0
        while rf_trace.ndim < incoming[..., 0].ndim:
            rf_trace = rf_trace[..., None]
        features['rf_trace'] = rf_trace.expand_as(incoming[..., 0])
        return features

    def _leaves(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = next(iter(features.values())).shape
        device = next(iter(features.values())).device
        result = torch.empty(shape, dtype=torch.long, device=device)
        stack = [(self.tree, torch.ones(shape, dtype=torch.bool, device=device))]
        while stack:
            node, mask = stack.pop()
            if 'leaf' in node:
                result[mask] = int(node['leaf'])
                continue
            choose = features[str(node['feature'])] <= float(node['threshold'])
            stack.append((node['right'], mask & ~choose))
            stack.append((node['left'], mask & choose))
        return result
__all__ = ['CausalObservableTreeEncoder']
