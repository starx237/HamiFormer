import copy
import torch
from torch import nn
from .routing_runtime import install_pre_gate_refiner

def _quantize(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)

def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.square(value).sum(-1) + 1e-12)

def _cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(-1) / (_norm(left) * _norm(right)).clamp_min(1e-08)

class HamiBalls1RoutedExpert(nn.Module):
    per_object_gate = True
    component_gate = True
    requires_residual_hidden = True

    def __init__(self, base_gate, payload):
        super().__init__()
        self.base_gate = base_gate
        self.component_history_gate = bool(getattr(base_gate, 'component_history_gate', False))
        self.tree = copy.deepcopy(payload['tree'])
        self.feature_names = list(payload['feature_names'])
        for name in ('ridge_weight', 'feature_mean', 'feature_scale'):
            self.register_buffer(name, payload[name].float().clone())
        alpha = tuple((float(value) for value in payload['component_alpha']))

        def adjust(observable, leaf):
            design = torch.cat((observable, torch.ones_like(observable[..., :1])), dim=-1)
            local = self.base_gate.etrg_leaf_gate_linear[leaf.long()]
            return observable.new_tensor(alpha) * torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
        install_pre_gate_refiner(self, self, lambda: self.ridge_weight, None, replace_base_hr=True, gate_logit_adjuster=adjust, candidate_blend_alpha=alpha)

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
        device = next(iter(features.values())).device
        result = torch.empty(shape, dtype=torch.long, device=device)
        stack = [(self.tree, torch.ones(shape, dtype=torch.bool, device=device))]
        while stack:
            node, use = stack.pop()
            if 'leaf' in node:
                result[use] = int(node['leaf'])
                continue
            choose = features[str(node['feature'])] <= float(node['threshold'])
            stack.append((node['right'], use & ~choose))
            stack.append((node['left'], use & choose))
        return result
