from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from torch import nn

def _norm_qp(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x[..., :3].norm(dim=-1), x[..., 3:].norm(dim=-1)), -1)

def formal_observable_features(noisy: torch.Tensor, d: torch.Tensor, h: torch.Tensor, previous_mixed: torch.Tensor, common_r: torch.Tensor, common_hidden: torch.Tensor, previous_gate: torch.Tensor, *, x0: torch.Tensor, static: torch.Tensor, tau: torch.Tensor, physical_fraction: torch.Tensor) -> torch.Tensor:
    vectors = (d - h, h - previous_mixed, d - previous_mixed, noisy - d, common_r)
    scalar = torch.stack((tau, physical_fraction), -1)[:, None].expand(-1, noisy.shape[1], -1)
    return torch.cat((noisy, d, h, previous_mixed, x0, *vectors, *(_norm_qp(v) for v in vectors), static, previous_gate, scalar, common_hidden), -1)

@dataclass
class FixedTree:
    feature: torch.Tensor
    threshold: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    node_to_leaf: torch.Tensor
    depth: int

    @classmethod
    def from_dict(cls, tree: dict, device=None):
        result = cls(*(torch.as_tensor(tree[key], device=device) for key in ('feature', 'threshold', 'children_left', 'children_right', 'node_to_leaf')), depth=int(tree['depth']))
        node_count = int(result.feature.numel())
        if not all((int(value.numel()) == node_count for value in (result.threshold, result.left, result.right, result.node_to_leaf))):
            raise ValueError('fixed CART arrays must have the same length')
        if int(tree['depth']) < 0:
            raise ValueError('fixed CART depth must be non-negative')
        result.feature = result.feature.long()
        result.left = result.left.long()
        result.right = result.right.long()
        result.node_to_leaf = result.node_to_leaf.long()
        return result

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        node = torch.zeros(len(flat), device=flat.device, dtype=torch.long)
        row = torch.arange(len(flat), device=flat.device)
        for _ in range(self.depth):
            f = self.feature[node]
            terminal = f < 0
            go_left = flat[row, f.clamp_min(0)] <= self.threshold[node]
            node = torch.where(terminal, node, torch.where(go_left, self.left[node], self.right[node]))
        leaf = self.node_to_leaf[node]
        if bool((leaf < 0).any()):
            raise RuntimeError('fixed CART traversal did not reach a registered leaf')
        return leaf.reshape(x.shape[:-1])

class CommonResidual(nn.Module):

    def __init__(self, input_dim: int, width: int=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.output = nn.Linear(width, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        hidden = self.net(x)
        return (self.output(hidden), hidden)

class FormalRouter(nn.Module):

    def __init__(self, input_dim: int, rank: int=24, hidden: int=24, qp_width: int=32, max_leaves: int=8, leaf_feature_dim: int=1):
        super().__init__()
        self.rank = int(rank)
        self.hidden = int(hidden)
        self.max_leaves = int(max_leaves)
        self.input = nn.Linear(input_dim, rank)
        self.temporal = nn.GRUCell(rank, hidden)
        self.scalar = nn.Linear(hidden, 1)
        self.qp = nn.Sequential(nn.Linear(hidden, qp_width), nn.SiLU(), nn.Linear(qp_width, 2))
        nn.init.zeros_(self.qp[-1].weight)
        nn.init.zeros_(self.qp[-1].bias)
        self.leaf_affine = nn.Parameter(torch.zeros(max_leaves, 2, leaf_feature_dim + 1))

    def step(self, x, old, leaf_feature, leaf):
        state = self.temporal(torch.nn.functional.silu(self.input(x)), old)
        logit = self.scalar(state).expand(-1, 2) + self.qp(state)
        design = torch.cat((leaf_feature, torch.ones_like(leaf_feature[..., :1])), -1)
        selected = self.leaf_affine[leaf]
        logit = logit + torch.matmul(selected, design.unsqueeze(-1)).squeeze(-1)
        return (torch.sigmoid(logit), state)

class HamiBalls2FormalPostHD(nn.Module):
    architecture = 'hamiballs2_plas_semantic_posthd_v1'

    def __init__(self, *, phase_scale: tuple[float, ...], d_token_dim: int=168, residual_width: int=64, gate_rank: int=24, gate_hidden: int=24, gate_qp_width: int=32, max_leaves: int=8, leaf_gate_standardized: bool=False):
        super().__init__()
        self.leaf_gate_standardized = bool(leaf_gate_standardized)
        if len(phase_scale) != 6 or min(phase_scale) <= 0:
            raise ValueError('phase_scale must contain six positive entries')
        self.register_buffer('phase_scale', torch.tensor(phase_scale, dtype=torch.float32))
        self.d_token_dim = int(d_token_dim)
        self.max_leaves = int(max_leaves)
        residual_input = d_token_dim + 5 * 6 + 3 * 6 + 6 + 2 + 2
        self.common_r = CommonResidual(residual_input, residual_width)
        self.observable_dim = 5 * 6 + 5 * 6 + 5 * 2 + 6 + 2 + 2 + residual_width
        gate_input = d_token_dim + 5 * 6 + 2 * 6 + 6 + 2 + 2 + residual_width
        self.router = FormalRouter(gate_input, gate_rank, gate_hidden, gate_qp_width, max_leaves, self.observable_dim)
        self.register_buffer('static_feature_mean', torch.zeros(6))
        self.register_buffer('static_feature_scale', torch.ones(6))
        self.register_buffer('ridge_feature_mean', torch.zeros(self.observable_dim))
        self.register_buffer('ridge_feature_scale', torch.ones(self.observable_dim))
        self.register_buffer('ridge_weight', torch.zeros(max_leaves, 2, self.observable_dim + 1, 3))
        self.register_buffer('component_alpha', torch.zeros(2))
        self.tree: dict | None = None

    def set_training_stage(self, stage: str) -> int:
        if stage not in {'common_r', 'scalar0', 'scalar1', 'final_qp', 'frozen'}:
            raise ValueError(f'unknown post-HD stage: {stage}')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == 'common_r':
            for parameter in self.common_r.parameters():
                parameter.requires_grad_(True)
        elif stage in {'scalar0', 'scalar1'}:
            for module in (self.router.input, self.router.temporal, self.router.scalar):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        elif stage == 'final_qp':
            for parameter in self.router.qp.parameters():
                parameter.requires_grad_(True)
            self.router.leaf_affine.requires_grad_(True)
        return sum((parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def install_fitted_state(self, *, tree: dict, feature_mean: torch.Tensor, feature_scale: torch.Tensor, ridge_weight: torch.Tensor, component_alpha: torch.Tensor, static_mean: torch.Tensor | None=None, static_scale: torch.Tensor | None=None):
        if feature_mean.shape != (self.observable_dim,) or feature_scale.shape != (self.observable_dim,):
            raise ValueError('observable normalization shape mismatch')
        if ridge_weight.shape != self.ridge_weight.shape or component_alpha.shape != (2,):
            raise ValueError('fitted residual shape mismatch')
        if not bool(((component_alpha >= 0) & (component_alpha <= 1)).all()):
            raise ValueError('component alpha must be in [0,1]')
        fixed = FixedTree.from_dict(tree, device=feature_mean.device)
        if int(fixed.node_to_leaf.max()) >= self.max_leaves:
            raise ValueError('fitted Tree uses more leaves than the model budget')
        if static_mean is not None and static_mean.shape != (6,):
            raise ValueError('static feature mean must have shape [6]')
        if static_scale is not None and static_scale.shape != (6,):
            raise ValueError('static feature scale must have shape [6]')
        self.tree = copy.deepcopy(tree)
        self.ridge_feature_mean.copy_(feature_mean)
        self.ridge_feature_scale.copy_(feature_scale.clamp_min(1e-06))
        self.ridge_weight.copy_(ridge_weight)
        self.component_alpha.copy_(component_alpha)
        if static_mean is not None:
            self.static_feature_mean.copy_(static_mean)
        if static_scale is not None:
            self.static_feature_scale.copy_(static_scale.clamp_min(1e-06))

    def _r_features(self, token, noisy, d, h, previous, x0, static, tau, time, previous_gate):
        scalar = torch.stack((tau, time), -1)[:, None].expand(-1, noisy.shape[1], -1)
        return torch.cat((token, noisy, d, h, previous, x0, d - h, h - previous, d - previous, static, scalar, previous_gate), -1)

    def _gate_features(self, token, noisy, d, h, hr, previous, static, tau, time, previous_gate, common_hidden):
        scalar = torch.stack((tau, time), -1)[:, None].expand(-1, noisy.shape[1], -1)
        return torch.cat((token, noisy, d, h, hr, previous, hr - d, (hr - d).abs(), static, scalar, previous_gate, common_hidden), -1)

    def forward(self, state, d_candidate, h_candidate, d_tokens, *, x0, attrs, node_graph, physical_time, tau, object_mask, jet_matrix=None, jet_offset=None, force_gate=None, previous_start=None, previous_gate_start=None, physical_fraction=None, detach_physical_history: bool=False):
        b, e, o, c = state.shape
        if c != 6 or d_tokens.shape != (b, e, o, self.d_token_dim):
            raise ValueError('state/token shape mismatch')
        if d_candidate.shape != state.shape or h_candidate.shape != state.shape or x0.shape != (b, o, 6):
            raise ValueError('candidate shape mismatch')
        if attrs.shape != (b, o, 3) or node_graph.shape != (b, o, 3) or object_mask.shape != (b, o):
            raise ValueError('static shape mismatch')
        if physical_time.shape != (b, e) or tau.shape != (b,):
            raise ValueError('time/tau shape mismatch')
        if previous_start is not None and previous_start.shape != (b, o, 6):
            raise ValueError('previous_start shape mismatch')
        if previous_gate_start is not None and previous_gate_start.shape != (b, o, 2):
            raise ValueError('previous_gate_start shape mismatch')
        if physical_fraction is not None and physical_fraction.shape != (b, e):
            raise ValueError('physical_fraction shape mismatch')
        if (jet_matrix is None) != (jet_offset is None):
            raise ValueError('matrix and offset must be supplied together')
        if jet_matrix is not None and (jet_matrix.shape != (b, e, 6 * o, 6 * o) or jet_offset.shape != (b, e, 6 * o)):
            raise ValueError('PLAS jet shape mismatch')
        if force_gate is not None:
            if torch.is_tensor(force_gate):
                if force_gate.shape != (b, e, o, 2) or not bool(((force_gate >= 0) & (force_gate <= 1)).all()):
                    raise ValueError('forced gate tensor must be [B,E,O,2] in [0,1]')
            elif not 0.0 <= float(force_gate) <= 1.0:
                raise ValueError('forced gate must lie in [0,1]')
        scale = self.phase_scale.to(state)
        static = (torch.cat((attrs, node_graph), -1) - self.static_feature_mean.to(state)) / self.static_feature_scale.to(state)
        static = static * object_mask[..., None].to(static)
        previous = x0 if previous_start is None else previous_start
        previous_gate = state.new_ones(b, o, 2) if previous_gate_start is None else previous_gate_start
        gate_hidden = state.new_zeros(b * o, self.router.hidden)
        outs = {k: [] for k in ('mixed', 'h_candidate', 'hr_candidate', 'common_residual', 'leaf_residual', 'gate', 'leaf', 'observable')}
        fixed = None if self.tree is None else FixedTree.from_dict(self.tree, state.device)
        denom = (physical_time[:, -1] - physical_time[:, 0]).clamp_min(1e-06)
        for k in range(e):
            noisy = state[:, k] / scale
            d = d_candidate[:, k] / scale
            if jet_matrix is None:
                h = h_candidate[:, k] / scale
            else:
                flat = torch.cat((previous[..., :3].reshape(b, -1), previous[..., 3:].reshape(b, -1)), -1)
                raw = (jet_matrix[:, k] @ flat.unsqueeze(-1)).squeeze(-1) + jet_offset[:, k]
                h = torch.cat((raw[..., :3 * o].reshape(b, o, 3), raw[..., 3 * o:].reshape(b, o, 3)), -1) / scale
            prev = previous / scale
            x0n = x0 / scale
            time = physical_fraction[:, k] if physical_fraction is not None else (physical_time[:, k] - physical_time[:, 0]) / denom
            base_r, hidden = self.common_r(self._r_features(d_tokens[:, k], noisy, d, h, prev, x0n, static, tau, time, previous_gate))
            observable = formal_observable_features(noisy, d, h, prev, base_r, hidden, previous_gate, x0=x0n, static=static, tau=tau, physical_fraction=time)
            leaf = torch.zeros(b, o, device=state.device, dtype=torch.long) if fixed is None else fixed.apply(observable)
            design = torch.cat((((observable - self.ridge_feature_mean) / self.ridge_feature_scale).clamp(-8, 8), torch.ones_like(observable[..., :1])), -1)
            selected = self.ridge_weight[leaf]
            leaf_r = torch.cat(tuple((torch.matmul(design.unsqueeze(-2), selected[..., j, :, :]).squeeze(-2) for j in range(2))), -1)
            if k == 0:
                leaf_r = torch.zeros_like(leaf_r)
            alpha = self.component_alpha.repeat_interleave(3)
            r = (1 - alpha) * base_r + alpha * leaf_r
            hr = (h + r) * scale
            gate_input = self._gate_features(d_tokens[:, k], noisy, d, h, hr / scale, prev, static, tau, time, previous_gate, hidden)
            leaf_gate_feature = design[..., :-1] if self.leaf_gate_standardized else observable
            gate, gate_hidden = self.router.step(gate_input.reshape(b * o, -1), gate_hidden, leaf_gate_feature.reshape(b * o, -1), leaf.reshape(-1))
            gate = gate.reshape(b, o, 2)
            if force_gate is not None:
                gate = force_gate[:, k].to(gate) if torch.is_tensor(force_gate) else torch.full_like(gate, float(force_gate))
            gate = gate * object_mask[..., None].to(gate)
            gate6 = gate.repeat_interleave(3, -1)
            mixed = gate6 * hr + (1 - gate6) * d_candidate[:, k]
            mask = object_mask[..., None].to(mixed)
            mixed = mixed * mask
            hr = hr * mask
            previous = mixed.detach() if detach_physical_history else mixed
            previous_gate = gate.detach() if detach_physical_history else gate
            for name, value in (('mixed', mixed), ('h_candidate', h * scale * mask), ('hr_candidate', hr), ('common_residual', base_r * scale * mask), ('leaf_residual', leaf_r * scale * mask), ('gate', gate), ('leaf', leaf), ('observable', observable * mask)):
                outs[name].append(value)
        return {k: torch.stack(v, 1) for k, v in outs.items()}

def parameter_count(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters()))
__all__ = ['HamiBalls2FormalPostHD', 'FixedTree', 'formal_observable_features', 'parameter_count']
