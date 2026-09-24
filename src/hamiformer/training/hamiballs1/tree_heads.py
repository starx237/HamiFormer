from __future__ import annotations
from hamiformer.utils.paths import project_root
import types
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_features as tree_tools
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1 import tree_training as hierarchical_tree
from hamiformer.training.hamiballs1.tree_encoder import CausalObservableTreeEncoder
from hamiformer.training.hamiballs1.tree_candidate import conditional_delta, install_pre_gate_refiner
OUTPUT = ROOT / 'outputs/hami1/tree_head_training_oblique_tree_control'
TREE_OUTPUT = OUTPUT / 'component_router_parent_oblique_tree.pt'
PARENT_ROWS = OUTPUT / 'component_router_parent_rows.npz'
FINAL_OUTPUT = OUTPUT / 'final_clean'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
SEALED_OUTPUT = OUTPUT / 'TREE_CONTROL_self_contained_main.pt'
REFERENCE_TREE = ROOT / 'outputs/hami1/tree_residual_matched_model' / 'component_router_parent_honest_tree.pt'

def _action_targets(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate((np.clip(b[:, :2] / np.maximum(a[:, :2], 1e-12), 0.0, 1.0), b[:, 2:4] / np.maximum(a[:, 2:3], 1e-08), b[:, 4:6] / np.maximum(a[:, 3:4], 1e-08)), axis=-1)

def _pls2_direction(x: np.ndarray, a: np.ndarray, b: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    targets = _action_targets(a, b)
    task_weight = (a[:, 0], a[:, 1], a[:, 2], a[:, 2], a[:, 3], a[:, 3])
    aggregate = sum((value[index] for value in task_weight)) + 1e-12
    aggregate /= max(float(aggregate.sum()), 1e-30)
    local_x = x[index].astype(np.float64, copy=False)
    mean = (aggregate[:, None] * local_x).sum(0)
    scale = np.sqrt((aggregate[:, None] * np.square(local_x - mean)).sum(0))
    scale = np.maximum(scale, 1e-05)
    z = np.clip((x.astype(np.float64, copy=False) - mean) / scale, -8.0, 8.0)
    cross = np.empty((x.shape[1], 6), np.float64)
    for task, weight_all in enumerate(task_weight):
        weight = weight_all[index] + 1e-12
        weight /= max(float(weight.sum()), 1e-30)
        value = targets[index, task]
        value_mean = float((weight * value).sum())
        value_scale = np.sqrt(float((weight * np.square(value - value_mean)).sum()))
        value_scale = max(value_scale, 1e-12)
        cross[:, task] = (weight[:, None] * z[index] * ((value - value_mean) / value_scale)[:, None]).sum(0)
    left, singular, _right = np.linalg.svd(cross, full_matrices=False)
    direction = left[:, 0]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return (mean, scale, direction, singular.tolist())

def _grow_oblique(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, risk_scale: np.ndarray, source: np.ndarray, index: np.ndarray, *, depth: int=0, max_depth: int=clean.DEPTH) -> dict[str, object]:
    leaf: dict[str, object] = {'leaf': -1, 'depth': depth}
    if depth >= max_depth or len(index) < 8192:
        return leaf
    mean, scale, direction, singular = _pls2_direction(x, a, b, index)
    axis = np.clip((x - mean) / scale, -8.0, 8.0) @ direction
    parent = tree_tools._quad_impurity(a, b, c, index, risk_scale)
    best = None
    for threshold in np.unique(np.quantile(axis[index], np.linspace(0.1, 0.9, 9))):
        choose = axis[index] <= threshold
        left, right = (index[choose], index[~choose])
        if min(len(left), len(right)) < 4096:
            continue
        if min(len(np.unique(source[left])), len(np.unique(source[right]))) < 24:
            continue
        gain = parent - tree_tools._quad_impurity(a, b, c, left, risk_scale) - tree_tools._quad_impurity(a, b, c, right, risk_scale)
        candidate = (float(gain), min(len(left), len(right)), float(threshold), left, right)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] <= 0.0:
        return leaf
    gain, _support, threshold, left, right = best
    top = np.argsort(np.abs(direction))[::-1][:8]
    return {'threshold': threshold, 'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(), 'direction': direction.tolist(), 'leading_singular_values': singular, 'quadratic_impurity_reduction': gain, 'top_coefficients': top.astype(int).tolist(), 'depth': depth, 'left': _grow_oblique(x, a, b, c, risk_scale, source, left, depth=depth + 1, max_depth=max_depth), 'right': _grow_oblique(x, a, b, c, risk_scale, source, right, depth=depth + 1, max_depth=max_depth)}

class ObliqueTreeEncoder(CausalObservableTreeEncoder):

    def _leaves(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.stack([features[name] for name in self.feature_names], dim=-1)
        shape = x.shape[:-1]
        result = torch.empty(shape, dtype=torch.long, device=x.device)
        stack = [(self.tree, torch.ones(shape, dtype=torch.bool, device=x.device))]
        while stack:
            node, use = stack.pop()
            if 'leaf' in node:
                result[use] = int(node['leaf'])
                continue
            if 'direction' in node:
                mean = x.new_tensor(node['feature_mean'])
                scale = x.new_tensor(node['feature_scale'])
                direction = x.new_tensor(node['direction'])
                axis = (torch.clamp((x - mean) / scale, -8.0, 8.0) * direction).sum(-1)
                choose = axis <= float(node['threshold'])
            else:
                lookup = self.feature_names.index(str(node['feature']))
                choose = x[..., lookup] <= float(node['threshold'])
            stack.append((node['right'], use & ~choose))
            stack.append((node['left'], use & choose))
        return result

class MatchedObliqueHierarchicalResidual(hierarchical_tree.MatchedHierarchicalResidual):

    def _initialize(self, candidate, collector_module) -> None:
        payload = torch.load(TREE_OUTPUT, map_location='cpu', weights_only=False)
        device = next(candidate.parameters()).device
        self.encoder = ObliqueTreeEncoder(tree=payload['tree'], feature_names=list(payload['feature_names']), feature_mean=payload['feature_mean'], feature_scale=payload['feature_scale']).to(device).eval()
        self.reachable_leaf_count = int(payload['leaf_count'])
        self.alpha_qp = (float(payload['component_calibration']['q']['alpha']), float(payload['component_calibration']['p']['alpha']))
        width = hierarchical_tree.final.INPUT_DIM
        self.xtx = torch.zeros(self.leaf_capacity, width, width, device=device)
        self.xty = torch.zeros(self.leaf_capacity, 2, width, 2, device=device)
        collector_module.register_buffer('_integrated_ridge_weight', torch.zeros(self.leaf_capacity, 2, width, 2, device=device))
        self.collector = collector_module
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        candidate.register_parameter('etrg_leaf_gate_linear', torch.nn.Parameter(torch.zeros(self.leaf_capacity, 2, hierarchical_tree.DESIGN_DIM, device=device)))
        candidate._declared_extra_trainable_parameters = self.leaf_capacity * 2 * hierarchical_tree.DESIGN_DIM
        candidate._declared_trainable_parameters_override = self.leaf_capacity * 2 * hierarchical_tree.DESIGN_DIM
        candidate.register_buffer('_etrg_active_ridge_weight', torch.zeros_like(collector_module._integrated_ridge_weight), persistent=False)
        self.candidate = candidate

        def adjust(observable: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
            design = torch.cat((observable, torch.ones_like(observable[..., :1])), dim=-1)
            local = candidate.etrg_leaf_gate_linear[leaf.long()]
            return observable.new_tensor(self.alpha_qp) * torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
        install_pre_gate_refiner(candidate, self.encoder, lambda: candidate._etrg_active_ridge_weight, None, replace_base_hr=True, gate_logit_adjuster=adjust, candidate_blend_alpha=self.alpha_qp)

        def blended_parent_candidates(module, *, edge, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, previous_g, residual_hidden, base_gate, **_unused):
            delta, _leaf = conditional_delta(self.encoder, module._integrated_ridge_weight, edge=int(edge), previous_mixed=previous_mixed, h_candidate=h_candidate, base_hr_candidate=hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_g, residual_hidden=residual_hidden)
            strong = h_candidate + delta
            alpha = strong.new_tensor(self.alpha_qp).repeat_interleave(2)
            return (hr_candidate + alpha * (strong - hr_candidate), base_gate)
        collector_module.refine_candidates = types.MethodType(blended_parent_candidates, collector_module)
