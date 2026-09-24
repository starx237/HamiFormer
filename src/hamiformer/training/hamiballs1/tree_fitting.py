from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1 import tree_carriers as tree_carrier_support
from hamiformer.training.hamiballs1 import tree_refinement as refiner_training
from hamiformer.training.hamiballs1.component_tree import QPLateTreeController
from hamiformer.training.hamiballs1.stage_tree import StageBoundaryHierarchicalController
OUTPUT = ROOT / 'outputs/hami1/model_tree'
TF_OUTPUT = OUTPUT / 'teacher_forced_r0'
TREE_OUTPUT = OUTPUT / 'scalar1_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
MAX_LEAVES = 8
RIDGE_RELATIVE = 0.01

def _renumber(tree: dict[str, object]) -> int:
    count = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        if 'leaf' in node:
            node['leaf'] = count
            count += 1
        else:
            stack.append(node['right'])
            stack.append(node['left'])
    return count

def _replace_leaf(tree: dict[str, object], leaf: int, feature: str, threshold: float) -> dict[str, object]:
    result = copy.deepcopy(tree)
    stack = [result]
    found = False
    while stack:
        node = stack.pop()
        if int(node.get('leaf', -1)) == leaf:
            depth = int(node.get('depth', 0))
            node.clear()
            node.update({'feature': feature, 'threshold': threshold, 'left': {'leaf': -1, 'depth': depth + 1}, 'right': {'leaf': -1, 'depth': depth + 1}})
            found = True
            break
        if 'leaf' not in node:
            stack.extend((node['right'], node['left']))
    if not found:
        raise ValueError(f'leaf {leaf} not found')
    _renumber(result)
    return result

def _ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray | None=None) -> np.ndarray:
    if weight is None:
        gram = x.T @ x
        rhs = x.T @ y
    else:
        gram = x.T @ (weight[:, None] * x)
        rhs = x.T @ (weight[:, None] * y)
    lam = RIDGE_RELATIVE * max(float(np.diag(gram).mean()), 1e-12)
    penalty = np.eye(x.shape[1], dtype=np.float64) * lam
    penalty[-1, -1] = lam * 0.01
    return np.linalg.solve(gram + penalty, rhs)

def _quadratic_ridge(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gram = x.T @ (a[:, None] * x)
    rhs = x.T @ b
    lam = RIDGE_RELATIVE * max(float(np.diag(gram).mean()), 1e-12)
    penalty = np.eye(x.shape[1], dtype=np.float64) * lam
    penalty[-1, -1] = lam * 0.01
    return np.linalg.solve(gram + penalty, rhs)

def _fit_leaf_model(design: np.ndarray, h: np.ndarray, d: np.ndarray, target: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rw = np.zeros((2, design.shape[1], 2), np.float64)
    gw = np.zeros((2, design.shape[1]), np.float64)
    x = design[use]
    for component, sl in ((0, slice(0, 2)), (1, slice(2, 4))):
        rw[component] = _ridge(x, target[use, sl] - h[use, sl])
        hr = h[use, sl] + x @ rw[component]
        gap = hr - d[use, sl]
        a = np.square(gap).sum(-1)
        b = (gap * (target[use, sl] - d[use, sl])).sum(-1)
        gw[component] = _quadratic_ridge(x, a, b).reshape(-1)
    return (rw, gw)

def _predict_leaf_model(design: np.ndarray, h: np.ndarray, d: np.ndarray, index: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rw, gw = model
    result = np.empty((len(index), 4), np.float64)
    x = design[index]
    for component, sl in ((0, slice(0, 2)), (1, slice(2, 4))):
        hr = h[index, sl] + x @ rw[component]
        gate = np.clip(x @ gw[component], 0.0, 1.0)
        result[:, sl] = d[index, sl] + gate[:, None] * (hr - d[index, sl])
    return result

def _metrics(source: np.ndarray, target: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict[str, float]:
    old_q = np.square(old[:, :2] - target[:, :2]).sum(-1)
    new_q = np.square(new[:, :2] - target[:, :2]).sum(-1)
    old_p = np.square(old[:, 2:] - target[:, 2:]).sum(-1)
    new_p = np.square(new[:, 2:] - target[:, 2:]).sum(-1)
    old_total, new_total = (old_q + old_p, new_q + new_p)
    values = np.unique(source)
    source_win = np.mean([new_total[source == value].sum() < old_total[source == value].sum() for value in values])
    return {'q_ratio': float(new_q.sum() / max(float(old_q.sum()), 1e-30)), 'p_ratio': float(new_p.sum() / max(float(old_p.sum()), 1e-30)), 'total_ratio': float(new_total.sum() / max(float(old_total.sum()), 1e-30)), 'point_win': float(np.mean(new_total < old_total)), 'source_win': float(source_win), 'sources': int(len(values)), 'cells': int(len(source))}

def _fit_tree_models(tree: dict[str, object], data: dict[str, object], design: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    leaf = clean._assign(tree, np.asarray(data['x'], np.float32), list(data['names']))
    h = np.asarray(data['h_candidate'], np.float64)
    d = np.asarray(data['d_candidate'], np.float64)
    target = np.asarray(data['target'], np.float64)
    models = [_fit_leaf_model(design, h, d, target, fit & (leaf == region)) for region in range(int(leaf.max()) + 1)]
    return (leaf, models)

def _predict_tree(leaf: np.ndarray, models: list[tuple[np.ndarray, np.ndarray]], data: dict[str, object], design: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.flatnonzero(use)
    prediction = np.empty((len(index), 4), np.float64)
    h = np.asarray(data['h_candidate'], np.float64)
    d = np.asarray(data['d_candidate'], np.float64)
    local_leaf = leaf[index]
    for region, model in enumerate(models):
        local = local_leaf == region
        prediction[local] = _predict_leaf_model(design, h, d, index[local], model)
    return (index, prediction)

def _model_score(tree: dict[str, object], data: dict[str, object], design: np.ndarray, fit: np.ndarray, evaluate: np.ndarray) -> dict[str, float]:
    leaf, models = _fit_tree_models(tree, data, design, fit)
    index, prediction = _predict_tree(leaf, models, data, design, evaluate)
    target = np.asarray(data['target'], np.float64)[index]
    d = np.asarray(data['d_candidate'], np.float64)[index]
    row = _metrics(np.asarray(data['source'])[index], target, d, prediction)
    return {'q_gain_vs_D_percent': 100.0 * (1.0 - row['q_ratio']), 'p_gain_vs_D_percent': 100.0 * (1.0 - row['p_ratio']), 'total_gain_vs_D_percent': 100.0 * (1.0 - row['total_ratio']), 'point_win_vs_D': row['point_win'], 'source_win_vs_D': row['source_win']}

def _best_refinement(tree: dict[str, object], data: dict[str, object], design: np.ndarray, fit: np.ndarray, select: np.ndarray, candidate_names: list[str], responsibility: np.ndarray, source_weight: np.ndarray) -> tuple[dict[str, object] | None, dict[str, object]]:
    x = np.asarray(data['x'], np.float32)
    names = list(data['names'])
    lookup = {name: index for index, name in enumerate(names)}
    leaf = clean._assign(tree, x, names)
    h = np.asarray(data['h_candidate'], np.float64)
    d = np.asarray(data['d_candidate'], np.float64)
    target = np.asarray(data['target'], np.float64)
    source = np.asarray(data['source'], np.int64)
    candidates = []
    for region in range(int(leaf.max()) + 1):
        parent_fit = fit & (leaf == region)
        parent_select = select & (leaf == region)
        if min(int(parent_fit.sum()), int(parent_select.sum())) < 1024:
            continue
        parent_model = _fit_leaf_model(design, h, d, target, parent_fit)
        parent_index = np.flatnonzero(parent_select)
        parent_prediction = _predict_leaf_model(design, h, d, parent_index, parent_model)
        for name in candidate_names:
            column = lookup[name]
            values = x[parent_fit, column]
            best_proxy = None
            for threshold in np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9))):
                left_fit = parent_fit & (x[:, column] <= threshold)
                right_fit = parent_fit & ~left_fit
                if min(int(left_fit.sum()), int(right_fit.sum())) < 512:
                    continue
                if min(len(np.unique(source[left_fit])), len(np.unique(source[right_fit]))) < 12:
                    continue
                proxy = 0.0
                for side in (left_fit, right_fit):
                    w = source_weight[side]
                    y = responsibility[side]
                    mean = (w[:, None] * y).sum(0) / max(float(w.sum()), 1e-30)
                    proxy += float((w[:, None] * np.square(y - mean)).sum())
                if best_proxy is None or proxy < best_proxy[0]:
                    best_proxy = (proxy, float(threshold), left_fit, right_fit)
            if best_proxy is None:
                continue
            _, threshold, left_fit, right_fit = best_proxy
            child_models = (_fit_leaf_model(design, h, d, target, left_fit), _fit_leaf_model(design, h, d, target, right_fit))
            child = x[parent_index, column] <= threshold
            prediction = np.empty_like(parent_prediction)
            prediction[child] = _predict_leaf_model(design, h, d, parent_index[child], child_models[0])
            prediction[~child] = _predict_leaf_model(design, h, d, parent_index[~child], child_models[1])
            score = _metrics(source[parent_index], target[parent_index], parent_prediction, prediction)
            if score['q_ratio'] <= 1.0 and score['p_ratio'] <= 1.0 and (score['point_win'] >= 0.5) and (score['source_win'] >= 0.5):
                candidates.append((-score['source_win'], -score['point_win'], score['total_ratio'], name, threshold, region, score))
    if not candidates:
        return (None, {'admissible_candidates': 0})
    best = min(candidates)
    _, _, _, name, threshold, region, score = best
    refined = _replace_leaf(tree, int(region), str(name), float(threshold))
    return (refined, {'admissible_candidates': len(candidates), 'split_leaf': int(region), 'feature': name, 'threshold': threshold, 'disjoint_selection': score})

class HonestModelTreeController(tree_carrier_support.DeliveryScalarController):

    def _fit_tree_without_clearing(self) -> dict[str, object]:
        started = time.perf_counter()
        data = self._combine(self.bootstrap_records)
        source = np.asarray(data['source'], np.int64)
        fold = clean._fold(source)
        fit, select, head, safety = (fold == 0, fold == 1, fold == 2, fold == 3)
        initial_tree, _leaf, feature_names = clean._tree(data, fit | select)
        lookup = {name: index for index, name in enumerate(data['names'])}
        raw = np.asarray(data['x'], np.float32)[:, [lookup[n] for n in feature_names]]
        fit_mean = raw[fit].mean(0)
        fit_scale = np.maximum(raw[fit].std(0), 1e-05)
        fit_design = np.concatenate((np.clip((raw - fit_mean) / fit_scale, -8.0, 8.0), np.ones((len(raw), 1), np.float32)), axis=-1).astype(np.float64)
        responsibility, _ = refiner_training._responsibility(data, fit)
        source_weight = refiner_training._source_equal_weight(source, fit | select)
        responsibility, _, _ = refiner_training._weighted_standardize(responsibility, source_weight, fit)
        tree = copy.deepcopy(initial_tree)
        while _renumber(tree) < MAX_LEAVES:
            refined, row = _best_refinement(tree, data, fit_design, fit, select, feature_names, responsibility, source_weight)
            if refined is None:
                break
            tree = refined
        mean = raw[head].mean(0)
        scale = np.maximum(raw[head].std(0), 1e-05)
        design = self._design(data, feature_names, mean, scale)
        leaf = self._leaf_ids(tree, data)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        fitted = self._solve_np(design, residual, leaf, head)
        deployed, calibration = StageBoundaryHierarchicalController._global_calibrate(fitted, design, residual, leaf, source, safety)
        payload = {'schema': 'hamiformer.hamiballs.model_tree.v1', 'status': 'COMPLETE', 'source_stage': 'scalar1', 'sources': int(len(np.unique(source))), 'tree': copy.deepcopy(tree), 'leaf_count': _renumber(tree), 'feature_names': feature_names, 'feature_mean': torch.from_numpy(mean), 'feature_scale': torch.from_numpy(scale), 'residual_weight': torch.from_numpy(deployed).float(), 'component_calibration': calibration, 'split_admission': 'q/p pooled nonharm AND cell/source majority on disjoint fold1', 'wall_seconds': time.perf_counter() - started}
        torch.save(payload, TREE_OUTPUT)
        return {k: v for k, v in payload.items() if k not in {'feature_mean', 'feature_scale', 'residual_weight'}}
