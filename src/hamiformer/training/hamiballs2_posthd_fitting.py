from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable
import numpy as np

def derive_seed(master_seed: int, namespace: str, index: int=0) -> int:
    payload = f'hamiballs2-posthd|{int(master_seed)}|{namespace}|{int(index)}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') & 2147483647

def source_fold(source_ids: np.ndarray, *, master_seed: int, folds: int=4) -> np.ndarray:
    if folds < 2:
        raise ValueError('at least two source folds are required')
    salt = derive_seed(master_seed, 'tree_source_folds')
    return np.asarray([int.from_bytes(hashlib.sha256(f'{salt}|{int(value)}'.encode()).digest()[:4], 'little') % folds for value in np.asarray(source_ids).reshape(-1)], dtype=np.int8)

def source_equal_weights(source_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    source_ids = np.asarray(source_ids).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if source_ids.shape != selected.shape:
        raise ValueError('source ids and selection must align')
    weight = np.zeros(len(source_ids), dtype=np.float64)
    active = np.unique(source_ids[selected])
    for source in active:
        local = selected & (source_ids == source)
        weight[local] = 1.0 / max(int(local.sum()), 1)
    if len(active):
        weight /= len(active)
    return weight

def cart_responsibility(d_candidate: np.ndarray, hr_candidate: np.ndarray, h_candidate: np.ndarray, target: np.ndarray, *, qp_scale: np.ndarray, include_endpoint_advantage: bool=False) -> np.ndarray:
    values = tuple((np.asarray(value, dtype=np.float64) for value in (d_candidate, hr_candidate, h_candidate, target)))
    if any((value.shape != values[0].shape for value in values)) or values[0].shape[-1] != 6:
        raise ValueError('Tree responsibility states must align with width six')
    qp_scale = np.asarray(qp_scale, dtype=np.float64)
    if qp_scale.shape != (2,) or np.any(qp_scale <= 0):
        raise ValueError('Tree q/p scales must be positive')
    routing = []
    advantages = []
    for component in (slice(0, 3), slice(3, 6)):
        delta = values[1][..., component] - values[0][..., component]
        error = values[3][..., component] - values[0][..., component]
        denominator = np.square(delta).sum(-1)
        routing.append(np.clip((error * delta).sum(-1) / np.maximum(denominator, 1e-30), 0.0, 1.0))
        if include_endpoint_advantage:
            d_mse = np.square(values[3][..., component] - values[0][..., component]).mean(-1)
            hr_mse = np.square(values[3][..., component] - values[1][..., component]).mean(-1)
            advantages.append((d_mse - hr_mse) / np.maximum(d_mse + hr_mse, 1e-30))
    residual = values[3] - values[2]
    residual[..., :3] /= qp_scale[0]
    residual[..., 3:] /= qp_scale[1]
    parts = [np.stack(routing, -1)]
    if include_endpoint_advantage:
        parts.append(np.stack(advantages, -1))
    parts.append(residual)
    return np.concatenate(parts, -1)

@dataclass(frozen=True)
class TreeFitResult:
    tree: dict[str, object]
    selected_depth: int
    selected_leaf_count: int
    fold_report: dict[str, object]

def _serialize_tree(model) -> dict[str, object]:
    leaf_nodes = np.flatnonzero(model.tree_.children_left < 0)
    node_to_leaf = np.full(model.tree_.node_count, -1, dtype=np.int64)
    node_to_leaf[leaf_nodes] = np.arange(len(leaf_nodes), dtype=np.int64)
    return {'feature': model.tree_.feature.tolist(), 'threshold': model.tree_.threshold.tolist(), 'children_left': model.tree_.children_left.tolist(), 'children_right': model.tree_.children_right.tolist(), 'node_to_leaf': node_to_leaf.tolist(), 'depth': int(model.get_depth()), 'node_count': int(model.tree_.node_count), 'leaf_count': int(model.get_n_leaves())}

def fit_source_balanced_cart(features: np.ndarray, responsibility: np.ndarray, source_ids: np.ndarray, *, master_seed: int, max_depth: int=4, max_leaves: int=8, feature_budget: int=32) -> TreeFitResult:
    from sklearn.tree import DecisionTreeRegressor
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(responsibility, dtype=np.float32)
    source = np.asarray(source_ids).reshape(-1)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or (len(x) != len(source)):
        raise ValueError('Tree rows must align')
    fold = source_fold(source, master_seed=master_seed)
    fit, select = (fold == 0, fold == 1)
    if not bool(fit.any()) or not bool(select.any()):
        raise ValueError('Tree fitting requires nonempty fold0 and fold1')
    center = np.median(y[fit], axis=0)
    dispersion = np.median(np.abs(y[fit] - center), axis=0)
    y = np.clip((y - center) / np.maximum(dispersion, 1e-06), -8.0, 8.0)
    x_center = np.median(x[fit], axis=0)
    x_dispersion = np.median(np.abs(x[fit] - x_center), axis=0)
    standardized_x = np.clip((x - x_center) / np.maximum(x_dispersion, 1e-06), -8.0, 8.0)
    fit_weight = source_equal_weights(source, fit)[fit, None]
    covariance = (fit_weight * standardized_x[fit]).T @ y[fit]
    score = np.linalg.norm(covariance, axis=1)
    selected_features = np.sort(np.argsort(score)[-min(int(feature_budget), x.shape[1]):])
    tree_x = x[:, selected_features]
    candidates: list[tuple[float, int, object]] = []
    random_state = derive_seed(master_seed, 'cart_tie_break')
    minimum_leaf = max(2, int(math.ceil(math.sqrt(int(fit.sum())))))
    for depth in range(1, max_depth + 1):
        model = DecisionTreeRegressor(criterion='squared_error', max_depth=depth, max_leaf_nodes=max_leaves, min_samples_leaf=minimum_leaf, random_state=random_state)
        weights = source_equal_weights(source, fit)
        model.fit(tree_x[fit], y[fit], sample_weight=weights[fit])
        prediction = model.predict(tree_x[select])
        select_weights = source_equal_weights(source, select)[select, None]
        risk = float((select_weights * np.square(prediction - y[select])).sum())
        candidates.append((risk, depth, model))
    _, selected_depth, _ = min(candidates, key=lambda row: (row[0], row[1]))
    refit = fit | select
    final = DecisionTreeRegressor(criterion='squared_error', max_depth=selected_depth, max_leaf_nodes=max_leaves, min_samples_leaf=max(2, int(math.ceil(math.sqrt(int(refit.sum()))))), random_state=random_state)
    weights = source_equal_weights(source, refit)
    final.fit(tree_x[refit], y[refit], sample_weight=weights[refit])
    serialized = _serialize_tree(final)
    serialized['feature'] = [int(selected_features[index]) if index >= 0 else -2 for index in serialized['feature']]
    return TreeFitResult(tree=serialized, selected_depth=selected_depth, selected_leaf_count=int(final.get_n_leaves()), fold_report={'structure_fit': 0, 'structure_select': 1, 'readout_statistics': 2, 'component_calibration': 3, 'candidate_select_risk': {str(depth): risk for risk, depth, _ in candidates}, 'source_balanced': True, 'tree_random_state': random_state, 'responsibility_center': center.tolist(), 'responsibility_scale': dispersion.tolist(), 'selected_feature_indices': selected_features.tolist(), 'feature_budget': int(feature_budget)})

def apply_serialized_tree(tree: dict[str, object], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features)
    feature = np.asarray(tree['feature'], dtype=np.int64)
    threshold = np.asarray(tree['threshold'], dtype=np.float64)
    left = np.asarray(tree['children_left'], dtype=np.int64)
    right = np.asarray(tree['children_right'], dtype=np.int64)
    node_to_leaf = np.asarray(tree['node_to_leaf'], dtype=np.int64)
    node = np.zeros(len(x), dtype=np.int64)
    for _ in range(int(tree['depth'])):
        selected = feature[node]
        terminal = selected < 0
        choose_left = x[np.arange(len(x)), np.maximum(selected, 0)] <= threshold[node]
        node = np.where(terminal, node, np.where(choose_left, left[node], right[node]))
    result = node_to_leaf[node]
    if np.any(result < 0):
        raise RuntimeError('serialized CART traversal did not terminate')
    return result

class HierarchicalRidgeAccumulator:

    def __init__(self, design_dim: int, *, max_leaves: int=8) -> None:
        if design_dim < 2 or max_leaves < 1:
            raise ValueError('invalid hierarchical ridge dimensions')
        self.design_dim = int(design_dim)
        self.max_leaves = int(max_leaves)
        self._rows: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def add(self, design: np.ndarray, target: np.ndarray, leaf: np.ndarray, source_ids: np.ndarray) -> None:
        x = np.asarray(design, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        leaf = np.asarray(leaf, dtype=np.int64).reshape(-1)
        source = np.asarray(source_ids, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.design_dim or y.shape != (len(x), 6):
            raise ValueError('hierarchical ridge design/target shape mismatch')
        if len(leaf) != len(x) or len(source) != len(x):
            raise ValueError('hierarchical ridge row labels do not align')
        if np.any(leaf < 0) or np.any(leaf >= self.max_leaves):
            raise ValueError('hierarchical ridge leaf is outside the fixed budget')
        for source_id in np.unique(source):
            use = source == source_id
            self._rows.setdefault(int(source_id), []).append((x[use].copy(), y[use].copy(), leaf[use].copy()))

    @property
    def source_count(self) -> int:
        return len(self._rows)

    @property
    def row_count(self) -> int:
        return sum((len(x) for chunks in self._rows.values() for x, _, _ in chunks))

    def solve(self, *, ridge_relative: float=0.01, intercept_multiplier: float=0.01) -> tuple[np.ndarray, dict[str, object]]:
        if not self._rows:
            raise ValueError('cannot solve an empty hierarchical ridge')
        d, groups = (self.design_dim, self.max_leaves + 1)
        width = d * groups
        result = np.zeros((self.max_leaves, 2, d, 3), dtype=np.float64)
        diagnostics: dict[str, object] = {}
        for component, label, slc in ((0, 'q', slice(0, 3)), (1, 'p', slice(3, 6))):
            gram = np.zeros((width, width), dtype=np.float64)
            rhs = np.zeros((width, 3), dtype=np.float64)
            for chunks in self._rows.values():
                x = np.concatenate([row[0] for row in chunks], axis=0)
                y = np.concatenate([row[1] for row in chunks], axis=0)[:, slc]
                leaf = np.concatenate([row[2] for row in chunks], axis=0)
                source_weight = 1.0 / len(x)
                common = slice(0, d)
                common_gram = source_weight * (x.T @ x)
                common_rhs = source_weight * (x.T @ y)
                gram[common, common] += common_gram
                rhs[common] += common_rhs
                for region in np.unique(leaf):
                    use = leaf == region
                    local_x, local_y = (x[use], y[use])
                    local_gram = source_weight * (local_x.T @ local_x)
                    local_rhs = source_weight * (local_x.T @ local_y)
                    local = slice((int(region) + 1) * d, (int(region) + 2) * d)
                    gram[common, local] += local_gram
                    gram[local, common] += local_gram
                    gram[local, local] += local_gram
                    rhs[local] += local_rhs
            lam = float(ridge_relative) * max(float(np.diag(gram[:d, :d]).mean()), 1e-12)
            penalty = np.full(width, lam, dtype=np.float64)
            for block in range(groups):
                penalty[(block + 1) * d - 1] *= float(intercept_multiplier)
            coefficient = np.linalg.solve(gram + np.diag(penalty), rhs).reshape(groups, d, 3)
            shared = coefficient[0]
            for region in range(self.max_leaves):
                result[region, component] = shared + coefficient[region + 1]
            diagnostics[label] = {'lambda': lam, 'common_l2': float(np.linalg.norm(shared)), 'leaf_deviation_l2': [float(np.linalg.norm(coefficient[i + 1])) for i in range(self.max_leaves)]}
        diagnostics.update({'sources': self.source_count, 'rows': self.row_count, 'source_equal': True, 'cumulative': True, 'ridge_relative': float(ridge_relative), 'intercept_multiplier': float(intercept_multiplier)})
        return (result, diagnostics)

def calibrate_component_alpha(prediction: np.ndarray, target: np.ndarray, source_ids: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source_ids).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if prediction.shape != target.shape or prediction.shape != (len(source), 6):
        raise ValueError('component calibration rows must align')
    weights = source_equal_weights(source, selected)
    alpha = np.zeros(2, dtype=np.float64)
    report: dict[str, object] = {}
    for index, (name, component) in enumerate((('q', slice(0, 3)), ('p', slice(3, 6)))):
        p, y, w = (prediction[:, component], target[:, component], weights[:, None])
        numerator = float((w * p * y).sum())
        denominator = float((w * p * p).sum())
        alpha[index] = np.clip(numerator / max(denominator, 1e-30), 0.0, 1.0)
        before = float((w * y * y).sum())
        after = float((w * np.square(y - alpha[index] * p)).sum())
        report[name] = {'alpha': float(alpha[index]), 'before_sse': before, 'after_sse': after}
    return (alpha, report)
__all__ = ['HierarchicalRidgeAccumulator', 'TreeFitResult', 'apply_serialized_tree', 'calibrate_component_alpha', 'cart_responsibility', 'derive_seed', 'fit_source_balanced_cart', 'source_equal_weights', 'source_fold']
