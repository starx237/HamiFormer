from __future__ import annotations
from hamiformer.utils.paths import project_root
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
ROOT = project_root()
CARRIER = ROOT / 'outputs/hami1/router_train_carriers'
OUTPUT = ROOT / 'outputs/hami1/router'
LEDGER = ROOT / 'data/hamiballs_canonical_v2/packs/train48_stride48/rows.jsonl'
NOISE = (1942634267, 2035767743)
SHRINKS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)

@dataclass
class Node:
    depth: int
    feature: int | None = None
    threshold: float | None = None
    left: 'Node | None' = None
    right: 'Node | None' = None
    leaf: int | None = None

def _weighted_stats(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = (w[:, None] * y).sum(0) / max(float(w.sum()), 1e-20)
    var = (w[:, None] * np.square(y - mean)).sum(0) / max(float(w.sum()), 1e-20)
    return (mean, np.sqrt(np.maximum(var, 1e-12)))

def _select_features(x: np.ndarray, y: np.ndarray, w: np.ndarray, count: int) -> np.ndarray:
    xm, xs = _weighted_stats(x, w)
    ym, ys = _weighted_stats(y, w)
    covariance = (w[:, None, None] * (x - xm)[:, :, None] * (y - ym)[:, None, :]).sum(0) / w.sum()
    score = np.max(np.abs(covariance / np.maximum(xs[:, None] * ys[None], 1e-12)), axis=1)
    return np.argsort(score)[::-1][:count]

def _select_features_quad(x: np.ndarray, a: np.ndarray, b: np.ndarray, index: np.ndarray, count: int) -> np.ndarray:
    targets = np.concatenate((np.clip(b[:, :2] / np.maximum(a[:, :2], 1e-12), 0.0, 1.0), b[:, 2:4] / np.maximum(a[:, 2:3], 1e-08), b[:, 4:6] / np.maximum(a[:, 3:4], 1e-08)), axis=-1)
    scores = np.zeros(x.shape[1], np.float64)
    task_weights = (a[:, 0], a[:, 1], a[:, 2], a[:, 2], a[:, 3], a[:, 3])
    for column, local_weight in enumerate(task_weights):
        w = local_weight[index] + 1e-12
        xm, xs = _weighted_stats(x[index], w)
        ym, ys = _weighted_stats(targets[index, column:column + 1], w)
        covariance = (w[:, None] * (x[index] - xm) * (targets[index, column] - ym[0])[:, None]).sum(0) / w.sum()
        scores = np.maximum(scores, np.abs(covariance) / np.maximum(xs * ys[0], 1e-12))
    return np.argsort(scores)[::-1][:count]

def _quad_optimal(a: np.ndarray, b: np.ndarray, index: np.ndarray) -> np.ndarray:
    action = np.zeros(6, np.float64)
    action[:2] = np.clip(b[index, :2].sum(0) / np.maximum(a[index, :2].sum(0), 1e-20), 0.0, 1.0)
    action[2:4] = b[index, 2:4].sum(0) / max(float(a[index, 2].sum()), 1e-20)
    action[4:6] = b[index, 4:6].sum(0) / max(float(a[index, 3].sum()), 1e-20)
    return action

def _quad_risk(a: np.ndarray, b: np.ndarray, c: np.ndarray, index: np.ndarray, action: np.ndarray) -> np.ndarray:
    rows = np.empty(4, np.float64)
    rows[0] = (c[index, 0] + a[index, 0] * action[0] ** 2 - 2.0 * b[index, 0] * action[0]).sum()
    rows[1] = (c[index, 1] + a[index, 1] * action[1] ** 2 - 2.0 * b[index, 1] * action[1]).sum()
    rows[2] = (c[index, 2] + a[index, 2] * np.square(action[2:4]).sum() - 2.0 * (b[index, 2:4] * action[2:4]).sum(-1)).sum()
    rows[3] = (c[index, 3] + a[index, 3] * np.square(action[4:6]).sum() - 2.0 * (b[index, 4:6] * action[4:6]).sum(-1)).sum()
    return rows

def _quad_impurity(a: np.ndarray, b: np.ndarray, c: np.ndarray, index: np.ndarray, scale: np.ndarray) -> float:
    return float((_quad_risk(a, b, c, index, _quad_optimal(a, b, index)) / scale).sum())

def _quad_oracle_risk(a: np.ndarray, b: np.ndarray, c: np.ndarray, index: np.ndarray) -> np.ndarray:
    local_a, local_b, local_c = (a[index], b[index], c[index])
    gate = np.clip(local_b[:, :2] / np.maximum(local_a[:, :2], 1e-20), 0.0, 1.0)
    r_q = local_b[:, 2:4] / np.maximum(local_a[:, 2:3], 1e-20)
    r_p = local_b[:, 4:6] / np.maximum(local_a[:, 3:4], 1e-20)
    rows = np.empty(4, np.float64)
    rows[0] = (local_c[:, 0] + local_a[:, 0] * gate[:, 0] ** 2 - 2.0 * local_b[:, 0] * gate[:, 0]).sum()
    rows[1] = (local_c[:, 1] + local_a[:, 1] * gate[:, 1] ** 2 - 2.0 * local_b[:, 1] * gate[:, 1]).sum()
    rows[2] = (local_c[:, 2] + local_a[:, 2] * np.square(r_q).sum(-1) - 2.0 * (local_b[:, 2:4] * r_q).sum(-1)).sum()
    rows[3] = (local_c[:, 3] + local_a[:, 3] * np.square(r_p).sum(-1) - 2.0 * (local_b[:, 4:6] * r_p).sum(-1)).sum()
    return rows

def _grow_quad(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, scale: np.ndarray, source: np.ndarray, index: np.ndarray, depth: int=0, max_depth: int=3) -> Node:
    node = Node(depth=depth)
    if depth >= max_depth or len(index) < 8192:
        return node
    parent = _quad_impurity(a, b, c, index, scale)
    best = None
    for feature in range(x.shape[-1]):
        value = x[index, feature]
        for threshold in np.unique(np.quantile(value, np.linspace(0.1, 0.9, 9))):
            choose = value <= threshold
            left, right = (index[choose], index[~choose])
            if min(len(left), len(right)) < 4096:
                continue
            if min(len(np.unique(source[left])), len(np.unique(source[right]))) < 24:
                continue
            gain = parent - _quad_impurity(a, b, c, left, scale) - _quad_impurity(a, b, c, right, scale)
            candidate = (gain, min(len(left), len(right)), feature, float(threshold), left, right)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] <= 0.0:
        return node
    _, _, node.feature, node.threshold, left, right = best
    node.left = _grow_quad(x, a, b, c, scale, source, left, depth + 1, max_depth)
    node.right = _grow_quad(x, a, b, c, scale, source, right, depth + 1, max_depth)
    return node

def _impurity(y: np.ndarray, w: np.ndarray, index: np.ndarray) -> float:
    local_w = w[index]
    local_y = y[index]
    mean = (local_w[:, None] * local_y).sum(0) / max(float(local_w.sum()), 1e-20)
    return float((local_w[:, None] * np.square(local_y - mean)).sum())

def _grow(x: np.ndarray, y: np.ndarray, w: np.ndarray, source: np.ndarray, index: np.ndarray, depth: int=0, max_depth: int=3) -> Node:
    node = Node(depth=depth)
    if depth >= max_depth or len(index) < 8192:
        return node
    parent = _impurity(y, w, index)
    best = None
    for feature in range(x.shape[-1]):
        value = x[index, feature]
        for threshold in np.unique(np.quantile(value, np.linspace(0.1, 0.9, 9))):
            choose = value <= threshold
            left, right = (index[choose], index[~choose])
            if min(len(left), len(right)) < 4096:
                continue
            if min(len(np.unique(source[left])), len(np.unique(source[right]))) < 24:
                continue
            gain = parent - _impurity(y, w, left) - _impurity(y, w, right)
            candidate = (gain, min(len(left), len(right)), feature, float(threshold), left, right)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] <= 0.0:
        return node
    _, _, node.feature, node.threshold, left, right = best
    node.left = _grow(x, y, w, source, left, depth + 1, max_depth)
    node.right = _grow(x, y, w, source, right, depth + 1, max_depth)
    return node

def _assign_leaves(root: Node, x: np.ndarray) -> tuple[np.ndarray, list[Node]]:
    result = np.empty(len(x), np.int16)
    leaves = []
    stack = [(root, np.arange(len(x), dtype=np.int64))]
    while stack:
        node, index = stack.pop()
        if node.left is None or node.right is None:
            node.leaf = len(leaves)
            leaves.append(node)
            result[index] = node.leaf
            continue
        choose = x[index, node.feature] <= node.threshold
        stack.append((node.right, index[~choose]))
        stack.append((node.left, index[choose]))
    return (result, leaves)

def _tree_rows(node: Node, names: list[str]) -> dict[str, object]:
    if node.left is None or node.right is None:
        return {'leaf': node.leaf, 'depth': node.depth}
    return {'feature': names[node.feature], 'threshold': node.threshold, 'left': _tree_rows(node.left, names), 'right': _tree_rows(node.right, names)}
