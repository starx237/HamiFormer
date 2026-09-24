from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_features as tree_tools
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1 import tree_carriers as tree_carrier_support
from hamiformer.training.hamiballs1.component_tree import QPLateTreeController
from hamiformer.training.hamiballs1.stage_tree import StageBoundaryHierarchicalController
OUTPUT = ROOT / 'outputs/hami1/conflict_aware_tree'
TF_OUTPUT = OUTPUT / 'teacher_forced_r0'
TREE_OUTPUT = OUTPUT / 'scalar1_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
TARGET_NAMES = ('gate_q', 'gate_p', 'r_q_dir0', 'r_q_dir1', 'r_q_log_amplitude', 'r_p_dir0', 'r_p_dir1', 'r_p_log_amplitude')

def _source_equal_weight(source: np.ndarray, use: np.ndarray) -> np.ndarray:
    weight = np.zeros(len(source), np.float64)
    values, counts = np.unique(source[use], return_counts=True)
    for value, count in zip(values, counts):
        weight[use & (source == value)] = 1.0 / max(int(count), 1)
    mean = float(weight[use].mean())
    weight[use] /= max(mean, 1e-30)
    return weight

def _responsibility(data: dict[str, object], grow: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    target = np.asarray(data['target'], np.float64)
    h = np.asarray(data['h_candidate'], np.float64)
    d = np.asarray(data['d_candidate'], np.float64)
    residual = target - h
    rows = []
    scales: dict[str, float] = {}
    gate_rows = []
    for label, sl in (('q', slice(0, 2)), ('p', slice(2, 4))):
        gap = h[:, sl] - d[:, sl]
        alpha = np.clip((gap * (target[:, sl] - d[:, sl])).sum(-1) / np.maximum(np.square(gap).sum(-1), 1e-20), 0.0, 1.0)
        gate_rows.append(alpha[:, None])
        local = residual[:, sl]
        norm = np.sqrt(np.square(local).sum(-1))
        positive = norm[grow & (norm > 0)]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-12)
        scales[f'r_{label}_amplitude_median'] = scale
        direction = local / np.maximum(norm[:, None], 1e-12)
        amplitude = np.log1p(norm / scale)[:, None]
        rows.append(np.concatenate((direction, amplitude), axis=-1))
    return (np.concatenate((gate_rows[0], gate_rows[1], rows[0], rows[1]), axis=-1), scales)

def _weighted_standardize(y: np.ndarray, weight: np.ndarray, use: np.ndarray):
    w = weight[use]
    local = y[use]
    mean = (w[:, None] * local).sum(0) / max(float(w.sum()), 1e-30)
    var = (w[:, None] * np.square(local - mean)).sum(0) / max(float(w.sum()), 1e-30)
    scale = np.sqrt(np.maximum(var, 1e-12))
    return ((y - mean) / scale, mean, scale)

def _leaf_conflict(tree: dict[str, object], data: dict[str, object], y: np.ndarray, source_weight: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    leaf = clean._assign(tree, np.asarray(data['x'], np.float32), list(data['names']))
    rows = []
    total = 0.0
    for region in sorted((int(v) for v in np.unique(leaf))):
        use = mask & (leaf == region)
        w = source_weight[use]
        local = y[use]
        mean = (w[:, None] * local).sum(0) / max(float(w.sum()), 1e-30)
        variance = float((w[:, None] * np.square(local - mean)).sum())
        total += variance
        rows.append({'leaf': region, 'cells': int(use.sum()), 'sources': int(len(np.unique(np.asarray(data['source'])[use]))), 'normalized_conflict': variance / max(float(w.sum()), 1e-30), 'gate_q_mean': float(mean[0]), 'gate_p_mean': float(mean[1]), 'r_q_direction_resultant': float(np.linalg.norm(mean[2:4])), 'r_q_log_amplitude_mean': float(mean[4]), 'r_p_direction_resultant': float(np.linalg.norm(mean[5:7])), 'r_p_log_amplitude_mean': float(mean[7])})
    return {'weighted_conflict': total, 'weighted_conflict_per_source_weight': total / max(float(source_weight[mask].sum()), 1e-30), 'leaves': rows}

class ConflictAwareScalarController(tree_carrier_support.DeliveryScalarController):

    def _fit_tree_without_clearing(self) -> dict[str, object]:
        started = time.perf_counter()
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        source = np.asarray(data['source'], np.int64)
        x = np.asarray(data['x'], np.float32)
        names = list(data['names'])
        allowed = clean._clean_columns(names)
        source_weight = _source_equal_weight(source, masks['grow'] | masks['safety'])
        responsibility, amplitude_scales = _responsibility(data, masks['grow'])
        standardized, target_mean, target_scale = _weighted_standardize(responsibility, source_weight, masks['grow'])
        selected_local = tree_tools._select_features(x[:, allowed], standardized, source_weight, min(clean.FEATURE_COUNT, len(allowed)))
        selected = allowed[selected_local]
        root = tree_tools._grow(x[:, selected], standardized, source_weight, source, np.flatnonzero(masks['grow']), max_depth=clean.DEPTH)
        _leaf, leaves = tree_tools._assign_leaves(root, x[:, selected])
        tree = tree_tools._tree_rows(root, [names[i] for i in selected])
        baseline_tree, _baseline_leaf, _baseline_names = clean._tree(data, masks['grow'])
        comparison = {'pooled_quadratic_tree': _leaf_conflict(baseline_tree, data, standardized, source_weight, masks['safety']), 'conflict_aware_tree': _leaf_conflict(tree, data, standardized, source_weight, masks['safety'])}
        chosen_names = [names[i] for i in selected]
        lookup = {name: index for index, name in enumerate(names)}
        raw = x[:, [lookup[name] for name in chosen_names]]
        mean = raw[masks['head']].mean(0)
        scale = np.maximum(raw[masks['head']].std(0), 1e-05)
        design = self._design(data, chosen_names, mean, scale)
        leaf = self._leaf_ids(tree, data)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        fitted = self._solve_np(design, residual, leaf, masks['head'])
        deployed, calibration = StageBoundaryHierarchicalController._global_calibrate(fitted, design, residual, leaf, source, masks['safety'])
        payload = {'schema': 'hamiformer.hamiballs.conflict_aware_tree.v1', 'status': 'COMPLETE', 'source_stage': 'scalar1', 'sources': int(len(np.unique(source))), 'tree': copy.deepcopy(tree), 'leaf_count': len(leaves), 'feature_names': chosen_names, 'feature_mean': torch.from_numpy(mean), 'feature_scale': torch.from_numpy(scale), 'residual_weight': torch.from_numpy(deployed).float(), 'component_calibration': calibration, 'tree_target_names': list(TARGET_NAMES), 'tree_target_mean': target_mean.tolist(), 'tree_target_scale': target_scale.tolist(), 'residual_amplitude_scales': amplitude_scales, 'source_weighting': 'each physical source has equal total weight', 'tree_comparison': comparison, 'wall_seconds': time.perf_counter() - started}
        torch.save(payload, TREE_OUTPUT)
        return {key: value for key, value in payload.items() if key not in {'feature_mean', 'feature_scale', 'residual_weight'}}
