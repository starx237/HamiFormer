from __future__ import annotations
from hamiformer.utils.paths import project_root
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_features as feature_library
OUTPUT = ROOT / 'outputs/hami1/clean_tree_unified_r'
LEDGER = ROOT / 'data/hamiballs_canonical_v2/packs/train48_stride48/rows.jsonl'
DEPTH = 3
FEATURE_COUNT = 32
WIDTH = 16
UPDATES = 400
BATCH = 8192

def _fold(source: np.ndarray) -> np.ndarray:
    ledger = [json.loads(line) for line in LEDGER.read_text(encoding='utf-8').splitlines() if line.strip()]
    scenes = np.asarray([row['scene_id'] for row in ledger])
    return np.asarray([hashlib.sha256(scenes[int(value)].encode()).digest()[0] % 4 for value in source], np.int8)

def _clean_columns(names: list[str]) -> np.ndarray:
    allowed = []
    for index, name in enumerate(names):
        if name.startswith('observable71_'):
            coordinate = int(name.rsplit('_', 1)[1])
            if 31 <= coordinate <= 42:
                allowed.append(index)
            continue
        if name.startswith('gap_') and (not name.startswith('gap_accept_')) or name.startswith(('Hstep_', 'Dstep_', 'Hstep_Dstep_')):
            allowed.append(index)
            continue
        if name in {'mass', 'radius', 'attr2', 'rf_trace'}:
            allowed.append(index)
    return np.asarray(allowed, np.int64)

def _pure_quadratics(data: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(data['target'], np.float64)
    h = np.asarray(data['h_candidate'], np.float64)
    d = np.asarray(data['d_candidate'], np.float64)
    gap = h - d
    target_d = target - d
    residual = target - h
    rows = len(target)
    a = np.ones((rows, 4), np.float64)
    b = np.zeros((rows, 6), np.float64)
    c = np.zeros((rows, 4), np.float64)
    for component, sl in ((0, slice(0, 2)), (1, slice(2, 4))):
        a[:, component] = np.square(gap[:, sl]).sum(-1)
        b[:, component] = (gap[:, sl] * target_d[:, sl]).sum(-1)
        c[:, component] = np.square(target_d[:, sl]).sum(-1)
        action = 2 + 2 * component
        b[:, action:action + 2] = residual[:, sl]
        c[:, 2 + component] = np.square(residual[:, sl]).sum(-1)
    return (a, b, c)

def _tree(data: dict[str, object], grow: np.ndarray) -> tuple[dict[str, object], np.ndarray, list[str]]:
    x = np.asarray(data['x'], np.float32)
    names = list(data['names'])
    source = np.asarray(data['source'], np.int64)
    allowed = _clean_columns(names)
    a, b, c = _pure_quadratics(data)
    grow_index = np.flatnonzero(grow)
    rng = np.random.default_rng(42043)
    if len(grow_index) > 180000:
        grow_index = np.sort(rng.choice(grow_index, 180000, replace=False))
    local_selected = feature_library._select_features_quad(x[:, allowed], a, b, grow_index, min(FEATURE_COUNT, len(allowed)))
    selected = allowed[local_selected]
    global_risk = feature_library._quad_risk(a, b, c, grow_index, feature_library._quad_optimal(a, b, grow_index))
    oracle_risk = feature_library._quad_oracle_risk(a, b, c, grow_index)
    scale = np.maximum(global_risk - oracle_risk, 1e-20)
    root = feature_library._grow_quad(x[:, selected], a, b, c, scale, source, grow_index, max_depth=DEPTH)
    leaf, leaves = feature_library._assign_leaves(root, x[:, selected])
    return (feature_library._tree_rows(root, [names[i] for i in selected]), leaf, [names[i] for i in selected])

def _assign(tree: dict[str, object], x: np.ndarray, names: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(names)}
    result = np.empty(len(x), np.int64)
    stack = [(tree, np.arange(len(x), dtype=np.int64))]
    while stack:
        node, index = stack.pop()
        if 'leaf' in node:
            result[index] = int(node['leaf'])
            continue
        choose = x[index, lookup[str(node['feature'])]] <= float(node['threshold'])
        stack.append((node['right'], index[~choose]))
        stack.append((node['left'], index[choose]))
    return result

class CompactLeafResidual(nn.Module):

    def __init__(self, input_dim: int, leaves: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, WIDTH), nn.SiLU())
        self.weight = nn.Parameter(torch.zeros(leaves, 2, WIDTH, 2))
        self.bias = nn.Parameter(torch.zeros(leaves, 2, 2))

    def forward(self, x: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        local_weight = self.weight[leaf]
        local_bias = self.bias[leaf]
        value = torch.einsum('bi,bcij->bcj', hidden, local_weight) + local_bias
        return torch.cat((value[:, 0], value[:, 1]), dim=-1)
