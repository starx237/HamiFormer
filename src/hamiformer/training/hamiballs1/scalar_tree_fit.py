from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = project_root()
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1 import residual_control as teacher_control_support
from hamiformer.training.hamiballs1.component_tree import QPLateTreeController
from hamiformer.training.hamiballs1.stage_tree import StageBoundaryHierarchicalController
TeacherControl_ROOT = ROOT / 'outputs/hami1/teacher_forced_residual_control'
OUTPUT = ROOT / 'outputs/hami1/scalar_tree_fit'
TREE_OUTPUT = OUTPUT / 'scalar1_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'

class Scalar1TreeController(teacher_control_support.TeacherForcedColdStartController):

    def _fit_tree_without_clearing(self) -> dict[str, object]:
        started = time.perf_counter()
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        tree, leaf, names = clean._tree(data, masks['grow'])
        lookup = {name: index for index, name in enumerate(data['names'])}
        raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
        mean = raw[masks['head']].mean(0)
        scale = np.maximum(raw[masks['head']].std(0), 1e-05)
        design = self._design(data, names, mean, scale)
        leaf = self._leaf_ids(tree, data)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        fitted = self._solve_np(design, residual, leaf, masks['head'])
        deployed, calibration = StageBoundaryHierarchicalController._global_calibrate(fitted, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        payload = {'schema': 'hamiformer.hamiballs.scalar_model_tree.v1', 'status': 'COMPLETE', 'source_stage': 'scalar1', 'sources': int(len(np.unique(data['source']))), 'tree': copy.deepcopy(tree), 'leaf_count': int(leaf.max()) + 1, 'feature_names': names, 'feature_mean': torch.from_numpy(mean), 'feature_scale': torch.from_numpy(scale), 'residual_weight': torch.from_numpy(deployed).float(), 'component_calibration': calibration, 'wall_seconds': time.perf_counter() - started}
        torch.save(payload, TREE_OUTPUT)
        return {key: value for key, value in payload.items() if key not in {'feature_mean', 'feature_scale', 'residual_weight'}}

    @torch.no_grad()
    def finish_stage(self, stage: str) -> None:
        if stage == 'scalar1':
            report = self._fit_tree_without_clearing()
            print(json.dumps({'scalar_tree_support_same_run_tree': report}), flush=True)
        super().finish_stage(stage)

class SameRunTreePureLeafResidual(teacher_control_support.NonfatalCompactUnifiedResidualGateOffset):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        payload = torch.load(TREE_OUTPUT, map_location='cpu', weights_only=False)
        if payload.get('status') != 'COMPLETE':
            raise ValueError('same-run scalar1 tree artifact is incomplete')
        names = list(payload['feature_names'])
        if len(names) != 32 or any((name.startswith(('gate_', 'r_')) for name in names)):
            raise ValueError('same-run tree/readout basis is not causal compact32')
        device = self.encoder.feature_mean.device
        dtype = self.encoder.feature_mean.dtype
        self.encoder.tree = copy.deepcopy(payload['tree'])
        self.encoder.feature_names = names
        self.encoder.feature_mean = payload['feature_mean'].to(device=device, dtype=dtype)
        self.encoder.feature_scale = payload['feature_scale'].to(device=device, dtype=dtype)
        print(json.dumps({'scalar_tree_support_installed_same_run_tree': {'leaf_count': payload['leaf_count'], 'feature_count': len(names), 'tree_artifact_sha256': sha256_file(TREE_OUTPUT)}}), flush=True)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload.update({'schema': 'hamiformer.hamiballs.etrg.ScalarTree.same_run_scalar1_tree.v1', 'tree_and_statistics_source': 'same_run_scalar1_refresh_records', 'tree_artifact': str(TREE_OUTPUT), 'tree_artifact_sha256': sha256_file(TREE_OUTPUT)})
        torch.save(payload, RIDGE_OUTPUT)
