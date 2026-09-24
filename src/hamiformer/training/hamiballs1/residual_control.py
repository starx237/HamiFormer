from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_schedule as light
from hamiformer.training.hamiballs1 import readout as compact_gate_offset_support
from hamiformer.training.hamiballs1.tree_encoder import CausalObservableTreeEncoder
from hamiformer.training.hamiballs1.component_tree import QPLateTreeController
OUTPUT = ROOT / 'outputs/hami1/teacher_forced_residual_control'
TF_OUTPUT = OUTPUT / 'teacher_forced_r0'
FINAL_OUTPUT = OUTPUT / 'final'
R_REGISTRATION = light.R_REGISTRATION
GATE_REGISTRATION = light.GATE_REGISTRATION

class TeacherForcedStateEncoder(CausalObservableTreeEncoder):
    apply_on_edge_zero = True

    def _features(self, **kwargs):
        previous = kwargs['previous_mixed'].to(torch.float16).to(torch.float32)
        result = super()._features(**kwargs)
        result.update({f'previous_{index}': previous[..., index] for index in range(4)})
        return result

class TeacherForcedColdStartController(QPLateTreeController):

    def __init__(self, path: Path) -> None:
        super().__init__(gate_mode='linear')
        with np.load(path, allow_pickle=False) as payload:
            names = [str(value) for value in payload['feature_names'].tolist()]
            mean = torch.from_numpy(payload['feature_mean']).float()
            scale = torch.from_numpy(payload['feature_scale']).float()
            weight = torch.from_numpy(payload['ridge_weight']).float()
            alpha = torch.tensor([float(payload['calibration_q']), float(payload['calibration_p'])], dtype=torch.float32)
        expected = ['previous_0', 'previous_1', 'previous_2', 'previous_3', 'Hstep_0', 'Hstep_1', 'Hstep_2', 'Hstep_3', 'mass', 'radius', 'attr2']
        if names != expected or weight.shape != (2, len(names) + 1, 2):
            raise ValueError('TF-r0 artifact feature contract drifted')
        self.encoder = TeacherForcedStateEncoder(tree={'depth': 0, 'leaf': 0}, feature_names=names, feature_mean=mean, feature_scale=scale)
        self.active_weight = torch.zeros(8, 2, len(names) + 1, 2)
        self.active_weight[0] = alpha[:, None, None] * weight
        self.tf_r0_artifact = str(path)

    def finish_stage(self, stage: str) -> None:
        super().finish_stage(stage)
        if stage == 'scalar0':
            self.encoder.apply_on_edge_zero = False

class NonfatalCompactUnifiedResidualGateOffset(compact_gate_offset_support.CompactUnifiedResidualGateOffset):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        for handle in self._gradient_hooks:
            handle.remove()
        self._gradient_hooks = []
