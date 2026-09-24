from __future__ import annotations
from hamiformer.utils.paths import project_root
import json
from pathlib import Path
import sys
import torch
ROOT = project_root()
from hamiformer.training import hami1_random_streams as seeds
from hamiformer.training.hamiballs1 import scalar_tree_fit as scalar_tree_support
OUTPUT = ROOT / 'outputs/hami1/tree_carrier_training'
TF_OUTPUT = OUTPUT / 'teacher_forced_r0'
TREE_OUTPUT = OUTPUT / 'scalar1_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
CELL_SAMPLE_NAMESPACE = 'etrg_same_run_cell_subsample'

class DeliveryScalarController(scalar_tree_support.Scalar1TreeController):

    def _cell_sample_seed(self, noise_slot: int) -> int:
        counter = 2 * int(self.global_block) + int(noise_slot)
        return seeds.derive_seed(CELL_SAMPLE_NAMESPACE, counter)

    @torch.no_grad()
    def finish_stage(self, stage: str) -> None:
        if stage != 'scalar1':
            super().finish_stage(stage)
            return
        data = self._combine(self.bootstrap_records)
        grow_rows = int(self._fold_masks(data)['grow'].sum())
        if grow_rows > 180000:
            raise RuntimeError('delivery CART unexpectedly entered the unaudited subsampling branch')
        tree_report = self._fit_tree_without_clearing()
        sampled_rows = int(sum((len(value['source']) for value in self.bootstrap_records)))
        self.bootstrap_records.clear()
        row = {'boundary_after': 'scalar1', 'mode': 'fit_model_tree', 'tree': tree_report, 'grow_rows': grow_rows, 'released_training_rows': sampled_rows}
        self.history.append(row)
        print(json.dumps({'tree_carrier_support_scalar1_boundary': row}), flush=True)
