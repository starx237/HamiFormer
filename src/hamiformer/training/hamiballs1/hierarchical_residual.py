from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import scalar_tree_fit as scalar_tree_support
from hamiformer.training.hamiballs1 import leaf_router as leaf_gate_support
from hamiformer.training.hamiballs1 import ridge_training as final
ScalarTree_ROOT = ROOT / 'outputs/hami1/scalar_tree_fit'
OUTPUT = ROOT / 'outputs/hami1/hierarchical_residual'
QP_OUTPUT = OUTPUT / 'qp'
RIDGE_OUTPUT = OUTPUT / 'ridge_terminal.pt'

class RoutedHierarchicalResidual(leaf_gate_support.SameRunTreeLeafLinearGate):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        tree_payload = torch.load(scalar_tree_support.TREE_OUTPUT, map_location='cpu', weights_only=False)
        self.reachable_leaf_count = int(tree_payload['leaf_count'])
        if not 1 <= self.reachable_leaf_count <= 8:
            raise ValueError('same-run tree leaf count is outside depth-3 capacity')

    def _solve(self):
        leaves = self.reachable_leaf_count
        width = final.INPUT_DIM
        result = torch.zeros(self.leaf_capacity, 2, width, 2, device=self.xtx.device)
        size = (leaves + 1) * width
        grams = self.xtx[:leaves]
        common_gram = grams.sum(0)
        reference = (common_gram.diagonal().mean() / float(leaves)).clamp_min(1e-08)
        ridge = final.RIDGE_RELATIVE * reference
        identity = torch.eye(size, device=self.xtx.device)
        common = slice(0, width)
        for component in range(2):
            rhs = self.xty[:leaves, component]
            matrix = torch.zeros(size, size, device=self.xtx.device)
            target = torch.zeros(size, 2, device=self.xtx.device)
            matrix[common, common] = common_gram
            target[common] = rhs.sum(0)
            for leaf in range(leaves):
                local = slice((leaf + 1) * width, (leaf + 2) * width)
                matrix[common, local] = grams[leaf]
                matrix[local, common] = grams[leaf]
                matrix[local, local] = grams[leaf]
                target[local] = rhs[leaf]
            solution = torch.linalg.solve(matrix + ridge * identity, target)
            shared = solution[common]
            for leaf in range(leaves):
                local = slice((leaf + 1) * width, (leaf + 2) * width)
                result[leaf, component] = shared + solution[local]
        return result

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload.update({'schema': 'hamiformer.hamiballs.etrg.HierarchicalResidual.final_history_hierarchical_r.v1', 'residual_parameterization': 'same_final_history_common_plus_leaf_deviation_compiled_to_leaf_matrix', 'hierarchical_ridge': {'reachable_leaves': self.reachable_leaf_count, 'relative_penalty': final.RIDGE_RELATIVE, 'penalty_reference': 'mean reachable-leaf Gram diagonal', 'searched_hyperparameters': 0, 'deployed_as_combined_leaf_matrices': True, 'extra_inference_parameters': 0}})
        torch.save(payload, RIDGE_OUTPUT)
