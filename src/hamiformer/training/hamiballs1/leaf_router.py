from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import torch
ROOT = project_root()
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import scalar_tree_fit as scalar_tree_support
ScalarTree_ROOT = ROOT / 'outputs/hami1/scalar_tree_fit'
OUTPUT = ROOT / 'outputs/hami1/tree_leaf_gate'
QP_OUTPUT = OUTPUT / 'qp'
RIDGE_OUTPUT = OUTPUT / 'ridge_terminal.pt'
OBSERVABLE_DIM = 71
DESIGN_DIM = 72

class SameRunTreeLeafLinearGate(scalar_tree_support.SameRunTreePureLeafResidual):

    def _install_optional_gate_bias(self, candidate):
        del candidate
        return None

    def _install_optional_gate_adjuster(self, candidate):
        device = next(candidate.parameters()).device
        candidate.register_parameter('etrg_leaf_gate_linear', torch.nn.Parameter(torch.zeros(8, 2, DESIGN_DIM, device=device)))
        candidate._declared_extra_trainable_parameters = 8 * 2 * DESIGN_DIM

        def adjust(observable: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
            if observable.shape[-1] != OBSERVABLE_DIM:
                raise ValueError('LeafGate expected observable71')
            design = torch.cat((observable, torch.ones_like(observable[..., :1])), dim=-1)
            local = candidate.etrg_leaf_gate_linear[leaf.long()]
            return torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
        return adjust

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload.update({'schema': 'hamiformer.hamiballs.etrg.LeafGate.same_run_tree_leaf_linear_gate.v1', 'gate_conditioning': 'per_leaf_qp_affine_observable71_logit_offset', 'gate_conditioning_parameters': 8 * 2 * DESIGN_DIM, 'constant_leaf_bias_parameters_removed': 16, 'tree_artifact': str(scalar_tree_support.TREE_OUTPUT), 'tree_artifact_sha256': sha256_file(scalar_tree_support.TREE_OUTPUT)})
        torch.save(payload, RIDGE_OUTPUT)
