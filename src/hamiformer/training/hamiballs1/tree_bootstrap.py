from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1.tree_controller import SameRunETrController
R_REGISTRATION = ROOT / 'configs/hamiballs1/residual.json'
GATE_REGISTRATION = ROOT / 'configs/hamiballs1/gate.json'
NOISES = (1942634267, 2035767743)
CONTROLLER_CLASS = SameRunETrController
BOOTSTRAP_CONTRACT = {'initial_partition': 'one_leaf', 'depth3_fit_once_at_scalar0_refresh_block': 4, 'target': 'four normalized quadratic gate-q/p and clean-H-r-q/p tasks', 'source_disjoint_folds': True, 'event_or_contact_labels': False}
