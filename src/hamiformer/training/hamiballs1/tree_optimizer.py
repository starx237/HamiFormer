from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_carriers as tree_carrier_support
from hamiformer.training.hamiballs1 import hierarchical_residual as hierarchical_residual_support
SOURCE = tree_carrier_support.OUTPUT
SOURCE_LOG = ROOT / 'outputs/hami1/treecarrier_tree_carrier_.log'
OUTPUT = ROOT / 'outputs/hami1/stable_tree_training'
TF_OUTPUT = OUTPUT / 'teacher_forced_r0'
TREE_OUTPUT = OUTPUT / 'scalar1_tree.pt'
FINAL_OUTPUT = OUTPUT / 'final'
RIDGE_OUTPUT = FINAL_OUTPUT / 'ridge_terminal.pt'
BATCH_FINAL_PAIRED_NOISE = False
FINAL_INTEGRATED_RIDGE_CLASS = hierarchical_residual_support.RoutedHierarchicalResidual
