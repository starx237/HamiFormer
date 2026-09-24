from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_bootstrap as tree_bootstrap_support
OUTPUT = ROOT / 'outputs/hami1/late_tree'
R_REGISTRATION = tree_bootstrap_support.R_REGISTRATION
GATE_REGISTRATION = tree_bootstrap_support.GATE_REGISTRATION
NOISES = tree_bootstrap_support.NOISES
QP_BOOTSTRAP_UPDATES = 128
QP_FINAL_UPDATES = 200
REFRESH_EVERY = 8
PHYSICAL_SOURCES_PER_REFRESH = 32
HOLDOUT_SOURCES = 32
MAXIMUM_LR = 0.003
MINIMUM_LR = 0.0001
