from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1 import tree_optimizer as prefix
SOURCE = prefix.OUTPUT
OUTPUT = ROOT / 'outputs/hami1/disjoint_training_disjoint_source_no_harm'
NO_HARM_WEIGHT = 0.25
