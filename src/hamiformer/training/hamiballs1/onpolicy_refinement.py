from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import torch
from torch import nn
ROOT = project_root()
from hamiformer.training.hamiballs1 import ridge_features as moe
from hamiformer.training.hamiballs1 import refinement as frozen
OUTPUT = ROOT / 'outputs/hami1/regime_refiner_onpolicy'
UPDATES = 200
REFRESH = 8
BATCH_SOURCES = 64
START_OFFSET = 8320
SOURCE_SEED = 1570077070
NOISE_SEEDS = (1942634267, 2035767743)
FIELDS = (0, 5, 11, 17)
HEAD_LR = 0.001
GATE_LR = 0.01
WARMUP = 20
CELLS_PER_LEAF = 1024

class TrainableRegimeRefiner(frozen.RegimeRefinedGate):

    def __init__(self, base_gate: nn.Module) -> None:
        super().__init__(base_gate, enable_r=True, enable_gate=True)
        checkpoint = torch.load(frozen.CHECKPOINT, map_location='cpu', weights_only=False)
        width = int(checkpoint['width'])
        input_dim = int(checkpoint['input_dim'])
        self.q_heads = nn.ModuleList([moe.Head(input_dim, width) for _ in range(self.leaf_count)])
        self.p_heads = nn.ModuleList([moe.Head(input_dim, width) for _ in range(self.leaf_count)])
        self._buffers.pop('biases')
        self.biases = nn.Parameter(torch.zeros(self.leaf_count, 2))
