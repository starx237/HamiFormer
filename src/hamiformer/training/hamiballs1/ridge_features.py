from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import torch
from torch import nn
ROOT = project_root()
TREE_REPORT = ROOT / 'outputs/hami1/router/report.json'
OUTPUT = ROOT / 'outputs/hami1/moe'
LEDGER = ROOT / 'data/hamiballs_canonical_v2/packs/train48_stride48/rows.jsonl'
DEPTHS = (3,)
UPDATES = 400
BATCH = 8192
EVAL_EVERY = 20

class Head(nn.Module):

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)
