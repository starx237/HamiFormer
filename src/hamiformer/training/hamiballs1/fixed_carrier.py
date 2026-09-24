from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
import torch
ROOT = project_root()

class _FixedStream:

    def __init__(self, indices: torch.Tensor):
        self.indices = indices.cpu()
        self.batch_size = int(indices.numel())

    def next_indices(self) -> torch.Tensor:
        return self.indices.clone()
