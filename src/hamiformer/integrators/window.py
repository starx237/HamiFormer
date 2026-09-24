"""Window-level DLR factor construction and batched candidate application."""
import copy
from types import SimpleNamespace

import torch

from .lowrank import install_lowrank


class DLRWindow:
    def __init__(self, hamiltonian, attr_scale, step_size, *, rank=2, probes=4, compiled=True):
        model = copy.deepcopy(hamiltonian).eval().requires_grad_(False)
        if compiled and next(model.parameters()).device.type != 'cuda':
            raise ValueError('compiled DLR windows require CUDA')
        # Validation occurs on public inputs, outside captured derivative graphs.
        model._validate = lambda q, p, c: None
        self.runtime = SimpleNamespace(models={'h': model}, stats={}, prefetching=False)
        self.attr_scale = attr_scale
        self.step_size = step_size
        self.construct = install_lowrank(self.runtime, rank=rank, probes=probes,
            chunk=128, cache=True, batch_ad=True, jet_graph=compiled,
            compiled=compiled, install_symbols=False)

    def build(self, source_q, target_p, attrs):
        """q/p: [batch, edges, objects, 2]; attrs: [batch, objects, 3]."""
        if source_q.shape != target_p.shape or source_q.ndim != 4 or source_q.shape[-1] != 2:
            raise ValueError('q and p must have matching [batch, edges, objects, 2] shapes')
        if source_q.shape[2] * 2 != self.runtime.models['h'].state_dim:
            raise ValueError('window and Hamiltonian dimensions differ')
        if attrs.shape != (source_q.shape[0], source_q.shape[2], 3):
            raise ValueError('attribute shape mismatch')
        if not all(bool(torch.isfinite(v).all()) for v in (source_q, target_p, attrs)):
            raise ValueError('nonfinite window inputs')
        anchor = SimpleNamespace(source_q=source_q, target_p=target_p)
        with torch.no_grad():
            return self.construct(self.runtime.models['h'], anchor, attrs,
                                  attr_scale=self.attr_scale, step_size=self.step_size)

    def apply(self, factors, edge, previous):
        """Apply one edge to physical states [batch, objects, 4]."""
        if not 0 <= edge < factors.matrix.shape[1]:
            raise IndexError('edge outside window')
        if previous.shape != (factors.matrix.shape[0], self.runtime.models['h'].state_dim // 2, 4):
            raise ValueError('candidate state shape mismatch')
        return self.runtime.integrator_candidate(factors.matrix[:, edge], factors.offset[:, edge], previous, None)
