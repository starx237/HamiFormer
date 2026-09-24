from __future__ import annotations
import torch
from torch import nn
from .block_hamiltonian_expert import BlockHamiltonianOccurrences, BoundaryConditionMode, ParallelBlockHamiltonianExpert
from .g1y_assembly_hamiltonian import G1YAssemblyOutput

class G1YOverlappingBlockHamiltonianExpert(nn.Module):

    def __init__(self, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor, block_size: int=2, block_step: int=1, chart_coupling: str='sequential', hidden_size: int=128, depth: int=2, num_heads: int=4, mlp_ratio: float=4.0, dropout: float=0.0, boundary_condition_mode: BoundaryConditionMode='annealed', boundary_anneal_end: float=0.98, force_float32: bool=True) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.block_size = int(block_size)
        self.block_step = int(block_step)
        self.chart_coupling = str(chart_coupling)
        self.assembly_mode = f'block_b{self.block_size}_s{self.block_step}_{self.chart_coupling}'
        self.core = ParallelBlockHamiltonianExpert(num_objects=num_objects, future_steps=future_steps, q_dim=q_dim, attr_dim=attr_dim, block_size=block_size, block_step=block_step, q_scale=q_scale, p_scale=p_scale, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, chart_coupling=chart_coupling, boundary_condition_mode=boundary_condition_mode, boundary_anneal_end=boundary_anneal_end, force_float32=force_float32)

    @property
    def q_scale(self) -> torch.Tensor:
        return self.core.q_scale

    @property
    def p_scale(self) -> torch.Tensor:
        return self.core.p_scale

    @property
    def num_windows(self) -> int:
        return self.core.num_windows

    @property
    def source_indices(self) -> torch.Tensor:
        return self.core.source_indices

    @property
    def target_indices(self) -> torch.Tensor:
        return self.core.target_indices

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool | None=None) -> G1YAssemblyOutput:
        output = self.core(noisy_future, tau, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=create_graph)
        return G1YAssemblyOutput(clean=output.clean, occurrences=output.occurrences, disagreement=output.disagreement, factor_calls=self.num_windows, sequential_depth=2 if self.chart_coupling == 'sequential' else 1)

    def clean_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> BlockHamiltonianOccurrences:
        return self.core.clean_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=create_graph)

    def noisy_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, noise_scale: float, create_graph: bool, generator: torch.Generator | None=None) -> BlockHamiltonianOccurrences:
        return self.core.noisy_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, noise_scale=noise_scale, create_graph=create_graph, generator=generator)

    def _clean_relation_objectives(self, candidate_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, preserve_state_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch = candidate_future.shape[0]
        full = torch.cat([x0[:, None], candidate_future], dim=1)
        signal = torch.ones(batch, self.future_steps + 1, device=full.device, dtype=full.dtype)
        occurrences = self.core.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=preserve_state_graph, detach_state_inputs=not preserve_state_graph)
        q = full[..., :self.q_dim]
        p = full[..., self.q_dim:]
        q_scale = self.q_scale.to(full).view(1, 1, 1, 1, -1)
        p_scale = self.p_scale.to(full).view(1, 1, 1, 1, -1)
        plus_residuals = torch.cat([((occurrences.q_plus - q[:, occurrences.target_indices]) / q_scale).flatten(1), ((occurrences.p_plus - p[:, occurrences.source_indices]) / p_scale).flatten(1)], dim=1)
        minus_residuals = torch.cat([((occurrences.q_minus - q[:, occurrences.source_indices]) / q_scale).flatten(1), ((occurrences.p_minus - p[:, occurrences.target_indices]) / p_scale).flatten(1)], dim=1)
        return (0.5 * plus_residuals.square().sum(dim=1), 0.5 * minus_residuals.square().sum(dim=1))
