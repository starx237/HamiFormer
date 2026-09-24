from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .block_hamiltonian_expert import BlockHamiltonianOccurrences, ParallelBlockHamiltonianExpert

@dataclass
class WavefrontHamiltonianOutput:
    clean: torch.Tensor
    occurrences: BlockHamiltonianOccurrences

@dataclass(frozen=True)
class _WavefrontLayout:
    source_write_mask: torch.Tensor
    target_write_mask: torch.Tensor
    owner_window: torch.Tensor
    owner_position: torch.Tensor
    owner_is_target: torch.Tensor
    owner_stage: torch.Tensor
    selected_window_indices: torch.Tensor

def _build_wavefront_layout(*, future_steps: int, block_size: int, block_step: int, num_windows: int) -> _WavefrontLayout:
    num_states = future_steps + 1
    max_start = num_states - block_size - block_step
    starts = list(range(0, max_start + 1, block_step))
    if starts[-1] != max_start:
        starts.append(max_start)
    source_write_mask = torch.zeros(num_windows, block_size, dtype=torch.bool)
    target_write_mask = torch.zeros(num_windows, block_size, dtype=torch.bool)
    owner_window = torch.full((future_steps,), -1, dtype=torch.long)
    owner_position = torch.full((future_steps,), -1, dtype=torch.long)
    owner_is_target = torch.zeros(future_steps, dtype=torch.bool)
    owner_stage = torch.full((future_steps,), -1, dtype=torch.long)
    owned = torch.zeros(num_states, dtype=torch.bool)
    owned[0] = True
    for stage, start in enumerate(starts):
        source = torch.arange(start, start + block_size)
        target = source + block_step
        if stage == 0:
            for local_position, state_index in enumerate(source.tolist()):
                if state_index > 0 and (not bool(owned[state_index])):
                    source_write_mask[start, local_position] = True
                    slot = state_index - 1
                    owner_window[slot] = start
                    owner_position[slot] = local_position
                    owner_is_target[slot] = False
                    owner_stage[slot] = stage
                    owned[state_index] = True
        for local_position, state_index in enumerate(target.tolist()):
            if state_index <= future_steps and (not bool(owned[state_index])):
                target_write_mask[start, local_position] = True
                slot = state_index - 1
                owner_window[slot] = start
                owner_position[slot] = local_position
                owner_is_target[slot] = True
                owner_stage[slot] = stage
                owned[state_index] = True
    if not bool(owned.all()):
        missing = torch.nonzero(~owned, as_tuple=False).flatten().tolist()
        raise RuntimeError(f'Wavefront owner没有覆盖全部phase states: {missing}')
    if bool((owner_window < 0).any()) or bool((owner_position < 0).any()):
        raise RuntimeError('Wavefront owner索引构造不完整')
    if int(source_write_mask.sum() + target_write_mask.sum()) != future_steps:
        raise RuntimeError('每个未来phase state必须恰好拥有一个写owner')
    if bool((owner_window >= num_windows).any()):
        raise RuntimeError('Wavefront owner引用了不存在的core window')
    return _WavefrontLayout(source_write_mask=source_write_mask, target_write_mask=target_write_mask, owner_window=owner_window, owner_position=owner_position, owner_is_target=owner_is_target, owner_stage=owner_stage, selected_window_indices=torch.tensor(starts, dtype=torch.long))

class WavefrontHamiltonianExpert(nn.Module):

    def __init__(self, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, block_size: int, block_step: int, q_scale: torch.Tensor, p_scale: torch.Tensor, hidden_size: int=128, depth: int=2, num_heads: int=4, mlp_ratio: float=4.0, dropout: float=0.0, force_float32: bool=True) -> None:
        super().__init__()
        self.core = ParallelBlockHamiltonianExpert(num_objects=num_objects, future_steps=future_steps, q_dim=q_dim, attr_dim=attr_dim, block_size=block_size, block_step=block_step, q_scale=q_scale, p_scale=p_scale, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, chart_coupling='sequential', force_float32=force_float32)
        layout = _build_wavefront_layout(future_steps=future_steps, block_size=block_size, block_step=block_step, num_windows=self.core.num_windows)
        self.register_buffer('source_write_mask', layout.source_write_mask, persistent=False)
        self.register_buffer('target_write_mask', layout.target_write_mask, persistent=False)
        self.register_buffer('owner_window', layout.owner_window, persistent=False)
        self.register_buffer('owner_position', layout.owner_position, persistent=False)
        self.register_buffer('owner_is_target', layout.owner_is_target, persistent=False)
        self.register_buffer('owner_stage', layout.owner_stage, persistent=False)
        self.register_buffer('selected_window_indices', layout.selected_window_indices, persistent=False)

    @property
    def future_steps(self) -> int:
        return self.core.future_steps

    @property
    def q_dim(self) -> int:
        return self.core.q_dim

    @property
    def q_scale(self) -> torch.Tensor:
        return self.core.q_scale

    @property
    def p_scale(self) -> torch.Tensor:
        return self.core.p_scale

    @property
    def num_groups(self) -> int:
        return int(self.owner_stage.max().item()) + 1

    def _validate_signal(self, signal: torch.Tensor, batch: int) -> None:
        expected = (batch, self.future_steps + 1)
        if signal.shape != expected:
            raise ValueError('signal 必须为 [B,F+1]')
        if not bool(torch.isfinite(signal).all().item()):
            raise ValueError('signal 含非有限值')
        if bool(((signal < 0) | (signal > 1)).any().item()):
            raise ValueError('signal 必须位于 [0,1]')
        if not bool((signal[:, 0] == 1).all().item()):
            raise ValueError('已知初态的 signal 必须严格为 1')

    def _owner_clean(self, occurrences: BlockHamiltonianOccurrences) -> torch.Tensor:
        window = self.owner_window
        position = self.owner_position
        q_from_source = occurrences.q_minus[:, window, position]
        p_from_source = occurrences.p_plus[:, window, position]
        q_from_target = occurrences.q_plus[:, window, position]
        p_from_target = occurrences.p_minus[:, window, position]
        role = self.owner_is_target.view(1, self.future_steps, 1, 1)
        q_owner = torch.where(role, q_from_target, q_from_source)
        p_owner = torch.where(role, p_from_target, p_from_source)
        return torch.cat([q_owner, p_owner], dim=-1)

    def forward(self, noisy_future: torch.Tensor, signal: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool | None=None) -> WavefrontHamiltonianOutput:
        batch = noisy_future.shape[0]
        self._validate_signal(signal, batch)
        dummy_tau = noisy_future.new_zeros(batch)
        self.core._validate_inputs(noisy_future, dummy_tau, x0, attrs, physical_time)
        if create_graph is None:
            create_graph = self.training
        full = torch.cat([x0[:, None], noisy_future], dim=1)
        occurrences = self.core.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=True, create_graph=create_graph, source_write_mask=self.source_write_mask, target_write_mask=self.target_write_mask)
        clean = self._owner_clean(occurrences)
        if not create_graph:
            clean = clean.detach()
            occurrences = occurrences.detached()
        return WavefrontHamiltonianOutput(clean=clean, occurrences=occurrences)

    def clean_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> BlockHamiltonianOccurrences:
        return self.core.clean_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=create_graph)

    def noisy_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, noise_scale: float, create_graph: bool, generator: torch.Generator | None=None) -> BlockHamiltonianOccurrences:
        return self.core.noisy_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, noise_scale=noise_scale, create_graph=create_graph, generator=generator)
