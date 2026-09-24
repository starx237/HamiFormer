from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
BoundaryConditionMode = Literal['none', 'constant', 'annealed']

@dataclass
class BlockHamiltonianOccurrences:
    q_plus: torch.Tensor
    p_plus: torch.Tensor
    q_minus: torch.Tensor
    p_minus: torch.Tensor
    source_indices: torch.Tensor
    target_indices: torch.Tensor

    def detached(self) -> 'BlockHamiltonianOccurrences':
        return BlockHamiltonianOccurrences(q_plus=self.q_plus.detach(), p_plus=self.p_plus.detach(), q_minus=self.q_minus.detach(), p_minus=self.p_minus.detach(), source_indices=self.source_indices, target_indices=self.target_indices)

@dataclass
class BlockHamiltonianOutput:
    clean: torch.Tensor
    disagreement: torch.Tensor
    occurrences: BlockHamiltonianOccurrences
    q_coverage: torch.Tensor
    p_coverage: torch.Tensor

class TemporalScalarTransformer(nn.Module):

    def __init__(self, *, num_objects: int, q_dim: int, attr_dim: int, block_size: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float, boundary_condition_dim: int=0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError('hidden_size 必须能被 num_heads 整除')
        if boundary_condition_dim < 0:
            raise ValueError('boundary_condition_dim 不能为负')
        self.num_objects = int(num_objects)
        self.q_dim = int(q_dim)
        self.attr_dim = int(attr_dim)
        self.block_size = int(block_size)
        self.boundary_condition_dim = int(boundary_condition_dim)
        phase_token_dim = self.num_objects * self.q_dim
        condition_dim = self.num_objects * self.attr_dim + self.block_size
        self.q_embedding = nn.Linear(phase_token_dim, hidden_size)
        self.p_embedding = nn.Linear(phase_token_dim, hidden_size)
        self.condition_embedding = nn.Linear(condition_dim, hidden_size)
        if self.boundary_condition_dim > 0:
            self.boundary_condition_embedding: nn.Linear | None = nn.Linear(self.boundary_condition_dim, hidden_size, bias=False)
            nn.init.zeros_(self.boundary_condition_embedding.weight)
        else:
            self.boundary_condition_embedding = None
        self.signal_embedding = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + 2 * self.block_size, hidden_size))
        self.type_embedding = nn.Parameter(torch.zeros(3, hidden_size))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=int(hidden_size * mlp_ratio), dropout=dropout, activation='relu', batch_first=True, norm_first=False) for _ in range(depth)])
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, q: torch.Tensor, p: torch.Tensor, *, q_signal: torch.Tensor, p_signal: torch.Tensor, attrs: torch.Tensor, delta_time: torch.Tensor, boundary_condition: torch.Tensor | None=None, boundary_weight: torch.Tensor | None=None) -> torch.Tensor:
        if q.shape != p.shape or q.ndim != 4:
            raise ValueError('q/p 必须是相同 shape 的 [N,b,K,d_q]')
        n, b, k, d = q.shape
        if (b, k, d) != (self.block_size, self.num_objects, self.q_dim):
            raise ValueError('q/p 的 block、object 或 coordinate 维与构造参数不一致')
        if q_signal.shape != (n, b) or p_signal.shape != (n, b):
            raise ValueError('q_signal/p_signal 必须为 [N,b]')
        if attrs.shape != (n, k, self.attr_dim):
            raise ValueError('attrs 必须为 [N,K,attr_dim]')
        if delta_time.shape != (n, b):
            raise ValueError('delta_time 必须为 [N,b]')
        if self.boundary_condition_embedding is None:
            if boundary_condition is not None or boundary_weight is not None:
                raise ValueError('未启用边界条件时不得传入 boundary tensor')
        else:
            if boundary_condition is None or boundary_weight is None:
                raise ValueError('启用边界条件时必须同时提供 condition 与 weight')
            if boundary_condition.shape != (n, self.boundary_condition_dim):
                raise ValueError('boundary_condition shape 与构造参数不一致')
            if boundary_weight.shape not in {(n,), (n, 1)}:
                raise ValueError('boundary_weight 必须为 [N] 或 [N,1]')
        q_token = self.q_embedding(q.flatten(2))
        p_token = self.p_embedding(p.flatten(2))
        condition = torch.cat([attrs.flatten(1), delta_time], dim=-1)
        query_token = self.condition_embedding(condition)[:, None]
        if self.boundary_condition_embedding is not None:
            assert boundary_condition is not None
            assert boundary_weight is not None
            boundary_query = self.boundary_condition_embedding(boundary_condition)
            query_token = query_token + (boundary_query * boundary_weight.reshape(n, 1))[:, None]
        tokens = torch.cat([query_token, q_token, p_token], dim=1)
        query_signal = torch.ones(n, 1, device=q.device, dtype=q.dtype)
        signal = torch.cat([query_signal, q_signal, p_signal], dim=1)
        tokens = tokens + self.signal_embedding(signal[..., None])
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        type_ids = torch.cat([torch.zeros(1, device=q.device, dtype=torch.long), torch.ones(b, device=q.device, dtype=torch.long), torch.full((b,), 2, device=q.device, dtype=torch.long)])
        tokens = tokens + self.type_embedding[type_ids][None]
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                tokens = layer(tokens)
        return self.output(tokens[:, 0]).squeeze(-1)

class ParallelBlockHamiltonianExpert(nn.Module):

    def __init__(self, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, block_size: int, block_step: int, q_scale: torch.Tensor, p_scale: torch.Tensor, hidden_size: int=128, depth: int=2, num_heads: int=4, mlp_ratio: float=4.0, dropout: float=0.0, chart_coupling: str='sequential', boundary_condition_mode: BoundaryConditionMode='none', boundary_anneal_end: float=1.0, force_float32: bool=True) -> None:
        super().__init__()
        if not 0 < block_step <= block_size:
            raise ValueError('必须满足 0 < block_step <= block_size')
        if future_steps + 1 < block_size + block_step:
            raise ValueError('完整轨迹必须至少容纳一个长度 b+s 的 local union')
        if chart_coupling not in {'sequential', 'independent'}:
            raise ValueError('chart_coupling 只能是 sequential 或 independent')
        if boundary_condition_mode not in {'none', 'constant', 'annealed'}:
            raise ValueError('boundary_condition_mode 只能是 none、constant 或 annealed')
        if not 0.0 < boundary_anneal_end <= 1.0:
            raise ValueError('boundary_anneal_end 必须位于 (0,1]')
        if q_scale.shape != (q_dim,) or p_scale.shape != (q_dim,):
            raise ValueError('q_scale/p_scale 必须为 [q_dim]')
        if bool((q_scale <= 0).any().item()) or bool((p_scale <= 0).any().item()):
            raise ValueError('q_scale/p_scale 必须严格为正')
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.state_dim = 2 * self.q_dim
        self.attr_dim = int(attr_dim)
        self.block_size = int(block_size)
        self.block_step = int(block_step)
        self.chart_coupling = chart_coupling
        self.boundary_condition_mode: BoundaryConditionMode = boundary_condition_mode
        self.boundary_anneal_end = float(boundary_anneal_end)
        self.force_float32 = bool(force_float32)
        self.register_buffer('q_scale', q_scale.float().clone())
        self.register_buffer('p_scale', p_scale.float().clone())
        num_states = self.future_steps + 1
        num_windows = num_states - self.block_size - self.block_step + 1
        starts = torch.arange(num_windows, dtype=torch.long)
        offsets = torch.arange(self.block_size, dtype=torch.long)
        source_indices = starts[:, None] + offsets[None]
        target_indices = source_indices + self.block_step
        self.register_buffer('source_indices', source_indices, persistent=False)
        self.register_buffer('target_indices', target_indices, persistent=False)
        boundary_condition_dim = self.num_objects * self.state_dim + 2 * self.block_size if self.boundary_condition_mode != 'none' else 0
        network_kwargs = dict(num_objects=self.num_objects, q_dim=self.q_dim, attr_dim=self.attr_dim, block_size=self.block_size, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, boundary_condition_dim=boundary_condition_dim)
        self.h_plus = TemporalScalarTransformer(**network_kwargs)
        self.h_minus = TemporalScalarTransformer(**network_kwargs)

    @property
    def num_windows(self) -> int:
        return int(self.source_indices.shape[0])

    def _autocast_context(self, reference: torch.Tensor):
        if reference.device.type in {'cuda', 'cpu'}:
            return torch.autocast(device_type=reference.device.type, enabled=False)
        return nullcontext()

    def _validate_inputs(self, future: torch.Tensor, tau: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> None:
        batch = future.shape[0]
        if future.shape != (batch, self.future_steps, self.num_objects, self.state_dim):
            raise ValueError('future 必须为 [B,F,K,2*d_q]')
        if x0.shape != (batch, self.num_objects, self.state_dim):
            raise ValueError('x0 必须为 [B,K,2*d_q]')
        if attrs.shape != (batch, self.num_objects, self.attr_dim):
            raise ValueError('attrs 必须为 [B,K,attr_dim]')
        if physical_time.shape != (batch, self.future_steps + 1):
            raise ValueError('physical_time 必须为 [B,F+1]')
        if tau.shape != (batch,) or bool(((tau < 0) | (tau >= 1)).any().item()):
            raise ValueError('tau 必须为 [B] 且满足 0 <= tau < 1')
        if not bool(torch.isfinite(future).all().item()):
            raise ValueError('future 含非有限值')

    def _extract(self, value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return value[:, indices]

    def _window_conditions(self, attrs: torch.Tensor, physical_time: torch.Tensor, *, source_indices: torch.Tensor | None=None, target_indices: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if source_indices is None:
            source_indices = self.source_indices
        if target_indices is None:
            target_indices = self.target_indices
        if source_indices.shape != target_indices.shape:
            raise ValueError('source_indices 与 target_indices shape 必须一致')
        batch = attrs.shape[0]
        num_windows = int(source_indices.shape[0])
        attrs_windows = attrs[:, None].expand(batch, num_windows, self.num_objects, self.attr_dim)
        source_time = self._extract(physical_time, source_indices)
        target_time = self._extract(physical_time, target_indices)
        delta_time = target_time - source_time
        if bool((delta_time <= 0).any().item()):
            raise ValueError('source-target 相对时间必须严格为正')
        return (attrs_windows, delta_time)

    def rf_boundary_strength(self, tau: torch.Tensor) -> torch.Tensor | None:
        if self.boundary_condition_mode == 'none':
            return None
        if tau.ndim != 1:
            raise ValueError('tau 必须为一维 batch tensor')
        if self.boundary_condition_mode == 'constant':
            return torch.ones_like(tau)
        return (1.0 - tau / self.boundary_anneal_end).clamp(0.0, 1.0)

    def _boundary_conditions(self, full_state: torch.Tensor, *, q_signal: torch.Tensor, p_signal: torch.Tensor, physical_time: torch.Tensor, source_indices: torch.Tensor, target_indices: torch.Tensor, boundary_strength: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.boundary_condition_mode == 'none':
            if boundary_strength is not None:
                raise ValueError('none 模式不得传入 boundary_strength')
            return (None, None)
        batch = full_state.shape[0]
        num_windows, block = source_indices.shape
        x0 = full_state[:, 0]
        q0 = x0[..., :self.q_dim] / self.q_scale.to(x0).view(1, 1, -1)
        p0 = x0[..., self.q_dim:] / self.p_scale.to(x0).view(1, 1, -1)
        normalized_x0 = torch.cat([q0, p0], dim=-1).flatten(1)
        normalized_x0 = normalized_x0[:, None].expand(batch, num_windows, normalized_x0.shape[-1])
        origin_time = physical_time[:, :1, None]
        source_offset = self._extract(physical_time, source_indices) - origin_time
        target_offset = self._extract(physical_time, target_indices) - origin_time
        condition = torch.cat([normalized_x0, source_offset.reshape(batch, num_windows, block), target_offset.reshape(batch, num_windows, block)], dim=-1)
        if self.boundary_condition_mode == 'constant':
            weight = condition.new_ones(batch, num_windows)
        elif boundary_strength is not None:
            if boundary_strength.shape == (batch,):
                weight = boundary_strength[:, None].expand(batch, num_windows)
            elif boundary_strength.shape == (batch, num_windows):
                weight = boundary_strength
            else:
                raise ValueError('boundary_strength 必须为 [B] 或 [B,W]')
            weight = weight.to(condition).clamp(0.0, 1.0)
        else:
            q_source_signal = self._extract(q_signal, source_indices)
            p_source_signal = self._extract(p_signal, source_indices)
            q_target_signal = self._extract(q_signal, target_indices)
            p_target_signal = self._extract(p_signal, target_indices)
            local_signal = torch.stack([q_source_signal, p_source_signal, q_target_signal, p_target_signal], dim=0).mean(dim=(0, 3))
            weight = (1.0 - local_signal).to(condition).clamp(0.0, 1.0)
        return (condition.flatten(0, 1), weight.flatten(0, 1))

    def _right_update(self, q_source: torch.Tensor, p_target: torch.Tensor, *, q_signal: torch.Tensor, p_signal: torch.Tensor, attrs: torch.Tensor, delta_time: torch.Tensor, boundary_condition: torch.Tensor | None, boundary_weight: torch.Tensor | None, create_graph: bool, detach_inputs: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32 if self.force_float32 else q_source.dtype
        with torch.enable_grad(), self._autocast_context(q_source):
            if detach_inputs:
                q_input = q_source.detach().to(dtype=dtype).requires_grad_(True)
                p_input = p_target.detach().to(dtype=dtype).requires_grad_(True)
            else:
                q_input = q_source.to(dtype=dtype)
                p_input = p_target.to(dtype=dtype)
                if not q_input.requires_grad:
                    q_input = q_input.requires_grad_(True)
                if not p_input.requires_grad:
                    p_input = p_input.requires_grad_(True)
            scalar = self.h_plus(q_input, p_input, q_signal=q_signal.detach().to(dtype=dtype), p_signal=p_signal.detach().to(dtype=dtype), attrs=attrs.detach().to(dtype=dtype), delta_time=delta_time.detach().to(dtype=dtype), boundary_condition=boundary_condition.detach().to(dtype=dtype) if boundary_condition is not None else None, boundary_weight=boundary_weight.detach().to(dtype=dtype) if boundary_weight is not None else None)
            grad_q, grad_p = torch.autograd.grad(scalar.sum(), (q_input, p_input), create_graph=create_graph)
            q_target = q_input + grad_p
            p_source = p_input + grad_q
        return (q_target, p_source)

    def _left_update(self, q_target: torch.Tensor, p_source: torch.Tensor, *, q_signal: torch.Tensor, p_signal: torch.Tensor, attrs: torch.Tensor, delta_time: torch.Tensor, boundary_condition: torch.Tensor | None, boundary_weight: torch.Tensor | None, create_graph: bool, detach_inputs: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32 if self.force_float32 else q_target.dtype
        with torch.enable_grad(), self._autocast_context(q_target):
            if detach_inputs:
                q_input = q_target.detach().to(dtype=dtype).requires_grad_(True)
                p_input = p_source.detach().to(dtype=dtype).requires_grad_(True)
            else:
                q_input = q_target.to(dtype=dtype)
                p_input = p_source.to(dtype=dtype)
                if not q_input.requires_grad:
                    q_input = q_input.requires_grad_(True)
                if not p_input.requires_grad:
                    p_input = p_input.requires_grad_(True)
            scalar = self.h_minus(q_input, p_input, q_signal=q_signal.detach().to(dtype=dtype), p_signal=p_signal.detach().to(dtype=dtype), attrs=attrs.detach().to(dtype=dtype), delta_time=delta_time.detach().to(dtype=dtype), boundary_condition=boundary_condition.detach().to(dtype=dtype) if boundary_condition is not None else None, boundary_weight=boundary_weight.detach().to(dtype=dtype) if boundary_weight is not None else None)
            grad_q, grad_p = torch.autograd.grad(scalar.sum(), (q_input, p_input), create_graph=create_graph)
            q_source = q_input - grad_p
            p_target = p_input - grad_q
        return (q_source, p_target)

    def _flatten_windows(self, value: torch.Tensor) -> torch.Tensor:
        return value.flatten(0, 1)

    def _unflatten_windows(self, value: torch.Tensor, batch: int, *, num_windows: int | None=None) -> torch.Tensor:
        if num_windows is None:
            num_windows = self.num_windows
        return value.unflatten(0, (batch, num_windows))

    def _to_normalized(self, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = full[..., :self.q_dim] / self.q_scale.to(full).view(1, 1, 1, -1)
        p = full[..., self.q_dim:] / self.p_scale.to(full).view(1, 1, 1, -1)
        return (q, p)

    def _to_raw_occurrences(self, q_plus: torch.Tensor, p_plus: torch.Tensor, q_minus: torch.Tensor, p_minus: torch.Tensor, *, source_indices: torch.Tensor | None=None, target_indices: torch.Tensor | None=None) -> BlockHamiltonianOccurrences:
        if source_indices is None:
            source_indices = self.source_indices
        if target_indices is None:
            target_indices = self.target_indices
        q_scale = self.q_scale.to(q_plus).view(1, 1, 1, 1, -1)
        p_scale = self.p_scale.to(p_plus).view(1, 1, 1, 1, -1)
        return BlockHamiltonianOccurrences(q_plus=q_plus * q_scale, p_plus=p_plus * p_scale, q_minus=q_minus * q_scale, p_minus=p_minus * p_scale, source_indices=source_indices, target_indices=target_indices)

    def chart_occurrences(self, full_state: torch.Tensor, *, q_signal: torch.Tensor, p_signal: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, sequential: bool, create_graph: bool, source_write_mask: torch.Tensor | None=None, target_write_mask: torch.Tensor | None=None, window_indices: torch.Tensor | None=None, detach_state_inputs: bool=True, boundary_strength: torch.Tensor | None=None) -> BlockHamiltonianOccurrences:
        batch, num_states, objects, state_dim = full_state.shape
        if (num_states, objects, state_dim) != (self.future_steps + 1, self.num_objects, self.state_dim):
            raise ValueError('full_state shape 与模型构造参数不一致')
        if q_signal.shape != (batch, num_states) or p_signal.shape != (batch, num_states):
            raise ValueError('q_signal/p_signal 必须为 [B,F+1]')
        if window_indices is None:
            source_indices = self.source_indices
            target_indices = self.target_indices
        else:
            if window_indices.ndim != 1 or window_indices.dtype != torch.long:
                raise ValueError('window_indices 必须是 long 型一维张量')
            if window_indices.numel() == 0:
                raise ValueError('window_indices 不能为空')
            if bool(((window_indices < 0) | (window_indices >= self.num_windows)).any().item()):
                raise ValueError('window_indices 超出 local factor 范围')
            source_indices = self.source_indices[window_indices]
            target_indices = self.target_indices[window_indices]
        num_windows = int(source_indices.shape[0])
        q, p = self._to_normalized(full_state)
        q_source = self._extract(q, source_indices)
        p_source = self._extract(p, source_indices)
        q_target = self._extract(q, target_indices)
        p_target = self._extract(p, target_indices)
        q_source_signal = self._extract(q_signal, source_indices)
        p_source_signal = self._extract(p_signal, source_indices)
        q_target_signal = self._extract(q_signal, target_indices)
        p_target_signal = self._extract(p_signal, target_indices)
        attrs_windows, delta_time = self._window_conditions(attrs, physical_time, source_indices=source_indices, target_indices=target_indices)
        boundary_condition, boundary_weight = self._boundary_conditions(full_state, q_signal=q_signal, p_signal=p_signal, physical_time=physical_time, source_indices=source_indices, target_indices=target_indices, boundary_strength=boundary_strength)
        q_plus, p_plus = self._right_update(self._flatten_windows(q_source), self._flatten_windows(p_target), q_signal=self._flatten_windows(q_source_signal), p_signal=self._flatten_windows(p_target_signal), attrs=self._flatten_windows(attrs_windows), delta_time=self._flatten_windows(delta_time), boundary_condition=boundary_condition, boundary_weight=boundary_weight, create_graph=create_graph, detach_inputs=detach_state_inputs)
        q_plus = self._unflatten_windows(q_plus, batch, num_windows=num_windows)
        p_plus = self._unflatten_windows(p_plus, batch, num_windows=num_windows)
        if (source_write_mask is None) != (target_write_mask is None):
            raise ValueError('source_write_mask 与 target_write_mask 必须同时提供或同时省略')
        if source_write_mask is not None:
            expected_mask_shape = (num_windows, self.block_size)
            if source_write_mask.shape != expected_mask_shape:
                raise ValueError('source_write_mask 必须为 [W,b]')
            if target_write_mask is None or target_write_mask.shape != expected_mask_shape:
                raise ValueError('target_write_mask 必须为 [W,b]')
            if source_write_mask.dtype != torch.bool or target_write_mask.dtype != torch.bool:
                raise ValueError('source/target write mask 必须是 bool')
        if sequential:
            if source_write_mask is None:
                target_use_proposal = (target_indices != 0).view(1, num_windows, self.block_size, 1, 1)
                source_use_proposal = (source_indices != 0).view(1, num_windows, self.block_size, 1, 1)
            else:
                target_use_proposal = target_write_mask.view(1, num_windows, self.block_size, 1, 1)
                source_use_proposal = source_write_mask.view(1, num_windows, self.block_size, 1, 1)
            left_q_input = torch.where(target_use_proposal, q_plus, q_target)
            left_p_input = torch.where(source_use_proposal, p_plus, p_source)
        else:
            if source_write_mask is not None:
                raise ValueError('write masks 只允许用于 sequential chart coupling')
            left_q_input = q_target
            left_p_input = p_source
        q_minus, p_minus = self._left_update(self._flatten_windows(left_q_input), self._flatten_windows(left_p_input), q_signal=self._flatten_windows(q_target_signal), p_signal=self._flatten_windows(p_source_signal), attrs=self._flatten_windows(attrs_windows), delta_time=self._flatten_windows(delta_time), boundary_condition=boundary_condition, boundary_weight=boundary_weight, create_graph=create_graph, detach_inputs=detach_state_inputs)
        q_minus = self._unflatten_windows(q_minus, batch, num_windows=num_windows)
        p_minus = self._unflatten_windows(p_minus, batch, num_windows=num_windows)
        return self._to_raw_occurrences(q_plus, p_plus, q_minus, p_minus, source_indices=source_indices, target_indices=target_indices)

    def clean_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> BlockHamiltonianOccurrences:
        batch = clean_future.shape[0]
        full = torch.cat([x0[:, None], clean_future], dim=1)
        signal = torch.ones(batch, self.future_steps + 1, device=full.device, dtype=full.dtype)
        return self.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=create_graph)

    def noisy_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, noise_scale: float, create_graph: bool, generator: torch.Generator | None=None) -> BlockHamiltonianOccurrences:
        if noise_scale <= 0:
            raise ValueError('noise_scale 必须为正')
        batch = clean_future.shape[0]
        clean_full = torch.cat([x0[:, None], clean_future], dim=1)
        num_states = self.future_steps + 1
        q_signal = torch.rand(batch, num_states, device=clean_full.device, dtype=clean_full.dtype, generator=generator)
        p_signal = torch.rand(batch, num_states, device=clean_full.device, dtype=clean_full.dtype, generator=generator)
        q_signal[:, 0] = 1.0
        p_signal[:, 0] = 1.0
        q_clean, p_clean = (clean_full[..., :self.q_dim], clean_full[..., self.q_dim:])
        q_noise = torch.randn(q_clean.shape, device=q_clean.device, dtype=q_clean.dtype, generator=generator)
        p_noise = torch.randn(p_clean.shape, device=p_clean.device, dtype=p_clean.dtype, generator=generator)
        q_noise = q_noise * noise_scale * self.q_scale.to(q_noise).view(1, 1, 1, -1)
        p_noise = p_noise * noise_scale * self.p_scale.to(p_noise).view(1, 1, 1, -1)
        q_weight = q_signal[..., None, None]
        p_weight = p_signal[..., None, None]
        q_corrupt = q_weight * q_clean + (1.0 - q_weight) * q_noise
        p_corrupt = p_weight * p_clean + (1.0 - p_weight) * p_noise
        corrupted_full = torch.cat([q_corrupt, p_corrupt], dim=-1)
        corrupted_full = torch.cat([x0[:, None], corrupted_full[:, 1:]], dim=1)
        return self.chart_occurrences(corrupted_full, q_signal=q_signal, p_signal=p_signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=create_graph)

    @staticmethod
    def _aggregate(first: torch.Tensor, first_indices: torch.Tensor, second: torch.Tensor, second_indices: torch.Tensor, *, num_states: int, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, windows, block, objects, dim = first.shape
        if second.shape != first.shape:
            raise ValueError('两组 occurrence shape 必须一致')
        values = torch.cat([first.reshape(batch, windows * block, objects, dim), second.reshape(batch, windows * block, objects, dim)], dim=1)
        indices = torch.cat([first_indices.reshape(-1), second_indices.reshape(-1)])
        total = values.new_zeros(batch, num_states, objects, dim).index_add(1, indices, values)
        squared_total = values.new_zeros(batch, num_states, objects, dim).index_add(1, indices, values.square())
        ones = values.new_ones(indices.numel())
        coverage = values.new_zeros(num_states).index_add(0, indices, ones)
        if bool((coverage <= 0).any().item()):
            raise RuntimeError('occurrence aggregation 存在未覆盖的时间状态')
        denominator = coverage.view(1, num_states, 1, 1)
        mean = total / denominator
        variance = (squared_total / denominator - mean.square()).clamp_min(0.0)
        normalized_variance = variance / scale.to(variance).view(1, 1, 1, -1).square()
        return (mean, coverage, normalized_variance)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool | None=None) -> BlockHamiltonianOutput:
        self._validate_inputs(noisy_future, tau, x0, attrs, physical_time)
        if create_graph is None:
            create_graph = self.training
        full = torch.cat([x0[:, None], noisy_future], dim=1)
        signal = tau[:, None].expand(-1, self.future_steps + 1).clone()
        signal[:, 0] = 1.0
        occurrences = self.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=self.chart_coupling == 'sequential', create_graph=create_graph, boundary_strength=self.rf_boundary_strength(tau))
        q_mean, q_coverage, q_variance = self._aggregate(occurrences.q_plus, occurrences.target_indices, occurrences.q_minus, occurrences.source_indices, num_states=self.future_steps + 1, scale=self.q_scale)
        p_mean, p_coverage, p_variance = self._aggregate(occurrences.p_plus, occurrences.source_indices, occurrences.p_minus, occurrences.target_indices, num_states=self.future_steps + 1, scale=self.p_scale)
        q_full = torch.cat([x0[:, None, ..., :self.q_dim], q_mean[:, 1:]], dim=1)
        p_full = torch.cat([x0[:, None, ..., self.q_dim:], p_mean[:, 1:]], dim=1)
        clean = torch.cat([q_full[:, 1:], p_full[:, 1:]], dim=-1)
        disagreement = torch.cat([q_variance, p_variance], dim=-1)
        disagreement = disagreement.mean(dim=(1, 2, 3)).sqrt()
        if not create_graph:
            clean = clean.detach()
            disagreement = disagreement.detach()
            occurrences = occurrences.detached()
        return BlockHamiltonianOutput(clean=clean, disagreement=disagreement, occurrences=occurrences, q_coverage=q_coverage.detach(), p_coverage=p_coverage.detach())
