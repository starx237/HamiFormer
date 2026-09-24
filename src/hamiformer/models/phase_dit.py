from __future__ import annotations
import torch
from torch import nn
from .common import ContinuousRoPE, ManualSelfAttention, RMSNorm, SinusoidalMLP, SwiGLU, modulate

class FactorizedPhaseBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, num_register_tokens: int, q_dim: int, dropout: float, qk_norm: bool, mlp_inner_dim: int | None=None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_register_tokens = int(num_register_tokens)
        self.object_norm = RMSNorm(hidden_size)
        self.temporal_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.object_attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm, use_sdpa=True)
        self.temporal_attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm, use_sdpa=True)
        head_dim = hidden_size // num_heads
        self.object_rope = ContinuousRoPE(head_dim, coord_dim=q_dim)
        self.temporal_rope = ContinuousRoPE(head_dim, coord_dim=1)
        self.mlp = SwiGLU(hidden_size, mlp_ratio, dropout, inner_dim=mlp_inner_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, registers: torch.Tensor, rf_condition: torch.Tensor, q_coords: torch.Tensor, physical_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects, hidden = tokens.shape
        if registers.shape != (batch, self.num_register_tokens, hidden):
            raise ValueError('registers 必须为 [B,R,D]')
        params = self.modulation(rf_condition).chunk(9, dim=-1)
        shift_o, scale_o, gate_o, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = params
        object_tokens = tokens.reshape(batch * frames, objects, hidden)
        object_shift = shift_o[:, None, :].expand(batch, frames, hidden).reshape(batch * frames, hidden)
        object_scale = scale_o[:, None, :].expand(batch, frames, hidden).reshape(batch * frames, hidden)
        object_input = modulate(self.object_norm(object_tokens), object_shift, object_scale)
        object_coords = q_coords.reshape(batch * frames, objects, q_coords.shape[-1])
        object_registers = registers[:, None].expand(batch, frames, -1, -1)
        object_registers = object_registers.reshape(batch * frames, self.num_register_tokens, hidden)
        if self.num_register_tokens > 0:
            object_register_input = modulate(self.object_norm(object_registers), object_shift, object_scale)
            object_input = torch.cat([object_register_input, object_input], dim=1)
            zero_coords = object_coords.new_zeros(batch * frames, self.num_register_tokens, object_coords.shape[-1])
            object_coords = torch.cat([zero_coords, object_coords], dim=1)
        object_output = self.object_attention(object_input, rope=self.object_rope, coords=object_coords)
        if self.num_register_tokens > 0:
            register_delta = object_output[:, :self.num_register_tokens]
            register_delta = register_delta.reshape(batch, frames, self.num_register_tokens, hidden)
            registers = (registers[:, None] + gate_o[:, None, None, :] * register_delta).mean(dim=1)
            object_output = object_output[:, self.num_register_tokens:]
        object_output = object_output.reshape(batch, frames, objects, hidden)
        tokens = tokens + gate_o[:, None, None, :] * object_output
        temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(batch * objects, frames, hidden)
        temporal_shift = shift_t[:, None, :].expand(batch, objects, hidden).reshape(batch * objects, hidden)
        temporal_scale = scale_t[:, None, :].expand(batch, objects, hidden).reshape(batch * objects, hidden)
        temporal_input = modulate(self.temporal_norm(temporal_tokens), temporal_shift, temporal_scale)
        time_coords = physical_time[:, None, :, None].expand(batch, objects, frames, 1)
        time_coords = time_coords.reshape(batch * objects, frames, 1)
        temporal_registers = registers[:, None].expand(batch, objects, -1, -1)
        temporal_registers = temporal_registers.reshape(batch * objects, self.num_register_tokens, hidden)
        if self.num_register_tokens > 0:
            temporal_register_input = modulate(self.temporal_norm(temporal_registers), temporal_shift, temporal_scale)
            temporal_input = torch.cat([temporal_register_input, temporal_input], dim=1)
            zero_time = time_coords.new_zeros(batch * objects, self.num_register_tokens, 1)
            time_coords = torch.cat([zero_time, time_coords], dim=1)
        temporal_output = self.temporal_attention(temporal_input, rope=self.temporal_rope, coords=time_coords)
        if self.num_register_tokens > 0:
            register_delta = temporal_output[:, :self.num_register_tokens]
            register_delta = register_delta.reshape(batch, objects, self.num_register_tokens, hidden)
            registers = (registers[:, None] + gate_t[:, None, None, :] * register_delta).mean(dim=1)
            temporal_output = temporal_output[:, self.num_register_tokens:]
        temporal_output = temporal_output.reshape(batch, objects, frames, hidden).permute(0, 2, 1, 3)
        tokens = tokens + gate_t[:, None, None, :] * temporal_output
        flat = tokens.reshape(batch, frames * objects, hidden)
        mlp_input = modulate(self.mlp_norm(flat), shift_m, scale_m)
        flat = flat + gate_m.unsqueeze(1) * self.mlp(mlp_input)
        if self.num_register_tokens > 0:
            register_input = modulate(self.mlp_norm(registers), shift_m, scale_m)
            registers = registers + gate_m.unsqueeze(1) * self.mlp(register_input)
        return (flat.reshape(batch, frames, objects, hidden), registers)

class PhysiFormerAxisPhaseBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, num_register_tokens: int, q_dim: int, dropout: float, qk_norm: bool, *, attention_axis: str, mlp_inner_dim: int | None=None) -> None:
        super().__init__()
        if attention_axis not in {'spatial', 'temporal', 'object'}:
            raise ValueError('attention_axis 必须为 spatial/temporal/object')
        self.attention_axis = attention_axis
        self.num_register_tokens = int(num_register_tokens)
        self.num_heads = int(num_heads)
        self.attention_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.attention = ManualSelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm, use_sdpa=True)
        head_dim = hidden_size // num_heads
        self.rope = ContinuousRoPE(head_dim, coord_dim=1 if attention_axis == 'temporal' else q_dim)
        self.mlp = SwiGLU(hidden_size, mlp_ratio, dropout, inner_dim=mlp_inner_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _group_tokens_and_coords(self, tokens: torch.Tensor, q_coords: torch.Tensor, temporal_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch, frames, objects, hidden = tokens.shape
        if self.attention_axis == 'spatial':
            grouped = tokens.reshape(batch * frames, objects, hidden)
            coords = q_coords.reshape(batch * frames, objects, q_coords.shape[-1])
            return (grouped, coords, frames)
        if self.attention_axis == 'temporal':
            grouped = tokens.permute(0, 2, 1, 3).reshape(batch * objects, frames, hidden)
            coords = temporal_coords[:, None, :, None].expand(batch, objects, frames, 1)
            return (grouped, coords.reshape(batch * objects, frames, 1), objects)
        grouped = tokens.reshape(batch * frames * objects, 1, hidden)
        coords = q_coords.reshape(batch * frames * objects, 1, q_coords.shape[-1])
        return (grouped, coords, frames * objects)

    def _restore_tokens(self, grouped: torch.Tensor, *, batch: int, frames: int, objects: int, hidden: int) -> torch.Tensor:
        if self.attention_axis == 'spatial':
            return grouped.reshape(batch, frames, objects, hidden)
        if self.attention_axis == 'temporal':
            return grouped.reshape(batch, objects, frames, hidden).permute(0, 2, 1, 3)
        return grouped.reshape(batch, frames, objects, hidden)

    def forward(self, tokens: torch.Tensor, registers: torch.Tensor, rf_condition: torch.Tensor, q_coords: torch.Tensor, temporal_coords: torch.Tensor, object_mask: torch.Tensor | None=None, spatial_attention_bias: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects, hidden = tokens.shape
        if object_mask is not None and object_mask.shape != (batch, objects):
            raise ValueError('object_mask 必须为 [B,K]')
        if spatial_attention_bias is not None and spatial_attention_bias.shape != (batch, self.num_heads, objects, objects):
            raise ValueError('spatial_attention_bias 必须为 [B,H,K,K]')
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(rf_condition).chunk(6, dim=-1)
        grouped, coords, groups_per_sample = self._group_tokens_and_coords(tokens, q_coords, temporal_coords)
        group_batch = grouped.shape[0]
        group_shift = shift_a[:, None, :].expand(batch, groups_per_sample, hidden).reshape(group_batch, hidden)
        group_scale = scale_a[:, None, :].expand(batch, groups_per_sample, hidden).reshape(group_batch, hidden)
        attention_input = modulate(self.attention_norm(grouped), group_shift, group_scale)
        keep_mask: torch.Tensor | None = None
        grouped_bias: torch.Tensor | None = None
        if self.attention_axis == 'spatial':
            if object_mask is not None:
                keep_mask = object_mask[:, None, :].expand(batch, frames, objects).reshape(batch * frames, objects)
            if spatial_attention_bias is not None:
                grouped_bias = spatial_attention_bias[:, None].expand(batch, frames, self.num_heads, objects, objects).reshape(batch * frames, self.num_heads, objects, objects)
        if self.num_register_tokens > 0:
            register_groups = registers[:, None].expand(batch, groups_per_sample, self.num_register_tokens, hidden).reshape(group_batch, self.num_register_tokens, hidden)
            register_input = modulate(self.attention_norm(register_groups), group_shift, group_scale)
            attention_input = torch.cat([register_input, attention_input], dim=1)
            register_coords = coords.new_zeros(group_batch, self.num_register_tokens, coords.shape[-1])
            coords = torch.cat([register_coords, coords], dim=1)
            if keep_mask is not None:
                keep_mask = torch.cat([torch.ones(group_batch, self.num_register_tokens, device=keep_mask.device, dtype=torch.bool), keep_mask], dim=1)
            if grouped_bias is not None:
                padded_bias = grouped_bias.new_zeros(group_batch, self.num_heads, objects + self.num_register_tokens, objects + self.num_register_tokens)
                padded_bias[:, :, self.num_register_tokens:, self.num_register_tokens:] = grouped_bias
                grouped_bias = padded_bias
        attention_output = self.attention(attention_input, rope=self.rope, coords=coords, keep_mask=keep_mask, attention_bias=grouped_bias)
        if self.num_register_tokens > 0:
            register_delta = attention_output[:, :self.num_register_tokens]
            register_delta = register_delta.reshape(batch, groups_per_sample, self.num_register_tokens, hidden)
            if object_mask is not None and self.attention_axis in {'temporal', 'object'}:
                group_valid = object_mask
                if self.attention_axis == 'object':
                    group_valid = object_mask[:, None, :].expand(batch, frames, objects).reshape(batch, frames * objects)
                weight = group_valid.to(register_delta).unsqueeze(-1).unsqueeze(-1)
                register_delta = (register_delta * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
            else:
                register_delta = register_delta.mean(dim=1)
            registers = registers + gate_a.unsqueeze(1) * register_delta
            attention_output = attention_output[:, self.num_register_tokens:]
        token_delta = self._restore_tokens(attention_output, batch=batch, frames=frames, objects=objects, hidden=hidden)
        tokens = tokens + gate_a[:, None, None, :] * token_delta
        flat_tokens = tokens.reshape(batch, frames * objects, hidden)
        combined = torch.cat([registers, flat_tokens], dim=1)
        mlp_input = modulate(self.mlp_norm(combined), shift_m, scale_m)
        combined = combined + gate_m.unsqueeze(1) * self.mlp(mlp_input)
        registers = combined[:, :self.num_register_tokens]
        tokens = combined[:, self.num_register_tokens:].reshape(batch, frames, objects, hidden)
        if object_mask is not None:
            tokens = tokens * object_mask[:, None, :, None].to(tokens)
        return (tokens, registers)

class PhaseDiT(nn.Module):

    def __init__(self, *, state_dim: int, q_dim: int, attr_dim: int, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float, num_register_tokens: int, mlp_inner_dim: int | None=None, dropout: float=0.0, qk_norm: bool=True, block_attn_pattern: tuple[str, ...] | None=None, temporal_rope_mode: str='physical_time', auxiliary_state_dim: int=0, background_state_dim: int=0) -> None:
        super().__init__()
        if state_dim != 2 * q_dim:
            raise ValueError('PhaseDiT 首版要求 state_dim=2*q_dim')
        self.state_dim = int(state_dim)
        self.q_dim = int(q_dim)
        if auxiliary_state_dim < 0:
            raise ValueError('auxiliary_state_dim 不得为负')
        if background_state_dim < 0:
            raise ValueError('background_state_dim 不得为负')
        if auxiliary_state_dim > 0 and background_state_dim > 0:
            raise ValueError('auxiliary_state_dim 与 background_state_dim 不能同时启用')
        self.auxiliary_state_dim = int(auxiliary_state_dim)
        self.background_state_dim = int(background_state_dim)
        self.block_attn_pattern = block_attn_pattern
        if mlp_inner_dim is not None and int(mlp_inner_dim) < 8:
            raise ValueError('mlp_inner_dim must be >= 8 when provided')
        self.mlp_inner_dim = None if mlp_inner_dim is None else int(mlp_inner_dim)
        if temporal_rope_mode not in {'physical_time', 'frame_index'}:
            raise ValueError('temporal_rope_mode 必须为 physical_time 或 frame_index')
        self.temporal_rope_mode = temporal_rope_mode
        self.state_projection = nn.Linear(state_dim + self.auxiliary_state_dim, hidden_size)
        self.background_projection = nn.Linear(self.background_state_dim, hidden_size, bias=False) if self.background_state_dim > 0 else None
        if self.background_projection is not None:
            nn.init.zeros_(self.background_projection.weight)
        if block_attn_pattern is None:
            self.condition_projection = nn.Sequential(nn.Linear(state_dim + attr_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.initial_q_projection = None
            self.initial_p_projection = None
            self.attr_projection = None
            self.phase_token = None
        else:
            if not block_attn_pattern:
                raise ValueError('block_attn_pattern 不能为空')
            if any((axis not in {'spatial', 'temporal', 'object'} for axis in block_attn_pattern)):
                raise ValueError('block_attn_pattern 含未知 attention axis')
            self.condition_projection = None
            self.initial_q_projection = nn.Linear(q_dim, hidden_size)
            self.initial_p_projection = nn.Linear(q_dim, hidden_size)
            self.attr_projection = nn.Sequential(nn.Linear(attr_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.phase_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.rf_time_embedding = SinusoidalMLP(hidden_size)
        self.physical_time_embedding = SinusoidalMLP(hidden_size) if block_attn_pattern is None else None
        if block_attn_pattern is None:
            self.blocks = nn.ModuleList([FactorizedPhaseBlock(hidden_size, num_heads, mlp_ratio, num_register_tokens, q_dim, dropout, qk_norm, self.mlp_inner_dim) for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([PhysiFormerAxisPhaseBlock(hidden_size, num_heads, mlp_ratio, num_register_tokens, q_dim, dropout, qk_norm, attention_axis=block_attn_pattern[index % len(block_attn_pattern)], mlp_inner_dim=self.mlp_inner_dim) for index in range(depth)])
        self.num_register_tokens = int(num_register_tokens)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size) * 0.02)
        else:
            self.register_tokens = None
        self.final_norm = RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output = nn.Linear(hidden_size, state_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_tokens(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, auxiliary_future: torch.Tensor | None=None, background_future: torch.Tensor | None=None, object_mask: torch.Tensor | None=None, spatial_attention_bias: torch.Tensor | None=None) -> torch.Tensor:
        if noisy_future.ndim != 4 or noisy_future.shape[-1] != self.state_dim:
            raise ValueError('noisy_future 必须为 [B,F,K,state_dim]')
        batch, frames, objects, _ = noisy_future.shape
        if x0.shape != (batch, objects, self.state_dim):
            raise ValueError('x0 必须为 [B,K,state_dim]')
        if attrs.shape[:2] != (batch, objects):
            raise ValueError('attrs 必须为 [B,K,attr_dim]')
        if physical_time.shape != (batch, frames + 1):
            raise ValueError('physical_time 必须包含初态，为 [B,F+1]')
        if tau.shape != (batch,):
            raise ValueError('tau 必须为 [B]')
        if object_mask is not None:
            if object_mask.shape != (batch, objects):
                raise ValueError('object_mask 必须为 [B,K]')
            object_mask = object_mask.to(device=noisy_future.device, dtype=torch.bool)
        if spatial_attention_bias is not None and self.block_attn_pattern is None:
            raise ValueError('reference factorized block 不接受 spatial_attention_bias')
        if self.auxiliary_state_dim == 0:
            if auxiliary_future is not None:
                raise ValueError('auxiliary_future requires auxiliary_state_dim > 0')
            projected_state = noisy_future
        else:
            expected_auxiliary = (batch, frames, objects, self.auxiliary_state_dim)
            if auxiliary_future is None or auxiliary_future.shape != expected_auxiliary:
                raise ValueError(f'auxiliary_future 必须为 [B,F,K,{self.auxiliary_state_dim}]')
            projected_state = torch.cat([noisy_future, auxiliary_future], dim=-1)
        if self.background_state_dim == 0:
            if background_future is not None:
                raise ValueError('无 background 输入的 PhaseDiT 不接受 background_future；请显式构造 background_state_dim>0 的模型')
            projected_background: torch.Tensor | None = None
        else:
            expected_background = (batch, frames, objects, self.background_state_dim)
            if background_future is None or background_future.shape != expected_background:
                raise ValueError(f'background_future 必须为 [B,F,K,{self.background_state_dim}]')
            projected_background = background_future
        rf_condition = self.rf_time_embedding(tau)
        if self.block_attn_pattern is None:
            if self.condition_projection is None:
                raise RuntimeError('reference condition_projection 未初始化')
            object_condition = self.condition_projection(torch.cat([x0, attrs], dim=-1))
        else:
            if self.initial_q_projection is None or self.initial_p_projection is None or self.attr_projection is None:
                raise RuntimeError('canonical condition projections 未初始化')
            object_condition = self.initial_q_projection(x0[..., :self.q_dim]) + self.initial_p_projection(x0[..., self.q_dim:]) + self.attr_projection(attrs)
        object_condition = object_condition[:, None, :, :].expand(batch, frames, objects, -1)
        relative_time = physical_time[:, 1:] - physical_time[:, :1]
        if self.block_attn_pattern is None:
            if self.physical_time_embedding is None:
                raise RuntimeError('reference physical_time_embedding 未初始化')
            time_embedding = self.physical_time_embedding(relative_time.reshape(-1))
            time_embedding = time_embedding.reshape(batch, frames, 1, -1)
        else:
            time_embedding = 0.0
        tokens = self.state_projection(projected_state) + object_condition + time_embedding
        if projected_background is not None:
            if self.background_projection is None:
                raise RuntimeError('background_projection 未初始化')
            tokens = tokens + self.background_projection(projected_background)
        if self.phase_token is not None:
            tokens = tokens + self.phase_token[:, None, :, :]
        if object_mask is not None:
            tokens = tokens * object_mask[:, None, :, None].to(tokens)
        q_coords = noisy_future[..., :self.q_dim]
        if self.register_tokens is None:
            registers = tokens.new_empty(batch, 0, tokens.shape[-1])
        else:
            registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        if self.temporal_rope_mode == 'frame_index':
            temporal_coords = torch.arange(frames, device=noisy_future.device, dtype=noisy_future.dtype)[None].expand(batch, frames)
        else:
            temporal_coords = relative_time
        for block in self.blocks:
            if self.block_attn_pattern is None:
                tokens, registers = block(tokens, registers, rf_condition, q_coords, temporal_coords)
            else:
                tokens, registers = block(tokens, registers, rf_condition, q_coords, temporal_coords, object_mask=object_mask, spatial_attention_bias=spatial_attention_bias)
        shift, scale = self.final_modulation(rf_condition).chunk(2, dim=-1)
        flat = tokens.reshape(batch, frames * objects, -1)
        flat = modulate(self.final_norm(flat), shift, scale)
        tokens = flat.reshape(batch, frames, objects, -1)
        if object_mask is not None:
            tokens = tokens * object_mask[:, None, :, None].to(tokens)
        return tokens

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, auxiliary_future: torch.Tensor | None=None, background_future: torch.Tensor | None=None, object_mask: torch.Tensor | None=None, spatial_attention_bias: torch.Tensor | None=None) -> torch.Tensor:
        tokens = self.encode_tokens(noisy_future, tau, x0=x0, attrs=attrs, physical_time=physical_time, auxiliary_future=auxiliary_future, background_future=background_future, object_mask=object_mask, spatial_attention_bias=spatial_attention_bias)
        batch, frames, objects, hidden = tokens.shape
        output = self.output(tokens.reshape(batch, frames * objects, hidden)).reshape(batch, frames, objects, self.state_dim)
        if object_mask is not None:
            output = output * object_mask[:, None, :, None].to(output)
        return output

def phase_dit_architecture_kwargs(model_config: dict) -> dict:
    pattern = model_config.get('block_attn_pattern')
    if pattern is None:
        return {}
    if not isinstance(pattern, (list, tuple)):
        raise ValueError('model.block_attn_pattern 必须为字符串列表')
    return {'block_attn_pattern': tuple((str(axis) for axis in pattern)), 'temporal_rope_mode': str(model_config.get('temporal_rope_mode', 'frame_index')), 'mlp_inner_dim': None if model_config.get('mlp_inner_dim') is None else int(model_config['mlp_inner_dim'])}

def initialize_nested_wide_phase_dit(narrow: PhaseDiT, wide: PhaseDiT) -> dict[str, int | bool]:
    if type(narrow) is not PhaseDiT or type(wide) is not PhaseDiT:
        raise TypeError('nested initialisation requires two PhaseDiT instances')
    if len(narrow.blocks) != len(wide.blocks):
        raise ValueError('nested PhaseDiT models must have equal depth')
    narrow_inner = {int(block.mlp.inner_dim) for block in narrow.blocks}
    wide_inner = {int(block.mlp.inner_dim) for block in wide.blocks}
    if len(narrow_inner) != 1 or len(wide_inner) != 1:
        raise ValueError('nested PhaseDiT requires a uniform FFN width per model')
    source_inner = next(iter(narrow_inner))
    target_inner = next(iter(wide_inner))
    if target_inner < source_inner:
        raise ValueError('wide PhaseDiT FFN must not be narrower than source')
    source_state = narrow.state_dict()
    target_state = wide.state_dict()
    skip_fragments = ('.mlp.input.weight', '.mlp.input.bias', '.mlp.output.weight')
    with torch.no_grad():
        for name, source in source_state.items():
            target = target_state.get(name)
            if target is None or any((fragment in name for fragment in skip_fragments)):
                continue
            if source.shape != target.shape:
                raise ValueError(f'non-FFN nested tensor shape mismatch for {name}: {tuple(source.shape)} vs {tuple(target.shape)}')
            target.copy_(source)
        wide.load_state_dict(target_state)
        for source_block, target_block in zip(narrow.blocks, wide.blocks):
            source_mlp = source_block.mlp
            target_mlp = target_block.mlp
            target_mlp.input.weight[:source_inner].copy_(source_mlp.input.weight[:source_inner])
            target_mlp.input.weight[target_inner:target_inner + source_inner].copy_(source_mlp.input.weight[source_inner:])
            target_mlp.input.bias[:source_inner].copy_(source_mlp.input.bias[:source_inner])
            target_mlp.input.bias[target_inner:target_inner + source_inner].copy_(source_mlp.input.bias[source_inner:])
            target_mlp.output.weight[:, :source_inner].copy_(source_mlp.output.weight)
            target_mlp.output.weight[:, source_inner:target_inner].zero_()
            target_mlp.output.bias.copy_(source_mlp.output.bias)
    return {'narrow_inner_dim': source_inner, 'wide_inner_dim': target_inner, 'added_inner_units_per_block': target_inner - source_inner, 'function_preserving': True, 'extra_output_columns_zero': True}
