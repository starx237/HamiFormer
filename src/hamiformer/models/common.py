from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight

class SinusoidalMLP(nn.Module):

    def __init__(self, hidden_size: int, frequency_dim: int=256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(nn.Linear(self.frequency_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1:
            raise ValueError(f'连续标量嵌入期望 [N]，实际 {tuple(value.shape)}')
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=value.device, dtype=torch.float32) / max(half - 1, 1))
        angles = value.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if embedding.shape[-1] < self.frequency_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.frequency_dim - embedding.shape[-1]))
        return self.mlp(embedding.to(dtype=value.dtype))

class ContinuousRoPE(nn.Module):

    def __init__(self, head_dim: int, coord_dim: int, base: float=10000.0) -> None:
        super().__init__()
        if coord_dim <= 0:
            raise ValueError('coord_dim 必须为正')
        axis_dim = head_dim // coord_dim // 2 * 2
        if axis_dim < 2:
            raise ValueError('每个坐标轴至少需要两个 head dimensions')
        self.coord_dim = int(coord_dim)
        self.axis_dim = int(axis_dim)
        half = axis_dim // 2
        inv_freq = base ** (-torch.arange(half, dtype=torch.float32) / max(half, 1))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even, odd = (pair[..., 0], pair[..., 1])
        cos, sin = (torch.cos(angle), torch.sin(angle))
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.flatten(-2)

    def _apply_one(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        cursor = 0
        for axis in range(self.coord_dim):
            part = x[..., cursor:cursor + self.axis_dim]
            angle = (coords[:, None, :, axis, None].float() * self.inv_freq[None, None, None, :]).to(dtype=part.dtype)
            pieces.append(self._rotate_pairs(part, angle))
            cursor += self.axis_dim
        if cursor < x.shape[-1]:
            pieces.append(x[..., cursor:])
        return torch.cat(pieces, dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.ndim != 3 or coords.shape[0] != q.shape[0] or coords.shape[1] != q.shape[2]:
            raise ValueError('RoPE coords 必须为 [N,L,C] 并与 q/k 序列一致')
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(f'RoPE 期望坐标维 {self.coord_dim}，实际 {coords.shape[-1]}')
        return (self._apply_one(q, coords), self._apply_one(k, coords))

class ManualSelfAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, *, dropout: float=0.0, qk_norm: bool=True, use_sdpa: bool=False) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError('hidden_size 必须能被 num_heads 整除')
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qk_norm = bool(qk_norm)
        self.use_sdpa = bool(use_sdpa)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _head_rms(x: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)

    def forward(self, x: torch.Tensor, *, rope: ContinuousRoPE | None=None, coords: torch.Tensor | None=None, keep_mask: torch.Tensor | None=None, attention_bias: torch.Tensor | None=None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'attention 输入必须为 [N,L,D]，实际 {tuple(x.shape)}')
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            q, k = (self._head_rms(q), self._head_rms(k))
        if rope is not None:
            if coords is None:
                raise ValueError('启用 RoPE 时必须提供 coords')
            q, k = rope(q, k, coords)
        if keep_mask is not None and keep_mask.shape != (batch, length):
            raise ValueError('keep_mask 必须为 [N,L]')
        if attention_bias is not None:
            if attention_bias.ndim != 4:
                raise ValueError('attention_bias 必须为 [N,H|1,L,L]')
            if attention_bias.shape[0] != batch or attention_bias.shape[1] not in {1, self.num_heads} or attention_bias.shape[2:] != (length, length):
                raise ValueError('attention_bias 形状与 attention 不一致')
        if self.use_sdpa:
            attention_mask = attention_bias
            if keep_mask is not None:
                if attention_mask is None:
                    attention_mask = torch.zeros(batch, 1, length, length, device=x.device, dtype=q.dtype)
                attention_mask = attention_mask.masked_fill(~keep_mask[:, None, None, :], torch.finfo(attention_mask.dtype).min)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_bias is not None:
                logits = logits + attention_bias.to(dtype=logits.dtype)
            if keep_mask is not None:
                logits = logits.masked_fill(~keep_mask[:, None, None, :], torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
            weights = self.dropout(weights)
            output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch, length, self.hidden_size)
        output = self.output(output)
        if keep_mask is not None:
            output = output * keep_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output

class SwiGLU(nn.Module):

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float=0.0, *, inner_dim: int | None=None) -> None:
        super().__init__()
        if inner_dim is not None and int(inner_dim) < 8:
            raise ValueError('SwiGLU inner_dim must be >= 8')
        inner = max(8, int(hidden_size * mlp_ratio)) if inner_dim is None else int(inner_dim)
        self.inner_dim = inner
        self.input = nn.Linear(hidden_size, 2 * inner)
        self.output = nn.Linear(inner, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(self.dropout(value * torch.nn.functional.silu(gate)))

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
