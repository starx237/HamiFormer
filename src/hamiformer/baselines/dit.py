from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .physiformer import HamiBalls2WideD

def sinusoidal(value: torch.Tensor, width: int) -> torch.Tensor:
    frequency = torch.exp(-math.log(10000) * torch.arange(width // 2, device=value.device, dtype=torch.float32) / (width // 2))
    phase = value.float()[..., None] * frequency
    return torch.cat((phase.cos(), phase.sin()), dim=-1)

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

class DiTBlock(nn.Module):

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-06)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-06)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate='tanh'), nn.Linear(4 * width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition, bias):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, -1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        b, length, width = h.shape
        q, k, v = self.qkv(h).reshape(b, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.0)
        h = self.proj(h.transpose(1, 2).reshape(b, length, width))
        x = x + gate_a[:, None] * h
        return x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))

class HamiBalls2DiT(nn.Module):
    _graph_inputs = HamiBalls2WideD._graph_inputs

    def __init__(self, *, hidden_size: int, depth: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads or hidden_size % 2:
            raise ValueError('even hidden_size must be divisible by num_heads')
        self.width = hidden_size
        self.num_heads = num_heads
        self.edge_bias = nn.Linear(3, num_heads, bias=False)
        self.input = nn.Linear(6, hidden_size)
        self.initial_condition = nn.Linear(6 + 7, hidden_size)
        self.timestep = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output = nn.Linear(hidden_size, 6)
        self.apply(self._initialize)
        nn.init.zeros_(self.edge_bias.weight)
        nn.init.normal_(self.timestep[0].weight, std=0.02)
        nn.init.normal_(self.timestep[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, noisy_future, tau, *, x0, attrs, physical_time, object_mask, spring_mask, spring_k, spring_rest_length):
        b, frames, objects, _ = noisy_future.shape
        if physical_time.shape != (b, frames + 1):
            raise ValueError('physical_time must include initial frame and each future frame')
        attrs, graph_bias = self._graph_inputs(attrs, object_mask, spring_mask, spring_k, spring_rest_length)
        static = self.initial_condition(torch.cat((x0, attrs), -1))
        elapsed = physical_time[:, 1:] - physical_time[:, :1]
        position = sinusoidal(elapsed, self.width)
        x = self.input(noisy_future) + static[:, None] + position[:, :, None]
        x = x.reshape(b, frames * objects, self.width)
        condition = self.timestep(sinusoidal(tau.reshape(b), 256))
        bias = graph_bias[:, :, None, :, None, :].expand(b, self.num_heads, frames, objects, frames, objects).reshape(b, self.num_heads, frames * objects, frames * objects)
        keys = object_mask[:, None, :].expand(b, frames, objects).reshape(b, -1)
        bias = bias.masked_fill(~keys[:, None, None, :], float('-inf'))
        for block in self.blocks:
            x = block(x, condition, bias)
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        prediction = self.output(modulate(self.norm_final(x), shift, scale))
        return prediction.reshape(b, frames, objects, 6) * object_mask[:, None, :, None]
