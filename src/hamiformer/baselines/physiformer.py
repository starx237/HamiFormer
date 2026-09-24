from __future__ import annotations
import torch
from torch import nn
from hamiformer.models.phase_dit import PhaseDiT

class HamiBalls2WideD(nn.Module):

    def __init__(self, *, hidden_size: int, depth: int, num_heads: int, mlp_inner_dim: int, num_register_tokens: int=2) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.edge_bias = nn.Linear(3, num_heads, bias=False)
        nn.init.zeros_(self.edge_bias.weight)
        self.backbone = PhaseDiT(state_dim=6, q_dim=3, attr_dim=7, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=2.0, mlp_inner_dim=mlp_inner_dim, num_register_tokens=num_register_tokens, dropout=0.0, qk_norm=True, block_attn_pattern=('spatial', 'temporal', 'object', 'temporal'), temporal_rope_mode='physical_time')

    def _graph_inputs(self, attrs: torch.Tensor, object_mask: torch.Tensor, spring_mask: torch.Tensor, spring_k: torch.Tensor, spring_rest_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = object_mask.to(dtype=attrs.dtype)
        edge = spring_mask.to(dtype=attrs.dtype)
        log_k = torch.log1p(spring_k) * edge
        rest = spring_rest_length * edge
        degree = edge.sum(dim=-1)
        denominator = degree.clamp_min(1.0)
        node_graph = torch.stack([degree / max(object_mask.shape[-1] - 1, 1), log_k.sum(dim=-1) / denominator, rest.sum(dim=-1) / denominator], dim=-1)
        enriched_attrs = torch.cat([attrs, mask.unsqueeze(-1), node_graph], dim=-1)
        enriched_attrs = enriched_attrs * mask.unsqueeze(-1)
        pair_features = torch.stack([edge, log_k, rest], dim=-1)
        pair_bias = self.edge_bias(pair_features).permute(0, 3, 1, 2)
        pair_bias = pair_bias * edge.unsqueeze(1)
        return (enriched_attrs, pair_bias)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, object_mask: torch.Tensor, spring_mask: torch.Tensor, spring_k: torch.Tensor, spring_rest_length: torch.Tensor) -> torch.Tensor:
        enriched_attrs, pair_bias = self._graph_inputs(attrs, object_mask, spring_mask, spring_k, spring_rest_length)
        return self.backbone(noisy_future, tau, x0=x0, attrs=enriched_attrs, physical_time=physical_time, object_mask=object_mask, spatial_attention_bias=pair_bias)

    def forward_with_tokens(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, object_mask: torch.Tensor, spring_mask: torch.Tensor, spring_k: torch.Tensor, spring_rest_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enriched_attrs, pair_bias = self._graph_inputs(attrs, object_mask, spring_mask, spring_k, spring_rest_length)
        tokens = self.backbone.encode_tokens(noisy_future, tau, x0=x0, attrs=enriched_attrs, physical_time=physical_time, object_mask=object_mask, spatial_attention_bias=pair_bias)
        prediction = self.backbone.output(tokens)
        prediction = prediction * object_mask[:, None, :, None].to(prediction)
        return (prediction, tokens)

def parameter_count(module: nn.Module) -> int:
    return sum((parameter.numel() for parameter in module.parameters()))
__all__ = ['HamiBalls2WideD', 'parameter_count']
