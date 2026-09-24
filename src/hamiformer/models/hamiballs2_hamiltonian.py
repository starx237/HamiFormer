from __future__ import annotations
import math
import torch
from torch import nn

class _MaskedGraphAttention(nn.Module):

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError('attention width must be divisible by heads')
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor, object_mask: torch.Tensor, pair_bias: torch.Tensor) -> torch.Tensor:
        if value.shape[:-1] != object_mask.shape:
            raise ValueError('object mask does not align with Hamiltonian tokens')
        *prefix, objects, width = value.shape
        flat_batch = math.prod(prefix) if prefix else 1
        packed = self.qkv(value.reshape(flat_batch, objects, width)).reshape(flat_batch, objects, 3, self.heads, self.head_width)
        query, key, content = packed.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = query @ key.transpose(-1, -2) * self.head_width ** (-0.5)
        scores = scores + pair_bias.reshape(flat_batch, self.heads, objects, objects)
        mask = object_mask.reshape(flat_batch, objects)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ content
        result = self.output(attended.transpose(1, 2).reshape(flat_batch, objects, width))
        return (result * mask[..., None].to(result)).reshape(*prefix, objects, width)

class _HamiltonianBlock(nn.Module):

    def __init__(self, width: int, heads: int, expansion: float) -> None:
        super().__init__()
        hidden = max(width, int(round(width * expansion)))
        self.attention_norm = nn.LayerNorm(width)
        self.attention = _MaskedGraphAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward_in = nn.Linear(width, hidden)
        self.feedforward_out = nn.Linear(hidden, width, bias=False)

    def forward(self, value: torch.Tensor, object_mask: torch.Tensor, pair_bias: torch.Tensor) -> torch.Tensor:
        mask = object_mask[..., None].to(value)
        value = (value + self.attention(self.attention_norm(value), object_mask, pair_bias)) * mask
        value = (value + self.feedforward_out(torch.nn.functional.silu(self.feedforward_in(self.feedforward_norm(value))))) * mask
        return value

class _SymmetricPairEnergy(nn.Module):

    def __init__(self, hidden_size: int, num_objects: int, mode: str) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError('pair-energy hidden size must be positive')
        if mode not in {'general_hamiltonian', 'configuration_potential', 'radial_configuration_potential', 'distance_configuration_potential'}:
            raise ValueError(f'unsupported pair-energy mode: {mode}')
        self.mode = mode
        input_size = 29 if mode == 'general_hamiltonian' else 10 if mode in {'radial_configuration_potential', 'distance_configuration_potential'} else 12
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.0001)
        self.register_buffer('upper_triangle', torch.triu(torch.ones(num_objects, num_objects, dtype=torch.bool), diagonal=1), persistent=False)

    def _ordered_energy(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.input_projection(features))
        hidden = torch.nn.functional.silu(self.hidden_projection(hidden))
        return self.output_projection(hidden).squeeze(-1)

    def forward(self, q_tokens: torch.Tensor, p_tokens: torch.Tensor, node: torch.Tensor, edge_features: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        objects = q_tokens.shape[-2]
        valid_pair = object_mask.unsqueeze(-1) & object_mask.unsqueeze(-2)
        active_relation = edge_features[..., 0] > 0.5
        if self.mode == 'general_hamiltonian':
            endpoint = torch.cat((q_tokens, p_tokens, node), dim=-1)
            left = endpoint.unsqueeze(-2).expand(*endpoint.shape[:-2], objects, objects, 13)
            right = endpoint.unsqueeze(-3).expand(*endpoint.shape[:-2], objects, objects, 13)
            forward = torch.cat((left, right, edge_features), dim=-1)
            reverse = torch.cat((right, left, edge_features.transpose(-3, -2)), dim=-1)
        else:
            q_left = q_tokens.unsqueeze(-2).expand(*q_tokens.shape[:-2], objects, objects, 3)
            q_right = q_tokens.unsqueeze(-3).expand(*q_tokens.shape[:-2], objects, objects, 3)
            intrinsic = node[..., :3]
            left = intrinsic.unsqueeze(-2).expand(*intrinsic.shape[:-2], objects, objects, 3)
            right = intrinsic.unsqueeze(-3).expand(*intrinsic.shape[:-2], objects, objects, 3)
            relative_q = q_left - q_right
            if self.mode in {'radial_configuration_potential', 'distance_configuration_potential'}:
                squared_distance = relative_q.square().sum(dim=-1, keepdim=True)
                if self.mode == 'distance_configuration_potential':
                    distance_needed = valid_pair & active_relation
                    radial = torch.sqrt(squared_distance + (~distance_needed).unsqueeze(-1).to(squared_distance))
                else:
                    radial = squared_distance
                forward = torch.cat((radial, left, right, edge_features), dim=-1)
                reverse = torch.cat((radial, right, left, edge_features.transpose(-3, -2)), dim=-1)
            else:
                forward = torch.cat((relative_q, left, right, edge_features), dim=-1)
                reverse = torch.cat((-relative_q, right, left, edge_features.transpose(-3, -2)), dim=-1)
        pair_energy = 0.5 * (self._ordered_energy(forward) + self._ordered_energy(reverse))
        selected = valid_pair & active_relation & self.upper_triangle.to(object_mask.device)
        return (pair_energy * selected.to(pair_energy)).sum(dim=(-2, -1))

class _SeparableKineticEnergy(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError('kinetic hidden size must be positive')
        self.input_projection = nn.Linear(6, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.0001)

    def forward(self, p_tokens: torch.Tensor, intrinsic_attrs: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        value = torch.cat((p_tokens, intrinsic_attrs), dim=-1)
        value = torch.nn.functional.silu(self.input_projection(value))
        value = torch.nn.functional.silu(self.hidden_projection(value))
        energy = self.output_projection(value).squeeze(-1)
        return (energy * object_mask.to(energy)).sum(dim=-1)

class HamiBalls2ContinuousHamiltonian(nn.Module):
    architecture = 'hamiballs2_masked_graph_continuous_hamiltonian_v1'

    def __init__(self, *, hidden_size: int=36, depth: int=2, heads: int=4, expansion: float=2.0, pair_energy_hidden_size: int | None=None, pair_energy_mode: str='general_hamiltonian', explicit_relations_exclusive_to_pair_energy: bool=False, global_hamiltonian_mode: str='general', kinetic_hidden_size: int | None=None, q_scale: tuple[float, float, float], p_scale: tuple[float, float, float], num_objects: int=10) -> None:
        super().__init__()
        if hidden_size % heads or min(hidden_size, depth, heads, num_objects) < 1:
            raise ValueError('invalid HamiBalls-2 Hamiltonian dimensions')
        self.num_objects = int(num_objects)
        self.coordinate_dim = 3
        self.state_dim = 3 * self.num_objects
        self.context_dim = 10
        self.heads = int(heads)
        if global_hamiltonian_mode not in {'general', 'separable'}:
            raise ValueError(f'unsupported global Hamiltonian mode: {global_hamiltonian_mode}')
        if global_hamiltonian_mode == 'general' and kinetic_hidden_size is not None:
            raise ValueError('kinetic hidden size is only valid for separable Hamiltonians')
        self.global_hamiltonian_mode = global_hamiltonian_mode
        if explicit_relations_exclusive_to_pair_energy and pair_energy_hidden_size is None:
            raise ValueError('exclusive explicit relations require a pair-energy branch')
        self.explicit_relations_exclusive_to_pair_energy = bool(explicit_relations_exclusive_to_pair_energy)
        self.register_buffer('q_scale', torch.tensor(q_scale, dtype=torch.float32))
        self.register_buffer('p_scale', torch.tensor(p_scale, dtype=torch.float32))
        global_input_size = 10 if global_hamiltonian_mode == 'separable' else 13
        self.input_projection = nn.Linear(global_input_size, hidden_size)
        self.edge_bias = nn.Linear(3, heads, bias=False)
        nn.init.zeros_(self.edge_bias.weight)
        self.blocks = nn.ModuleList((_HamiltonianBlock(hidden_size, heads, expansion) for _ in range(depth)))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.0001)
        self.pair_energy = None if pair_energy_hidden_size is None else _SymmetricPairEnergy(int(pair_energy_hidden_size), self.num_objects, pair_energy_mode)
        self.kinetic_energy = _SeparableKineticEnergy(int(kinetic_hidden_size or hidden_size)) if global_hamiltonian_mode == 'separable' else None
        self.architecture = 'hamiballs2_separable_global_plus_distance_dynamic_pair_potential_v7' if self.global_hamiltonian_mode == 'separable' and self.explicit_relations_exclusive_to_pair_energy and (pair_energy_mode == 'distance_configuration_potential') else 'hamiballs2_separable_global_plus_radial_dynamic_pair_potential_v6' if self.global_hamiltonian_mode == 'separable' and self.explicit_relations_exclusive_to_pair_energy and (pair_energy_mode == 'radial_configuration_potential') else 'hamiballs2_separable_global_potential_plus_dynamic_pair_potential_v5' if self.global_hamiltonian_mode == 'separable' and self.explicit_relations_exclusive_to_pair_energy and (pair_energy_mode == 'configuration_potential') else 'hamiballs2_orthogonal_global_h_plus_symmetric_dynamic_pair_potential_v4' if self.explicit_relations_exclusive_to_pair_energy and pair_energy_mode == 'configuration_potential' else 'hamiballs2_orthogonal_global_attention_plus_symmetric_dynamic_pair_energy_v3' if self.explicit_relations_exclusive_to_pair_energy else 'hamiballs2_global_attention_plus_symmetric_dynamic_pair_energy_v2' if self.pair_energy is not None else 'hamiballs2_masked_graph_continuous_hamiltonian_v1'

    def forward(self, q: torch.Tensor, p: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self._validate(q, p, context)
        shape = (*q.shape[:-1], self.num_objects, self.coordinate_dim)
        q_tokens = q.reshape(shape) / self.q_scale.to(q)
        p_tokens = p.reshape(shape) / self.p_scale.to(p)
        node = context[..., :, 0, :7]
        object_mask = node[..., 3] > 0.5
        edge_features = context[..., :, :, 7:10]
        if self.explicit_relations_exclusive_to_pair_energy:
            global_node = torch.cat((node[..., :4], torch.zeros_like(node[..., 4:])), dim=-1)
            pair_bias = torch.zeros_like(self.edge_bias(edge_features)).movedim(-1, -3)
        else:
            global_node = node
            pair_bias = self.edge_bias(edge_features).movedim(-1, -3)
        global_input = torch.cat((q_tokens, global_node), dim=-1) if self.global_hamiltonian_mode == 'separable' else torch.cat((q_tokens, p_tokens, global_node), dim=-1)
        value = self.input_projection(global_input)
        value = value * object_mask[..., None].to(value)
        for block in self.blocks:
            value = block(value, object_mask, pair_bias)
        energy = self.output_projection(self.output_norm(value)).squeeze(-1)
        total = (energy * object_mask.to(energy)).sum(dim=-1)
        if self.kinetic_energy is not None:
            total = total + self.kinetic_energy(p_tokens, global_node[..., :3], object_mask)
        if self.pair_energy is not None:
            total = total + self.pair_energy(q_tokens, p_tokens, global_node, edge_features, object_mask)
        return total

    def _validate(self, q: torch.Tensor, p: torch.Tensor, context: torch.Tensor) -> None:
        if q.shape != p.shape or q.shape[-1] != self.state_dim:
            raise ValueError('Hamiltonian q/p must end in the flattened 30D canonical state')
        if context.shape != (*q.shape[:-1], self.num_objects, self.num_objects, 10):
            raise ValueError('Hamiltonian context has the wrong graph tensor shape')
        if not bool(torch.isfinite(q).all() and torch.isfinite(p).all() and torch.isfinite(context).all()):
            raise ValueError('Hamiltonian inputs must be finite')

def graph_context(attrs: torch.Tensor, object_mask: torch.Tensor, spring_mask: torch.Tensor, spring_k: torch.Tensor, spring_rest_length: torch.Tensor) -> torch.Tensor:
    mask = object_mask.to(dtype=attrs.dtype)
    edge = spring_mask.to(dtype=attrs.dtype)
    log_k = torch.log1p(spring_k) * edge
    rest = spring_rest_length * edge
    degree = edge.sum(dim=-1)
    denominator = degree.clamp_min(1.0)
    node_graph = torch.stack((degree / max(object_mask.shape[-1] - 1, 1), log_k.sum(dim=-1) / denominator, rest.sum(dim=-1) / denominator), dim=-1)
    node = torch.cat((attrs, mask.unsqueeze(-1), node_graph), dim=-1)
    node = node * mask.unsqueeze(-1)
    node_pair = node.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], node.shape[-2], 7)
    pair = torch.stack((edge, log_k, rest), dim=-1)
    return torch.cat((node_pair, pair), dim=-1)

def parameter_count(module: nn.Module) -> int:
    return sum((parameter.numel() for parameter in module.parameters()))
__all__ = ['HamiBalls2ContinuousHamiltonian', 'graph_context', 'parameter_count']
