from __future__ import annotations
import torch
from torch import nn

def hamiballs2_node_graph_features(object_mask: torch.Tensor, spring_mask: torch.Tensor, spring_k: torch.Tensor, spring_rest_length: torch.Tensor) -> torch.Tensor:
    mask = object_mask.to(spring_k)
    edge = spring_mask.to(spring_k)
    degree = edge.sum(dim=-1)
    denominator = degree.clamp_min(1.0)
    result = torch.stack((degree / max(object_mask.shape[-1] - 1, 1), (torch.log1p(spring_k) * edge).sum(dim=-1) / denominator, (spring_rest_length * edge).sum(dim=-1) / denominator), dim=-1)
    return result * mask[..., None]

class _ObjectTemporalCore(nn.Module):

    def __init__(self, feature_dim: int, width: int, output_dim: int, tree_leaves: int) -> None:
        super().__init__()
        self.width = int(width)
        self.input = nn.Linear(feature_dim, width)
        self.global_context = nn.Linear(2 * width, width)
        self.tree_adapter = nn.Linear(tree_leaves, width, bias=False) if tree_leaves else None
        if self.tree_adapter is not None:
            nn.init.zeros_(self.tree_adapter.weight)
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def step(self, feature: torch.Tensor, mask: torch.Tensor, hidden: torch.Tensor | None, leaf: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, objects = feature.shape[:2]
        local = torch.nn.functional.silu(self.input(feature))
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(local)
        mean = (local * mask[..., None].to(local)).sum(dim=1) / denominator
        masked = local.masked_fill(~mask[..., None], torch.finfo(local.dtype).min)
        maximum = masked.amax(dim=1)
        encoded = local + self.global_context(torch.cat((mean, maximum), dim=-1))[:, None]
        if self.tree_adapter is not None:
            if leaf is None:
                raise ValueError('Tree-conditioned core requires leaf one-hot inputs')
            encoded = encoded + self.tree_adapter(leaf)
        encoded = torch.nn.functional.silu(encoded) * mask[..., None].to(encoded)
        flat_hidden = None if hidden is None else hidden.reshape(1, batch * objects, self.width)
        temporal, next_flat = self.temporal(encoded.reshape(batch * objects, 1, self.width), flat_hidden)
        temporal = temporal[:, 0].reshape(batch, objects, self.width)
        next_hidden = next_flat.reshape(1, batch, objects, self.width)
        return (self.output(temporal), temporal, next_hidden)

class HamiBalls2TemporalDualExpert(nn.Module):
    architecture = 'hamiballs2_temporal_dual_expert_v1'

    def __init__(self, *, phase_scale: tuple[float, ...], residual_width: int=64, gate_width: int=48, tree_leaves: int=0) -> None:
        super().__init__()
        if len(phase_scale) != 6 or min(phase_scale) <= 0:
            raise ValueError('phase_scale must contain six positive entries')
        self.register_buffer('phase_scale', torch.tensor(phase_scale, dtype=torch.float32))
        self.tree_leaves = int(tree_leaves)
        residual_features = 7 * 6 + 6 + 2
        self.residual = _ObjectTemporalCore(residual_features, residual_width, 6, self.tree_leaves)
        gate_features = 7 * 6 + 6 + 2 + 2 + residual_width
        self.gate = _ObjectTemporalCore(gate_features, gate_width, 2, self.tree_leaves)

    def forward(self, state: torch.Tensor, d_candidate: torch.Tensor, h_candidate: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, node_graph: torch.Tensor, physical_time: torch.Tensor, tau: torch.Tensor, object_mask: torch.Tensor, leaf_one_hot: torch.Tensor | None=None, jet_matrix: torch.Tensor | None=None, jet_offset: torch.Tensor | None=None, force_gate: float | None=None, component_gate_scale: torch.Tensor | None=None, per_edge_gate_scale: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        if state.shape != d_candidate.shape or state.shape != h_candidate.shape:
            raise ValueError('state and expert candidates must align')
        batch, edges, objects, channels = state.shape
        if channels != 6 or x0.shape != (batch, objects, 6):
            raise ValueError('HamiBalls-2 phase shapes are invalid')
        if attrs.shape != (batch, objects, 3) or node_graph.shape != (batch, objects, 3):
            raise ValueError('static per-object context has invalid shape')
        if physical_time.shape != (batch, edges) or tau.shape != (batch,):
            raise ValueError('time/tau shapes are invalid')
        if object_mask.shape != (batch, objects):
            raise ValueError('object mask shape is invalid')
        if self.tree_leaves:
            if leaf_one_hot is None or leaf_one_hot.shape != (batch, edges, objects, self.tree_leaves):
                raise ValueError('leaf one-hot shape is invalid')
        if (jet_matrix is None) != (jet_offset is None):
            raise ValueError('jet matrix and offset must be supplied together')
        if jet_matrix is not None:
            if jet_matrix.shape != (batch, edges, 60, 60) or jet_offset.shape != (batch, edges, 60):
                raise ValueError('PASS jet shapes are invalid')
        if per_edge_gate_scale is not None and per_edge_gate_scale.shape != (batch, edges, objects, 2):
            raise ValueError('per-edge gate scale must be [B,E,K,2]')
        scale = self.phase_scale.to(state)
        static = torch.cat((attrs, node_graph), dim=-1)
        previous_hr = x0
        previous_mixed = x0
        previous_gate = state.new_ones(batch, objects, 2)
        residual_hidden = None
        gate_hidden = None
        hr_values, mixed_values, residual_values, gate_values, hidden_values = ([], [], [], [], [])
        for edge in range(edges):
            leaf = None if leaf_one_hot is None else leaf_one_hot[:, edge]
            s = state[:, edge] / scale
            d = d_candidate[:, edge] / scale
            if jet_matrix is None:
                h_value = h_candidate[:, edge]
            else:
                canonical_previous = torch.cat((previous_mixed[..., :3].reshape(batch, 30), previous_mixed[..., 3:].reshape(batch, 30)), dim=-1)
                canonical_h = (jet_matrix[:, edge] @ canonical_previous.unsqueeze(-1)).squeeze(-1) + jet_offset[:, edge]
                h_value = torch.cat((canonical_h[..., :30].reshape(batch, objects, 3), canonical_h[..., 30:].reshape(batch, objects, 3)), dim=-1)
            h = h_value / scale
            phr = previous_hr / scale
            relative_time = (physical_time[:, edge] - physical_time[:, 0]) / (physical_time[:, -1] - physical_time[:, 0]).clamp_min(1e-06)
            scalar = torch.stack((tau, relative_time), dim=-1)[:, None].expand(-1, objects, -1)
            r_feature = torch.cat((s, d, h, phr, d - h, d - phr, h - phr, static, scalar), dim=-1)
            residual_normalized, residual_state, residual_hidden = self.residual.step(r_feature, object_mask, residual_hidden, leaf)
            residual_value = residual_normalized * scale
            hr = h_value + residual_value
            pm = previous_mixed / scale
            hrn = hr / scale
            disagreement = hrn - d
            gate_feature = torch.cat((s, d, h, hrn, pm, disagreement, disagreement.abs(), static, scalar, previous_gate, residual_state), dim=-1)
            logits, _, gate_hidden = self.gate.step(gate_feature, object_mask, gate_hidden, leaf)
            gate = torch.sigmoid(logits)
            if force_gate is not None:
                if not 0.0 <= force_gate <= 1.0:
                    raise ValueError('forced gate must lie in [0,1]')
                gate = torch.full_like(gate, float(force_gate))
            if component_gate_scale is not None:
                if component_gate_scale.shape != (2,) or not bool(((component_gate_scale >= 0.0) & (component_gate_scale <= 1.0)).all()):
                    raise ValueError('component gate scale must be a length-two value in [0,1]')
                gate = gate * component_gate_scale.to(gate)
            if per_edge_gate_scale is not None:
                local_scale = per_edge_gate_scale[:, edge]
                if not bool(((local_scale >= 0.0) & (local_scale <= 1.0)).all()):
                    raise ValueError('per-edge gate scale must lie in [0,1]')
                gate = gate * local_scale.to(gate)
            gate6 = torch.cat((gate[..., :1].expand(-1, -1, 3), gate[..., 1:].expand(-1, -1, 3)), dim=-1)
            mixed = gate6 * hr + (1.0 - gate6) * d_candidate[:, edge]
            mask = object_mask[..., None].to(mixed)
            hr, mixed, residual_value = (hr * mask, mixed * mask, residual_value * mask)
            previous_hr, previous_mixed, previous_gate = (hr, mixed, gate)
            hr_values.append(hr)
            mixed_values.append(mixed)
            residual_values.append(residual_value)
            gate_values.append(gate)
            hidden_values.append(residual_state)
        return {'hr_candidate': torch.stack(hr_values, dim=1), 'mixed': torch.stack(mixed_values, dim=1), 'residual': torch.stack(residual_values, dim=1), 'gate': torch.stack(gate_values, dim=1), 'residual_hidden': torch.stack(hidden_values, dim=1)}

def parameter_count(module: nn.Module) -> int:
    return sum((parameter.numel() for parameter in module.parameters()))
__all__ = ['HamiBalls2TemporalDualExpert', 'hamiballs2_node_graph_features', 'parameter_count']
