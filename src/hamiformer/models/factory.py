from __future__ import annotations
from dataclasses import dataclass
from hamiformer.config import ExperimentConfig
from .direct_vector_expert import DirectVectorExpert
from .hamiltonian_expert import HamiltonianExpert
from .phase_dit import PhaseDiT

@dataclass
class ModelBundle:
    d: PhaseDiT
    h: HamiltonianExpert
    s: DirectVectorExpert | None

def build_models(config: ExperimentConfig) -> ModelBundle:
    data = config.data
    d_cfg = config.phase_d
    h_cfg = config.hamiltonian_h
    s_cfg = config.matched_s
    d_model = PhaseDiT(state_dim=data.state_dim, q_dim=data.q_dim, attr_dim=data.attr_dim, hidden_size=d_cfg.hidden_size, depth=d_cfg.depth, num_heads=d_cfg.num_heads, mlp_ratio=d_cfg.mlp_ratio, num_register_tokens=d_cfg.num_register_tokens, dropout=d_cfg.dropout, qk_norm=d_cfg.qk_norm)
    h_model = HamiltonianExpert(q_dim=data.q_dim, state_dim=data.state_dim, attr_dim=data.attr_dim, hidden_size=h_cfg.hidden_size, depth=h_cfg.depth, num_heads=h_cfg.num_heads, mlp_ratio=h_cfg.mlp_ratio, dropout=h_cfg.dropout, force_float32=h_cfg.force_float32)
    s_model = None
    if s_cfg.enabled:
        s_model = DirectVectorExpert(q_dim=data.q_dim, state_dim=data.state_dim, attr_dim=data.attr_dim, hidden_size=s_cfg.hidden_size, depth=s_cfg.depth, num_heads=s_cfg.num_heads, mlp_ratio=s_cfg.mlp_ratio, dropout=s_cfg.dropout)
    return ModelBundle(d=d_model, h=h_model, s=s_model)
