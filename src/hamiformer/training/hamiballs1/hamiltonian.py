from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
from typing import Any
import torch
ROOT = project_root()
from hamiformer.physics.continuous_hamiltonian import TokenConditionalContinuousHamiltonian

def _build_continuous_h(config: dict[str, Any], state_scale: torch.Tensor, *, device: torch.device) -> TokenConditionalContinuousHamiltonian:
    dataset, hcfg = (config['dataset'], config['hamiltonian'])
    q_dim = int(dataset['q_dim'])
    return TokenConditionalContinuousHamiltonian(num_objects=int(dataset['num_objects']), spatial_tokens=1, coordinate_dim=q_dim, token_context_dim=int(dataset['attr_dim']), hidden_size=int(hcfg['hidden_size']), depth=int(hcfg['depth']), heads=int(hcfg['heads']), expansion=float(hcfg['expansion']), q_scale=tuple((float(value) for value in state_scale[:q_dim])), p_scale=tuple((float(value) for value in state_scale[q_dim:])), spatial_attention_mode=str(hcfg['spatial_attention_mode']), object_attention_mode=str(hcfg['object_attention_mode'])).to(device=device, dtype=torch.float32)
