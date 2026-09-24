from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hamiformer.models.hamiballs_committed import HamiBallsPerObjectCompactCommittedGate
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable


def _freeze(module: nn.Module, *, name: str) -> str:
    set_trainable(module, False)
    module.eval()
    digest = module_digest(module)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError(f'frozen {name} retained a trainable parameter')
    return digest


def _build_gate(
    config: dict[str, Any],
    registration: dict[str, Any],
    *,
    device: torch.device,
) -> HamiBallsPerObjectCompactCommittedGate:
    seed = int(config['seed']) + 251
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observables = registration['gate'].get('observables')
    model = HamiBallsPerObjectCompactCommittedGate(
        token_dim=int(config['model']['hidden_size']),
        state_dim=int(config['model']['state_dim']),
        attr_dim=int(config['dataset']['attr_dim']),
        residual_hidden_dim=int(config['residual']['hidden_size']),
        rank=int(registration['gate']['rank']),
        candidate_step_observables=observables == 'candidate_absolute_steps_and_qp_norms_v1',
        function_preserving_candidate_step_observables=(
            observables == 'candidate_absolute_steps_and_qp_norms_function_preserving_v2'
        ),
        function_preserving_pairwise_relations=(
            observables == 'learned_pairwise_relations_function_preserving_v1'
        ),
    ).to(device=device, dtype=torch.float32)
    model.temporal.flatten_parameters()
    if torch.count_nonzero(model.output.weight).item() or torch.count_nonzero(model.output.bias).item():
        raise AssertionError('fresh gate is not a neutral zero-logit gate')
    if not torch.equal(model.log_temperature.detach(), torch.zeros_like(model.log_temperature.detach())):
        raise AssertionError('fresh gate temperature is not neutral')
    return model
