from __future__ import annotations
from hamiformer.utils.paths import project_root
import math
import torch
SCHEMA = 'hamiformer.hamiballs.scalar_gate.fresh_metric_readout_gate.v1'
ROLE = 'scalar_gate-fresh-standard-readout-single-metric-objective'
READOUT_SEED = 4353

def initialise_standard_readout(state: dict[str, torch.Tensor], *, seed: int=READOUT_SEED) -> dict[str, torch.Tensor]:
    result = {name: value.detach().clone() for name, value in state.items()}
    weight = result.get('output.weight')
    bias = result.get('output.bias')
    temperature = result.get('log_temperature')
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != 1 or (not isinstance(bias, torch.Tensor)) or (bias.shape != (1,)) or (not isinstance(temperature, torch.Tensor)) or (temperature.numel() != 1):
        raise ValueError('neutral gate readout schema drifted')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    bound = 1.0 / math.sqrt(float(weight.shape[1]))
    weight.uniform_(-bound, bound, generator=generator)
    bias.zero_()
    temperature.zero_()
    if torch.count_nonzero(weight).item() == 0:
        raise AssertionError('standard readout unexpectedly remained zero')
    return result
