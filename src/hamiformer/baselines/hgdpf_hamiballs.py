from __future__ import annotations
from dataclasses import dataclass
import torch
from .hgdpf import HGDPFConfig, HGDPFPerceiver, SeparableHamiltonianNetwork
HAMIBALLS_OBJECTS = 5
HAMIBALLS_Q_DIM_PER_OBJECT = 2
HAMIBALLS_STATE_DIM = 20
HAMIBALLS_CANONICAL_Q_DIM = 10
HAMIBALLS_ATTRIBUTE_DIM = 15
HAMIBALLS_ACTION_DIM = 10

def hamiballs_hgdpf_config(parameter_mode: str='paper') -> HGDPFConfig:
    if parameter_mode not in {'paper', 'matched_wide_d'}:
        raise ValueError(f'unknown HG-DPF parameter mode: {parameter_mode}')
    matched = parameter_mode == 'matched_wide_d'
    return HGDPFConfig(state_dim=HAMIBALLS_STATE_DIM, action_dim=HAMIBALLS_ACTION_DIM, attribute_dim=HAMIBALLS_ATTRIBUTE_DIM, latent_count=64 if matched else 256, hidden_dim=64 if matched else 256, encoder_self_blocks=8, decoder_self_blocks=4, num_heads=2 if matched else 8, head_dim=32, feedforward_expansion=4, diffusion_fourier_bands=64, time_sinusoidal_bands=32, adaln_dim=64 if matched else 256, dropout=0.0, ffn_activation='gelu')

def make_hamiballs_hgdpf_models(parameter_mode: str='paper') -> tuple[HGDPFPerceiver, SeparableHamiltonianNetwork]:
    hnn_width = 248 if parameter_mode == 'matched_wide_d' else 1024
    return (HGDPFPerceiver(hamiballs_hgdpf_config(parameter_mode)), SeparableHamiltonianNetwork(HAMIBALLS_CANONICAL_Q_DIM, attribute_dim=HAMIBALLS_ATTRIBUTE_DIM, width=hnn_width, hidden_layers=4))

@dataclass(frozen=True)
class HGDPFTrainingBatch:
    diffusion_steps: torch.Tensor
    context_states: torch.Tensor
    context_times: torch.Tensor
    query_states: torch.Tensor
    query_times: torch.Tensor
    actions: torch.Tensor
    attributes: torch.Tensor
    target_noise: torch.Tensor

def pack_hamiballs_phase(phase: torch.Tensor, *, q_dim: int=2) -> torch.Tensor:
    if phase.ndim < 2 or phase.shape[-1] != 2 * q_dim:
        raise ValueError('phase must end in [objects,2*q_dim]')
    p = phase[..., q_dim:].flatten(start_dim=-2)
    q = phase[..., :q_dim].flatten(start_dim=-2)
    return torch.cat([p, q], dim=-1)

def unpack_hamiballs_phase(canonical: torch.Tensor, *, num_objects: int=5, q_dim: int=2) -> torch.Tensor:
    canonical_dim = num_objects * q_dim
    if canonical.shape[-1] != 2 * canonical_dim:
        raise ValueError('canonical state has the wrong final dimension')
    p = canonical[..., :canonical_dim].reshape(*canonical.shape[:-1], num_objects, q_dim)
    q = canonical[..., canonical_dim:].reshape(*canonical.shape[:-1], num_objects, q_dim)
    return torch.cat([q, p], dim=-1)

def flatten_hamiballs_attributes(attrs: torch.Tensor) -> torch.Tensor:
    if attrs.ndim != 3:
        raise ValueError('attrs must be [B,objects,attribute_dim]')
    return attrs.flatten(start_dim=1)

def scaled_query_cardinalities(source_edges: int, *, original_edges: int=1000, original_cardinalities: tuple[int, ...]=tuple(range(100, 1001, 100))) -> tuple[int, ...]:
    if source_edges < 2 or original_edges < 2:
        raise ValueError('trajectory horizons must be at least two')
    values = {max(2, min(source_edges, int(round(source_edges * n / original_edges)))) for n in original_cardinalities}
    return tuple(sorted(values))

def compute_state_normalization(canonical_phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if canonical_phase.ndim != 3:
        raise ValueError('canonical_phase must be [sources,time,state]')
    mean = canonical_phase.mean(dim=(0, 1))
    scale = canonical_phase.std(dim=(0, 1), unbiased=False)
    floor = torch.finfo(canonical_phase.dtype).eps
    if bool((scale <= floor).any()) or not bool(torch.isfinite(scale).all()):
        raise ValueError('degenerate state normalization')
    return (mean, scale)

def compute_attribute_normalization(attributes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if attributes.ndim != 2:
        raise ValueError('attributes must be [sources,attribute_dim]')
    mean = attributes.mean(dim=0)
    scale = attributes.std(dim=0, unbiased=False)
    floor = torch.finfo(attributes.dtype).eps
    if bool((scale <= floor).any()) or not bool(torch.isfinite(scale).all()):
        raise ValueError('degenerate attribute normalization')
    return (mean, scale)

def make_dpf_training_batch(*, model: HGDPFPerceiver, phase: torch.Tensor, times: torch.Tensor, attributes: torch.Tensor, state_mean: torch.Tensor, state_scale: torch.Tensor, attribute_mean: torch.Tensor, attribute_scale: torch.Tensor, alpha_bar: torch.Tensor, query_points: int, generator: torch.Generator) -> HGDPFTrainingBatch:
    if phase.ndim != 3 or times.shape != phase.shape[:2]:
        raise ValueError('phase/times must align as [B,T,state] and [B,T]')
    batch, frames, state_dim = phase.shape
    if state_dim != model.config.state_dim or not 1 <= query_points < frames:
        raise ValueError('invalid state dimension or query cardinality')
    if attributes.shape != (batch, model.config.attribute_dim):
        raise ValueError('attributes do not match the model')
    if state_mean.shape != (state_dim,) or state_scale.shape != (state_dim,):
        raise ValueError('state normalization shape mismatch')
    normalized = (phase - state_mean.to(phase)) / state_scale.to(phase)
    normalized_attrs = (attributes - attribute_mean.to(attributes)) / attribute_scale.to(attributes)
    clean_initial = normalized[:, 0]
    clean_query = normalized[:, 1:query_points + 1]
    query_times = times[:, 1:query_points + 1]
    diffusion_steps = torch.randint(0, alpha_bar.numel(), (batch,), device=phase.device, generator=generator)
    alpha = alpha_bar.to(phase)[diffusion_steps].reshape(batch, 1, 1)
    query_noise = torch.randn(clean_query.shape, device=phase.device, dtype=phase.dtype, generator=generator)
    noisy_query = alpha.sqrt() * clean_query + (1.0 - alpha).sqrt() * query_noise
    context_points = int(torch.randint(1, query_points, (), device=phase.device, generator=generator).item())
    clean_context = clean_query[:, :context_points]
    context_noise = torch.randn(clean_context.shape, device=phase.device, dtype=phase.dtype, generator=generator)
    noisy_context = alpha.sqrt() * clean_context + (1.0 - alpha).sqrt() * context_noise
    context_states = torch.cat([clean_initial[:, None], noisy_context], dim=1)
    context_times = torch.cat([times[:, :1], query_times[:, :context_points]], dim=1)
    action_dim = model.config.action_dim
    actions = torch.zeros(batch, query_points, action_dim, device=phase.device, dtype=phase.dtype)
    return HGDPFTrainingBatch(diffusion_steps=diffusion_steps, context_states=context_states, context_times=context_times, query_states=noisy_query, query_times=query_times, actions=actions, attributes=normalized_attrs, target_noise=query_noise)

def dpf_noise_prediction_loss(model: HGDPFPerceiver, batch: HGDPFTrainingBatch) -> torch.Tensor:
    prediction = model(diffusion_steps=batch.diffusion_steps, context_states=batch.context_states, context_times=batch.context_times, query_states=batch.query_states, query_times=batch.query_times, actions=batch.actions, attributes=batch.attributes)
    if prediction.shape != batch.target_noise.shape:
        raise RuntimeError('DPF prediction and target noise shapes differ')
    return (prediction - batch.target_noise).square().mean()

def published_parameter_ledger(dpf: HGDPFPerceiver, hnn: torch.nn.Module, *, wide_d_parameters: int) -> dict[str, float | int]:
    dpf_parameters = sum((parameter.numel() for parameter in dpf.parameters()))
    hnn_parameters = sum((parameter.numel() for parameter in hnn.parameters()))
    total = dpf_parameters + hnn_parameters
    if wide_d_parameters <= 0:
        raise ValueError('wide_d_parameters must be positive')
    return {'dpf_parameters': dpf_parameters, 'hnn_parameters': hnn_parameters, 'guided_total_parameters': total, 'wide_d_parameters': int(wide_d_parameters), 'unguided_dpf_over_wide_d': dpf_parameters / wide_d_parameters, 'guided_hgdpf_over_wide_d': total / wide_d_parameters}
__all__ = ['HAMIBALLS_ACTION_DIM', 'HAMIBALLS_ATTRIBUTE_DIM', 'HAMIBALLS_CANONICAL_Q_DIM', 'HAMIBALLS_OBJECTS', 'HAMIBALLS_Q_DIM_PER_OBJECT', 'HAMIBALLS_STATE_DIM', 'HGDPFTrainingBatch', 'compute_attribute_normalization', 'compute_state_normalization', 'dpf_noise_prediction_loss', 'flatten_hamiballs_attributes', 'hamiballs_hgdpf_config', 'make_hamiballs_hgdpf_models', 'make_dpf_training_batch', 'pack_hamiballs_phase', 'published_parameter_ledger', 'scaled_query_cardinalities', 'unpack_hamiballs_phase']
