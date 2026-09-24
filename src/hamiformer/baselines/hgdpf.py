from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal
import torch
from torch import nn
GuidanceMode = Literal['none', 'gradient', 'snis']

@dataclass(frozen=True)
class HGDPFConfig:
    state_dim: int
    action_dim: int
    attribute_dim: int = 0
    latent_count: int = 256
    hidden_dim: int = 256
    encoder_self_blocks: int = 8
    decoder_self_blocks: int = 4
    num_heads: int = 8
    head_dim: int = 32
    feedforward_expansion: int = 4
    diffusion_fourier_bands: int = 64
    time_sinusoidal_bands: int = 32
    adaln_dim: int = 256
    dropout: float = 0.0
    ffn_activation: Literal['gelu', 'celu'] = 'gelu'

    def __post_init__(self) -> None:
        positive = {'state_dim': self.state_dim, 'latent_count': self.latent_count, 'hidden_dim': self.hidden_dim, 'encoder_self_blocks': self.encoder_self_blocks, 'decoder_self_blocks': self.decoder_self_blocks, 'num_heads': self.num_heads, 'head_dim': self.head_dim, 'feedforward_expansion': self.feedforward_expansion, 'diffusion_fourier_bands': self.diffusion_fourier_bands, 'time_sinusoidal_bands': self.time_sinusoidal_bands, 'adaln_dim': self.adaln_dim}
        if any((int(value) <= 0 for value in positive.values())):
            raise ValueError(f'positive HG-DPF dimensions required: {positive}')
        if self.action_dim < 0 or self.attribute_dim < 0:
            raise ValueError('action_dim and attribute_dim must be non-negative')
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError('hidden_dim must equal num_heads * head_dim')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must lie in [0,1)')

def cosine_alpha_bar(diffusion_steps: int, *, schedule_offset: float=0.008, dtype: torch.dtype=torch.float64) -> torch.Tensor:
    if diffusion_steps < 2:
        raise ValueError('diffusion_steps must be at least two')
    if not 0.0 < schedule_offset < 1.0:
        raise ValueError('schedule_offset must lie in (0,1)')
    points = torch.linspace(0.0, 1.0, diffusion_steps + 1, dtype=dtype)
    values = torch.cos((points + schedule_offset) / (1.0 + schedule_offset) * math.pi / 2.0).square()
    values = values / values[0]
    betas = (1.0 - values[1:] / values[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0)

def ddim_leading_timesteps(diffusion_steps: int, inference_steps: int) -> torch.Tensor:
    if not 1 <= inference_steps <= diffusion_steps:
        raise ValueError('invalid DDIM inference step count')
    ratio = diffusion_steps // inference_steps
    return (torch.arange(inference_steps, dtype=torch.long) * ratio).flip(0)

def _sinusoidal_features(values: torch.Tensor, bands: int) -> torch.Tensor:
    if values.ndim < 1:
        raise ValueError('values must have at least one dimension')
    frequencies = torch.exp(torch.linspace(0.0, math.log(10000.0), bands, device=values.device, dtype=values.dtype))
    angles = values.unsqueeze(-1) / frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

def _activation(name: str) -> nn.Module:
    if name == 'gelu':
        return nn.GELU()
    if name == 'celu':
        return nn.CELU()
    raise ValueError(f'unsupported activation: {name}')

class _FeedForward(nn.Module):

    def __init__(self, width: int, expansion: int, dropout: float, activation: str) -> None:
        super().__init__()
        inner = width * expansion
        self.net = nn.Sequential(nn.Linear(width, inner), _activation(activation), nn.Dropout(dropout), nn.Linear(inner, width), nn.Dropout(dropout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)

class _CrossAttentionBlock(nn.Module):

    def __init__(self, width: int, heads: int, expansion: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = _FeedForward(width, expansion, dropout, activation)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.query_norm(query), self.context_norm(context), self.context_norm(context), need_weights=False)
        query = query + attended
        return query + self.ff(self.ff_norm(query))

class _SelfAttentionBlock(nn.Module):

    def __init__(self, width: int, heads: int, expansion: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = _FeedForward(width, expansion, dropout, activation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + attended
        return value + self.ff(self.ff_norm(value))

class _AdaLNSelfAttentionBlock(nn.Module):

    def __init__(self, width: int, heads: int, expansion: int, dropout: float, activation: str, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.CELU(), nn.Linear(condition_dim, 2 * width))
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = _FeedForward(width, expansion, dropout, activation)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if value.shape != condition.shape:
            raise ValueError('AdaLN condition must align with query tokens')
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        modulated = self.norm(value) * (1.0 + gamma) + beta
        attended, _ = self.attention(modulated, modulated, modulated, need_weights=False)
        value = value + attended
        return value + self.ff(self.ff_norm(value))

class _TwoLayerEmbedding(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.CELU(), nn.Linear(output_dim, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)

class HGDPFPerceiver(nn.Module):

    def __init__(self, config: HGDPFConfig) -> None:
        super().__init__()
        self.config = config
        token_dim = config.state_dim + 2 * config.diffusion_fourier_bands + 2 * config.time_sinusoidal_bands
        self.token_projection = nn.Sequential(nn.Linear(token_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim))
        self.latents = nn.Parameter(torch.randn(config.latent_count, config.hidden_dim) / math.sqrt(config.hidden_dim))
        self.encoder_cross = _CrossAttentionBlock(config.hidden_dim, config.num_heads, config.feedforward_expansion, config.dropout, config.ffn_activation)
        self.encoder_self = nn.ModuleList([_SelfAttentionBlock(config.hidden_dim, config.num_heads, config.feedforward_expansion, config.dropout, config.ffn_activation) for _ in range(config.encoder_self_blocks)])
        self.decoder_cross = _CrossAttentionBlock(config.hidden_dim, config.num_heads, config.feedforward_expansion, config.dropout, config.ffn_activation)
        self.previous_state_embedding = _TwoLayerEmbedding(config.state_dim, config.adaln_dim)
        self.action_embedding = _TwoLayerEmbedding(config.action_dim, config.adaln_dim) if config.action_dim > 0 else None
        interaction_inputs = config.adaln_dim * (2 if config.action_dim > 0 else 1)
        self.interaction_embedding = _TwoLayerEmbedding(interaction_inputs, config.adaln_dim)
        self.first_condition = nn.Parameter(torch.zeros(config.adaln_dim))
        self.attribute_embedding = _TwoLayerEmbedding(config.attribute_dim, config.hidden_dim) if config.attribute_dim > 0 else None
        self.attribute_condition = _TwoLayerEmbedding(config.attribute_dim, config.adaln_dim) if config.attribute_dim > 0 else None
        if config.adaln_dim != config.hidden_dim:
            self.condition_projection = nn.Linear(config.adaln_dim, config.hidden_dim)
        else:
            self.condition_projection = nn.Identity()
        self.decoder_self = nn.ModuleList([_AdaLNSelfAttentionBlock(config.hidden_dim, config.num_heads, config.feedforward_expansion, config.dropout, config.ffn_activation, config.hidden_dim) for _ in range(config.decoder_self_blocks)])
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.state_dim))

    def _tokens(self, states: torch.Tensor, times: torch.Tensor, diffusion_steps: torch.Tensor, attributes: torch.Tensor | None) -> torch.Tensor:
        batch, points, state_dim = states.shape
        if state_dim != self.config.state_dim or times.shape != (batch, points):
            raise ValueError('state/time token dimensions do not align')
        if diffusion_steps.shape != (batch,):
            raise ValueError('diffusion_steps must be [B]')
        step_values = diffusion_steps.to(states.dtype)
        step_features = _sinusoidal_features(step_values, self.config.diffusion_fourier_bands)[:, None].expand(-1, points, -1)
        time_features = _sinusoidal_features(times.to(states), self.config.time_sinusoidal_bands)
        tokens = self.token_projection(torch.cat([states, step_features, time_features], dim=-1))
        if self.attribute_embedding is not None:
            if attributes is None or attributes.shape != (batch, self.config.attribute_dim):
                raise ValueError('flattened attributes must be [B,attribute_dim]')
            tokens = tokens + self.attribute_embedding(attributes.to(states))[:, None]
        elif attributes is not None and attributes.shape[-1] != 0:
            raise ValueError('attributes supplied to an unconditional HG-DPF')
        return tokens

    def _causal_condition(self, query_states: torch.Tensor, actions: torch.Tensor | None, attributes: torch.Tensor | None) -> torch.Tensor:
        batch, points, _ = query_states.shape
        state_features = self.previous_state_embedding(query_states)
        if self.action_embedding is not None:
            if actions is None or actions.shape != (batch, points, self.config.action_dim):
                raise ValueError('actions must align with query states')
            action_features = self.action_embedding(actions.to(query_states))
            fused = self.interaction_embedding(torch.cat([state_features[:, :-1], action_features[:, :-1]], dim=-1))
        else:
            if actions is not None and actions.shape[-1] != 0:
                raise ValueError('actions supplied to an action-free HG-DPF')
            fused = self.interaction_embedding(state_features[:, :-1])
        first = self.first_condition.to(query_states)[None, None].expand(batch, 1, -1)
        condition = torch.cat([first, fused], dim=1)
        if self.attribute_condition is not None:
            assert attributes is not None
            condition = condition + self.attribute_condition(attributes.to(query_states))[:, None]
        return self.condition_projection(condition)

    def forward(self, *, diffusion_steps: torch.Tensor, context_states: torch.Tensor, context_times: torch.Tensor, query_states: torch.Tensor, query_times: torch.Tensor, actions: torch.Tensor | None=None, attributes: torch.Tensor | None=None) -> torch.Tensor:
        if context_states.shape[0] != query_states.shape[0]:
            raise ValueError('context/query batch sizes differ')
        context = self._tokens(context_states, context_times, diffusion_steps, attributes)
        query = self._tokens(query_states, query_times, diffusion_steps, attributes)
        latents = self.latents.to(query)[None].expand(query.shape[0], -1, -1)
        latents = self.encoder_cross(latents, context)
        for block in self.encoder_self:
            latents = block(latents)
        query = self.decoder_cross(query, latents)
        condition = self._causal_condition(query_states, actions, attributes)
        for block in self.decoder_self:
            query = block(query, condition)
        return self.output(query)

class _HamiltonianMLP(nn.Module):

    def __init__(self, input_dim: int, width: int, layers: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(current, width), nn.CELU()])
            current = width
        modules.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)

class SeparableHamiltonianNetwork(nn.Module):

    def __init__(self, q_dim: int, *, attribute_dim: int=0, width: int=1024, hidden_layers: int=4) -> None:
        super().__init__()
        if min(q_dim, width, hidden_layers) <= 0 or attribute_dim < 0:
            raise ValueError('invalid HNN dimensions')
        self.q_dim = int(q_dim)
        self.attribute_dim = int(attribute_dim)
        self.kinetic = _HamiltonianMLP(2 * q_dim + attribute_dim, width, hidden_layers)
        self.potential = _HamiltonianMLP(q_dim + attribute_dim, width, hidden_layers)

    def forward(self, p: torch.Tensor, q: torch.Tensor, attributes: torch.Tensor | None=None) -> torch.Tensor:
        if p.shape != q.shape or p.shape[-1] != self.q_dim:
            raise ValueError('p and q must align on the canonical dimension')
        if self.attribute_dim:
            if attributes is None or attributes.shape != (*q.shape[:-1], self.attribute_dim):
                raise ValueError('attributes must align with p/q leading dimensions')
            kinetic_input = torch.cat([p, q, attributes.to(q)], dim=-1)
            potential_input = torch.cat([q, attributes.to(q)], dim=-1)
        else:
            if attributes is not None and attributes.shape[-1] != 0:
                raise ValueError('attributes supplied to an unconditional HNN')
            kinetic_input = torch.cat([p, q], dim=-1)
            potential_input = q
        return self.kinetic(kinetic_input) + self.potential(potential_input)

    def vector_field(self, p: torch.Tensor, q: torch.Tensor, attributes: torch.Tensor | None=None, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not p.requires_grad or not q.requires_grad:
            raise ValueError('p and q must require gradients for Hamiltonian differentiation')
        hamiltonian = self(p, q, attributes)
        d_h_dp, d_h_dq = torch.autograd.grad(hamiltonian.sum(), (p, q), create_graph=create_graph, retain_graph=create_graph)
        q_dot = d_h_dp
        p_dot_unforced = -d_h_dq
        return (q_dot, p_dot_unforced)

def _broadcast_attributes(attributes: torch.Tensor | None, *, batch: int, points: int, attribute_dim: int) -> torch.Tensor | None:
    if attribute_dim == 0:
        return None
    if attributes is None or attributes.shape != (batch, attribute_dim):
        raise ValueError('flattened attributes must be [B,attribute_dim]')
    return attributes[:, None].expand(-1, points, -1)

def hnn_derivative_loss(model: SeparableHamiltonianNetwork, *, p: torch.Tensor, q: torch.Tensor, p_dot_target: torch.Tensor, q_dot_target: torch.Tensor, actions: torch.Tensor | None=None, attributes: torch.Tensor | None=None) -> torch.Tensor:
    if p.shape != q.shape or p.shape != p_dot_target.shape or p.shape != q_dot_target.shape:
        raise ValueError('p/q and derivative targets must have identical shapes')
    if p.ndim != 2 or p.shape[-1] != model.q_dim:
        raise ValueError('differenced HNN samples must be [B,q_dim]')
    p_input = p.detach().requires_grad_(True)
    q_input = q.detach().requires_grad_(True)
    q_dot, p_dot_unforced = model.vector_field(p_input, q_input, attributes, create_graph=True)
    if actions is None:
        torque = torch.zeros_like(p_dot_unforced)
    else:
        if actions.shape != p.shape:
            raise ValueError('actions must align with the differenced phase points')
        torque = actions.to(p_dot_unforced)
    return ((q_dot - q_dot_target).square().sum(dim=-1) + (p_dot_unforced + torque - p_dot_target).square().sum(dim=-1)).mean()

def hnn_central_difference_loss(model: SeparableHamiltonianNetwork, trajectory_pq: torch.Tensor, times: torch.Tensor, *, actions: torch.Tensor | None=None, attributes: torch.Tensor | None=None) -> torch.Tensor:
    if trajectory_pq.ndim != 3 or trajectory_pq.shape[-1] != 2 * model.q_dim:
        raise ValueError('trajectory_pq must be [B,T,2*q_dim] in paper [p,q] order')
    batch, frames, _ = trajectory_pq.shape
    if frames < 3 or times.shape != (batch, frames):
        raise ValueError('at least three aligned time points are required')
    p, q = (trajectory_pq[..., :model.q_dim], trajectory_pq[..., model.q_dim:])
    denominator = (times[:, 2:] - times[:, :-2]).unsqueeze(-1)
    if bool((denominator <= 0).any()):
        raise ValueError('times must be strictly increasing')
    q_dot_target = (q[:, 2:] - q[:, :-2]) / denominator
    p_dot_target = (p[:, 2:] - p[:, :-2]) / denominator
    q_mid = q[:, 1:-1]
    p_mid = p[:, 1:-1]
    attrs_mid = _broadcast_attributes(attributes, batch=batch, points=frames - 2, attribute_dim=model.attribute_dim)
    if actions is None:
        torque = None
    else:
        if actions.shape != (batch, frames, model.q_dim):
            raise ValueError('actions must be [B,T,q_dim]')
        torque = actions[:, 1:-1]
    return hnn_derivative_loss(model, p=p_mid.reshape(-1, model.q_dim), q=q_mid.reshape(-1, model.q_dim), p_dot_target=p_dot_target.reshape(-1, model.q_dim), q_dot_target=q_dot_target.reshape(-1, model.q_dim), actions=None if torque is None else torque.reshape(-1, model.q_dim), attributes=None if attrs_mid is None else attrs_mid.reshape(-1, model.attribute_dim))

def hamiltonian_one_step_energy(model: SeparableHamiltonianNetwork, trajectory_pq: torch.Tensor, times: torch.Tensor, *, actions: torch.Tensor | None=None, attributes: torch.Tensor | None=None, create_graph: bool) -> torch.Tensor:
    if trajectory_pq.ndim != 3 or trajectory_pq.shape[-1] != 2 * model.q_dim:
        raise ValueError('trajectory_pq must be [B,T,2*q_dim] in paper [p,q] order')
    batch, frames, _ = trajectory_pq.shape
    if frames < 2 or times.shape != (batch, frames):
        raise ValueError('trajectory/time shapes do not align')
    p, q = (trajectory_pq[..., :model.q_dim], trajectory_pq[..., model.q_dim:])
    q_source = q[:, :-1]
    p_source = p[:, :-1]
    if not q_source.requires_grad or not p_source.requires_grad:
        q_source = q_source.requires_grad_(True)
        p_source = p_source.requires_grad_(True)
    attrs_edges = _broadcast_attributes(attributes, batch=batch, points=frames - 1, attribute_dim=model.attribute_dim)
    q_dot, p_dot_unforced = model.vector_field(p_source, q_source, attrs_edges, create_graph=create_graph)
    dt = (times[:, 1:] - times[:, :-1]).unsqueeze(-1).to(trajectory_pq)
    if bool((dt <= 0).any()):
        raise ValueError('times must be strictly increasing')
    if actions is None:
        torque = torch.zeros_like(p_dot_unforced)
    else:
        if actions.shape != (batch, frames, model.q_dim):
            raise ValueError('actions must be [B,T,q_dim]')
        torque = actions[:, :-1].to(p_dot_unforced)
    q_prediction = q_source + dt * q_dot
    p_prediction = p_source + dt * (p_dot_unforced + torque)
    edge_energy = (q[:, 1:] - q_prediction).square().sum(dim=-1) + (p[:, 1:] - p_prediction).square().sum(dim=-1)
    return edge_energy.sum(dim=-1)

def _extract(values: torch.Tensor, steps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.to(device=target.device, dtype=target.dtype)[steps].reshape(-1, 1, 1)

class HamiltonianGuidedDDIM:

    def __init__(self, denoiser: HGDPFPerceiver, hnn: SeparableHamiltonianNetwork, *, state_mean: torch.Tensor, state_scale: torch.Tensor, diffusion_steps: int=1000, inference_steps: int=20, context_ratio: float=0.5) -> None:
        if state_mean.shape != (denoiser.config.state_dim,) or state_scale.shape != state_mean.shape:
            raise ValueError('state normalization must match DPF state_dim')
        if bool((state_scale <= 0).any()) or not bool(torch.isfinite(state_scale).all()):
            raise ValueError('state_scale must be finite and positive')
        if not 0.0 < context_ratio < 1.0:
            raise ValueError('context_ratio must lie in (0,1)')
        if not 1 <= inference_steps <= diffusion_steps:
            raise ValueError('invalid inference step count')
        self.denoiser = denoiser
        self.hnn = hnn
        self.state_mean = state_mean.detach().clone()
        self.state_scale = state_scale.detach().clone()
        self.alpha_bar = cosine_alpha_bar(diffusion_steps)
        self.diffusion_steps = int(diffusion_steps)
        self.inference_steps = int(inference_steps)
        self.context_ratio = float(context_ratio)

    def _unnormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.state_scale.to(normalized) + self.state_mean.to(normalized)

    def _context(self, noisy: torch.Tensor, query_times: torch.Tensor, observed_initial: torch.Tensor, initial_time: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        batch, points, state_dim = noisy.shape
        count = max(1, int(math.floor(self.context_ratio * points)))
        scores = torch.rand(batch, points, device=noisy.device, generator=generator)
        indices = scores.topk(count, dim=1, largest=False).indices
        state_index = indices[..., None].expand(-1, -1, state_dim)
        chosen_states = noisy.gather(1, state_index)
        chosen_times = query_times.gather(1, indices)
        return (torch.cat([observed_initial[:, None], chosen_states], dim=1), torch.cat([initial_time[:, None], chosen_times], dim=1))

    def _predict_noise(self, noisy: torch.Tensor, step: int, query_times: torch.Tensor, observed_initial: torch.Tensor, initial_time: torch.Tensor, actions: torch.Tensor | None, attributes: torch.Tensor | None, *, generator: torch.Generator) -> torch.Tensor:
        context_states, context_times = self._context(noisy, query_times, observed_initial, initial_time, generator=generator)
        diffusion_steps = torch.full((noisy.shape[0],), step, device=noisy.device, dtype=torch.long)
        with torch.no_grad():
            return self.denoiser(diffusion_steps=diffusion_steps, context_states=context_states, context_times=context_times, query_states=noisy, query_times=query_times, actions=actions, attributes=attributes)

    def _energy(self, clean_normalized: torch.Tensor, query_times: torch.Tensor, observed_initial: torch.Tensor, initial_time: torch.Tensor, actions: torch.Tensor | None, attributes: torch.Tensor | None, *, create_graph: bool) -> torch.Tensor:
        full_state = torch.cat([self._unnormalize(observed_initial)[:, None], self._unnormalize(clean_normalized)], dim=1)
        full_time = torch.cat([initial_time[:, None], query_times], dim=1)
        if actions is None:
            full_actions = None
        else:
            zero_initial = torch.zeros(actions.shape[0], 1, actions.shape[-1], device=actions.device, dtype=actions.dtype)
            full_actions = torch.cat([zero_initial, actions], dim=1)
        return hamiltonian_one_step_energy(self.hnn, full_state, full_time, actions=full_actions, attributes=attributes, create_graph=create_graph)

    @staticmethod
    def _ddim_sigma(alpha_now: torch.Tensor, alpha_next: torch.Tensor, stochasticity: float) -> torch.Tensor:
        if stochasticity < 0.0:
            raise ValueError('DDIM stochasticity must be non-negative')
        variance = ((1.0 - alpha_next) / (1.0 - alpha_now) * (1.0 - alpha_now / alpha_next)).clamp_min(0.0)
        return float(stochasticity) * variance.sqrt()

    def sample(self, *, query_times: torch.Tensor, observed_initial: torch.Tensor, initial_time: torch.Tensor, actions: torch.Tensor | None=None, attributes: torch.Tensor | None=None, guidance: GuidanceMode='none', gradient_eta: float=0.01, snis_candidates: int=4, snis_lambda: float=1.0, snis_ddim_stochasticity: float=1.0, snis_recompute_clean: bool=True, generator: torch.Generator | None=None, proposal_generator: torch.Generator | None=None) -> torch.Tensor:
        if guidance not in {'none', 'gradient', 'snis'}:
            raise ValueError(f'unknown guidance mode: {guidance}')
        if query_times.ndim != 2 or observed_initial.shape != (query_times.shape[0], self.denoiser.config.state_dim):
            raise ValueError('query times and initial state do not align')
        if initial_time.shape != (query_times.shape[0],):
            raise ValueError('initial_time must be [B]')
        if generator is None:
            generator = torch.Generator(device=query_times.device)
            generator.manual_seed(0)
        if proposal_generator is None:
            proposal_generator = generator
        batch, points = query_times.shape
        noisy = torch.randn(batch, points, self.denoiser.config.state_dim, device=query_times.device, dtype=observed_initial.dtype, generator=generator)
        schedule = ddim_leading_timesteps(self.diffusion_steps, self.inference_steps).tolist()
        alpha = self.alpha_bar.to(noisy)
        for position, step_value in enumerate(schedule):
            step = int(step_value)
            next_step = int(schedule[position + 1]) if position + 1 < len(schedule) else -1
            epsilon = self._predict_noise(noisy, step, query_times, observed_initial, initial_time, actions, attributes, generator=generator)
            alpha_now = alpha[step]
            alpha_next = alpha[next_step] if next_step >= 0 else alpha.new_tensor(1.0)
            clean = (noisy - (1.0 - alpha_now).sqrt() * epsilon) / alpha_now.sqrt()
            if guidance == 'gradient':
                clean = clean.detach().requires_grad_(True)
                energy = self._energy(clean, query_times, observed_initial, initial_time, actions, attributes, create_graph=True)
                gradient = torch.autograd.grad(energy.sum(), clean)[0]
                clean = (clean - float(gradient_eta) * (1.0 - alpha_now).sqrt() * gradient).detach()
            deterministic = alpha_next.sqrt() * clean + (1.0 - alpha_next).sqrt() * epsilon
            if guidance != 'snis' or next_step < 0:
                noisy = deterministic
                continue
            if snis_candidates < 2 or snis_lambda <= 0.0:
                raise ValueError('SNIS requires at least two candidates and positive lambda')
            sigma = self._ddim_sigma(alpha_now, alpha_next, snis_ddim_stochasticity)
            direction_scale = (1.0 - alpha_next - sigma.square()).clamp_min(0.0).sqrt()
            stochastic_mean = alpha_next.sqrt() * clean + direction_scale * epsilon
            proposals = stochastic_mean[:, None] + sigma * torch.randn(batch, snis_candidates, points, noisy.shape[-1], device=noisy.device, dtype=noisy.dtype, generator=proposal_generator)
            flat = proposals.reshape(batch * snis_candidates, points, noisy.shape[-1])
            if snis_recompute_clean:
                repeated_times = query_times.repeat_interleave(snis_candidates, dim=0)
                repeated_initial = observed_initial.repeat_interleave(snis_candidates, dim=0)
                repeated_initial_time = initial_time.repeat_interleave(snis_candidates, dim=0)
                repeated_actions = None if actions is None else actions.repeat_interleave(snis_candidates, dim=0)
                repeated_attrs = None if attributes is None else attributes.repeat_interleave(snis_candidates, dim=0)
                proposal_epsilon = self._predict_noise(flat, next_step, repeated_times, repeated_initial, repeated_initial_time, repeated_actions, repeated_attrs, generator=generator)
                candidate_clean = (flat - (1.0 - alpha_next).sqrt() * proposal_epsilon) / alpha_next.sqrt()
                energy = self._energy(candidate_clean, repeated_times, repeated_initial, repeated_initial_time, repeated_actions, repeated_attrs, create_graph=False).reshape(batch, snis_candidates)
            else:
                repeated_times = query_times.repeat_interleave(snis_candidates, dim=0)
                repeated_initial = observed_initial.repeat_interleave(snis_candidates, dim=0)
                repeated_initial_time = initial_time.repeat_interleave(snis_candidates, dim=0)
                repeated_actions = None if actions is None else actions.repeat_interleave(snis_candidates, dim=0)
                repeated_attrs = None if attributes is None else attributes.repeat_interleave(snis_candidates, dim=0)
                energy = self._energy(flat, repeated_times, repeated_initial, repeated_initial_time, repeated_actions, repeated_attrs, create_graph=False).reshape(batch, snis_candidates)
            probabilities = torch.softmax(-float(snis_lambda) * energy, dim=1)
            selected = torch.multinomial(probabilities, 1, generator=proposal_generator).squeeze(1)
            noisy = proposals[torch.arange(batch, device=noisy.device), selected]
        return self._unnormalize(noisy)
