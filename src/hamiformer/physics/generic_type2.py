from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass(frozen=True)
class GenericTypeIINewtonResult:
    state: torch.Tensor
    residual_max: torch.Tensor
    mixed_singular_min: torch.Tensor
    mixed_singular_max: torch.Tensor
    mixed_condition: torch.Tensor
    iterations: int
    converged: bool

@dataclass(frozen=True)
class GenericTypeIIUnrolledResult:
    state: torch.Tensor
    p_next: torch.Tensor
    residual_max: torch.Tensor

@dataclass(frozen=True)
class GenericTypeIILinearization:
    matrix: torch.Tensor
    mixed_jacobian: torch.Tensor
    mixed_singular_min: torch.Tensor
    mixed_singular_max: torch.Tensor
    mixed_condition: torch.Tensor
    tangent_spectral_norm: torch.Tensor

@dataclass(frozen=True)
class GenericTypeIIJet:
    matrix: torch.Tensor
    offset: torch.Tensor
    source_graph: torch.Tensor
    target_graph: torch.Tensor
    mixed_jacobian: torch.Tensor
    mixed_singular_min: torch.Tensor
    mixed_singular_max: torch.Tensor
    mixed_condition: torch.Tensor
    tangent_spectral_norm: torch.Tensor
    mixed_singular_min_per_map: torch.Tensor
    mixed_singular_max_per_map: torch.Tensor
    mixed_condition_per_map: torch.Tensor
    tangent_spectral_norm_per_map: torch.Tensor
    tangent_spectral_norm_computed: bool

@dataclass(frozen=True)
class GenericTypeIIHealthBarrier:
    penalty: torch.Tensor
    mixed_singular_min: torch.Tensor
    mixed_singular_max: torch.Tensor
    mixed_condition: torch.Tensor
    tangent_spectral_norm: torch.Tensor

class _PreNormResidualBlock(nn.Module):

    def __init__(self, width: int, expansion: float) -> None:
        super().__init__()
        hidden = max(width, int(round(width * expansion)))
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, hidden)
        self.out_projection = nn.Linear(hidden, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.out_projection(torch.nn.functional.silu(self.in_projection(self.norm(value))))
        return value + residual

class _ExplicitMultiheadSelfAttention(nn.Module):

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width < 4 or heads < 1 or width % heads != 0:
            raise ValueError('attention width must be positive and divisible by heads')
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.width:
            raise ValueError('attention input must be [batch, tokens, width]')
        batch, tokens, _ = value.shape
        packed = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, content = packed.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = query @ key.transpose(-1, -2) * self.head_width ** (-0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ content
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, self.width))

def _attention_is_enabled(mode: str, token_count: int, name: str) -> bool:
    if mode not in {'auto', 'enabled', 'disabled'}:
        raise ValueError(f'{name}_attention_mode must be auto, enabled, or disabled')
    if token_count < 1:
        raise ValueError(f'{name} token count must be positive')
    return mode == 'enabled' or (mode == 'auto' and token_count > 1)

class _FactorizedTokenTransformerBlock(nn.Module):

    def __init__(self, width: int, heads: int, expansion: float, *, spatial_attention_enabled: bool, object_attention_enabled: bool) -> None:
        super().__init__()
        if expansion < 1.0:
            raise ValueError('transformer expansion must be at least one')
        hidden = max(width, int(round(width * expansion)))
        self.spatial_attention_enabled = bool(spatial_attention_enabled)
        self.object_attention_enabled = bool(object_attention_enabled)
        self.spatial_norm = nn.LayerNorm(width) if self.spatial_attention_enabled else None
        self.spatial_attention = _ExplicitMultiheadSelfAttention(width, heads) if self.spatial_attention_enabled else None
        self.object_norm = nn.LayerNorm(width) if self.object_attention_enabled else None
        self.object_attention = _ExplicitMultiheadSelfAttention(width, heads) if self.object_attention_enabled else None
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward_in = nn.Linear(width, hidden)
        self.feedforward_out = nn.Linear(hidden, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 4:
            raise ValueError('token tensor must end in [objects, spatial_tokens, width]')
        *prefix, objects, spatial_tokens, width = tokens.shape
        flat_batch = math.prod(prefix) if prefix else 1
        value = tokens.reshape(flat_batch, objects, spatial_tokens, width)
        if self.spatial_attention_enabled:
            assert self.spatial_norm is not None and self.spatial_attention is not None
            spatial = value.reshape(flat_batch * objects, spatial_tokens, width)
            spatial = spatial + self.spatial_attention(self.spatial_norm(spatial))
            value = spatial.reshape(flat_batch, objects, spatial_tokens, width)
        if self.object_attention_enabled:
            assert self.object_norm is not None and self.object_attention is not None
            object_tokens = value.transpose(1, 2).reshape(flat_batch * spatial_tokens, objects, width)
            object_tokens = object_tokens + self.object_attention(self.object_norm(object_tokens))
            value = object_tokens.reshape(flat_batch, spatial_tokens, objects, width).transpose(1, 2)
        value = value + self.feedforward_out(torch.nn.functional.silu(self.feedforward_in(self.feedforward_norm(value))))
        return value.reshape(*prefix, objects, spatial_tokens, width)

class GenericConditionalTypeIIGenerator(nn.Module):
    architecture = 'generic_conditional_type2'

    def __init__(self, *, state_dim: int, context_dim: int, hidden_size: int, depth: int, expansion: float=2.0, q_scale: tuple[float, ...] | None=None, p_scale: tuple[float, ...] | None=None, context_scale: tuple[float, ...] | None=None) -> None:
        super().__init__()
        if state_dim < 1 or context_dim < 1:
            raise ValueError('state_dim and context_dim must be positive')
        if hidden_size < 8 or depth < 1 or expansion < 1.0:
            raise ValueError('invalid generic Type-II MLP size')

        def scales(value: tuple[float, ...] | None, count: int, name: str) -> tuple[float, ...]:
            result = (1.0,) * count if value is None else tuple((float(item) for item in value))
            if len(result) != count or any((not math.isfinite(item) or item <= 0.0 for item in result)):
                raise ValueError(f'{name} must contain {count} finite positive scales')
            return result
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.register_buffer('q_scale', torch.tensor(scales(q_scale, state_dim, 'q_scale')))
        self.register_buffer('p_scale', torch.tensor(scales(p_scale, state_dim, 'p_scale')))
        self.register_buffer('context_scale', torch.tensor(scales(context_scale, context_dim, 'context_scale')))
        self.input_projection = nn.Linear(2 * state_dim + context_dim, hidden_size)
        self.blocks = nn.ModuleList((_PreNormResidualBlock(hidden_size, expansion) for _ in range(depth)))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.001)

    def _validate(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> None:
        if q.shape != p_next.shape or q.ndim < 1 or q.shape[-1] != self.state_dim:
            raise ValueError('q and p_next must align as [..., state_dim]')
        if context.shape != (*q.shape[:-1], self.context_dim):
            raise ValueError('context must align as [..., context_dim]')
        if not bool(torch.isfinite(q).all() and torch.isfinite(p_next).all() and torch.isfinite(context).all()):
            raise ValueError('generic Type-II inputs must be finite')

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self._validate(q, p_next, context)
        features = torch.cat([q / self.q_scale.to(q), p_next / self.p_scale.to(p_next), context / self.context_scale.to(context)], dim=-1)
        value = self.input_projection(features)
        for block in self.blocks:
            value = block(value)
        return self.output_projection(self.output_norm(value)).squeeze(-1)

class TokenConditionalTypeIIGenerator(nn.Module):
    architecture = 'token_conditional_type2'

    def __init__(self, *, num_objects: int, spatial_tokens: int, coordinate_dim: int, token_context_dim: int, hidden_size: int, depth: int, heads: int, expansion: float=2.0, q_scale: float | tuple[float, ...]=1.0, p_scale: float | tuple[float, ...]=1.0, spatial_attention_mode: str='auto', object_attention_mode: str='auto') -> None:
        super().__init__()
        if min(num_objects, spatial_tokens, coordinate_dim, token_context_dim, hidden_size, depth, heads) < 1:
            raise ValueError('token Type-II dimensions must be positive')
        if hidden_size % heads != 0 or expansion < 1.0:
            raise ValueError('invalid token Type-II transformer width/heads/expansion')
        self.num_objects = int(num_objects)
        self.spatial_tokens = int(spatial_tokens)
        self.coordinate_dim = int(coordinate_dim)
        self.token_context_dim = int(token_context_dim)
        self.state_dim = self.num_objects * self.spatial_tokens * self.coordinate_dim
        self.spatial_attention_mode = str(spatial_attention_mode)
        self.object_attention_mode = str(object_attention_mode)
        self.spatial_attention_enabled = _attention_is_enabled(self.spatial_attention_mode, self.spatial_tokens, 'spatial')
        self.object_attention_enabled = _attention_is_enabled(self.object_attention_mode, self.num_objects, 'object')
        self.context_dim = self.token_context_dim

        def coordinate_scale(value: float | tuple[float, ...], name: str) -> torch.Tensor:
            values = (float(value),) * self.coordinate_dim if isinstance(value, (float, int)) else tuple((float(item) for item in value))
            if len(values) != self.coordinate_dim or any((not math.isfinite(item) or item <= 0.0 for item in values)):
                raise ValueError(f'{name} must have coordinate_dim finite positive values')
            return torch.tensor(values)
        self.register_buffer('q_scale', coordinate_scale(q_scale, 'q_scale'))
        self.register_buffer('p_scale', coordinate_scale(p_scale, 'p_scale'))
        self.input_projection = nn.Linear(2 * self.coordinate_dim + self.token_context_dim, hidden_size)
        self.blocks = nn.ModuleList((_FactorizedTokenTransformerBlock(hidden_size, heads, expansion, spatial_attention_enabled=self.spatial_attention_enabled, object_attention_enabled=self.object_attention_enabled) for _ in range(depth)))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.0001)

    def _validate(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> None:
        if q.shape != p_next.shape or q.ndim < 1 or q.shape[-1] != self.state_dim:
            raise ValueError('token Type-II q and p_next must align as [..., flattened_state_dim]')
        expected_context = (*q.shape[:-1], self.num_objects, self.spatial_tokens, self.token_context_dim)
        if context.shape != expected_context:
            raise ValueError('token Type-II context must align as [..., objects, spatial_tokens, token_context_dim]')
        if not bool(torch.isfinite(q).all() and torch.isfinite(p_next).all() and torch.isfinite(context).all()):
            raise ValueError('token Type-II inputs must be finite')

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self._validate(q, p_next, context)
        shape = (*q.shape[:-1], self.num_objects, self.spatial_tokens, self.coordinate_dim)
        q_tokens = q.reshape(shape) / self.q_scale.to(q)
        p_tokens = p_next.reshape(shape) / self.p_scale.to(p_next)
        value = self.input_projection(torch.cat([q_tokens, p_tokens, context], dim=-1))
        for block in self.blocks:
            value = block(value)
        return self.output_projection(self.output_norm(value)).squeeze(-1).sum(dim=(-1, -2))

class ScalarContextTokenTypeIIGenerator(TokenConditionalTypeIIGenerator):
    architecture = 'scalar_context_token_type2'

    def __init__(self, *, context_dim: int, hidden_size: int, depth: int, heads: int, expansion: float=2.0, q_scale: float=1.0, p_scale: float=1.0, spatial_attention_mode: str='auto', object_attention_mode: str='auto') -> None:
        super().__init__(num_objects=1, spatial_tokens=1, coordinate_dim=1, token_context_dim=context_dim, hidden_size=hidden_size, depth=depth, heads=heads, expansion=expansion, q_scale=q_scale, p_scale=p_scale, spatial_attention_mode=spatial_attention_mode, object_attention_mode=object_attention_mode)
        self.theta_dim = int(context_dim)

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if q.shape != p_next.shape:
            raise ValueError('scalar token Type-II q and p_next must align')
        if context.shape != (*q.shape, self.theta_dim):
            raise ValueError('scalar token Type-II context must align as [..., context_dim]')
        return super().forward(q.unsqueeze(-1), p_next.unsqueeze(-1), context.unsqueeze(-2).unsqueeze(-2))

def _validate_type2_inputs(generator: GenericConditionalTypeIIGenerator, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor, step_size: float) -> None:
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError('step_size must be finite and positive')
    generator._validate(q, p_next, context)

def type2_vector_eom(generator: GenericConditionalTypeIIGenerator, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor, *, step_size: float, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_type2_inputs(generator, q, p_next, context, step_size)
    with torch.enable_grad():
        q_value = q.clone().requires_grad_(True)
        p_value = p_next.clone().requires_grad_(True)
        generating_value = (q_value * p_value).sum(dim=-1) + q_value.new_tensor(step_size) * generator(q_value, p_value, context)
        source_p, target_q = torch.autograd.grad(generating_value.sum(), (q_value, p_value), create_graph=create_graph, retain_graph=create_graph)
    return (source_p, target_q)

def _component_jacobian_reference(value: torch.Tensor, wrt: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    rows = []
    for component in range(value.shape[-1]):
        derivative = torch.autograd.grad(value[..., component].sum(), wrt, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
        rows.append(torch.zeros_like(wrt) if derivative is None else derivative)
    return torch.stack(rows, dim=-2)

def _component_jacobian(value: torch.Tensor, wrt: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    components = int(value.shape[-1])
    basis = torch.eye(components, device=value.device, dtype=value.dtype)
    grad_outputs = basis.reshape(components, *[1] * (value.ndim - 1), components).expand(components, *value.shape)
    derivative = torch.autograd.grad(value, wrt, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=True, allow_unused=True, is_grads_batched=True)[0]
    if derivative is None:
        derivative = torch.zeros((components, *wrt.shape), device=wrt.device, dtype=wrt.dtype)
    return derivative.movedim(0, -2)

def _joint_type2_jacobians(source_p: torch.Tensor, target_q: torch.Tensor, q_value: torch.Tensor, p_value: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    components = int(source_p.shape[-1])
    basis = torch.eye(2 * components, device=source_p.device, dtype=source_p.dtype)
    source_basis = basis[:, :components].reshape(2 * components, *[1] * (source_p.ndim - 1), components).expand(2 * components, *source_p.shape)
    target_basis = basis[:, components:].reshape(2 * components, *[1] * (target_q.ndim - 1), components).expand(2 * components, *target_q.shape)
    derivative_q, derivative_p = torch.autograd.grad((source_p, target_q), (q_value, p_value), grad_outputs=(source_basis, target_basis), create_graph=create_graph, retain_graph=True, is_grads_batched=True)
    derivative_q = derivative_q.movedim(0, -2)
    derivative_p = derivative_p.movedim(0, -2)
    return (derivative_q[..., :components, :], derivative_p[..., :components, :], derivative_p[..., components:, :])

def _type2_second_derivatives(generator: GenericConditionalTypeIIGenerator, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor, *, step_size: float, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_type2_inputs(generator, q, p_next, context, step_size)
    with torch.enable_grad():
        q_value = q.clone().requires_grad_(True)
        p_value = p_next.clone().requires_grad_(True)
        generating_value = (q_value * p_value).sum(dim=-1) + q_value.new_tensor(step_size) * generator(q_value, p_value, context)
        source_p, target_q = torch.autograd.grad(generating_value.sum(), (q_value, p_value), create_graph=True, retain_graph=True)
        qq, qp, pp = _joint_type2_jacobians(source_p, target_q, q_value, p_value, create_graph=create_graph)
    return (source_p, target_q, qq, qp, pp)

def _matrix_health_per_map(matrix: torch.Tensor, mixed: torch.Tensor, *, compute_tangent: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    singular_values = torch.linalg.svdvals(mixed)
    mixed_min = singular_values.amin(dim=-1)
    mixed_max = singular_values.amax(dim=-1)
    mixed_condition = mixed_max / mixed_min
    if compute_tangent:
        tangent_norm = torch.linalg.svdvals(matrix).amax(dim=-1)
    else:
        finite = torch.isfinite(matrix).all(dim=(-2, -1))
        tangent_norm = torch.where(finite, torch.zeros_like(mixed_min), torch.full_like(mixed_min, float('inf')))
    return (mixed_min, mixed_max, mixed_condition, tangent_norm)

def _matrix_health(matrix: torch.Tensor, mixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    per_map = _matrix_health_per_map(matrix, mixed)
    return (per_map[0].amin(), per_map[1].amax(), per_map[2].amax(), per_map[3].amax())

def _linearization_from_type2_second_derivatives(qq: torch.Tensor, qp: torch.Tensor, pp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = qp.shape[-1]
    identity = torch.eye(dimension, device=qp.device, dtype=qp.dtype).expand(*qp.shape[:-2], dimension, dimension)
    inverse = torch.linalg.solve(qp, identity)
    inverse_qq = torch.linalg.solve(qp, qq)
    top_left = qp.transpose(-1, -2) - pp @ inverse_qq
    top_right = pp @ inverse
    bottom_left = -inverse_qq
    matrix = torch.cat([torch.cat([top_left, top_right], dim=-1), torch.cat([bottom_left, inverse], dim=-1)], dim=-2)
    return (matrix, qp)

def type2_vector_linearization(generator: GenericConditionalTypeIIGenerator, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor, *, step_size: float) -> GenericTypeIILinearization:
    _, _, qq, qp, pp = _type2_second_derivatives(generator, q, p_next, context, step_size=step_size, create_graph=False)
    matrix, mixed = _linearization_from_type2_second_derivatives(qq, qp, pp)
    per_map = _matrix_health_per_map(matrix, mixed)
    mixed_min, mixed_max = (per_map[0].amin(), per_map[1].amax())
    mixed_condition, tangent_norm = (per_map[2].amax(), per_map[3].amax())
    return GenericTypeIILinearization(matrix=matrix, mixed_jacobian=mixed, mixed_singular_min=mixed_min, mixed_singular_max=mixed_max, mixed_condition=mixed_condition, tangent_spectral_norm=tangent_norm)

def type2_vector_jets(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, create_graph: bool=False, detach: bool=True, compute_tangent_spectral_norm: bool=True) -> GenericTypeIIJet:
    if not all((math.isfinite(value) and value > 0.0 for value in (mixed_singular_floor, mixed_condition_limit, tangent_spectral_norm_limit))):
        raise ValueError('GFJP Type-II health limits must be finite and positive')
    source_p, target_q, qq, qp, pp = _type2_second_derivatives(generator, q_anchor, p_next_anchor, context, step_size=step_size, create_graph=create_graph)
    matrix, mixed = _linearization_from_type2_second_derivatives(qq, qp, pp)
    per_map = _matrix_health_per_map(matrix, mixed, compute_tangent=compute_tangent_spectral_norm)
    mixed_min, mixed_max = (per_map[0].amin(), per_map[1].amax())
    mixed_condition, tangent_norm = (per_map[2].amax(), per_map[3].amax())
    if not bool(torch.isfinite(mixed_min).item()) or not bool(torch.isfinite(mixed_condition).item()) or (not bool(torch.isfinite(tangent_norm).item())) or (float(mixed_min.detach()) < mixed_singular_floor) or (float(mixed_condition.detach()) > mixed_condition_limit) or (float(tangent_norm.detach()) > tangent_spectral_norm_limit):
        raise RuntimeError(f'GFJP Type-II jet health failed: sigma_min={float(mixed_min):.3e}, sigma_max={float(mixed_max):.3e}, condition={float(mixed_condition):.3e}, tangent_norm={float(tangent_norm):.3e}')
    source_graph = torch.cat([q_anchor, source_p], dim=-1)
    target_graph = torch.cat([target_q, p_next_anchor], dim=-1)
    offset = target_graph - (matrix @ source_graph.unsqueeze(-1)).squeeze(-1)
    if detach:
        matrix, offset = (matrix.detach(), offset.detach())
        source_graph, target_graph = (source_graph.detach(), target_graph.detach())
        mixed, mixed_min, mixed_max = (mixed.detach(), mixed_min.detach(), mixed_max.detach())
        mixed_condition, tangent_norm = (mixed_condition.detach(), tangent_norm.detach())
        per_map = tuple((value.detach() for value in per_map))
    return GenericTypeIIJet(matrix=matrix, offset=offset, source_graph=source_graph, target_graph=target_graph, mixed_jacobian=mixed, mixed_singular_min=mixed_min, mixed_singular_max=mixed_max, mixed_condition=mixed_condition, tangent_spectral_norm=tangent_norm, mixed_singular_min_per_map=per_map[0], mixed_singular_max_per_map=per_map[1], mixed_condition_per_map=per_map[2], tangent_spectral_norm_per_map=per_map[3], tangent_spectral_norm_computed=bool(compute_tangent_spectral_norm))

def type2_vector_health_barrier(generator: nn.Module, q_anchor: torch.Tensor, p_next_anchor: torch.Tensor, context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float, tangent_spectral_norm_limit: float, safety_margin: float=1.25) -> GenericTypeIIHealthBarrier:
    if not all((math.isfinite(value) and value > 0.0 for value in (mixed_singular_floor, mixed_condition_limit, tangent_spectral_norm_limit, safety_margin))):
        raise ValueError('Type-II health-barrier limits must be finite and positive')
    _, _, qq, qp, pp = _type2_second_derivatives(generator, q_anchor, p_next_anchor, context, step_size=step_size, create_graph=True)
    matrix, mixed = _linearization_from_type2_second_derivatives(qq, qp, pp)
    mixed_min, mixed_max, mixed_condition, tangent_norm = _matrix_health(matrix, mixed)
    if not bool(torch.isfinite(mixed_min).item() and torch.isfinite(mixed_condition).item() and torch.isfinite(tangent_norm).item()):
        raise FloatingPointError('non-finite Type-II Hessian/tangent in GFJP health barrier')
    protected_floor = mixed_min.new_tensor(mixed_singular_floor * safety_margin)
    protected_condition = mixed_condition.new_tensor(mixed_condition_limit / safety_margin)
    protected_tangent = tangent_norm.new_tensor(tangent_spectral_norm_limit / safety_margin)
    penalty = torch.nn.functional.relu((protected_floor - mixed_min) / protected_floor).square() + torch.nn.functional.relu(mixed_condition / protected_condition - 1.0).square() + torch.nn.functional.relu(tangent_norm / protected_tangent - 1.0).square()
    return GenericTypeIIHealthBarrier(penalty=penalty, mixed_singular_min=mixed_min, mixed_singular_max=mixed_max, mixed_condition=mixed_condition, tangent_spectral_norm=tangent_norm)

def solve_generic_type2_step_unrolled(generator: nn.Module, state: torch.Tensor, context: torch.Tensor, *, step_size: float, iterations: int=4, damping: float=1.0) -> GenericTypeIIUnrolledResult:
    if state.ndim < 1 or state.shape[-1] != 2 * generator.state_dim:
        raise ValueError('state must be [..., 2 * state_dim] in [q,p] order')
    if iterations < 1 or not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError('iterations and step_size must be positive and finite')
    if not 0.0 < damping <= 1.0:
        raise ValueError('damping must lie in (0,1]')
    dimension = generator.state_dim
    q = state[..., :dimension]
    source_p = state[..., dimension:]
    generator._validate(q, source_p, context)
    with torch.enable_grad():
        p_next = source_p.clone().requires_grad_(True)
        for _ in range(iterations):
            predicted_p, _ = type2_vector_eom(generator, q, p_next, context, step_size=step_size, create_graph=True)
            mixed = _component_jacobian(predicted_p, p_next, create_graph=True)
            delta = torch.linalg.solve(mixed, (predicted_p - source_p).unsqueeze(-1)).squeeze(-1)
            p_next = p_next - damping * delta
        predicted_p, target_q = type2_vector_eom(generator, q, p_next, context, step_size=step_size, create_graph=True)
        result_state = torch.cat([target_q, p_next], dim=-1)
        residual_max = (predicted_p - source_p).abs().amax()
    return GenericTypeIIUnrolledResult(state=result_state, p_next=p_next, residual_max=residual_max)

def solve_generic_type2_step_newton(generator: GenericConditionalTypeIIGenerator, state: torch.Tensor, context: torch.Tensor, *, step_size: float, mixed_singular_floor: float, mixed_condition_limit: float=20.0, max_iterations: int=12, tolerance: float=1e-07, damping: float=1.0) -> GenericTypeIINewtonResult:
    if state.ndim < 1 or state.shape[-1] != 2 * generator.state_dim:
        raise ValueError('state must be [..., 2 * state_dim] in [q, p] order')
    if max_iterations < 1 or tolerance <= 0.0 or mixed_singular_floor <= 0.0 or (not math.isfinite(mixed_condition_limit)) or (mixed_condition_limit <= 0.0):
        raise ValueError('invalid Newton numerical contract')
    if not 0.0 < damping <= 1.0:
        raise ValueError('damping must lie in (0, 1]')
    if not bool(torch.isfinite(state).all() and torch.isfinite(context).all()):
        raise ValueError('state and context must be finite')
    dimension = generator.state_dim
    generator._validate(state[..., :dimension], state[..., dimension:], context)
    q = state[..., :dimension].detach()
    source_p = state[..., dimension:].detach()
    fixed_context = context.detach()
    p_next = source_p.clone()
    residual_max = torch.full((), math.inf, dtype=state.dtype, device=state.device)
    mixed_min = torch.full((), math.inf, dtype=state.dtype, device=state.device)
    mixed_max = torch.full((), math.inf, dtype=state.dtype, device=state.device)
    mixed_condition = torch.full((), math.inf, dtype=state.dtype, device=state.device)
    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        predicted_p, _, _, mixed, _ = _type2_second_derivatives(generator, q, p_next, fixed_context, step_size=step_size, create_graph=False)
        residual = predicted_p.detach() - source_p
        residual_max = residual.abs().amax()
        singular_values = torch.linalg.svdvals(mixed.detach())
        mixed_min = singular_values.amin()
        mixed_max = singular_values.amax()
        mixed_condition = (singular_values.amax(dim=-1) / singular_values.amin(dim=-1)).amax()
        if not bool(torch.isfinite(mixed_min).item()) or not bool(torch.isfinite(mixed_condition).item()) or float(mixed_min) < mixed_singular_floor or (float(mixed_condition) > mixed_condition_limit):
            raise RuntimeError(f'Type-II mixed Jacobian is singular/ill-conditioned: sigma_min={float(mixed_min):.3e}, sigma_max={float(mixed_max):.3e}, condition={float(mixed_condition):.3e}')
        if bool((residual_max <= tolerance).item()):
            converged = True
            break
        delta = torch.linalg.solve(mixed.detach(), residual.unsqueeze(-1)).squeeze(-1)
        p_next = (p_next - damping * delta).detach()
    predicted_p, target_q = type2_vector_eom(generator, q, p_next, fixed_context, step_size=step_size, create_graph=False)
    residual_max = (predicted_p.detach() - source_p).abs().amax()
    converged = bool((residual_max <= tolerance).item())
    return GenericTypeIINewtonResult(state=torch.cat([target_q.detach(), p_next.detach()], dim=-1), residual_max=residual_max.detach(), mixed_singular_min=mixed_min.detach(), mixed_singular_max=mixed_max.detach(), mixed_condition=mixed_condition.detach(), iterations=iterations, converged=converged)
