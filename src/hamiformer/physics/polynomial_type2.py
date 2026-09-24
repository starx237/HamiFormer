from __future__ import annotations
import math
import torch
from torch import nn

class ScalarPolynomialTypeIIGenerator(nn.Module):
    architecture = 'scalar_polynomial_type2'

    def __init__(self, *, q_degree: int, p_degree: int, context_degree: int, q_scale: float, p_scale: float, context_center: float, context_radius: float) -> None:
        super().__init__()
        if min(q_degree, p_degree, context_degree) < 0:
            raise ValueError('polynomial degrees must be nonnegative')
        if not all((math.isfinite(value) and value > 0.0 for value in (q_scale, p_scale, context_radius))):
            raise ValueError('polynomial coordinate scales must be finite and positive')
        if not math.isfinite(context_center):
            raise ValueError('context_center must be finite')
        self.q_degree = int(q_degree)
        self.p_degree = int(p_degree)
        self.context_degree = int(context_degree)
        self.state_dim = 1
        self.context_dim = 1
        self.register_buffer('q_scale', torch.tensor(float(q_scale)))
        self.register_buffer('p_scale', torch.tensor(float(p_scale)))
        self.register_buffer('context_center', torch.tensor(float(context_center)))
        self.register_buffer('context_radius', torch.tensor(float(context_radius)))
        self.coefficients = nn.Parameter(torch.zeros(self.q_degree + 1, self.p_degree + 1, self.context_degree + 1))

    def _validate(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> None:
        if q.shape != p_next.shape or q.ndim < 2 or q.shape[-1] != 1:
            raise ValueError('vector scalar-polynomial Type-II q/P must align as [...,1]')
        if context.shape != (*q.shape[:-1], 1):
            raise ValueError('vector scalar-polynomial context must align as [...,1]')
        if not bool(torch.isfinite(q).all() and torch.isfinite(p_next).all() and torch.isfinite(context).all()):
            raise ValueError('vector scalar-polynomial Type-II inputs must be finite')

    @staticmethod
    def _legendre_with_derivative(value: torch.Tensor, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = [torch.ones_like(value)]
        derivatives = [torch.zeros_like(value)]
        if degree >= 1:
            values.append(value)
            derivatives.append(torch.ones_like(value))
        for index in range(2, degree + 1):
            values.append(((2 * index - 1) * value * values[-1] - (index - 1) * values[-2]) / index)
            derivatives.append(((2 * index - 1) * (values[-2] + value * derivatives[-1]) - (index - 1) * derivatives[-2]) / index)
        return (torch.stack(values, dim=-1), torch.stack(derivatives, dim=-1))

    @staticmethod
    def _legendre(value: torch.Tensor, degree: int) -> torch.Tensor:
        return ScalarPolynomialTypeIIGenerator._legendre_with_derivative(value, degree)[0]

    def basis(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if q.shape != p_next.shape or context.shape != (*q.shape, 1):
            raise ValueError('scalar polynomial Type-II inputs must align as q, P, [...,1] context')
        if not bool(torch.isfinite(q).all() and torch.isfinite(p_next).all() and torch.isfinite(context).all()):
            raise ValueError('scalar polynomial Type-II inputs must be finite')
        q_values, q_derivative_normalized = self._legendre_with_derivative(q / self.q_scale.to(q), self.q_degree)
        p_values, p_derivative_normalized = self._legendre_with_derivative(p_next / self.p_scale.to(p_next), self.p_degree)
        context_values = self._legendre((context[..., 0] - self.context_center.to(context)) / self.context_radius.to(context), self.context_degree)
        return ((q_values, q_derivative_normalized / self.q_scale.to(q), p_values, p_derivative_normalized / self.p_scale.to(p_next)), context_values)

    def forward(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if q.ndim >= 2 and q.shape[-1] == 1 and (context.shape == (*q.shape[:-1], 1)):
            self._validate(q, p_next, context)
            return self._forward_scalar(q[..., 0], p_next[..., 0], context)
        return self._forward_scalar(q, p_next, context)

    def _forward_scalar(self, q: torch.Tensor, p_next: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        (q_values, _q_derivative, p_values, _p_derivative), context_values = self.basis(q, p_next, context)
        return torch.einsum('...i,...j,...k,ijk->...', q_values, p_values, context_values, self.coefficients)
