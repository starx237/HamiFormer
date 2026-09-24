import pytest
import torch

from hamiformer.integrators.runtime import candidate_function
from hamiformer.integrators.solver import Solver


@pytest.mark.parametrize('method', ('explicit_euler', 'symeuler2', 'plas', 'plas_s', 'plas_dc', 'plas_dc_f', 'plas_dc_s'))
def test_batched_candidate_matches_independent_equations(method):
    dtype = torch.float64
    state = torch.linspace(-.3, .6, 24, dtype=dtype).reshape(3, 2, 4)
    anchor = state + .1
    h = .03
    d = 4
    energy = lambda z: .5 * z.square().sum() + .1 * torch.sin(z.sum())
    gradient = torch.func.grad(energy)
    flat = lambda z: torch.cat((z[..., :2].flatten(1), z[..., 2:].flatten(1)), -1)
    def field(q, p, attrs):
        g = torch.func.vmap(gradient)(torch.cat((q.flatten(1), p.flatten(1)), -1))
        return g[:, d:].reshape_as(q), -g[:, :d].reshape_as(p)
    # Build the affine operator independently by applying the scalar map to a basis.
    matrices, offsets, expected = [], [], []
    for s, a in zip(flat(state), flat(anchor)):
        scalar = Solver('plas')
        zero = scalar.step(energy, torch.zeros_like(s), h, a)
        columns = [scalar.step(energy, v, h, a) - zero for v in torch.eye(8, dtype=dtype)]
        matrices.append(torch.stack(columns, -1))
        offsets.append(zero)
        expected.append(Solver(method).step(energy, s, h, a))
    result = candidate_function(method, field, h)(torch.stack(matrices), torch.stack(offsets), state, None)
    torch.testing.assert_close(flat(result), torch.stack(expected), atol=1e-12, rtol=1e-12)


def test_dc_accumulates_defect_in_double_precision():
    # Cancellation is lost if the float32 defect is formed before conversion.
    p = 2**24
    previous = torch.tensor([[[0., 0., p, p]]])
    matrix = torch.eye(4)[None]
    offset = torch.zeros(1, 4)
    def field(q, pp, attrs):
        return torch.zeros_like(q), torch.full_like(pp, .75)
    result = candidate_function('plas_dc', field, 1.)(matrix, offset, previous, None)
    # Position response makes the otherwise sub-ULP momentum correction visible.
    matrix[:, 0, 2] = 1.
    result = candidate_function('plas_dc', field, 1.)(matrix, offset, previous, None)
    assert result[0, 0, 0].item() == .75, 'DC must preserve a sub-ULP defect'


def test_symeuler_uses_two_momentum_updates_and_final_position_field():
    previous = torch.tensor([[[.4, .2, -.3, .1]]], dtype=torch.float64)
    def field(q, p, attrs):
        return q + 2*p, -3*q-p.square()
    h = .2
    q, p = previous[..., :2], previous[..., 2:]
    p1 = p + h*field(q,p,None)[1]
    p2 = p + h*field(q,p1,None)[1]
    expected = torch.cat((q+h*field(q,p2,None)[0], p2), -1)
    actual = candidate_function('symeuler2',field,h)(None,None,previous,None)
    torch.testing.assert_close(actual,expected)
