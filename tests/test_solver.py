import pytest
import torch
from hamiformer.integrators.solver import METHODS, Solver

@pytest.mark.parametrize('method', METHODS)
def test_harmonic_step(method):
    state = torch.tensor([0.7, 0.2], dtype=torch.float64)
    energy = lambda z: z.square().sum() / 2
    result = Solver(method).step(energy, state, 0.1, generator=torch.Generator().manual_seed(42))
    expected = torch.tensor([0.72, 0.13] if method == 'explicit_euler' else [0.713, 0.13], dtype=torch.float64)
    torch.testing.assert_close(result, expected)

@pytest.mark.parametrize('method', ('plas', 'plas_s', 'plas_dc', 'plas_dc_f', 'plas_dc_s'))
def test_coupled_quadratic(method):
    matrix = torch.tensor([[2.0, 0.3, 0.2, 0.1], [0.3, 3.0, -0.1, 0.4], [0.2, -0.1, 1.0, 0.2], [0.1, 0.4, 0.2, 2.0]], dtype=torch.float64)
    state = torch.tensor([0.2, -0.3, 0.4, 0.1], dtype=torch.float64)
    anchor = state + 0.15
    energy = lambda z: z @ matrix @ z / 2
    h = 0.05
    pp = torch.linalg.solve(torch.eye(2, dtype=state.dtype) + h * matrix[:2, 2:], state[2:] - h * matrix[:2, :2] @ state[:2])
    expected = torch.cat((state[:2] + h * (matrix[2:, :2] @ state[:2] + matrix[2:, 2:] @ pp), pp))
    torch.testing.assert_close(Solver(method).step(energy, state, h, anchor), expected)

def test_sparse_distinct_anchor_counter():
    solver = Solver('plas_s', refresh_interval=2)
    state = torch.tensor([0.4, 0.2], dtype=torch.float64)
    energy = lambda z: z.pow(4).sum()
    solver.step(energy, state, 0.01, state)
    first = solver._hessian.clone()
    solver.step(energy, state, 0.01, state.clone())
    assert solver._calls == 1
    solver.step(energy, state, 0.01, state + 0.1)
    torch.testing.assert_close(solver._hessian, first)
    solver.step(energy, state, 0.01, state + 0.2)
    assert not torch.equal(solver._hessian, first)

def test_frozen_jet_keeps_anchor():
    solver = Solver('plas_dc_f')
    state = torch.tensor([0.4, 0.2], dtype=torch.float64)
    energy = lambda z: z.pow(4).sum()
    expected = solver.step(energy, state, 0.01, state)
    actual = solver.step(energy, state, 0.01, state + 0.2)
    torch.testing.assert_close(actual, expected)
