import pytest
import torch

from hamiformer.integrators.solver import Solver


def energy(z):
    weights = torch.arange(1, z.numel() + 1, dtype=z.dtype, device=z.device)
    return (weights * z.square()).sum() / 2 + 0.1 * torch.sin(z.sum())


def reference_step(state, anchor, h, rank=2, probes=4, seed=42):
    generator = torch.Generator(device=anchor.device).manual_seed(seed)
    signs = (2 * torch.randint(0, 2, (anchor.numel(), probes),
                              device=anchor.device, generator=generator) - 1).to(anchor)
    omega = torch.randn(anchor.numel(), min(rank, anchor.numel()),
                        device=anchor.device, dtype=anchor.dtype, generator=generator)
    hessian = torch.func.hessian(energy)(anchor)
    diagonal = (signs * (hessian @ signs)).mean(1)
    remainder = hessian - torch.diag(diagonal)
    basis = torch.linalg.qr(remainder @ omega, mode='reduced').Q
    core = basis.T @ remainder @ basis
    curvature = torch.diag(diagonal) + basis @ ((core + core.T) / 2) @ basis.T
    gradient = torch.func.grad(energy)(anchor)
    m = state.numel() // 2
    dq = state[:m] - anchor[:m]
    dp = torch.linalg.solve(torch.eye(m, dtype=anchor.dtype, device=anchor.device)
                            + h * curvature[:m, m:],
                            state[m:] - anchor[m:] - h * (gradient[:m] + curvature[:m, :m] @ dq))
    return torch.cat((state[:m] + h * (gradient[m:] + curvature[m:, :m] @ dq
                                     + curvature[m:, m:] @ dp), anchor[m:] + dp))


def test_dlr_default_matches_four_probe_dense_reference():
    state = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64)
    anchor = state + 0.12
    solver = Solver('plas_dlr')
    assert (solver.rank, solver.diagonal_probes) == (2, 4), 'DLR defaults must match the paper'
    torch.testing.assert_close(solver.step(energy, state, 0.03, anchor),
                               reference_step(state, anchor, 0.03), atol=1e-12, rtol=1e-12)


def test_dlr_reuses_directions_after_anchor_changes_and_chunk_reset():
    state = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64)
    generator = torch.Generator().manual_seed(73)
    solver = Solver('plas_dlr')
    solver.step(energy, state, 0.02, state, generator)
    rng_after_initialization = generator.get_state().clone()
    for anchor in (state + 0.1, state - 0.2, state + 0.3):
        torch.testing.assert_close(solver.step(energy, state, 0.02, anchor, generator),
                                   reference_step(state, anchor, 0.02, seed=73), atol=1e-12, rtol=1e-12)
        solver.reset()
    assert torch.equal(generator.get_state(), rng_after_initialization), 'Steps and resets must not redraw probes'


def test_dlr_default_directions_do_not_consume_global_rng():
    state = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    first = Solver('plas_dlr').step(energy, state, 0.03, state + 0.1)
    second = Solver('plas_dlr').step(energy, state, 0.03, state + 0.1)
    assert torch.equal(before, torch.random.get_rng_state()), 'Default probes must use an isolated generator'
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_dlr_fixed_anchor_map_is_symplectic():
    state = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64)
    anchor = state + 0.1
    solver = Solver('plas_dlr')
    solver.step(energy, state, 0.03, anchor)
    jacobian = torch.func.jacrev(lambda z: solver.step(energy, z, 0.03, anchor))(state)
    eye = torch.eye(3, dtype=state.dtype)
    zero = torch.zeros_like(eye)
    canonical = torch.cat((torch.cat((zero, eye), 1), torch.cat((-eye, zero), 1)), 0)
    torch.testing.assert_close(jacobian.T @ canonical @ jacobian, canonical, atol=1e-12, rtol=1e-12)


def test_dlr_full_rank_matches_dense_plas():
    state = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64)
    anchor = state + 0.1
    actual = Solver('plas_dlr', rank=6).step(energy, state, 0.03, anchor)
    expected = Solver('plas').step(energy, state, 0.03, anchor)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_dlr_directions_follow_state_shape_and_dtype(dtype):
    solver = Solver('plas_dlr')
    for dimension in (6, 8, 6):
        state = torch.linspace(-0.3, 0.5, dimension, dtype=dtype)
        actual = solver.step(energy, state, 0.03, state + 0.1)
        torch.testing.assert_close(actual, reference_step(state, state + 0.1, 0.03))
