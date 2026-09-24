from dataclasses import dataclass
import torch
METHODS = ('explicit_euler', 'symeuler2', 'plas', 'plas_s', 'plas_dc', 'plas_dc_f', 'plas_dc_s', 'plas_dlr')

@dataclass
class Solver:
    method: str = 'plas'
    refresh_interval: int = 16
    rank: int = 2
    diagonal_probes: int = 4

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(self.method)
        if min(self.refresh_interval, self.rank, self.diagonal_probes) < 1:
            raise ValueError('positive solver settings required')
        self._dlr_directions = {}
        self.reset()

    def reset(self):
        """Clear anchor-dependent state while retaining fixed DLR directions."""
        self._hessian = None
        self._calls = 0
        self._anchor = None
        self._gradient = None

    def _probe_directions(self, anchor, generator):
        key = (anchor.numel(), anchor.device, anchor.dtype, self.rank, self.diagonal_probes)
        if key not in self._dlr_directions:
            if generator is None:
                generator = torch.Generator(device=anchor.device).manual_seed(42)
            signs = (2 * torch.randint(0, 2, (anchor.numel(), self.diagonal_probes),
                                     device=anchor.device, generator=generator) - 1).to(anchor)
            omega = torch.randn(anchor.numel(), min(self.rank, anchor.numel()),
                                device=anchor.device, dtype=anchor.dtype, generator=generator)
            self._dlr_directions[key] = (signs, omega)
        return self._dlr_directions[key]

    def step(self, energy, state, step_size, anchor=None, generator=None):
        if state.ndim != 1 or state.numel() % 2:
            raise ValueError('canonical state must have shape [2d]')
        d = state.numel() // 2
        q, p = (state[:d], state[d:])
        h = step_size
        grad = torch.func.grad(energy)
        if self.method == 'explicit_euler':
            g = grad(state)
            return torch.cat((q + h * g[d:], p - h * g[:d]))
        if self.method == 'symeuler2':
            pp = p
            for _ in range(2):
                pp = p - h * grad(torch.cat((q, pp)))[:d]
            return torch.cat((q + h * grad(torch.cat((q, pp)))[d:], pp))
        a = state if anchor is None else anchor
        if a.shape != state.shape:
            raise ValueError('anchor and state shapes must match')
        if self.method == 'plas_dc_f' and self._anchor is not None:
            a = self._anchor
        g = grad(a)
        qa, pa = (a[:d], a[d:])
        dq = q - qa
        if self.method == 'plas_dlr':
            _, pullback = torch.func.vjp(grad, a)
            action = torch.func.vmap(lambda v: pullback(v)[0], in_dims=1, out_dims=1)
            signs, omega = self._probe_directions(a, generator)
            diagonal = (signs * action(signs)).mean(1)
            basis = torch.linalg.qr(action(omega) - diagonal[:, None] * omega, mode='reduced').Q
            core = basis.T @ (action(basis) - diagonal[:, None] * basis)
            core = (core + core.T) / 2
            u, v = (basis[:d], basis[d:])
            rhs = p - pa - h * (g[:d] + diagonal[:d] * dq + u @ (core @ (u.T @ dq)))
            small = torch.eye(core.shape[0], device=a.device, dtype=a.dtype) + h * (v.T @ u) @ core
            dp = rhs - h * u @ (core @ torch.linalg.solve(small, v.T @ rhs))
            return torch.cat((q + h * (g[d:] + diagonal[d:] * dp + v @ (core @ (u.T @ dq + v.T @ dp))), pa + dp))
        refresh = self.method not in ('plas_s', 'plas_dc_s', 'plas_dc_f') or self._hessian is None
        if self.method in ('plas_s', 'plas_dc_s'):
            distinct = self._anchor is None or not torch.equal(a, self._anchor)
            refresh = refresh or (distinct and self._calls % self.refresh_interval == 0)
        else:
            distinct = True
        if refresh:
            self._hessian = torch.func.hessian(energy)(a)
        if distinct:
            self._calls += 1
        self._anchor = a.detach().clone()
        hh = self._hessian
        if hh.shape != (2 * d, 2 * d):
            raise ValueError('reset solver when system dimension changes')
        b = torch.eye(d, device=a.device, dtype=a.dtype) + h * hh[:d, d:]
        pp = torch.linalg.solve(b, p - h * (g[:d] + hh[:d, :d] @ dq - hh[:d, d:] @ pa))
        if self.method in ('plas_dc', 'plas_dc_f', 'plas_dc_s'):
            current = grad(torch.cat((q, pp)))
            if self.method in ('plas_dc', 'plas_dc_s'):
                # Accumulate the defect and correction in double precision.
                inverse = torch.linalg.solve(b, torch.eye(d, device=b.device, dtype=b.dtype))
                response = torch.cat((h * hh[d:, d:] @ inverse, inverse), dim=0).double()
                defect = p.double() - h * current[:d].double() - pp.double()
                delta = response @ defect
                return torch.cat(((q.double() + h * current[d:].double() + delta[:d]).to(q.dtype),
                                  (pp.double() + delta[d:]).to(pp.dtype)))
            delta = torch.linalg.solve(b, p - h * current[:d] - pp)
            return torch.cat((q + h * current[d:] + h * hh[d:, d:] @ delta, pp + delta))
        return torch.cat((q + h * (g[d:] + hh[d:, :d] @ dq + hh[d:, d:] @ (pp - pa)), pp))

    def install_h1(self, models, *, compiled=True):
        """Install the batched candidate and window construction in H1 inference.

        Use one solver per process. CUDA execution captures derivative and mixed
        scan graphs; CPU execution uses the same candidate equations eagerly.
        """
        from .runtime import install_h1
        return install_h1(self, models, compiled=compiled)

    def bind_dlr(self, hamiltonian, attr_scale, step_size, *, compiled=True):
        """Create batched DLR window factors without installing a model pipeline.

        The Hamiltonian accepts flattened q/p and normalized object context.
        Build factors once for a window anchor, then call apply for each edge.
        """
        if self.method != 'plas_dlr':
            raise ValueError('bind_dlr requires the plas_dlr method')
        from .window import DLRWindow
        return DLRWindow(hamiltonian, attr_scale, step_size, rank=self.rank,
                         probes=self.diagonal_probes, compiled=compiled)

    def install_h2(self, collector):
        """Install the HamiBalls-2 PLAS window runtime."""
        if self.method != 'plas':
            raise ValueError('HamiBalls-2 inference supports PLAS')
        from hamiformer.inference.hamiballs2 import Hami2Runtime
        return Hami2Runtime(collector, 'exact-chol')

def rollout(energy, initial, steps, step_size, method='plas', anchors=None, **kwargs):
    solver = Solver(method, **kwargs)
    state = initial
    states = []
    for i in range(steps):
        state = solver.step(energy, state, step_size, None if anchors is None else anchors[i])
        states.append(state)
    return torch.stack(states)
