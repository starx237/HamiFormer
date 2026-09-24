from hamiformer.utils.paths import project_root
import copy
import re
import time
from types import MethodType
import torch
from torch.nn import functional as F

class GraphCall:

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args):
        key = tuple(((tuple(v.shape), tuple(v.stride()), v.dtype) for v in args))
        if key not in self.cache:
            aa = tuple((v.detach().clone() for v in args))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.fn(*aa)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                values = self.fn(*aa)
            self.cache[key] = (aa, graph, values)
        aa, graph, values = self.cache[key]
        for dst, src in zip(aa, args):
            dst.copy_(src)
        graph.replay()
        return tuple((v.clone() for v in values))

class Hami2Runtime:

    def __init__(self, collector, mode='exact'):
        self.c = collector
        self.mode = mode
        self.stats = {}
        self.count = 0
        self.old_hess = None
        self.h = copy.deepcopy(collector.h)

        def validate(obj, q, p, c):
            if q.shape != p.shape or q.shape[-1] != obj.state_dim:
                raise ValueError('H2 state shape')
            if c.shape != (*q.shape[:-1], obj.num_objects, obj.num_objects, 10):
                raise ValueError('H2 graph shape')
        self.h._validate = MethodType(validate, self.h)
        from hamiformer.inference.derivatives import CompiledDerivatives
        step = collector.frame_dt
        self.deriv = CompiledDerivatives(self.h, step, plain_norm=True)
        self.grad = CompiledDerivatives(self.h, step, plain_norm=True, only_gradient=True)
        self.deriv_graph = GraphCall(lambda q, p, c: self._chunks(q, p, c))
        self.grad_graph = GraphCall(self.grad)
        self.scan_graph = GraphCall(self.scan)
        self.edge = torch.compile(self.edge_forward, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        self.static_cache = {}
        self.period = int(re.search('lag(\\d+)', mode).group(1)) if 'lag' in mode else 1
        self.stats['hessian_refresh_interval'] = self.period
        if int(collector.cfg['hamiltonian']['plas_substeps_per_frame']) != 1:
            raise ValueError('only one physical substep is supported')

    def reset(self):
        self.count = 0
        self.old_hess = None

    def _chunks(self, q, p, c):
        pieces = [self.deriv(q[i:i + 128], p[i:i + 128], c[i:i + 128]) for i in range(0, len(q), 128)]
        return tuple((torch.cat([part[j] for part in pieces], 0) for j in range(5)))

    def tree_leaf(self, x):
        tree = self.c.model.tree

        def visit(node):
            leaf = int(tree['node_to_leaf'][node])
            if leaf >= 0:
                return torch.full_like(x[..., 0], leaf, dtype=torch.long)
            return torch.where(x[..., int(tree['feature'][node])] <= float(tree['threshold'][node]), visit(int(tree['children_left'][node])), visit(int(tree['children_right'][node])))
        return visit(0)

    def edge_forward(self, noisy, d, token, matrix, offset, previous, previous_gate, gate_hidden, x0, static, tau, time, mask, initial):
        m = self.c.model
        b, o, _ = previous.shape
        scale = m.phase_scale
        flat = torch.cat((previous[..., :3].reshape(b, -1), previous[..., 3:].reshape(b, -1)), -1)
        raw = (matrix @ flat.unsqueeze(-1)).squeeze(-1) + offset
        h = torch.cat((raw[..., :3 * o].reshape(b, o, 3), raw[..., 3 * o:].reshape(b, o, 3)), -1) / scale
        noisy = noisy / scale
        dn = d / scale
        prev = previous / scale
        x0n = x0 / scale
        base, hidden = m.common_r(m._r_features(token, noisy, dn, h, prev, x0n, static, tau, time, previous_gate))
        from hamiformer.models.hamiballs2_posthd_formal import formal_observable_features
        obs = formal_observable_features(noisy, dn, h, prev, base, hidden, previous_gate, x0=x0n, static=static, tau=tau, physical_fraction=time)
        leaf = self.tree_leaf(obs)
        design = torch.cat((((obs - m.ridge_feature_mean) / m.ridge_feature_scale).clamp(-8, 8), torch.ones_like(obs[..., :1])), -1)
        selected = m.ridge_weight[leaf]
        leaf_r = torch.cat(tuple((torch.matmul(design.unsqueeze(-2), selected[..., j, :, :]).squeeze(-2) for j in range(2))), -1)
        if initial:
            leaf_r = torch.zeros_like(leaf_r)
        alpha = m.component_alpha.repeat_interleave(3)
        r = (1 - alpha) * base + alpha * leaf_r
        hr = (h + r) * scale
        inp = m._gate_features(token, noisy, dn, h, hr / scale, prev, static, tau, time, previous_gate, hidden).reshape(b * o, -1)
        router = m.router
        v = F.silu(router.input(inp))
        cell = router.temporal
        ir, iz, inn = F.linear(v, cell.weight_ih, cell.bias_ih).chunk(3, -1)
        hr0, hz, hn = F.linear(gate_hidden, cell.weight_hh, cell.bias_hh).chunk(3, -1)
        reset = torch.sigmoid(ir + hr0)
        update = torch.sigmoid(iz + hz)
        new = torch.tanh(inn + reset * hn)
        gh = (1 - update) * new + update * gate_hidden
        logits = router.scalar(gh).expand(-1, 2) + router.qp(gh)
        leaf_features = design[..., :-1] if m.leaf_gate_standardized else obs
        obsflat = leaf_features.reshape(b * o, -1)
        gd = torch.cat((obsflat, torch.ones_like(obsflat[..., :1])), -1)
        logits = logits + (router.leaf_affine[leaf.reshape(-1)] @ gd.unsqueeze(-1)).squeeze(-1)
        gate = torch.sigmoid(logits).reshape(b, o, 2) * mask[..., None]
        gate6 = gate.repeat_interleave(3, -1)
        mixed = (gate6 * hr + (1 - gate6) * d) * mask[..., None]
        return (mixed, gate, gh)

    def scan(self, state, d, tokens, matrix, offset, x0, static, tau, times, mask):
        b, e, o, _ = state.shape
        prev = x0
        pg = state.new_ones(b, o, 2)
        hidden = state.new_zeros(b * o, self.c.model.router.hidden)
        values = []
        denom = (times[:, -1] - times[:, 0]).clamp_min(1e-06)
        for k in range(e):
            prev, pg, hidden = self.edge(state[:, k], d[:, k], tokens[:, k], matrix[:, k], offset[:, k], prev, pg, hidden, x0, static, tau, (times[:, k] - times[:, 0]) / denom, mask, k == 0)
            values.append(prev)
        return (torch.stack(values, 1),)

    def jets(self, d, batch, context, anchor=None):
        profiling = getattr(self, 'profile', False)
        if profiling:
            torch.cuda.synchronize()
        stamp = time.perf_counter()

        def mark(name):
            nonlocal stamp
            if profiling:
                torch.cuda.synchronize()
                now = time.perf_counter()
                self.stats.setdefault('jet_profile', {}).setdefault(name, []).append(now - stamp)
                stamp = now
        b, e, o, _ = d.shape
        source = torch.cat((batch['phase'][:, :1].float(), d[:, :-1]), 1)
        if anchor is None:
            anchor_q, anchor_p = (source[..., :3], d[..., 3:])
        else:
            anchor_q, anchor_p = anchor
            if anchor_q.shape != (b, e, o, 3) or anchor_p.shape != (b, e, o, 3):
                raise ValueError('H2 anchor shape')
        q = anchor_q.reshape(-1, 3 * o)
        p = anchor_p.reshape(-1, 3 * o)
        ctx = context[:, None].expand(-1, e, -1, -1, -1).reshape(b * e, o, o, 10)
        if self.count % self.period == 0 or self.old_hess is None:
            sp, sq, qq, qp, pp = self.deriv_graph(q, p, ctx)
            self.old_hess = (qq, qp, pp)
            self.stats['full_hessian_calls'] = self.stats.get('full_hessian_calls', 0) + 1
        else:
            sp, sq = self.grad_graph(q, p, ctx)
            qq, qp, pp = self.old_hess
        self.count += 1
        mark('derivatives_and_anchors')
        dim = 3 * o
        eye = torch.eye(dim, device=d.device, dtype=d.dtype).expand_as(qp)
        inverse, info = torch.linalg.solve_ex(qp, eye)
        inverse_qq, info2 = torch.linalg.solve_ex(qp, qq)
        matrix = torch.cat((torch.cat((qp.transpose(-1, -2) - pp @ inverse_qq, pp @ inverse), -1), torch.cat((-inverse_qq, inverse), -1)), -2)
        valid = torch.isfinite(q).all() & torch.isfinite(p).all() & torch.isfinite(ctx).all() & (info == 0).all() & (info2 == 0).all()
        if not bool(valid):
            raise FloatingPointError('invalid H2 affine solve/input')
        sg = torch.cat((q, sp), -1)
        tg = torch.cat((sq, p), -1)
        offset = tg - (matrix @ sg.unsqueeze(-1)).squeeze(-1)
        mark('linear_solve_and_offset')
        cfg = self.c.cfg['hamiltonian']
        delta = (qp - eye).abs()
        rho = (delta.sum(-1).amax(-1) * delta.sum(-2).amax(-1)).sqrt()
        low = 1 - rho
        high = 1 + rho
        certified = (low >= float(cfg['mixed_singular_floor']) + 1e-05) & (high / low.clamp_min(1e-12) <= float(cfg['mixed_condition_limit']) - 0.0001)
        healthy = certified.clone()
        if not bool(certified.all()):
            singular = torch.linalg.svdvals(qp[~certified])
            lo = singular.amin(-1)
            hi = singular.amax(-1)
            healthy[~certified] = (lo >= float(cfg['mixed_singular_floor'])) & (hi / lo <= float(cfg['mixed_condition_limit'])) & torch.isfinite(singular).all(-1)
        healthy = healthy & torch.isfinite(matrix).all((-2, -1))
        mark('mixed_health')
        if self.c.exact_tangent_gate:
            if 'chol' in self.mode:
                from hamiformer.inference.spectral import tangent_health
                tangent = tangent_health(matrix, float(cfg['tangent_diagnostic_limit']), self.stats)
            else:
                from hamiformer.inference.mixed_rollout import _tangent_health_with_certified_bounds
                tangent, ambiguous = _tangent_health_with_certified_bounds(matrix, float(cfg['tangent_diagnostic_limit']))
            healthy = healthy & tangent
        mark('tangent_health')
        reset = source + 1.0 * (d - source)
        reset_flat = torch.cat((reset[..., :3].flatten(2), reset[..., 3:].flatten(2)), -1).reshape(b * e, 6 * o)
        matrix = torch.where(healthy[:, None, None], matrix, torch.zeros_like(matrix))
        offset = torch.where(healthy[:, None], offset, reset_flat)
        self.stats['d_reset_maps'] = self.stats.get('d_reset_maps', 0) + int((~healthy).sum())
        return (matrix.reshape(b, e, 6 * o, 6 * o), offset.reshape(b, e, 6 * o))

    @torch.no_grad()
    def field_with_anchor(self, state, tau, batch, anchor=None):
        from hamiformer.models.hamiballs2_hamiltonian import graph_context
        from hamiformer.models.hamiballs2_dual_expert import hamiballs2_node_graph_features
        key = id(batch)
        if key not in self.static_cache:
            node = hamiballs2_node_graph_features(batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float())
            context = graph_context(batch['attrs'].float(), batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float())
            model = self.c.model
            mask = batch['object_mask'].to(state)
            static = (torch.cat((batch['attrs'].float(), node), -1) - model.static_feature_mean) / model.static_feature_scale
            self.static_cache[key] = (batch, context, static * mask[..., None])
        _, context, static = self.static_cache[key]
        d, tokens = self.c._wide(state, tau, batch)
        matrix, offset = self.jets(d, batch, context, anchor=anchor)
        result = self.scan_graph(state, d, tokens, matrix, offset, batch['phase'][:, 0].float(), static, tau, batch['time'][:, 1:].float(), batch['object_mask'].to(state))[0]
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('nonfinite H2 mixed field')
        b, e, o, _ = result.shape
        predecessor = torch.cat((batch['phase'][:, :1], result[:, :-1]), 1)
        flat_previous = torch.cat((predecessor[..., :3].reshape(b, e, -1), predecessor[..., 3:].reshape(b, e, -1)), -1)
        h_target = (torch.matmul(matrix, flat_previous.unsqueeze(-1)).squeeze(-1) + offset)[..., 3 * o:].reshape(b, e, o, 3)
        return (result, (predecessor[..., :3].detach(), h_target.detach()))

    @torch.no_grad()
    def field(self, state, tau, batch):
        return self.field_with_anchor(state, tau, batch)[0]
SUPPORTED_PROFILES = {'exact': 'exact-chol', 'lagged_hessian4': 'lag4-chol'}

@torch.no_grad()
def sample_hami2_stateful_pf_rf_heun(runtime, source, batch, *, num_steps=20, t_eps=0.05):
    from hamiformer.flow.rectified_flow import clean_to_velocity
    if int(num_steps) < 2:
        raise ValueError('H2 sampling requires at least two RF intervals')
    runtime.reset()
    state = source.clone()
    committed = None
    cold_intervals = min(2, int(num_steps) - 1)
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=state.device, dtype=state.dtype)
    for index in range(num_steps):
        left, right = grid[index:index + 2]
        tau = left.expand(len(state))
        if index < cold_intervals:
            clean, _, next_anchor = runtime.c._d_only_field(state, tau, batch)
        else:
            clean, next_anchor = runtime.field_with_anchor(state, tau, batch, committed)
        velocity = clean_to_velocity(clean, state, tau, t_eps=t_eps)
        if index == num_steps - 1:
            state = state + (right - left) * velocity
            break
        proposal = state + (right - left) * velocity
        if index < cold_intervals:
            right_clean, _, _ = runtime.c._d_only_field(proposal, right.expand(len(state)), batch)
        else:
            right_clean, _ = runtime.field_with_anchor(proposal, right.expand(len(state)), batch, committed)
        right_velocity = clean_to_velocity(right_clean, proposal, right.expand(len(state)), t_eps=t_eps)
        state = state + 0.5 * (right - left) * (velocity + right_velocity)
        committed = next_anchor
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError('stateful H2 compiled sampler produced NaN/Inf')
    return state
