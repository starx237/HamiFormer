from hamiformer.utils.paths import project_root
import torch
import re
from torch import nn
from torch.nn import functional as F

class SingleStepGRU(nn.Module):

    def __init__(self, gru):
        super().__init__()
        if gru.num_layers != 1 or gru.bidirectional or gru.dropout != 0:
            raise ValueError('only the frozen one-layer GRU is supported')
        for name in ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'):
            self.register_buffer(name, getattr(gru, name).detach())
        self.hidden_size = gru.hidden_size

    def forward(self, x, h=None):
        h = x.new_zeros(x.shape[0], self.hidden_size) if h is None else h[0]
        xi = F.linear(x[:, 0], self.weight_ih_l0, self.bias_ih_l0)
        hi = F.linear(h, self.weight_hh_l0, self.bias_hh_l0)
        ir, iz, inn = xi.chunk(3, -1)
        hr, hz, hn = hi.chunk(3, -1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        new = torch.tanh(inn + reset * hn)
        result = new + update * (h - new)
        return (result[:, None], result[None])

class CompiledScan:

    def __init__(self, runtime):
        self.runtime = runtime
        self.r = runtime.models['r']
        self.g = runtime.models['gate']
        self.g.base_gate.temporal = SingleStepGRU(self.g.base_gate.temporal)
        self.checked = False
        self.preproject = 'preproj' in runtime.mode
        if self.preproject:
            self.token_projection = self.g.base_gate.token_adapter
            self.g.base_gate.token_adapter = nn.Identity()
            self.static_projection = torch.compile(self._static_projection, fullgraph=True, options={'triton.cudagraphs': False})
        torch._dynamo.config.cache_size_limit = 64
        self.step = torch.compile(self._step, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        group_match = re.search('group(\\d+)', runtime.mode)
        self.group = int(group_match.group(1)) if group_match else 1
        if self.group > 1:
            self.group_step = torch.compile(self._group, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})

    def _step(self, cold, matrix, offset, previous, previous_g, hidden, context, token, noisy, x0, d, attrs, tau, physical_time, scale, static):
        from hamiformer.physics.hamiballs_type2 import apply_hamiballs_affine_jet
        candidate_step = getattr(self.runtime, 'integrator_candidate', None)
        h = (apply_hamiballs_affine_jet(matrix, offset, previous * scale, q_dim=2) if candidate_step is None else candidate_step(matrix, offset, previous * scale, attrs)) / scale
        route = previous_g.mean(-1) if self.r.per_object_previous_g else previous_g.mean(-1).mean(1)
        if static is None:
            ri, obs, gain, drms, rrms, context = self.r.forward_step_with_context_diagnostics(token, noisy, x0, previous, h, d, attrs, tau, physical_time, route, context=context)
            gate_token = token
        else:
            raw, context = self.r.raw_step_features_with_context(token, noisy, x0, previous, h, d, attrs, tau, physical_time, route, context=context)
            obs = self.r.normalized_features(raw)
            b, n = previous.shape[:2]
            rroute = route if route.ndim == 2 else route[:, None].expand(b, n)
            scalar = torch.stack((tau[:, None].expand(b, n), physical_time[:, None].expand(b, n), 1 - rroute), dim=-1)
            dynamic = torch.cat((noisy, x0, previous, h, d, attrs, scalar), dim=-1)
            feature = F.silu(F.linear(dynamic, self.r.network[0].weight[:, self.r.token_dim:]) + static[..., :32])
            feature = F.silu(self.r.network[2](feature))
            direction = self.r.network[4](feature)
            ri, gain, drms, rrms = self.r._parameterize_direction(direction, feature, disagreement=d - h)
            self.runtime._finite('preprojected residual observable', obs)
            gate_token = static[..., 32:]
        base = h + ri
        gate, hidden, hr, leaf = self.runtime.pure_gate(0 if cold else 1, self.g, gate_token, noisy, x0, previous, h, base, d, attrs, tau, physical_time, previous_g, hidden, residual_hidden=obs)
        weight = torch.cat((gate[..., 0, None].expand(-1, -1, 2), gate[..., 1, None].expand(-1, -1, 2)), dim=-1)
        mixed = d + weight * (hr - d)
        return (h, hr - h, gain, drms, rrms, obs, hr, gate, mixed, base, hidden, context, leaf)

    def _static_projection(self, tokens):
        rvalue = F.linear(tokens, self.r.network[0].weight[:, :self.r.token_dim], self.r.network[0].bias)
        gvalue = F.linear(tokens, self.token_projection.weight, self.token_projection.bias)
        return torch.cat((rvalue, gvalue), dim=-1)

    def _group(self, cold, matrix, offset, previous, pg, hidden, context, tokens, noisy, x0, d, attrs, tau, times, scale, static):
        rows = []
        leaves = []
        for j in range(matrix.shape[1]):
            v = self._step(cold and j == 0, matrix[:, j], offset[:, j], previous, pg, hidden, context, tokens[:, j], noisy[:, j], x0, d[:, j], attrs, tau, times[:, j], scale, None if static is None else static[:, j])
            rows.append(v[:10])
            hidden, context, leaf = v[10:]
            previous = v[8]
            pg = v[7]
            leaves.append(leaf)
        return (*[torch.stack([r[k] for r in rows], 1) for k in range(10)], hidden, context, torch.stack(leaves, 1))

    def __call__(self, jets, **kw):
        from hamiformer.models.hamiballs_committed import HamiBallsCommittedRollout
        for key in ('initial_gate_hidden', 'exogenous_gate', 'gate_policy'):
            if kw.get(key) is not None:
                raise ValueError('unsupported compiled scan control: ' + key)
        if kw['q_dim'] != 2:
            raise ValueError('only PLAS q2/p2 recurrence is supported')
        d = kw['d_candidate']
        x0 = kw['x0']
        b, t, n, _ = d.shape
        previous = kw.get('initial_previous_mixed')
        if previous is None:
            previous = x0
        pg = kw.get('initial_previous_g')
        if pg is None:
            pg = d.new_ones(b, n, 2)
        elif pg.ndim == 2:
            pg = pg[..., None].expand(-1, -1, 2)
        hidden = None
        context = None
        rows = [[] for _ in range(10)]
        prevs = []
        leaves = []
        initial = previous
        static = self.static_projection(kw['d_tokens']) if self.preproject else None
        for edge in range(0, t, self.group):
            prevs.append(previous)
            if self.group > 1:
                sl = slice(edge, min(edge + self.group, t))
                values = self.group_step(edge == 0, jets.matrix[:, sl], jets.offset[:, sl], previous, pg, hidden, context, kw['d_tokens'][:, sl], kw['noisy'][:, sl], x0, d[:, sl], kw['attrs'], kw['tau'], kw['physical_time'][:, sl], kw['state_scale'].reshape(1, 1, -1), None if static is None else static[:, sl])
                for group, val in zip(rows, values[:10]):
                    group.append(val)
                hidden, context, leaf = values[10:]
                previous = values[8][:, -1].contiguous()
                pg = values[7][:, -1].contiguous()
                leaves.append(leaf)
                continue
            edge_matrix = jets.matrix[:, edge]
            edge_offset = jets.offset[:, edge]
            separate_candidate = getattr(self.runtime, 'separate_candidate', None)
            if separate_candidate is not None:
                edge_offset = separate_candidate(edge_matrix, edge_offset, previous * kw['state_scale'].reshape(1, 1, -1), kw['attrs'])
            step_args = (edge == 0, edge_matrix, edge_offset, previous, pg, hidden, context, kw['d_tokens'][:, edge], kw['noisy'][:, edge], x0, d[:, edge], kw['attrs'], kw['tau'], kw['physical_time'][:, edge], kw['state_scale'].reshape(1, 1, -1), None if static is None else static[:, edge])
            if not self.checked:
                self._step(*step_args)
                self.checked = True
                print('EAGER_FUNCTIONAL_EDGE_OK', flush=True)
            values = self.step(*step_args)
            for group, val in zip(rows, values[:10]):
                group.append(val)
            hidden, context, leaf = values[10:]
            previous = values[8]
            pg = values[7]
            leaves.append(leaf)
        rows = [torch.cat(row, dim=1) if self.group > 1 else torch.stack(row, dim=1) for row in rows]
        h, ri, gain, drms, rrms, obs, hr, gate, mixed, base = rows
        predecessors = torch.cat((initial[:, None], mixed[:, :-1]), 1) if self.group > 1 else torch.stack(prevs, 1)
        self.runtime.last_leaf = torch.cat(leaves, 1) if self.group > 1 else torch.stack(leaves, 1)
        return HamiBallsCommittedRollout(h_candidate=h, innovation=ri, residual_gain=gain, residual_direction_rms=drms, innovation_rms=rrms, residual_hidden=obs, hr_candidate=hr, d_candidate=d, gate=gate, mixed=mixed, previous_mixed=predecessors, final_state=previous, final_previous_g=pg.mean(-1), gate_hidden=None, final_residual_context=context, base_hr_candidate=base, base_gate=gate)
