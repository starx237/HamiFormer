from hamiformer.utils.paths import project_root
from dataclasses import fields, is_dataclass, replace
from types import MethodType
import sys
import time
import inspect
import re
import torch

def _map(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if is_dataclass(value) and (not isinstance(value, type)):
        return replace(value, **{f.name: _map(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple((_map(v, fn) for v in value))
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    return value

def _tensor_list(value):
    result = []
    _map(value, lambda x: result.append(x))
    return result

class PLASRuntime:

    def __init__(self, models, mode):
        self.models = models
        self.mode = mode
        self.stats = {}
        self.cache = {}
        self.derivative_cache = {}
        self.prefetching = False
        self.pending_checks = []
        self.pending_health = None
        self.gradient_cache = {}
        self.derivative_ordinal = 0
        self.lagged_hessian = None
        self._checks = None
        self.original_finite = []
        from hamiformer.models import hamiballs_committed as committed
        original = committed._finite
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module, '_finite', None) is original:
                self.original_finite.append((module, original))
                module._finite = self._finite
        gate = models['gate']

        def leaves(obj, features):
            template = next(iter(features.values()))

            def visit(node):
                if 'leaf' in node:
                    return torch.full_like(template, int(node['leaf']), dtype=torch.long)
                return torch.where(features[str(node['feature'])] <= float(node['threshold']), visit(node['left']), visit(node['right']))
            return visit(obj.tree)
        gate._leaves = MethodType(leaves, gate)
        self._prepare_gate(gate)
        from hamiformer.training import hamiballs_formal as training
        self.original_scale_views = training._state_scale_views

        def scale_views(scale, state):
            scale = scale.to(state)
            return (scale.reshape(1, 1, -1), scale.reshape(1, 1, 1, -1))
        training._state_scale_views = scale_views
        if not bool(torch.isfinite(models['state_scale']).all() & (models['state_scale'] > 0).all()):
            raise ValueError('invalid immutable state scale')
        self.original_scan = training.rollout_with_affine_jets
        if 'compile' in mode:
            from hamiformer.inference.scan import CompiledScan
            self.original_scan = CompiledScan(self)
        original_symbol = training.rollout_with_affine_jets
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module, 'rollout_with_affine_jets', None) is original_symbol:
                module.rollout_with_affine_jets = self.scan
        if mode not in ('graph-scan', 'compile-scan'):
            self._prepare_derivatives()
        if 'bounds' in mode:
            from hamiformer.inference.health import install
            install(self)
        if 'cache' in mode or 'overlap' in mode:
            self._prepare_jet_cache()
        if 'overlap' in mode:
            self._prepare_overlap()

    def _prepare_jet_cache(self):
        from hamiformer.training import hamiballs_formal as training
        original = training.learned_hamiballs_affine_jets
        self.jet_entries = {}

        def cached(generator, anchor, attrs, **kwargs):

            def tensor_key(t):
                return (t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()), t.dtype)
            signature = (id(generator), tensor_key(anchor.source_q), tensor_key(anchor.target_p), tensor_key(attrs), tuple(((k, id(v) if torch.is_tensor(v) else v) for k, v in kwargs.items())))
            if signature in self.jet_entries:
                self.stats['jet_cache_hits'] = self.stats.get('jet_cache_hits', 0) + 1
                entry = self.jet_entries[signature]
                if not self.prefetching:
                    self._join_prefetch(entry)
                return entry['value']
            value = original(generator, anchor, attrs, **kwargs)
            entry = {'refs': (generator, anchor, attrs), 'value': value, 'checks': self.pending_checks, 'health': self.pending_health}
            self.pending_checks = []
            self.pending_health = None
            self.jet_entries[signature] = entry
            if len(self.jet_entries) > 3:
                self.jet_entries.pop(next(iter(self.jet_entries)))
            self.stats['jet_cache_misses'] = self.stats.get('jet_cache_misses', 0) + 1
            return value
        self.cached_jets = cached
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module, 'learned_hamiballs_affine_jets', None) is original:
                module.learned_hamiballs_affine_jets = cached

    def _join_prefetch(self, entry):
        if not entry['checks'] and entry['health'] is None:
            return
        torch.cuda.current_stream().wait_stream(self.h_stream)
        if entry['checks'] and (not bool(torch.stack(entry['checks']).all())):
            raise RuntimeError('nonfinite H input or failed affine linear solve')
        entry['checks'] = []
        if entry['health'] is not None:
            health = entry['health']()
            entry['health'] = None
            entry['value'] = replace(entry['value'], health=health)
        _map(entry['value'], lambda t: t.record_stream(torch.cuda.current_stream()))

    def _prepare_overlap(self):
        from hamiformer.physics import generic_type2 as gt
        from hamiformer.evaluation import hamiballs_formal as ev
        self.h_stream = torch.cuda.Stream()
        if 'bounds' not in self.mode:
            raise ValueError('overlap requires deferred bound/SVD health')

        def linearize(qq, qp, pp):
            dim = qp.shape[-1]
            identity = torch.eye(dim, device=qp.device, dtype=qp.dtype).expand(*qp.shape[:-2], dim, dim)
            inverse, info1 = torch.linalg.solve_ex(qp, identity, check_errors=False)
            inverse_qq, info2 = torch.linalg.solve_ex(qp, qq, check_errors=False)
            valid = (info1 == 0).all() & (info2 == 0).all()
            if self.prefetching:
                self.pending_checks.append(valid)
            elif not bool(valid):
                raise RuntimeError('affine linear solve failed')
            top_left = qp.transpose(-1, -2) - pp @ inverse_qq
            top_right = pp @ inverse
            return (torch.cat((torch.cat((top_left, top_right), -1), torch.cat((-inverse_qq, inverse), -1)), -2), qp)
        gt._linearization_from_type2_second_derivatives = linearize
        original = ev.sample_stateful_pf_rf_v1_heun

        def sampler(field, source, *args, **kwargs):
            self.derivative_ordinal = 0
            self.lagged_hessian = None
            closure = inspect.getclosurevars(field).nonlocals
            if closure.get('continuous_integrator_method') is not None or closure.get('hamiltonian') is None:
                raise ValueError('overlap supports learned affine-H field only')
            jet_kw = {k: closure[k] for k in ('attr_scale', 'step_size', 'mixed_singular_floor', 'mixed_condition_limit', 'tangent_spectral_norm_limit')}
            jet_kw['differentiable'] = closure['differentiable_h_jets']
            count = 0
            total_calls = 2 * (kwargs['num_steps'] - 1 - kwargs['cold_start_intervals']) + 1

            def prefetch(anchor):
                self.h_stream.wait_stream(torch.cuda.current_stream())
                self.prefetching = True
                try:
                    with torch.cuda.stream(self.h_stream):
                        self.cached_jets(closure['hamiltonian'], anchor, closure['attrs'], **jet_kw)
                finally:
                    self.prefetching = False

            def overlapped(state, tau, anchor):
                nonlocal count
                prefetch(anchor)
                result = field(state, tau, anchor)
                if 'lookahead' in self.mode and count % 2 == 0 and (count + 1 < total_calls):
                    if kwargs['commit_anchor_from'] != 'accepted_left':
                        raise ValueError('lookahead requires accepted-left anchor')
                    prefetch(result.next_anchor)
                count += 1
                return result
            return original(overlapped, source, *args, **kwargs)
        ev.sample_stateful_pf_rf_v1_heun = sampler

    def _prepare_derivatives(self):
        from hamiformer.physics import generic_type2 as gt
        model = self.models['h']
        self.original_validate = model._validate

        def validate(obj, q, p, context):
            if q.shape != p.shape or q.shape[-1] != obj.state_dim:
                raise ValueError('Hamiltonian state shape mismatch')
            if context.shape != (*q.shape[:-1], obj.num_objects, obj.spatial_tokens, obj.token_context_dim):
                raise ValueError('Hamiltonian context shape mismatch')
        model._validate = MethodType(validate, model)
        self.original_derivatives = gt._type2_second_derivatives
        self.compiled_derivatives = {}
        gt._type2_second_derivatives = self.derivatives

    def derivatives(self, generator, q, p, context, *, step_size, create_graph):
        if create_graph:
            raise ValueError('inference derivative graph cannot train')
        lag = re.search('lag(\\d+)', self.mode)
        if lag:
            period = int(lag.group(1))
            ordinal = self.derivative_ordinal
            self.derivative_ordinal += 1
            self.stats['hessian_refresh_interval'] = period
            if ordinal % period and self.lagged_hessian is not None:
                grads = self.gradients(generator, q, p, context, step_size=step_size)
                self.stats['lagged_hessian_uses'] = self.stats.get('lagged_hessian_uses', 0) + 1
                return (*grads, *self.lagged_hessian)
        key = (tuple(q.shape), tuple(context.shape), str(q.dtype), step_size)
        if key not in self.derivative_cache:
            self.original_validate(q, p, context)
            start = time.perf_counter()
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            if 'tf32' in self.mode:
                torch.backends.cuda.matmul.allow_tf32 = True
                self.stats['h_matmul_policy'] = 'TF32 in captured Hessian only; D/r/g remain FP32'
            aa = tuple((t.detach().clone() for t in (q, p, context)))
            step_tensor = q.new_tensor(step_size)
            compiled = None
            if 'func' in self.mode:
                from hamiformer.inference.derivatives import CompiledDerivatives
                compiled = CompiledDerivatives(generator, step_size, plain_norm='plainln' in self.mode, forward_over_reverse='fwd' in self.mode, batch_ad='batchad' in self.mode, raw_h='rawh' in self.mode, autotune='autotune' in self.mode, point_attention='pointattn' in self.mode, jac_chunk=int(re.search('jchunk(\\d+)', self.mode).group(1)) if re.search('jchunk(\\d+)', self.mode) else None)
                self.compiled_derivatives[key] = compiled
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def run():
                from hamiformer.physics.generic_type2 import _component_jacobian
                if compiled is not None:
                    chunk_match = re.search('hchunk(\\d+)', self.mode)
                    if chunk_match:
                        size = int(chunk_match.group(1))
                        pieces = [compiled(*(v[i:i + size] for v in aa)) for i in range(0, aa[0].shape[0], size)]
                        values = tuple((torch.cat([part[j] for part in pieces], 0) for j in range(5)))
                    else:
                        values = compiled(*aa)
                else:
                    with torch.enable_grad():
                        qv = aa[0].clone().requires_grad_(True)
                        pv = aa[1].clone().requires_grad_(True)
                        s = (qv * pv).sum(dim=-1) + step_tensor * generator(qv, pv, aa[2])
                        sp, sq = torch.autograd.grad(s.sum(), (qv, pv), create_graph=True, retain_graph=True)
                        qq = _component_jacobian(sp, qv, create_graph=False)
                        qp = _component_jacobian(sp, pv, create_graph=False)
                        pp = _component_jacobian(sq, pv, create_graph=False)
                        values = (sp, sq, qq, qp, pp)
                valid = torch.isfinite(torch.cat([t.reshape(-1) for t in aa])).all()
                return (tuple((t.detach() for t in values)), valid)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, valid = run()
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            self.derivative_cache[key] = (aa, graph, output, valid, step_tensor)
            self.stats['derivative_capture_seconds'] = time.perf_counter() - start
            print('DERIVATIVE_CAPTURE', self.stats, flush=True)
        aa, graph, output, valid, _step_tensor = self.derivative_cache[key]
        for dst, src in zip(aa, (q, p, context)):
            dst.copy_(src)
        graph.replay()
        if self.prefetching:
            self.pending_checks.append(valid)
        elif not bool(valid):
            raise FloatingPointError('nonfinite Hamiltonian input')
        values = tuple((t.clone() for t in output))
        if lag:
            self.lagged_hessian = values[2:]
        if 'audit' in self.mode or getattr(self, 'audit_derivatives', False):
            reference = self.original_derivatives(generator, q, p, context, step_size=step_size, create_graph=False)
            diff = [{'max': float((v - r).abs().max()), 'rms': float((v - r).square().mean().sqrt()), 'finite': bool(torch.isfinite(v).all())} for v, r in zip(values, reference)]
            self.stats.setdefault('derivative_audit', []).append(diff)
            print('DERIVATIVE_DIFF', diff, flush=True)
        return values

    def gradients(self, generator, q, p, context, *, step_size):
        key = (tuple(q.shape), tuple(context.shape), str(q.dtype), step_size)
        if key not in self.gradient_cache:
            from hamiformer.inference.derivatives import CompiledDerivatives
            start = time.perf_counter()
            fn = CompiledDerivatives(generator, step_size, plain_norm=True, only_gradient=True, autotune='autotune' in self.mode, point_attention='pointattn' in self.mode)
            aa = tuple((t.detach().clone() for t in (q, p, context)))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def run():
                values = fn(*aa)
                valid = torch.isfinite(torch.cat([v.reshape(-1) for v in aa])).all()
                return (tuple((v.detach() for v in values)), valid)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, valid = run()
            self.gradient_cache[key] = (aa, graph, output, valid, fn)
            self.stats['gradient_capture_seconds'] = time.perf_counter() - start
            print('GRADIENT_CAPTURE', self.stats, flush=True)
        aa, graph, output, valid, _fn = self.gradient_cache[key]
        for dst, src in zip(aa, (q, p, context)):
            dst.copy_(src)
        graph.replay()
        if self.prefetching:
            self.pending_checks.append(valid)
        elif not bool(valid):
            raise FloatingPointError('nonfinite first-derivative input')
        return tuple((v.clone() for v in output))

    def _prepare_gate(self, gate):
        from hamiformer.training.hamiballs1.tree_candidate import conditional_delta, _componentize
        closure = inspect.getclosurevars(gate.forward_step.__func__).nonlocals
        alpha = closure['candidate_blend_alpha']
        if not isinstance(alpha, tuple) or len(alpha) != 2 or (not closure['replace_base_hr']):
            raise ValueError('unrecognized frozen PLAS pre-gate contract')
        alpha4 = gate.ridge_weight.new_tensor(alpha).repeat_interleave(2)
        adjust = closure['gate_logit_adjuster']
        adjust_vars = inspect.getclosurevars(adjust).nonlocals
        alpha2 = gate.ridge_weight.new_tensor(adjust_vars['alpha'])
        original = closure['original_forward']
        encoder = closure['encoder']
        weight_getter = closure['weight_getter']
        component_history = closure['original_component_history']

        def pure_forward(edge, module, d_token, noisy_state, x0, previous_mixed, h_candidate, base_hr_candidate, d_candidate, attrs, tau, physical_time, previous_g, hidden=None, residual_hidden=None):
            previous_qp = _componentize(previous_g)
            delta, leaf = conditional_delta(encoder, weight_getter(), edge=edge, previous_mixed=previous_mixed, h_candidate=h_candidate, base_hr_candidate=base_hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_qp, residual_hidden=residual_hidden)
            strong_hr = h_candidate + delta
            corrected = base_hr_candidate + alpha4 * (strong_hr - base_hr_candidate)
            history = previous_qp if component_history else previous_qp.mean(dim=-1)
            value, next_hidden = original(d_token, noisy_state, x0, previous_mixed, h_candidate, corrected, d_candidate, attrs, tau, physical_time, history, hidden, residual_hidden=residual_hidden)
            design = torch.cat((residual_hidden, torch.ones_like(residual_hidden[..., :1])), dim=-1)
            local = module.base_gate.etrg_leaf_gate_linear[leaf.long()]
            bias = alpha2 * torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
            value = torch.sigmoid(torch.logit(value.clamp(1e-06, 1 - 1e-06)) + bias)
            return (value, next_hidden, corrected, leaf)
        self.pure_gate = pure_forward

        def forward(module, *args, **kwargs):
            value, next_hidden, corrected, leaf = pure_forward(int(getattr(module, '_etrg_runtime_edge', 0)), module, *args, **kwargs)
            module._etrg_last_corrected_hr = corrected
            module._etrg_last_leaf = leaf
            module._etrg_runtime_edge = int(getattr(module, '_etrg_runtime_edge', 0)) + 1
            return (value, next_hidden)
        gate.forward_step = MethodType(forward, gate)

    def _finite(self, name, value):
        if 'compile' in self.mode:
            torch._assert_async(torch.isfinite(value).all(), name)
            return
        if self._checks is None:
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(name)
        else:
            self._checks.append(value)

    def _execute(self, args, kwargs):
        self._checks = []
        try:
            result = self.original_scan(*args, **kwargs)
            check = torch.isfinite(torch.cat([x.reshape(-1) for x in self._checks])).all() if self._checks else torch.isfinite(result.mixed).all()
            return (result, check)
        finally:
            self._checks = None

    def scan(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('PLAS runtime is inference only')
        signature = tuple(((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in _tensor_list((args, kwargs))))
        key = (signature, tuple(((k, str(v)) for k, v in kwargs.items() if not torch.is_tensor(v) and (not isinstance(v, torch.nn.Module)))))
        if key not in self.cache:
            start = time.perf_counter()
            aa, kk = _map((args, kwargs), lambda t: t.detach().clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._execute(aa, kk)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, valid = self._execute(aa, kk)
            self.cache[key] = (aa, kk, graph, output, valid)
            self.stats['scan_graphs'] = len(self.cache)
            self.stats['scan_capture_seconds'] = self.stats.get('scan_capture_seconds', 0) + time.perf_counter() - start
            print('SCAN_CAPTURE', self.stats, flush=True)
        aa, kk, graph, output, valid = self.cache[key]
        for dst, src in zip(_tensor_list((aa, kk)), _tensor_list((args, kwargs))):
            dst.copy_(src)
        graph.replay()
        if not bool(valid):
            raise FloatingPointError('nonfinite value in captured mixed scan')
        return _map(output, lambda t: t.clone())

def install(models, mode='graph-scan'):
    return PLASRuntime(models, mode)
SUPPORTED_PROFILES = {'exact': 'compile-func-plainln-bounds-overlap-preproj-lookahead-hchunk512-batchad-autotune-pointattn', 'exact_reference': 'compile-func-plainln-bounds-overlap-preproj-lookahead-hchunk256', 'lagged_hessian16': 'compile-func-plainln-bounds-overlap-preproj-lookahead-hchunk512-batchad-autotune-pointattn-lag16', 'lagged_hessian16_reference': 'compile-func-plainln-bounds-overlap-preproj-lookahead-hchunk256-lag16'}

def install_profile(models, profile='exact'):
    if profile not in SUPPORTED_PROFILES:
        raise ValueError('unsupported profile: ' + profile)
    if '_plas_runtime' in models:
        raise RuntimeError('PLAS runtime already installed')
    runtime = install(models, SUPPORTED_PROFILES[profile])
    models['_plas_runtime'] = runtime
    return runtime
