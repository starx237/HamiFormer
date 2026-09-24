"""Batched Hamiltonian candidates and window-scoped inference construction."""
import copy
from types import MethodType, SimpleNamespace

import torch

_installed = False


def candidate_function(method, field, step_size, qdim=2):
    """Return a batched candidate with shape [batch, objects, 2 * qdim]."""
    from hamiformer.physics.hamiballs_type2 import apply_hamiballs_affine_jet
    h = step_size

    def candidate(matrix, offset, previous, attrs):
        if method in ('plas', 'plas_s'):
            return apply_hamiballs_affine_jet(matrix, offset, previous, q_dim=qdim)
        q, p = previous[..., :qdim], previous[..., qdim:]
        if method == 'explicit_euler':
            fq, fp = field(q, p, attrs)
            return torch.cat((q + h * fq, p + h * fp), -1)
        if method == 'symeuler2':
            pp = p
            for _ in range(2):
                _, fp = field(q, pp, attrs)
                pp = p + h * fp
            fq, _ = field(q, pp, attrs)
            return torch.cat((q + h * fq, pp), -1)
        dim = q.shape[1] * qdim
        if method == 'plas_dc_f':
            flat = torch.cat((q.flatten(1), p.flatten(1)), -1)
            pp = torch.baddbmm(offset[:, dim:, None], matrix[:, dim:], flat[..., None]).squeeze(-1).reshape_as(p)
        else:
            pp = apply_hamiballs_affine_jet(matrix, offset, previous, q_dim=qdim)[..., qdim:]
        fq, fp = field(q, pp, attrs)
        if method in ('plas_dc', 'plas_dc_s'):
            defect = (p.double() + h * fp.double() - pp.double()).flatten(1)
            delta = (matrix[:, :, dim:].double() @ defect[..., None]).squeeze(-1)
            return torch.cat(((q.double() + h * fq.double() + delta[:, :dim].reshape_as(q)).to(q.dtype),
                              (pp.double() + delta[:, dim:].reshape_as(pp)).to(p.dtype)), -1)
        defect = (p + h * fp - pp).flatten(1)
        base = torch.cat(((q + h * fq).flatten(1), pp.flatten(1)), -1)
        result = torch.baddbmm(base[..., None], matrix[:, :, dim:], defect[..., None]).squeeze(-1)
        return torch.cat((result[:, :dim].reshape_as(q), result[:, dim:].reshape_as(p)), -1)
    return candidate


def _gradient_candidate(solver, models, step_size, compiled):
    model = copy.deepcopy(models['h'])
    from hamiformer.inference.derivatives import pointwise_attention
    pointwise_attention(model)

    def norm(obj, x):
        centered = x - x.mean(-1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + obj.eps) * obj.weight + obj.bias
    for layer in model.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.forward = MethodType(norm, layer)
    # Shapes are checked before tracing; data-dependent validation stays outside AD.
    model._validate = MethodType(lambda obj, q, p, c: None, model)
    derivative = torch.func.grad(lambda q, p, c: model(q, p, c).sum(), argnums=(0, 1))
    functions = {}

    def call(matrix, offset, previous, attrs):
        signature = (previous.shape, previous.dtype, previous.device)
        if signature not in functions:
            q, p = previous[..., :2], previous[..., 2:]
            context = (attrs / models['attr_scale'].reshape(1, 1, -1)).unsqueeze(-2)
            models['h']._validate(q.flatten(1), p.flatten(1), context)
            from torch.fx.experimental.proxy_tensor import make_fx
            graph = make_fx(derivative)(q.flatten(1), p.flatten(1), context)

            def field(qq, pp, at):
                c = (at / models['attr_scale'].reshape(1, 1, -1)).unsqueeze(-2)
                gq, gp = graph(qq.flatten(1), pp.flatten(1), c)
                return gp.reshape_as(qq), -gq.reshape_as(pp)
            fn = candidate_function(solver.method, field, step_size)
            functions[signature] = (torch.compile(fn, fullgraph=True, dynamic=False,
                options={'triton.cudagraphs': False, 'max_autotune': True}) if compiled else fn)
        return functions[signature](matrix, offset, previous, attrs)
    return call


def install_h1(solver, models, *, compiled=True):
    global _installed
    from hamiformer.evaluation import hamiballs_formal as ev
    from hamiformer.training import hamiballs_formal as training
    from hamiformer.training.hamiballs1.sampling import expert_kwargs
    if _installed or '_plas_runtime' in models:
        raise RuntimeError('run each installed solver in a separate process')
    if compiled and models['state_scale'].device.type != 'cuda':
        raise ValueError('compiled window inference requires CUDA')
    method = solver.method
    serial = method in ('explicit_euler', 'symeuler2')
    frozen = method == 'plas_dc_f'
    learner = ev.learned_hamiballs_affine_jets
    sampler = ev.sample_stateful_pf_rf_v1_heun
    frozen_jets = []
    current_attrs = []
    eager_jets = {}

    def jets(model, anchor, attrs, **kwargs):
        current_attrs[:] = [attrs]
        if serial:
            template = anchor.source_q.new_zeros(*anchor.source_q.shape[:2], model.num_objects, 4)
            return ev.identity_hamiballs_affine_jets(template, q_dim=2)
        if frozen:
            if not frozen_jets:
                frozen_jets.append(learner(model, anchor, attrs, **kwargs))
            return frozen_jets[0]
        if not compiled and method in ('plas_s', 'plas_dc_s'):
            q, p = anchor.source_q, anchor.target_p
            key = (q.data_ptr(), q._version, p.data_ptr(), p._version)
            if key not in eager_jets:
                eager_jets[key] = (learner(model, anchor, attrs, **kwargs), anchor)
                if len(eager_jets) > 3:
                    eager_jets.pop(next(iter(eager_jets)))
            return eager_jets[key][0]
        return learner(model, anchor, attrs, **kwargs)

    def reset_sampler(*args, **kwargs):
        frozen_jets.clear()
        eager_jets.clear()
        return sampler(*args, **kwargs)
    if serial or frozen:
        ev.learned_hamiballs_affine_jets = jets
        ev.sample_stateful_pf_rf_v1_heun = reset_sampler
    if compiled:
        from hamiformer.inference.hamiballs1 import install
        mode = 'compile-func-plainln-bounds-preproj-hchunk512-batchad-autotune-pointattn'
        if not (serial or frozen):
            mode = 'compile-func-plainln-bounds-overlap-preproj-lookahead-hchunk512-batchad-autotune-pointattn'
        if method in ('plas_s', 'plas_dc_s'):
            mode += f'-lag{solver.refresh_interval}'
        if method == 'plas_dlr':
            mode = 'compile-plainln-bounds-overlap-preproj-lookahead'
        runtime = install(models, mode)
    else:
        runtime = SimpleNamespace(models=models, stats={}, prefetching=False)
        ev.sample_stateful_pf_rf_v1_heun = reset_sampler
        if not (serial or frozen):
            ev.learned_hamiballs_affine_jets = jets
    h = expert_kwargs(models)['step_size']
    if method == 'plas_dlr':
        from .lowrank import install_lowrank
        build = install_lowrank(runtime, rank=solver.rank, probes=solver.diagonal_probes,
            chunk=128, cache=True, batch_ad=True, jet_graph=compiled, compiled=compiled)
        if not compiled:
            def lowrank_jets(model, anchor, attrs, **kwargs):
                current_attrs[:] = [attrs]
                return build(model, anchor, attrs, **kwargs)
            ev.learned_hamiballs_affine_jets = lowrank_jets
    else:
        candidate = _gradient_candidate(solver, models, h, compiled)
        runtime.separate_candidate = candidate
        runtime.integrator_candidate = lambda matrix, offset, previous, attrs: offset
    if not compiled:
        if method in ('plas_s', 'plas_dc_s'):
            _install_eager_sparse(solver, ev)
        candidate = runtime.integrator_candidate if method == 'plas_dlr' else runtime.separate_candidate
        training.apply_hamiballs_affine_jet = lambda matrix, offset, previous, q_dim: candidate(matrix, offset, previous, current_attrs[0])
    models['_plas_runtime'] = runtime
    _installed = True
    return runtime


def _install_eager_sparse(solver, ev):
    from hamiformer.physics import generic_type2 as gt
    original = gt._type2_second_derivatives
    sampler = ev.sample_stateful_pf_rf_v1_heun
    cache = {}
    state = {'ordinal': 0, 'hessian': None}

    def reset(*args, **kwargs):
        cache.clear()
        state.update(ordinal=0, hessian=None)
        return sampler(*args, **kwargs)

    def derivatives(model, q, p, c, *, step_size, create_graph):
        key = (q.data_ptr(), p.data_ptr(), q._version, p._version)
        if key in cache:
            return cache[key][0]
        if state['ordinal'] % solver.refresh_interval == 0:
            values = original(model, q, p, c, step_size=step_size, create_graph=create_graph)
            state['hessian'] = values[2:]
        else:
            grad = torch.func.grad(lambda qq, pp: ((qq * pp).sum(-1) + step_size * model(qq, pp, c)).sum(), argnums=(0, 1))
            values = (*grad(q, p), *state['hessian'])
        state['ordinal'] += 1
        cache[key] = (values, q, p)
        if len(cache) > 3:
            cache.pop(next(iter(cache)))
        return values
    gt._type2_second_derivatives = derivatives
    ev.sample_stateful_pf_rf_v1_heun = reset
