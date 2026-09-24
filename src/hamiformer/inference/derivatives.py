from hamiformer.utils.paths import project_root
import torch
import copy
from types import MethodType

def pointwise_attention(model):

    def attention(obj, value):
        b, n, _ = value.shape
        packed = obj.qkv(value).reshape(b, n, 3, obj.heads, obj.head_width)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
        score = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1) * obj.head_width ** (-0.5)
        weight = torch.softmax(score, -1)
        attended = (weight.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
        return obj.output(attended.transpose(1, 2).reshape(b, n, obj.width))
    for layer in model.modules():
        if type(layer).__name__ == '_ExplicitMultiheadSelfAttention':
            layer.forward = MethodType(attention, layer)
    return model

class CompiledDerivatives:

    def __init__(self, model, step_size, plain_norm=False, forward_over_reverse=False, only_gradient=False, batch_ad=False, raw_h=False, jac_chunk=None, autotune=False, point_attention=False):
        if point_attention and (not plain_norm):
            model = copy.deepcopy(model)
        if plain_norm:
            model = copy.deepcopy(model)

            def norm(obj, x):
                centered = x - x.mean(-1, keepdim=True)
                out = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + obj.eps)
                return out * obj.weight + obj.bias
            for layer in model.modules():
                if isinstance(layer, torch.nn.LayerNorm):
                    layer.forward = MethodType(norm, layer)
        if point_attention:
            pointwise_attention(model)
        dim = model.state_dim

        def energy(z, c):
            q, p = (z[:dim], z[dim:])
            hv = model(q[None], p[None], c[None])[0]
            return hv if raw_h else (q * p).sum() + z.new_tensor(step_size) * hv
        grad = torch.func.grad(energy, argnums=0)

        def grad_aux(z, c):
            g = grad(z, c)
            return (g, g)
        transform = torch.func.jacfwd if forward_over_reverse else torch.func.jacrev
        jac = transform(grad_aux, argnums=0, has_aux=True, **{'chunk_size': jac_chunk} if jac_chunk and (not forward_over_reverse) else {})
        batched = torch.func.vmap(jac, in_dims=(0, 0))
        batched_grad = torch.func.vmap(grad, in_dims=(0, 0))

        def batch_energy(z, c):
            hv = model(z[:, :dim], z[:, dim:], c).sum()
            return hv if raw_h else (z[:, :dim] * z[:, dim:]).sum() + z.new_tensor(step_size) * hv
        batch_gradient = torch.func.grad(batch_energy, argnums=0)

        def forward(q, p, c):
            if only_gradient:
                g = batched_grad(torch.cat((q, p), dim=-1), c)
                return (g[:, :dim], g[:, dim:])
            z = torch.cat((q, p), dim=-1)
            if batch_ad:
                g, vjp = torch.func.vjp(lambda zz: batch_gradient(zz, c), z)
                seeds = torch.eye(2 * dim, device=z.device, dtype=z.dtype)[:, None, :].expand(-1, z.shape[0], -1)
                h = torch.func.vmap(vjp)(seeds)[0].permute(1, 0, 2)
            else:
                h, g = batched(z, c)
            if raw_h:
                step = q.new_tensor(step_size)
                identity = torch.eye(dim, device=q.device, dtype=q.dtype)[None]
                return (p + step * g[:, :dim], q + step * g[:, dim:], step * h[:, :dim, :dim], identity + step * h[:, :dim, dim:], step * h[:, dim:, dim:])
            return (g[:, :dim], g[:, dim:], h[:, :dim, :dim], h[:, :dim, dim:], h[:, dim:, dim:])
        self.forward = forward
        self.autotune = autotune
        self.functions = {}

    def __call__(self, q, p, c):
        key = (tuple(q.shape), tuple(c.shape))
        if key not in self.functions:
            from torch.fx.experimental.proxy_tensor import make_fx
            graph = make_fx(self.forward)(q, p, c)
            print('HESSIAN_FX_NODES', len(list(graph.graph.nodes)), flush=True)
            self.functions[key] = torch.compile(graph, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False, 'max_autotune': self.autotune})
        return self.functions[key](q, p, c)
