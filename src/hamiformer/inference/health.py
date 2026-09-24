from hamiformer.utils.paths import project_root
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class CertifiedJet:
    matrix: torch.Tensor
    offset: torch.Tensor
    source_graph: torch.Tensor
    target_graph: torch.Tensor
    mixed_jacobian: torch.Tensor
    mixed_singular_lower_bound: torch.Tensor
    mixed_singular_upper_bound: torch.Tensor
    mixed_condition_upper_bound: torch.Tensor
    tangent_norm_upper_bound: torch.Tensor
    diagnostic_kind: str = 'sufficient_induced_norm_bounds'

def install(runtime):
    from hamiformer.physics import generic_type2 as gt
    import sys
    original = gt.type2_vector_jets

    def jets(generator, q, p, context, *, step_size, mixed_singular_floor, mixed_condition_limit, tangent_spectral_norm_limit, create_graph=False, detach=True):
        if create_graph or not detach:
            raise ValueError('runtime health checks are inference only')
        sp, sq, qq, qp, pp = gt._type2_second_derivatives(generator, q, p, context, step_size=step_size, create_graph=False)
        matrix, mixed = gt._linearization_from_type2_second_derivatives(qq, qp, pp)

        def upper_norm(a):
            x = a.abs()
            return (x.sum(-1).amax(-1) * x.sum(-2).amax(-1)).sqrt()
        error = mixed - torch.eye(mixed.shape[-1], device=mixed.device, dtype=mixed.dtype)
        slack = torch.finfo(mixed.dtype).eps * 64 * matrix.shape[-1]
        e = upper_norm(error) * (1 + slack) + slack
        lower = (1 - e).amin()
        upper = (1 + e).amax()
        condition = ((1 + e) / (1 - e)).amax()
        tangent = (upper_norm(matrix) * (1 + slack) + slack).amax()
        valid = torch.isfinite(torch.stack((lower, upper, condition, tangent))).all()
        certificate = valid & (lower >= mixed_singular_floor) & (condition <= mixed_condition_limit) & (tangent <= tangent_spectral_norm_limit)
        source = torch.cat((q, sp), dim=-1)
        target = torch.cat((sq, p), dim=-1)
        offset = target - (matrix @ source.unsqueeze(-1)).squeeze(-1)
        result = CertifiedJet(matrix.detach(), offset.detach(), source.detach(), target.detach(), mixed.detach(), lower.detach(), upper.detach(), condition.detach(), tangent.detach())

        def finalize():
            if bool(certificate):
                runtime.stats['health_bound_passes'] = runtime.stats.get('health_bound_passes', 0) + 1
                return result
            runtime.stats['health_svd_fallbacks'] = runtime.stats.get('health_svd_fallbacks', 0) + 1
            minimum, maximum, cond, norm = gt._matrix_health(matrix, mixed)
            ok = torch.isfinite(torch.stack((minimum, maximum, cond, norm))).all() & (minimum >= mixed_singular_floor) & (cond <= mixed_condition_limit) & (norm <= tangent_spectral_norm_limit)
            if not bool(ok):
                raise RuntimeError('PLAS original spectral hard health failed')
            return gt.GenericTypeIIJet(matrix.detach(), offset.detach(), source.detach(), target.detach(), mixed.detach(), minimum.detach(), maximum.detach(), cond.detach(), norm.detach())
        if runtime.prefetching:
            runtime.pending_health = finalize
            return result
        return finalize()
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, 'type2_vector_jets', None) is original:
            module.type2_vector_jets = jets
