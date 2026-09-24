from hamiformer.utils.paths import project_root
import torch

def tangent_health(matrix, limit, stats=None):
    finite = torch.isfinite(matrix).all((-2, -1))
    gram = matrix.transpose(-1, -2) @ matrix
    lower = gram.diagonal(dim1=-2, dim2=-1).amax(-1)
    upper = gram.abs().sum(-1).amax(-1)
    limit2 = float(limit) ** 2
    healthy = finite & (upper <= limit2)
    unhealthy = ~finite | (lower > limit2)
    ambiguous = ~(healthy | unhealthy)
    if bool(ambiguous.any()):
        g = gram[ambiguous]
        n = g.shape[-1]
        eps = torch.finfo(g.dtype).eps
        eye = torch.eye(n, device=g.device, dtype=g.dtype)
        margin = eps * 64 * n * (g.abs().sum(-1).amax(-1) + limit2 + 1)
        shifted = (limit2 - margin)[:, None, None] * eye - g
        factor, info = torch.linalg.cholesky_ex(shifted, check_errors=False)
        residual = shifted - factor @ factor.transpose(-1, -2)
        error = residual.abs().sum(-1).amax(-1)
        rounding = eps * 8 * n * (shifted.abs().sum(-1).amax(-1) + factor.abs().sum(-1).amax(-1).square() + 1)
        certified = (info == 0) & torch.isfinite(error) & (error + rounding < margin)
        local = certified.clone()
        remaining = ~certified
        if bool(remaining.any()):
            eigen = torch.linalg.eigvalsh(g[remaining], UPLO='U')[..., -1]
            local[remaining] = torch.isfinite(eigen) & (eigen <= limit2)
        healthy[ambiguous] = local
        if stats is not None:
            stats['tangent_cholesky_certified'] = stats.get('tangent_cholesky_certified', 0) + int(certified.sum())
            stats['tangent_eigen_fallback'] = stats.get('tangent_eigen_fallback', 0) + int(remaining.sum())
    return healthy
