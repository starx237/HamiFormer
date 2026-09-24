from __future__ import annotations
import torch
from .artifact import RoutingArtifact

class HardRoutingPolicy:

    def __init__(self, artifact: RoutingArtifact) -> None:
        artifact.validate()
        self.artifact = artifact

    def bin_index(self, tau: torch.Tensor) -> torch.Tensor:
        index = torch.floor(tau * self.artifact.num_bins).long()
        return index.clamp(min=0, max=self.artifact.num_bins - 1)

    def parameters(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.bin_index(tau)
        thresholds = torch.as_tensor(self.artifact.thresholds, device=tau.device, dtype=tau.dtype)
        d_only = torch.as_tensor(self.artifact.d_only, device=tau.device, dtype=torch.bool)
        return (thresholds[index], d_only[index])

    def accept(self, tau: torch.Tensor, disagreement: torch.Tensor) -> torch.Tensor:
        threshold, d_only = self.parameters(tau)
        return ~d_only & torch.isfinite(disagreement) & (disagreement <= threshold)

class ThreeZoneSoftPolicy(HardRoutingPolicy):

    def __init__(self, artifact: RoutingArtifact) -> None:
        super().__init__(artifact)
        if artifact.soft_high_thresholds is None or artifact.soft_low_thresholds is None:
            raise ValueError('soft_three_zone 必须提供独立 high/low thresholds')

    def weight(self, tau: torch.Tensor, disagreement: torch.Tensor) -> torch.Tensor:
        index = self.bin_index(tau)
        high_all = torch.as_tensor(self.artifact.soft_high_thresholds, device=tau.device, dtype=tau.dtype)
        low_all = torch.as_tensor(self.artifact.soft_low_thresholds, device=tau.device, dtype=tau.dtype)
        d_only_all = torch.as_tensor(self.artifact.d_only, device=tau.device, dtype=torch.bool)
        high, low, d_only = (high_all[index], low_all[index], d_only_all[index])
        width = (low - high).clamp_min(torch.finfo(tau.dtype).eps)
        s = ((low - disagreement) / width).clamp(0.0, 1.0)
        smooth = 3.0 * s.pow(2) - 2.0 * s.pow(3)
        exact = torch.where(disagreement <= high, torch.ones_like(smooth), smooth)
        exact = torch.where(disagreement >= low, torch.zeros_like(exact), exact)
        return torch.where(d_only | ~torch.isfinite(disagreement), torch.zeros_like(exact), exact)
