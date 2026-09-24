from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class RoutingArtifact:
    num_bins: int
    thresholds: list[float]
    d_only: list[bool]
    model_hash: str
    stats_hash: str
    solver_hash: str
    config_hash: str
    audit: list[dict[str, float | int | bool]]
    cost_profile_hash: str = ''
    cost_device: str = ''
    cost_torch_version: str = ''
    structured_expert: str = 'h'
    calibration_data_hash: str = ''
    calibration_complete: bool = False
    calibration_seed: int = 0
    fit_records: int = 0
    audit_records: int = 0
    interval_stride: int = 1
    soft_high_thresholds: list[float] | None = None
    soft_low_thresholds: list[float] | None = None

    def validate(self) -> None:
        if self.structured_expert not in {'h', 's'}:
            raise ValueError('routing artifact 的 structured_expert 只能是 h 或 s')
        if self.interval_stride <= 0 or self.fit_records < 0 or self.audit_records < 0:
            raise ValueError('routing artifact 的 calibration 计数/stride 非法')
        if not all(self.d_only):
            if not self.calibration_complete or self.fit_records <= 0 or self.audit_records <= 0:
                raise ValueError('启用 structured route 的 artifact 必须来自完整且非空的 calibration')
            if not self.cost_profile_hash or not self.cost_device or (not self.cost_torch_version):
                raise ValueError('启用 structured route 的 artifact 缺少完整成本 provenance')
        if len(self.thresholds) != self.num_bins or len(self.d_only) != self.num_bins:
            raise ValueError('routing artifact 的 thresholds/d_only 长度错误')
        if self.soft_high_thresholds is not None or self.soft_low_thresholds is not None:
            if self.soft_high_thresholds is None or self.soft_low_thresholds is None:
                raise ValueError('soft high/low thresholds 必须同时存在')
            if len(self.soft_high_thresholds) != self.num_bins or len(self.soft_low_thresholds) != self.num_bins:
                raise ValueError('soft thresholds 长度错误')
            for high, low in zip(self.soft_high_thresholds, self.soft_low_thresholds, strict=True):
                if high > low:
                    raise ValueError('三段软融合要求 high-confidence threshold <= low-confidence threshold')

def save_routing_artifact(artifact: RoutingArtifact, path: str | Path) -> None:
    artifact.validate()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(asdict(artifact), handle, ensure_ascii=False, indent=2)

def load_routing_artifact(path: str | Path, *, expected_hashes: dict[str, str] | None=None, allow_incomplete: bool=False) -> RoutingArtifact:
    with Path(path).open('r', encoding='utf-8') as handle:
        artifact = RoutingArtifact(**json.load(handle))
    artifact.validate()
    if not artifact.calibration_complete and (not all(artifact.d_only)) and (not allow_incomplete):
        raise ValueError('routing artifact 来自 partial/debug calibration，禁止正式加载')
    if expected_hashes is not None:
        for name in ('model_hash', 'stats_hash', 'solver_hash', 'config_hash', 'calibration_data_hash'):
            expected = expected_hashes.get(name)
            if expected is not None and getattr(artifact, name) != expected:
                raise ValueError(f'routing artifact 的 {name} 与当前运行不匹配')
    return artifact

def validate_cost_runtime(artifact: RoutingArtifact, *, device_type: str, device_name: str, torch_version: str) -> None:
    expected_device = f'{device_type}:{device_name}'
    if artifact.cost_device and artifact.cost_device != expected_device:
        raise ValueError('routing artifact 的 cost device 与当前 runtime 不一致')
    if artifact.cost_torch_version and artifact.cost_torch_version != torch_version:
        raise ValueError('routing artifact 的 cost PyTorch version 与当前 runtime 不一致')
