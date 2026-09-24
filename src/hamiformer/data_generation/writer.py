from __future__ import annotations
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from hamiformer.data.schema import validate_phase_arrays
from hamiformer.utils import sha256_file
from .config import GeneratorConfig
from .plan import SceneTask
from .provenance import GENERATOR_IMPLEMENTATION_VERSION
from .simulator import SimulationResult

@dataclass(frozen=True)
class SampleArtifact:
    split: str
    scene_id: str
    sample_id: str
    npz_path: str
    npz_sha256: str
    reused: bool
    ball_ball_events: int | None
    ball_wall_events: int | None
    total_impulse: float | None
    events: list[dict[str, object]] | tuple[dict[str, object], ...] | None
    attempts: int | None
    closure_float64_max_error: float | None
    closure_float32_max_error: float | None
    seam_quiet: bool | None

    def manifest_record(self, manifest_dir: Path) -> dict[str, str]:
        relative = os.path.relpath(self.npz_path, start=manifest_dir)
        return {'sample_id': self.sample_id, 'scene_id': self.scene_id, 'npz_path': Path(relative).as_posix(), 'npz_sha256': self.npz_sha256}

    def event_record(self) -> dict[str, Any]:
        return {'sample_id': self.sample_id, 'scene_id': self.scene_id, 'ball_ball_events': self.ball_ball_events, 'ball_wall_events': self.ball_wall_events, 'total_impulse': self.total_impulse, 'events': self.events, 'attempts': self.attempts, 'closure_float64_max_error': self.closure_float64_max_error, 'closure_float32_max_error': self.closure_float32_max_error, 'seam_quiet': self.seam_quiet}

@dataclass(frozen=True)
class CompletionRecord:
    sample_id: str
    npz_sha256: str
    provenance_sha256: str

def sample_path(root: Path, task: SceneTask) -> Path:
    return root / 'samples' / task.split / f'{task.sample_id}.npz'

def sample_provenance_path(root: Path, task: SceneTask) -> Path:
    return root / 'metadata' / 'samples' / task.split / f'{task.sample_id}.json'

def _fsync_parent_directory(path: Path) -> None:
    if os.name == 'nt' or not hasattr(os, 'O_DIRECTORY'):
        return
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _atomic_replace_bytes(target: Path, payload_writer: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            payload_writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_parent_directory(target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

def _write_deterministic_npz_temp(target: Path, result: SimulationResult, *, compressed: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with zipfile.ZipFile(handle, mode='w', compression=compression, allowZip64=True) as archive:
                for name, array in (('phase', result.phase), ('attrs', result.attrs), ('time', result.time)):
                    payload = io.BytesIO()
                    np.save(payload, np.asarray(array), allow_pickle=False)
                    member = zipfile.ZipInfo(f'{name}.npy', date_time=(1980, 1, 1, 0, 0, 0))
                    member.compress_type = compression
                    member.create_system = 0
                    member.external_attr = 0
                    archive.writestr(member, payload.getvalue(), compress_type=compression)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

def write_json_atomic(target: Path, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')
    _atomic_replace_bytes(target, lambda handle: handle.write(encoded))

def write_jsonl_atomic(target: Path, rows: Iterable[dict[str, Any]]) -> None:
    encoded = ''.join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in rows)).encode('utf-8')
    _atomic_replace_bytes(target, lambda handle: handle.write(encoded))

def append_state_log(path: Path, artifact: SampleArtifact, semantic_hash: str, provenance_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'semantic_hash': semantic_hash, 'sample_id': artifact.sample_id, 'npz_sha256': artifact.npz_sha256, 'provenance_sha256': provenance_sha256}
    line = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    with path.open('ab') as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())

def load_state_log(path: Path, semantic_hash: str) -> dict[str, CompletionRecord]:
    result: dict[str, CompletionRecord] = {}
    if not path.is_file():
        return result
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    offset = 0
    append_missing_newline = False
    for line_number, encoded_line in enumerate(lines, start=1):
        terminated = encoded_line.endswith(b'\n')
        try:
            line = encoded_line.decode('utf-8')
        except UnicodeDecodeError as error:
            if line_number == len(lines) and (not terminated):
                with path.open('r+b') as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f'state log 第 {line_number} 行 UTF-8 损坏') from error
        if not line.strip():
            offset += len(encoded_line)
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines) and (not terminated):
                with path.open('r+b') as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f'state log 第 {line_number} 行 JSON/UTF-8 损坏')
        if set(raw) != {'semantic_hash', 'sample_id', 'npz_sha256', 'provenance_sha256'}:
            raise ValueError(f'state log 第 {line_number} 行字段集合非法')
        if raw.pop('semantic_hash') != semantic_hash:
            raise ValueError(f'state log 第 {line_number} 行属于另一份生成配置')
        record = CompletionRecord(**raw)
        result[record.sample_id] = record
        offset += len(encoded_line)
        if line_number == len(lines) and (not terminated):
            append_missing_newline = True
    if append_missing_newline:
        with path.open('ab') as handle:
            handle.write(b'\n')
            handle.flush()
            os.fsync(handle.fileno())
    return result

def validate_existing_sample(path: Path, task: SceneTask, config: GeneratorConfig) -> str:
    with np.load(path, allow_pickle=False) as sample:
        if set(sample.files) != {'phase', 'attrs', 'time'}:
            raise ValueError(f'已有样本字段不等于 phase/attrs/time: {path}')
        phase = np.asarray(sample['phase'])
        attrs = np.asarray(sample['attrs'])
        time = np.asarray(sample['time'])
    validate_phase_arrays(phase, attrs, time, num_objects=config.sampling.num_objects, future_steps=task.frames - 1, q_dim=2, attr_dim=3)
    return sha256_file(path)

def _provenance_payload(task: SceneTask, artifact: SampleArtifact, config: GeneratorConfig, implementation_hash: str) -> dict[str, Any]:
    return {'format_version': 1, 'semantic_hash': config.semantic_hash, 'implementation_version': GENERATOR_IMPLEMENTATION_VERSION, 'implementation_fingerprint': implementation_hash, 'task': asdict(task), 'artifact': {'npz_sha256': artifact.npz_sha256, 'ball_ball_events': artifact.ball_ball_events, 'ball_wall_events': artifact.ball_wall_events, 'total_impulse': artifact.total_impulse, 'events': list(artifact.events or ()), 'attempts': artifact.attempts, 'closure_float64_max_error': artifact.closure_float64_max_error, 'closure_float32_max_error': artifact.closure_float32_max_error, 'seam_quiet': artifact.seam_quiet}}

def _load_provenance_artifact(root: Path, task: SceneTask, config: GeneratorConfig, implementation_hash: str) -> tuple[SampleArtifact, str]:
    provenance_path = sample_provenance_path(root, task)
    with provenance_path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    required = {'format_version', 'semantic_hash', 'implementation_version', 'implementation_fingerprint', 'task', 'artifact'}
    if set(raw) != required or raw['format_version'] != 1:
        raise ValueError(f'样本 provenance 格式非法: {provenance_path}')
    if raw['semantic_hash'] != config.semantic_hash:
        raise ValueError(f'样本 provenance 属于另一份生成配置: {task.sample_id}')
    if raw['implementation_version'] != GENERATOR_IMPLEMENTATION_VERSION or raw['implementation_fingerprint'] != implementation_hash:
        raise ValueError(f'样本 provenance 属于另一版生成器实现: {task.sample_id}')
    if raw['task'] != asdict(task):
        raise ValueError(f'样本 provenance 的 task 身份不匹配: {task.sample_id}')
    diagnostic = raw['artifact']
    diagnostic_keys = {'npz_sha256', 'ball_ball_events', 'ball_wall_events', 'total_impulse', 'events', 'attempts', 'closure_float64_max_error', 'closure_float32_max_error', 'seam_quiet'}
    if not isinstance(diagnostic, dict) or set(diagnostic) != diagnostic_keys:
        raise ValueError(f'样本 provenance 的 artifact 字段非法: {task.sample_id}')
    if any((diagnostic[name] is None for name in ('ball_ball_events', 'ball_wall_events', 'total_impulse', 'events', 'attempts', 'closure_float64_max_error', 'closure_float32_max_error', 'seam_quiet'))):
        raise ValueError(f'样本 provenance 缺少 event/closure 诊断: {task.sample_id}')
    artifact = SampleArtifact(split=task.split, scene_id=task.scene_id, sample_id=task.sample_id, npz_path=str(sample_path(root, task).resolve()), npz_sha256=str(diagnostic['npz_sha256']), reused=True, ball_ball_events=int(diagnostic['ball_ball_events']), ball_wall_events=int(diagnostic['ball_wall_events']), total_impulse=float(diagnostic['total_impulse']), events=list(diagnostic['events']), attempts=int(diagnostic['attempts']), closure_float64_max_error=float(diagnostic['closure_float64_max_error']), closure_float32_max_error=float(diagnostic['closure_float32_max_error']), seam_quiet=bool(diagnostic['seam_quiet']))
    return (artifact, sha256_file(provenance_path))

def write_sample_atomic(root: Path, task: SceneTask, result: SimulationResult, config: GeneratorConfig, implementation_hash: str) -> SampleArtifact:
    target = sample_path(root, task)
    temporary = _write_deterministic_npz_temp(target, result, compressed=config.output.compressed)
    try:
        digest = validate_existing_sample(temporary, task, config)
        candidate = SampleArtifact(split=task.split, scene_id=task.scene_id, sample_id=task.sample_id, npz_path=str(target.resolve()), npz_sha256=digest, reused=False, ball_ball_events=result.ball_ball_events, ball_wall_events=result.ball_wall_events, total_impulse=result.total_impulse, events=result.events, attempts=result.attempts, closure_float64_max_error=result.closure_float64_max_error, closure_float32_max_error=result.closure_float32_max_error, seam_quiet=result.seam_quiet)
        payload = _provenance_payload(task, candidate, config, implementation_hash)
        provenance_path = sample_provenance_path(root, task)
        if target.exists():
            existing_digest = validate_existing_sample(target, task, config)
            if existing_digest != digest:
                raise FileExistsError(f'同名 final NPZ 与确定性重演不一致: {task.sample_id}')
            if not provenance_path.is_file():
                raise FileExistsError(f'同名 final NPZ 缺少完成 sidecar: {task.sample_id}')
            existing, _ = _load_provenance_artifact(root, task, config, implementation_hash)
            if _provenance_payload(task, existing, config, implementation_hash) != payload:
                raise FileExistsError(f'同名 final 的诊断 provenance 不一致: {task.sample_id}')
            return SampleArtifact(**{**asdict(existing), 'reused': True})
        if provenance_path.is_file():
            with provenance_path.open('r', encoding='utf-8') as handle:
                existing_payload = json.load(handle)
            if existing_payload != payload:
                raise FileExistsError(f'未提交 sidecar 与确定性重演不一致: {task.sample_id}')
        else:
            write_json_atomic(provenance_path, payload)
        os.replace(temporary, target)
        _fsync_parent_directory(target)
        temporary = None
        if validate_existing_sample(target, task, config) != digest:
            raise RuntimeError(f'NPZ 原子发布后的 hash 改变: {task.sample_id}')
        return candidate
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def artifact_from_existing(root: Path, task: SceneTask, config: GeneratorConfig, recorded: CompletionRecord | None, implementation_hash: str) -> SampleArtifact | None:
    target = sample_path(root, task)
    provenance_path = sample_provenance_path(root, task)
    if not target.is_file() and (not provenance_path.is_file()):
        if recorded is not None:
            raise ValueError(f'state log 声称完成但 NPZ/sidecar 均缺失: {task.sample_id}')
        return None
    if not target.is_file():
        artifact, provenance_digest = _load_provenance_artifact(root, task, config, implementation_hash)
        if recorded is not None and (recorded.sample_id != task.sample_id or recorded.npz_sha256 != artifact.npz_sha256 or recorded.provenance_sha256 != provenance_digest):
            raise ValueError(f'已完成日志与未提交 sidecar 不一致: {task.sample_id}')
        return None
    if not provenance_path.is_file():
        raise ValueError(f'orphan final NPZ 缺少 provenance sidecar: {task.sample_id}')
    artifact, provenance_digest = _load_provenance_artifact(root, task, config, implementation_hash)
    digest = validate_existing_sample(target, task, config)
    if artifact.npz_sha256 != digest:
        raise ValueError(f'样本 sidecar 与 final NPZ hash 不一致: {task.sample_id}')
    if recorded is not None:
        if recorded.sample_id != task.sample_id or recorded.npz_sha256 != digest or recorded.provenance_sha256 != provenance_digest:
            raise ValueError(f'已完成日志与 final artifact 不一致: {task.sample_id}')
    return artifact

def compute_train_rms(manifest_path: Path, records: list[dict[str, str]]) -> dict[str, Any]:
    q_sumsq = np.zeros(2, dtype=np.float64)
    p_sumsq = np.zeros(2, dtype=np.float64)
    count = 0
    for record in records:
        path = (manifest_path.parent / record['npz_path']).resolve()
        with np.load(path, allow_pickle=False) as sample:
            phase = np.asarray(sample['phase'], dtype=np.float64)
        q_sumsq += np.sum(np.square(phase[..., :2]), axis=(0, 1))
        p_sumsq += np.sum(np.square(phase[..., 2:]), axis=(0, 1))
        count += phase.shape[0] * phase.shape[1]
    if count <= 0:
        raise ValueError('train manifest 为空，不能计算 phase scales')
    q_scale = np.maximum(np.sqrt(q_sumsq / count), 1e-06)
    p_scale = np.maximum(np.sqrt(p_sumsq / count), 1e-06)
    return {'format_version': 1, 'train_manifest_sha256': sha256_file(manifest_path), 'q_scale': [float(value) for value in q_scale], 'p_scale': [float(value) for value in p_scale]}
