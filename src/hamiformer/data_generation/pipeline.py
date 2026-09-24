from __future__ import annotations
import concurrent.futures
import contextlib
import datetime as dt
import json
import multiprocessing
import os
import platform
import shutil
import socket
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import numpy as np
from hamiformer.utils import sha256_file
from .budget import GenerationBudget, estimate_generation_budget
from .config import GeneratorConfig
from .plan import SPLIT_ORDER, SceneTask, build_scene_plan
from .provenance import GENERATOR_IMPLEMENTATION_VERSION, implementation_fingerprint
from .simulator import pymunk_runtime_info, simulate_task
from .writer import SampleArtifact, append_state_log, artifact_from_existing, compute_train_rms, load_state_log, sample_path, sample_provenance_path, write_json_atomic, write_jsonl_atomic, write_sample_atomic
LOCK_FILENAME = '.generation.lock'

def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise ValueError(f'无法找到 output root 的已存在父目录: {path}')
        current = current.parent
    return current

def _validate_output_root(root: Path) -> None:
    resolved = root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError('output root 不能是文件系统根目录')

@contextlib.contextmanager
def _generation_lock(config: GeneratorConfig, implementation_hash: str) -> Iterator[None]:
    root = Path(config.output.root).expanduser().resolve()
    _validate_output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 384)
    lock_backend: str
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b'\x00')
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            lock_backend = 'msvcrt-byte-range'
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_backend = 'fcntl-flock'
    except (OSError, BlockingIOError) as error:
        os.close(descriptor)
        try:
            detail = lock_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            detail = '<锁持有期间当前平台不允许读取 owner payload>'
        raise RuntimeError(f'output root 已被另一个生成 coordinator 锁定: {lock_path}\n{detail}') from error
    token = uuid.uuid4().hex
    payload = {'format_version': 1, 'token': token, 'pid': os.getpid(), 'host': socket.gethostname(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'output_root': str(root), 'semantic_hash': config.semantic_hash, 'implementation_fingerprint': implementation_hash, 'lock_backend': lock_backend}
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        yield
    finally:
        try:
            if os.name == 'nt':
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

def check_disk_budget(config: GeneratorConfig, budget: GenerationBudget, credited_sample_paths: list[Path] | None=None) -> tuple[int, int]:
    root = Path(config.output.root).expanduser().resolve()
    _validate_output_root(root)
    free = shutil.disk_usage(_nearest_existing_parent(root)).free
    existing_allocated = 0
    for path in credited_sample_paths or ():
        resolved = path.resolve()
        if not resolved.is_file() or root not in resolved.parents:
            raise ValueError(f'预算 credit 不是当前 root 内已验证的 final NPZ: {resolved}')
        size = resolved.stat().st_size
        existing_allocated += (size + 4095) // 4096 * 4096
    reserve = int(config.output.reserve_free_gib * 1024 ** 3)
    required = budget.estimated_disk_bytes + reserve
    if free + existing_allocated < required:
        raise RuntimeError(f'预计生成需 {budget.estimated_disk_bytes / 1024 ** 2:.1f} MiB，且要求保留 {config.output.reserve_free_gib:.1f} GiB；当前空闲与已验证 root 占用之和不足')
    return (free, required)

def _runtime_payload(config: GeneratorConfig) -> dict[str, str]:
    return {**pymunk_runtime_info(config.runtime.required_pymunk_version), 'numpy_version': str(np.__version__), 'python_version': platform.python_version(), 'platform': platform.platform(), 'byteorder': sys.byteorder}

def _recover_interrupted_initial_metadata_publish(root: Path, metadata_path: Path) -> bool:
    allowed_directory = metadata_path.parent.resolve()
    temporary_files: list[Path] = []
    for path in root.rglob('*'):
        if path.name == LOCK_FILENAME:
            continue
        if path.is_dir():
            if path.resolve() != allowed_directory:
                return False
            continue
        if path.parent.resolve() != allowed_directory or not path.name.startswith(f'.{metadata_path.name}.') or (not path.name.endswith('.tmp')):
            return False
        temporary_files.append(path)
    for path in temporary_files:
        path.unlink()
    return True

def _prepare_root(config: GeneratorConfig, runtime: dict[str, str], implementation_hash: str) -> tuple[Path, Path]:
    root = Path(config.output.root).expanduser().resolve()
    _validate_output_root(root)
    metadata_path = root / 'metadata' / 'dataset_meta.json'
    expected = {'format_version': 1, 'dataset_id': config.semantic_hash[:16], 'semantic_hash': config.semantic_hash, 'semantic_config': config.semantic_payload(), 'implementation_version': GENERATOR_IMPLEMENTATION_VERSION, 'implementation_fingerprint': implementation_hash, 'runtime': runtime, 'stats_algorithm': 'train_all_states_objects_coordinatewise_rms_about_zero_float64', 'status': 'in_progress'}
    non_lock_entries = [path for path in root.iterdir() if path.name != LOCK_FILENAME]
    if non_lock_entries:
        if not metadata_path.is_file():
            if _recover_interrupted_initial_metadata_publish(root, metadata_path):
                write_json_atomic(metadata_path, expected)
                return (root, metadata_path)
            raise ValueError('output root 非空但没有 dataset_meta.json；拒绝猜测文件来源')
        with metadata_path.open('r', encoding='utf-8') as handle:
            current = json.load(handle)
        for key in ('format_version', 'dataset_id', 'semantic_hash', 'semantic_config', 'implementation_version', 'implementation_fingerprint', 'runtime'):
            if current.get(key) != expected[key]:
                raise ValueError(f'已有 dataset metadata 的 {key} 与当前配置不一致')
        status = current.get('status')
        if status == 'finalized_unvalidated':
            raise RuntimeError('Output directory is finalized; use a separate directory to generate another dataset.')
        if status != 'in_progress':
            raise ValueError(f'dataset metadata 状态不可恢复: {status!r}')
    else:
        write_json_atomic(metadata_path, expected)
    return (root, metadata_path)

def _worker_generate(task: SceneTask, config: GeneratorConfig, root_text: str, implementation_hash: str) -> SampleArtifact:
    root = Path(root_text)
    result = simulate_task(task, config)
    return write_sample_atomic(root, task, result, config, implementation_hash)

def _bounded_parallel(tasks: list[SceneTask], config: GeneratorConfig, root: Path, implementation_hash: str) -> Iterator[SampleArtifact]:
    if config.runtime.workers == 1:
        for task in tasks:
            yield _worker_generate(task, config, str(root), implementation_hash)
        return
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.runtime.workers, mp_context=context) as executor:
        iterator = iter(tasks)
        pending: dict[concurrent.futures.Future[SampleArtifact], SceneTask] = {}
        for _ in range(min(2 * config.runtime.workers, len(tasks))):
            task = next(iterator, None)
            if task is not None:
                pending[executor.submit(_worker_generate, task, config, str(root), implementation_hash)] = task
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                yield future.result()
                task = next(iterator, None)
                if task is not None:
                    pending[executor.submit(_worker_generate, task, config, str(root), implementation_hash)] = task

def _validate_artifact_set(tasks: list[SceneTask], artifacts: list[SampleArtifact], config: GeneratorConfig) -> None:
    expected = {task.sample_id: task for task in tasks}
    if len(expected) != len(tasks):
        raise RuntimeError('scene plan 自身含重复 sample_id')
    observed_ids = [artifact.sample_id for artifact in artifacts]
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(expected):
        raise RuntimeError('artifact sample_id 集合与 scene plan 不完全一致')
    expected_counts = asdict(config.splits)
    observed_counts = Counter((artifact.split for artifact in artifacts))
    if any((observed_counts[split] != expected_counts[split] for split in SPLIT_ORDER)):
        raise RuntimeError(f"artifact split 计数错误；expected={expected_counts}, observed={{{', '.join((f'{split!r}: {observed_counts[split]}' for split in SPLIT_ORDER))}}}")
    paths = [str(Path(artifact.npz_path).resolve()) for artifact in artifacts]
    hashes = [artifact.npz_sha256 for artifact in artifacts]
    if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise RuntimeError('artifact 出现重复 NPZ path 或内容 hash，拒绝发布 manifest')
    for artifact in artifacts:
        task = expected[artifact.sample_id]
        if artifact.split != task.split or artifact.scene_id != task.scene_id:
            raise RuntimeError(f'artifact 身份与 task 不一致: {artifact.sample_id}')
        diagnostics = (artifact.ball_ball_events, artifact.ball_wall_events, artifact.total_impulse, artifact.events, artifact.attempts, artifact.closure_float64_max_error, artifact.closure_float32_max_error, artifact.seam_quiet)
        if any((value is None for value in diagnostics)):
            raise RuntimeError(f'artifact 缺少 event/closure provenance: {artifact.sample_id}')
        event_count = int(artifact.ball_ball_events or 0) + int(artifact.ball_wall_events or 0)
        if task.collision_requirement == 'at_least' and event_count < config.sampling.minimum_collision_events:
            raise RuntimeError(f'强制碰撞 task 未满足接受条件: {artifact.sample_id}')
        if task.collision_requirement == 'zero' and event_count != 0:
            raise RuntimeError(f'zero-collision task 含碰撞: {artifact.sample_id}')

def _reject_unplanned_sample_files(root: Path, tasks: list[SceneTask]) -> None:
    expected = {sample_path(root, task).resolve() for task in tasks}
    sample_root = root / 'samples'
    observed = {path.resolve() for path in sample_root.rglob('*.npz')} if sample_root.is_dir() else set()
    extra = observed - expected
    if extra:
        preview = sorted((str(path) for path in extra))[:3]
        raise ValueError(f'samples 树含 scene plan 外的 final NPZ: {preview}')

def _finalize_dataset(*, root: Path, metadata_path: Path, tasks: list[SceneTask], artifacts: list[SampleArtifact], config: GeneratorConfig, budget: GenerationBudget, runtime: dict[str, str], implementation_hash: str, free_bytes: int, required_bytes: int) -> dict[str, object]:
    _validate_artifact_set(tasks, artifacts, config)
    grouped: dict[str, list[SampleArtifact]] = defaultdict(list)
    for artifact in artifacts:
        grouped[artifact.split].append(artifact)
    manifest_hashes: dict[str, str] = {}
    manifest_records: dict[str, list[dict[str, str]]] = {}
    for split in SPLIT_ORDER:
        ordered = sorted(grouped[split], key=lambda item: item.sample_id)
        manifest_path = root / 'manifests' / f'{split}.jsonl'
        records = [item.manifest_record(manifest_path.parent) for item in ordered]
        write_jsonl_atomic(manifest_path, records)
        write_jsonl_atomic(root / 'metadata' / 'events' / f'{split}.jsonl', [item.event_record() for item in ordered])
        manifest_hashes[split] = sha256_file(manifest_path)
        manifest_records[split] = records
    train_manifest = root / 'manifests' / 'train.jsonl'
    phase_scales = compute_train_rms(train_manifest, manifest_records['train'])
    stats_path = root / 'stats' / 'phase_scales.json'
    write_json_atomic(stats_path, phase_scales)
    summary: dict[str, object] = {'format_version': 1, 'semantic_hash': config.semantic_hash, 'implementation_version': GENERATOR_IMPLEMENTATION_VERSION, 'implementation_fingerprint': implementation_hash, 'samples': len(artifacts), 'new_samples': sum((not item.reused for item in artifacts)), 'reused_samples': sum((item.reused for item in artifacts)), 'diagnostics_known_samples': len(artifacts), 'ball_ball_events': sum((item.ball_ball_events or 0 for item in artifacts)), 'ball_wall_events': sum((item.ball_wall_events or 0 for item in artifacts)), 'realized_zero_collision_train': sum(((item.ball_ball_events or 0) + (item.ball_wall_events or 0) == 0 for item in grouped['train'])), 'max_closure_float64_error': max((item.closure_float64_max_error or 0.0 for item in artifacts), default=0.0), 'max_closure_float32_error': max((item.closure_float32_max_error or 0.0 for item in artifacts), default=0.0), 'seam_quiet_samples': sum((item.seam_quiet is True for item in artifacts)), 'seam_nonquiet_samples': sum((item.seam_quiet is False for item in artifacts)), 'budget': budget.as_dict(), 'free_bytes_before_generation': free_bytes, 'required_bytes_with_reserve': required_bytes, 'manifest_sha256': manifest_hashes, 'stats_sha256': sha256_file(stats_path)}
    write_json_atomic(root / 'metadata' / 'generation_summary.json', summary)
    finalized_metadata = {'format_version': 1, 'dataset_id': config.semantic_hash[:16], 'semantic_hash': config.semantic_hash, 'semantic_config': config.semantic_payload(), 'implementation_version': GENERATOR_IMPLEMENTATION_VERSION, 'implementation_fingerprint': implementation_hash, 'runtime': runtime, 'stats_algorithm': 'train_all_states_objects_coordinatewise_rms_about_zero_float64', 'status': 'finalized_unvalidated', 'manifest_sha256': manifest_hashes, 'stats_sha256': sha256_file(stats_path)}
    write_json_atomic(metadata_path, finalized_metadata)
    return summary

def generate_dataset(config: GeneratorConfig) -> dict[str, object]:
    implementation_hash = implementation_fingerprint()
    with _generation_lock(config, implementation_hash):
        return _generate_dataset_locked(config, implementation_hash)

def _generate_dataset_locked(config: GeneratorConfig, implementation_hash: str) -> dict[str, object]:
    budget = estimate_generation_budget(config)
    runtime = _runtime_payload(config)
    root, metadata_path = _prepare_root(config, runtime, implementation_hash)
    state_path = root / 'state' / 'completed.jsonl'
    recorded = load_state_log(state_path, config.semantic_hash)
    tasks = build_scene_plan(config)
    _reject_unplanned_sample_files(root, tasks)
    task_by_id = {task.sample_id: task for task in tasks}
    unknown_records = set(recorded) - set(task_by_id)
    if unknown_records:
        raise ValueError(f'state log 含 scene plan 之外的 sample: {sorted(unknown_records)[:3]}')
    artifacts: list[SampleArtifact] = []
    remaining: list[SceneTask] = []
    for task in tasks:
        artifact = artifact_from_existing(root, task, config, recorded.get(task.sample_id), implementation_hash)
        if artifact is None:
            remaining.append(task)
        else:
            artifacts.append(artifact)
            if task.sample_id not in recorded:
                append_state_log(state_path, artifact, config.semantic_hash, sha256_file(sample_provenance_path(root, task)))
    free_bytes, required_bytes = check_disk_budget(config, budget, [Path(artifact.npz_path) for artifact in artifacts])
    for artifact in _bounded_parallel(remaining, config, root, implementation_hash):
        artifacts.append(artifact)
        task = task_by_id[artifact.sample_id]
        if artifact.sample_id not in recorded:
            append_state_log(state_path, artifact, config.semantic_hash, sha256_file(sample_provenance_path(root, task)))
        print(f'generated {len(artifacts)}/{len(tasks)}: {artifact.sample_id}', flush=True)
    return _finalize_dataset(root=root, metadata_path=metadata_path, tasks=tasks, artifacts=artifacts, config=config, budget=budget, runtime=runtime, implementation_hash=implementation_hash, free_bytes=free_bytes, required_bytes=required_bytes)
