from __future__ import annotations
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
BUNDLE_SCHEMA = 'hamiformer.hamiballs.trajectory_bundle.v1'
VIDEO_MANIFEST_SCHEMA = 'hamiformer.hamiballs.trajectory_video_manifest.v1'
PREDICTION_PREFIX = 'pred_'
STATE_LAYOUT = ('qx', 'qy', 'px', 'py')
_METHOD_RE = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]*$')

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.ndim != 0:
        raise ValueError('trajectory bundle metadata must be scalar')
    return value.item()

def _finite_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != 'f' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite floating-point array')
    return np.array(array, dtype=np.float32, copy=True)

def _normalise_prediction_shape(name: str, prediction: np.ndarray, gt_phase: np.ndarray) -> tuple[np.ndarray, bool]:
    value = _finite_float(name, prediction)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape == gt_phase.shape:
        return (value, False)
    if value.shape == gt_phase.shape[1:]:
        raise ValueError(f'{name} contains one state, not a trajectory')
    if value.shape == (gt_phase.shape[0] - 1, *gt_phase.shape[1:]):
        return (np.concatenate((gt_phase[:1], value), axis=0), True)
    raise ValueError(f'{name} shape must be {gt_phase.shape} or future-only {(gt_phase.shape[0] - 1, *gt_phase.shape[1:])}, got {value.shape}')

def _parse_prediction_spec(spec: str) -> tuple[str, Path, str | None]:
    if '=' not in spec:
        raise ValueError('prediction spec must be NAME=PATH or NAME=PATH::KEY')
    name, location = spec.split('=', 1)
    if not _METHOD_RE.fullmatch(name):
        raise ValueError(f'invalid prediction method name: {name!r}')
    if '::' in location:
        path_text, key = location.rsplit('::', 1)
        if not key:
            raise ValueError('prediction spec has an empty NPZ key')
    else:
        path_text, key = (location, None)
    if not path_text:
        raise ValueError('prediction spec has an empty path')
    return (name, Path(path_text).expanduser().resolve(), key)

@dataclass(frozen=True)
class TrajectoryBundle:
    gt_phase: np.ndarray
    attrs: np.ndarray
    time: np.ndarray
    predictions: Mapping[str, np.ndarray]
    event_edge: np.ndarray | None
    metadata: Mapping[str, Any]
    input_files: tuple[Mapping[str, str], ...]
    prepended_initial_state: tuple[str, ...] = ()

    @property
    def frames(self) -> int:
        return int(self.gt_phase.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.gt_phase.shape[1])

    def select_methods(self, names: Sequence[str] | None) -> 'TrajectoryBundle':
        if names is None:
            return self
        ordered = tuple(dict.fromkeys(names))
        missing = set(ordered) - set(self.predictions)
        if missing:
            raise ValueError(f'trajectory bundle lacks methods: {sorted(missing)}')
        return replace(self, predictions={name: self.predictions[name] for name in ordered}, prepended_initial_state=tuple((name for name in self.prepended_initial_state if name in ordered)))

    def with_event_edge(self, event_edge: np.ndarray | None, *, source_path: str | Path | None=None) -> 'TrajectoryBundle':
        files = self.input_files
        if source_path is not None:
            path = Path(source_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            files = (*files, {'role': 'event_sidecar', 'path': str(path), 'sha256': _sha256(path)})
        return replace(self, event_edge=_validate_event_edge(event_edge, self.frames), input_files=files)

def _validate_event_edge(value: np.ndarray | None, frames: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (frames - 1,):
        raise ValueError(f'event_edge must have shape {(frames - 1,)}, got {array.shape}')
    if array.dtype.kind not in {'b', 'i', 'u'}:
        raise ValueError('event_edge must be boolean/integer')
    return np.array(array, dtype=bool, copy=True)

def _read_prediction_array(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if key is None:
            candidates = [name for name in ('phase', 'prediction', 'pred') if name in payload.files]
            if len(candidates) != 1:
                raise ValueError(f'prediction NPZ {path} needs an explicit ::KEY; candidates={candidates}')
            key = candidates[0]
        if key not in payload.files:
            raise ValueError(f'prediction NPZ {path} lacks key {key!r}')
        value = np.array(payload[key], copy=True)
    return (value, key)

def load_trajectory_bundle(input_path: str | Path, *, prediction_specs: Iterable[str]=()) -> TrajectoryBundle:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        gt_keys = [name for name in ('phase_gt', 'gt_phase', 'phase') if name in payload.files]
        if not gt_keys:
            raise ValueError('input NPZ lacks phase_gt/gt_phase/phase')
        gt_phase = _finite_float(gt_keys[0], payload[gt_keys[0]])
        if gt_phase.ndim == 4 and gt_phase.shape[0] == 1:
            gt_phase = gt_phase[0]
        if gt_phase.ndim != 3 or gt_phase.shape[-1] != 4 or gt_phase.shape[0] < 2:
            raise ValueError('GT phase must have shape [frames>=2, objects, 4]')
        if 'attrs' not in payload.files or 'time' not in payload.files:
            raise ValueError('input NPZ must contain attrs and time')
        attrs = _finite_float('attrs', payload['attrs'])
        time = _finite_float('time', payload['time'])
        if attrs.ndim != 2 or attrs.shape[0] != gt_phase.shape[1] or attrs.shape[1] < 2:
            raise ValueError('attrs must have shape [objects, >=2] with radius in column 1')
        if np.any(attrs[:, 1] <= 0.0):
            raise ValueError('all object radii must be positive')
        if time.shape != (gt_phase.shape[0],) or not np.all(np.diff(time) > 0.0):
            raise ValueError('time must be strictly increasing with one value per frame')
        predictions: dict[str, np.ndarray] = {}
        prepended: list[str] = []
        for key in sorted((name for name in payload.files if name.startswith(PREDICTION_PREFIX))):
            method = key[len(PREDICTION_PREFIX):]
            if not _METHOD_RE.fullmatch(method):
                raise ValueError(f'invalid method encoded by bundle key: {key!r}')
            value, added = _normalise_prediction_shape(key, payload[key], gt_phase)
            predictions[method] = value
            if added:
                prepended.append(method)
        event_edge = _validate_event_edge(payload['event_edge'], gt_phase.shape[0]) if 'event_edge' in payload.files else None
        metadata: dict[str, Any] = {}
        for key in ('schema', 'sample_id', 'source_index', 'noise_seed', 'split', 'steps'):
            if key in payload.files:
                metadata[key] = _scalar(payload[key])
        if 'schema' in metadata and str(metadata['schema']) != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported trajectory bundle schema: {metadata['schema']!r}")
    input_files: list[Mapping[str, str]] = [{'role': 'trajectory_bundle', 'path': str(path), 'sha256': _sha256(path)}]
    for spec in prediction_specs:
        method, prediction_path, key = _parse_prediction_spec(spec)
        if method in predictions:
            raise ValueError(f'duplicate prediction method: {method}')
        raw, resolved_key = _read_prediction_array(prediction_path, key)
        value, added = _normalise_prediction_shape(method, raw, gt_phase)
        predictions[method] = value
        if added:
            prepended.append(method)
        input_files.append({'role': f'prediction:{method}:{resolved_key}', 'path': str(prediction_path), 'sha256': _sha256(prediction_path)})
    return TrajectoryBundle(gt_phase=gt_phase, attrs=attrs, time=time, predictions=predictions, event_edge=event_edge, metadata=metadata, input_files=tuple(input_files), prepended_initial_state=tuple(prepended))

def _read_event_record(path: Path, sample_id: str | None) -> Mapping[str, Any]:
    text = path.read_text(encoding='utf-8')
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and 'events' in document:
        return document
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'event sidecar is empty: {path}')
    if len(rows) == 1 and isinstance(rows[0], dict) and ('events' in rows[0]):
        return rows[0]
    if sample_id is None:
        raise ValueError('event JSONL requires bundle sample_id metadata')
    matches = [row for row in rows if isinstance(row, dict) and row.get('sample_id') == sample_id]
    if len(matches) != 1:
        raise ValueError(f'event JSONL has {len(matches)} rows for sample_id={sample_id!r}')
    return matches[0]

def load_event_edges(path: str | Path, *, frames: int, sample_id: str | None, substeps: int) -> np.ndarray:
    if substeps < 1:
        raise ValueError('substeps must be positive')
    event_path = Path(path).expanduser().resolve()
    record = _read_event_record(event_path, sample_id)
    events = record.get('events')
    if not isinstance(events, list):
        raise ValueError('event sidecar record lacks an events list')
    result = np.zeros(frames - 1, dtype=bool)
    for event in events:
        if not isinstance(event, dict) or type(event.get('substep')) is not int:
            raise ValueError('event sidecar contains an invalid substep')
        edge = int(event['substep']) // substeps
        if not 0 <= edge < frames - 1:
            raise ValueError(f"event substep maps outside trajectory edges: {event['substep']}")
        result[edge] = True
    return result

def _bounds(bundle: TrajectoryBundle, box_half_extent: float | None) -> float:
    if box_half_extent is not None:
        if not math.isfinite(box_half_extent) or box_half_extent <= 0.0:
            raise ValueError('box_half_extent must be finite and positive')
        return float(box_half_extent)
    maximum = float(np.max(np.abs(bundle.gt_phase[..., :2])))
    for value in bundle.predictions.values():
        maximum = max(maximum, float(np.max(np.abs(value[..., :2]))))
    return max(maximum + float(np.max(bundle.attrs[:, 1])) + 0.05, 0.25)

def _central_spring_polyline(position: np.ndarray, radius: float, *, extent: float, coils: int=7) -> tuple[np.ndarray, np.ndarray]:
    if coils < 1:
        raise ValueError('spring coils must be positive')
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('spring position must be one finite qx/qy pair')
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError('spring radius must be finite and positive')
    distance = float(np.linalg.norm(point))
    anchor_radius = min(0.018 * extent, 0.02)
    if distance <= radius + anchor_radius + 1e-09:
        return (np.asarray((0.0, float(point[0]))), np.asarray((0.0, float(point[1]))))
    direction = point / distance
    normal = np.asarray((-direction[1], direction[0]))
    start = direction * anchor_radius
    end = point - direction * radius
    length = float(np.linalg.norm(end - start))
    fractions = np.linspace(0.0, 1.0, 2 * coils + 1)
    coordinates = start[None, :] + fractions[:, None] * (end - start)[None, :]
    amplitude = min(0.03 * extent, 0.18 * length)
    offsets = np.zeros_like(fractions)
    offsets[1:-1] = amplitude * np.where(np.arange(1, len(fractions) - 1) % 2 == 1, 1.0, -1.0)
    coordinates += offsets[:, None] * normal[None, :]
    return (coordinates[:, 0], coordinates[:, 1])

def render_trajectory_video(bundle: TrajectoryBundle, output_path: str | Path, *, layout: str='panels', fps: int=24, tail: int=16, max_edges: int | None=None, dpi: int=110, box_half_extent: float | None=1.0, show_springs: bool=True, title: str | None=None, manifest_path: str | Path | None=None, overwrite: bool=False) -> dict[str, Any]:
    if layout not in {'panels', 'overlay', 'comparison'}:
        raise ValueError('layout must be panels, overlay, or comparison')
    if fps < 1 or tail < 0 or dpi < 40:
        raise ValueError('fps must be positive, tail nonnegative, and dpi >= 40')
    source_edges = bundle.frames - 1
    if max_edges is not None:
        if isinstance(max_edges, bool) or not isinstance(max_edges, int):
            raise ValueError('max_edges must be an integer or None')
        if not 1 <= max_edges <= source_edges:
            raise ValueError(f'max_edges must be in [1, {source_edges}], got {max_edges}')
        bundle = replace(bundle, gt_phase=bundle.gt_phase[:max_edges + 1], time=bundle.time[:max_edges + 1], predictions={name: values[:max_edges + 1] for name, values in bundle.predictions.items()}, event_edge=None if bundle.event_edge is None else bundle.event_edge[:max_edges])
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {'.gif', '.mp4'}:
        raise ValueError('output extension must be .gif or .mp4')
    manifest = Path(manifest_path).expanduser().resolve() if manifest_path is not None else output.with_suffix(output.suffix + '.manifest.json')
    for path in (output, manifest):
        if path.exists() and (not overwrite):
            raise FileExistsError(f'refusing to overwrite existing output: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
    from matplotlib.patches import Circle, Rectangle
    suffix = output.suffix.lower()
    if suffix == '.mp4' and (not writers.is_available('ffmpeg')):
        raise RuntimeError('MP4 output requires an ffmpeg executable visible to Matplotlib; install ffmpeg or choose a .gif output')
    series = [('GT', bundle.gt_phase), *bundle.predictions.items()]
    prediction_series = list(bundle.predictions.items())
    if layout == 'comparison':

        def comparison_order(item: tuple[str, np.ndarray]) -> int:
            name = item[0].lower().replace('_', '-')
            return 0 if name == 'main' else 2 if name == 'wide-d' else 1
        prediction_series.sort(key=comparison_order)
    if layout == 'comparison' and (not prediction_series):
        raise ValueError('comparison layout requires at least one prediction')
    count = len(series)
    if layout == 'overlay':
        rows, columns = (1, 1)
    elif layout == 'comparison':
        rows, columns = (1, 1 + len(prediction_series))
    else:
        columns = min(3, count)
        rows = int(math.ceil(count / columns))
    figure, axes_value = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), squeeze=False, constrained_layout=True)
    axes = list(axes_value.flat)
    if layout == 'overlay':
        used_axes = axes[:1]
    elif layout == 'panels':
        used_axes = axes[:count]
    else:
        used_axes = axes[:columns]
    for axis in axes[len(used_axes):]:
        axis.set_visible(False)
    extent = _bounds(bundle, box_half_extent)
    palette = plt.get_cmap('tab10')
    object_colours = [palette(index % 10) for index in range(bundle.num_objects)]
    panels: list[dict[str, Any]] = []
    if layout == 'panels':
        panel_specs = [(axis, label, [(label, values)]) for axis, (label, values) in zip(used_axes, series)]
    elif layout == 'comparison':
        panel_specs = [(used_axes[0], 'GT', [('GT', bundle.gt_phase)]), *[(axis, label, [('GT', bundle.gt_phase), (label, values)]) for axis, (label, values) in zip(used_axes[1:], prediction_series)]]
    else:
        panel_specs = [(used_axes[0], 'GT and predictions', series)]
    line_styles = ('-', '--', ':', '-.')
    for panel_index, (axis, panel_label, traces) in enumerate(panel_specs):
        axis.add_patch(Rectangle((-extent, -extent), 2.0 * extent, 2.0 * extent, fill=False, linewidth=1.3, edgecolor='#4b5563'))
        axis.set_xlim(-1.05 * extent, 1.05 * extent)
        axis.set_ylim(-1.05 * extent, 1.05 * extent)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(color='#d1d5db', linewidth=0.45, alpha=0.6)
        axis.set_xlabel('qx')
        axis.set_ylabel('qy' if panel_index == 0 else '')
        axis.add_patch(Circle((0.0, 0.0), radius=min(0.018 * extent, 0.02), facecolor='#111827', edgecolor='none', zorder=5))
        panel_traces: list[dict[str, Any]] = []
        for trace_index, (label, values) in enumerate(traces):
            is_gt = label == 'GT'
            if layout == 'overlay':
                linestyle = line_styles[trace_index % len(line_styles)]
            elif layout == 'comparison':
                linestyle = '--' if is_gt and panel_label != 'GT' else '-'
            else:
                linestyle = '-' if is_gt else '--'
            render_springs = show_springs and (layout != 'overlay' or is_gt)
            circles: list[Any] = []
            lines: list[Any] = []
            springs: list[Any] = []
            for object_index in range(bundle.num_objects):
                colour = object_colours[object_index]
                circle = Circle(tuple(values[0, object_index, :2]), radius=float(bundle.attrs[object_index, 1]), facecolor=colour if is_gt else 'none', edgecolor=colour, linestyle=linestyle, alpha=0.82 if is_gt else 0.96, linewidth=1.65 if is_gt else 1.35, zorder=3 if is_gt else 4)
                axis.add_patch(circle)
                line = axis.plot([], [], color=colour, linestyle=linestyle, linewidth=1.25, alpha=0.8 if is_gt else 0.72, zorder=2 if is_gt else 3)[0]
                circles.append(circle)
                lines.append(line)
                if render_springs:
                    spring = axis.plot([], [], color=colour, linestyle=linestyle, linewidth=0.85, alpha=0.52 if is_gt else 0.68, zorder=1 if is_gt else 2)[0]
                    springs.append(spring)
            panel_traces.append({'label': label, 'values': values, 'circles': circles, 'lines': lines, 'springs': springs})
        panels.append({'axis': axis, 'label': panel_label, 'traces': panel_traces})
    if layout == 'overlay':
        legend_handles = []
        for index, (label, _values) in enumerate(series):
            legend_handles.append(used_axes[0].plot([], [], color='#111827', linestyle=line_styles[index % len(line_styles)], label=label)[0])
        used_axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8)
        used_axes[0].set_title('GT and predictions')
    suptitle = figure.suptitle('')

    def update(frame: int) -> list[Any]:
        artists: list[Any] = []
        left = max(0, frame - tail)
        for panel in panels:
            for trace in panel['traces']:
                values = trace['values']
                for object_index, (circle, line) in enumerate(zip(trace['circles'], trace['lines'])):
                    position = values[frame, object_index, :2]
                    circle.center = (float(position[0]), float(position[1]))
                    history = values[left:frame + 1, object_index, :2]
                    line.set_data(history[:, 0], history[:, 1])
                    artists.extend((circle, line))
                    if object_index < len(trace['springs']):
                        spring_x, spring_y = _central_spring_polyline(position, float(bundle.attrs[object_index, 1]), extent=extent)
                        spring = trace['springs'][object_index]
                        spring.set_data(spring_x, spring_y)
                        artists.append(spring)
            if layout == 'panels':
                label = panel['label']
                if label == 'GT':
                    panel['axis'].set_title('GT')
                else:
                    values = panel['traces'][0]['values']
                    q_error = values[:frame + 1, :, :2] - bundle.gt_phase[:frame + 1, :, :2]
                    q_rmse = float(np.sqrt(np.mean(np.square(q_error))))
                    panel['axis'].set_title(f'{label} | q-RMSE={q_rmse:.4f}')
            elif layout == 'comparison':
                label = panel['label']
                if label == 'GT':
                    panel['axis'].set_title('GT')
                else:
                    values = panel['traces'][1]['values']
                    q_error = values[:frame + 1, :, :2] - bundle.gt_phase[:frame + 1, :, :2]
                    q_rmse = float(np.sqrt(np.mean(np.square(q_error))))
                    panel['axis'].set_title(f'{label} vs GT | q-RMSE={q_rmse:.4f}', fontsize=9)
        event = bool(bundle.event_edge is not None and frame > 0 and bundle.event_edge[frame - 1])
        prefix = f'{title} | ' if title else ''
        style_legend = 'Comparison: GT filled/dashed; prediction open/solid | ' if layout == 'comparison' else ''
        suptitle.set_text(f'{prefix}{style_legend}frame {frame}/{bundle.frames - 1} | t={float(bundle.time[frame]):.3f}s' + (' | CONTACT' if event else ''))
        suptitle.set_color('#b91c1c' if event else '#111827')
        artists.append(suptitle)
        return artists
    animation = FuncAnimation(figure, update, frames=bundle.frames, interval=1000.0 / fps, blit=False, repeat=False)
    if suffix == '.mp4':
        writer: Any = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        encoder = 'matplotlib-ffmpeg-libx264'
    else:
        writer = PillowWriter(fps=fps)
        encoder = 'matplotlib-pillow'
    temporary = output.with_name(f'.{output.stem}.tmp-{os.getpid()}{output.suffix}')
    try:
        animation.save(str(temporary), writer=writer, dpi=dpi)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError('animation writer produced an empty file')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    payload: dict[str, Any] = {'schema': VIDEO_MANIFEST_SCHEMA, 'status': 'COMPLETE', 'input_files': list(bundle.input_files), 'bundle_metadata': dict(bundle.metadata), 'trajectory': {'frames': bundle.frames, 'edges': bundle.frames - 1, 'source_edges': source_edges, 'rendered_prefix_edges': max_edges, 'objects': bundle.num_objects, 'state_layout': list(STATE_LAYOUT), 'methods': list(bundle.predictions), 'event_edges': int(bundle.event_edge.sum()) if bundle.event_edge is not None else None, 'prepended_initial_state': list(bundle.prepended_initial_state)}, 'render': {'layout': layout, 'panel_grid': {'rows': rows, 'columns': columns, 'panel_order': ['GT', *[name for name, _ in prediction_series]] if layout == 'comparison' else [panel['label'] for panel in panels]}, 'comparison_style': {'GT': {'ball': 'filled', 'trajectory': 'dashed', 'spring': 'dashed'}, 'prediction': {'ball': 'open', 'trajectory': 'solid', 'spring': 'solid'}, 'reference_panel': 'GT solid'} if layout == 'comparison' else None, 'fps': fps, 'tail_frames': tail, 'max_edges': max_edges, 'dpi': dpi, 'box_half_extent': extent, 'central_springs': {'enabled': bool(show_springs), 'anchor': [0.0, 0.0], 'coils': 7, 'overlay_scope': 'GT_only' if layout == 'overlay' else 'all_panels'}, 'encoder': encoder, 'matplotlib_version': str(matplotlib.__version__), 'numpy_version': str(np.__version__), 'output_path': str(output), 'output_sha256': _sha256(output), 'output_bytes': output.stat().st_size}}
    _atomic_json(manifest, payload)
    payload['manifest_path'] = str(manifest)
    payload['manifest_sha256'] = _sha256(manifest)
    return payload
