"""Export full generated episodes to the evaluation HDF5 schema."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

H1_SPLITS = {'train': 'train', 'dev': 'validation', 'calibration_fit': 'calibration', 'test': 'extra'}
H2_KEYS = ('phase', 'attrs', 'object_mask', 'spring_mask', 'spring_k',
           'spring_rest_length', 'time', 'window_start', 'gravity', 'bounds')


def _records(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate(row, dataset):
    objects, qdim = (5, 2) if dataset == 'h1' else (10, 3)
    shapes = {'phase': (193, objects, 2 * qdim), 'attrs': (objects, 3),
              'time': (193,), 'contact': (192, objects)}
    if dataset == 'h2':
        shapes.update(object_mask=(objects,), spring_mask=(objects, objects),
                      spring_k=(objects, objects), spring_rest_length=(objects, objects))
    for key, shape in shapes.items():
        if np.shape(row[key]) != shape:
            raise ValueError(f'{key}: expected full-episode shape {shape}, got {np.shape(row[key])}')
        if not np.isfinite(row[key]).all():
            raise ValueError(f'{key}: nonfinite data')
    if not (np.diff(row['time']) > 0).all():
        raise ValueError('episode times must increase')
    if not np.isin(row['contact'], (0, 1)).all():
        raise ValueError('contact must be binary')
    if dataset == 'h2' and np.any(row['contact'][:, ~row['object_mask'].astype(bool)]):
        raise ValueError('padded object has a contact label')


def _write_row(group, row, index, count, dataset):
    _validate(row, dataset)
    for key, value in row.items():
        value = np.asarray(value)
        if index == 0:
            group.create_dataset(key, shape=(count, *value.shape), dtype=value.dtype,
                                 chunks=(min(32, count), *value.shape), compression='lzf')
        group[key][index] = value


def _export_h1(root, dst, splits, seeds):
    for target in splits:
        split = next(k for k, v in H1_SPLITS.items() if v == target)
        entries = []
        for seed in seeds:
            source = root / f'source_seed{seed}'
            manifest = source / 'manifests' / f'{split}.jsonl'
            records = _records(manifest)
            events = _records(source / 'metadata' / 'events' / f'{split}.jsonl')
            indexed = {r['sample_id']: r for r in events}
            if len(indexed) != len(events) or len({r['sample_id'] for r in records}) != len(records):
                raise ValueError('duplicate sample identifiers')
            if set(indexed) != {r['sample_id'] for r in records}:
                raise ValueError('event and sample identifiers do not match')
            entries.extend((manifest, r, indexed[r['sample_id']]) for r in records)
        if not entries:
            raise ValueError(f'empty {target} split')
        group = dst.create_group(target)
        for i, (manifest, record, diagnostics) in enumerate(entries):
            path = Path(record['npz_path'])
            if not path.is_absolute():
                path = manifest.parent / path
            with np.load(path, allow_pickle=False) as source:
                row = {k: source[k] for k in ('phase', 'attrs', 'time')}
                if 'frame_contact' in source:
                    row['contact'] = source['frame_contact']
            if 'contact' not in row:
                from .contact_labels import replay_contacts
                row['contact'] = replay_contacts(manifest.parent.parent, record, diagnostics, row['phase'], row['attrs'])
            _write_row(group, row, i, len(entries), 'h1')
        # The full episode pack defines the same seed-then-manifest ordering.
        pack = root / 'packs' / f'{target}192'
        if pack.is_dir():
            for key in ('phase', 'attrs', 'time'):
                reference = np.load(pack / f'{key}.npy', mmap_mode='r', allow_pickle=False)
                if reference.shape != group[key].shape:
                    raise ValueError('episode pack length mismatch')
                for i in range(0, len(entries), 32):
                    if not np.array_equal(reference[i:i + 32], group[key][i:i + 32]):
                        raise ValueError(f'episode pack ordering/content mismatch: {key}')


def _export_h2(root, dst, splits):
    for target in splits:
        split = 'expansion' if target == 'extra' else target
        records = _records(root / 'manifests' / f'{split}.jsonl')
        if not records:
            raise ValueError(f'empty {target} split')
        group = dst.create_group(target)
        loaded, arrays = None, None
        for i, record in enumerate(records):
            if 'shard_path' in record:
                path = root / record['shard_path']
                if path != loaded:
                    with np.load(path, allow_pickle=False) as source:
                        arrays = {k: source[k] for k in (*H2_KEYS, 'qa_frame_contact')}
                    loaded = path
                row_index = int(record['shard_row'])
                row = {k: arrays[k][row_index] for k in H2_KEYS}
                row['contact'] = arrays['qa_frame_contact'][row_index]
            else:
                with np.load(root / record['data_path'], allow_pickle=False) as source:
                    row = {k: source[k] for k in H2_KEYS}
                with np.load(root / record['qa_path'], allow_pickle=False) as qa:
                    row['contact'] = qa['frame_contact']
            _write_row(group, row, i, len(records), 'h2')


def export_hdf5(dataset, root, output, *, splits=('train', 'validation'), seeds=(40, 41, 42, 43)):
    """Preserve physical coordinates, full episodes and source ordering."""
    root, output = Path(root), Path(output)
    if dataset not in ('h1', 'h2'):
        raise ValueError('dataset must be h1 or h2')
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(H1_SPLITS.values()):
        raise ValueError('invalid or repeated splits')
    if not seeds or tuple(seeds) != tuple(sorted(set(seeds))):
        raise ValueError('physical seeds must be unique and increasing')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(name, 'w') as dst:
            dst.attrs['schema'] = 'hamiformer.episodes.v1'
            dst.attrs['dataset'] = dataset
            if dataset == 'h1':
                _export_h1(root, dst, splits, seeds)
            else:
                _export_h2(root, dst, splits)
        if output.exists():
            raise FileExistsError(output)
        os.replace(name, output)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('h1', 'h2'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=tuple(H1_SPLITS.values()), default=['train', 'validation'])
    args = parser.parse_args()
    export_hdf5(args.dataset, args.root, args.output, splits=args.splits)


if __name__ == '__main__':
    main()
