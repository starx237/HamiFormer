from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
from hamiformer.utils import sha256_file
PACK_SCHEMA = 'hamiformer.hamiballs2.crop-pack.v1'

def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f'empty HamiBalls-2 manifest: {path}')
    return records

def build_hamiballs2_crop_pack(manifest_path: str | Path, output_dir: str | Path, *, include_contact_labels: bool) -> Path:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    manifest_hash = sha256_file(manifest)
    metadata_path = output / 'metadata.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('schema') == PACK_SCHEMA and metadata.get('manifest_sha256') == manifest_hash and (bool(metadata.get('include_contact_labels')) == include_contact_labels):
            return output
    records = _read_manifest(manifest)
    dataset_root = manifest.parent.parent
    temporary = output.with_name(output.name + '.building')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    count = len(records)
    arrays: dict[str, np.memmap] = {'phase': open_memmap(temporary / 'phase.npy', mode='w+', dtype=np.float32, shape=(count, 49, 10, 6)), 'attrs': open_memmap(temporary / 'attrs.npy', mode='w+', dtype=np.float32, shape=(count, 10, 3)), 'object_mask': open_memmap(temporary / 'object_mask.npy', mode='w+', dtype=np.uint8, shape=(count, 10)), 'spring_mask': open_memmap(temporary / 'spring_mask.npy', mode='w+', dtype=np.uint8, shape=(count, 10, 10)), 'spring_k': open_memmap(temporary / 'spring_k.npy', mode='w+', dtype=np.float32, shape=(count, 10, 10)), 'spring_rest_length': open_memmap(temporary / 'spring_rest_length.npy', mode='w+', dtype=np.float32, shape=(count, 10, 10)), 'time': open_memmap(temporary / 'time.npy', mode='w+', dtype=np.float32, shape=(count, 49)), 'physical_seed': open_memmap(temporary / 'physical_seed.npy', mode='w+', dtype=np.int16, shape=(count,))}
    if include_contact_labels:
        arrays['contact'] = open_memmap(temporary / 'contact.npy', mode='w+', dtype=np.uint8, shape=(count, 48, 10))
    current_shard_path: Path | None = None
    current_shard: dict[str, np.ndarray] | None = None
    for index, record in enumerate(records):
        if record.get('storage_format') == 'hamiformer.hamiballs2.full-episode-npz-shard.v1':
            shard_path = dataset_root / str(record['shard_path'])
            if shard_path != current_shard_path:
                with np.load(shard_path, allow_pickle=False) as archive:
                    keys = ('phase', 'attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'time', 'window_start', 'qa_frame_contact')
                    current_shard = {key: archive[key] for key in keys}
                current_shard_path = shard_path
            row = int(record['shard_row'])
            start = int(current_shard['window_start'][row])
            arrays['phase'][index] = current_shard['phase'][row, start:start + 49]
            for key in ('attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length'):
                arrays[key][index] = current_shard[key][row]
            arrays['time'][index] = current_shard['time'][row, start:start + 49]
            if include_contact_labels:
                arrays['contact'][index] = current_shard['qa_frame_contact'][row, start:start + 48]
        else:
            data_path = dataset_root / str(record['data_path'])
            with np.load(data_path, allow_pickle=False) as sample:
                start = int(sample['window_start'])
                arrays['phase'][index] = sample['phase'][start:start + 49]
                for key in ('attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length'):
                    arrays[key][index] = sample[key]
                arrays['time'][index] = sample['time'][start:start + 49]
            if include_contact_labels:
                with np.load(dataset_root / str(record['qa_path']), allow_pickle=False) as qa:
                    arrays['contact'][index] = qa['frame_contact'][start:start + 48]
        arrays['physical_seed'][index] = int(record['physical_seed'])
        if (index + 1) % 1024 == 0 or index + 1 == count:
            print(f'crop-pack {index + 1}/{count}', flush=True)
    for array in arrays.values():
        array.flush()
    del arrays
    metadata = {'schema': PACK_SCHEMA, 'manifest': str(manifest), 'manifest_sha256': manifest_hash, 'samples': count, 'frames': 49, 'include_contact_labels': include_contact_labels}
    (temporary / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    if output.exists():
        shutil.rmtree(output)
    os.replace(temporary, output)
    return output

class HamiBalls2CropPackDataset(Dataset[dict[str, Any]]):

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        if self.metadata.get('schema') != PACK_SCHEMA:
            raise ValueError('unsupported HamiBalls-2 crop pack')
        self.arrays = {path.stem: np.load(path, mmap_mode='r', allow_pickle=False) for path in self.root.glob('*.npy')}

    def __len__(self) -> int:
        return int(self.metadata['samples'])

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = {key: torch.from_numpy(np.array(value[index], copy=True)) for key, value in self.arrays.items()}
        return result
__all__ = ['HamiBalls2CropPackDataset', 'build_hamiballs2_crop_pack']
