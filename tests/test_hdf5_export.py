import json
import h5py
import numpy as np
import pytest

from hamiformer.data.export_hdf5 import export_hdf5


def test_h1_export_preserves_seed_order_coordinates_and_event_alignment(tmp_path):
    expected = []
    for seed in (40,41):
        root = tmp_path/f'source_seed{seed}'
        (root/'manifests').mkdir(parents=True)
        (root/'metadata/events').mkdir(parents=True)
        records, events = [], []
        for index in (1,0):
            phase = np.full((193,5,4), seed+index, dtype=np.float32)
            expected.append(phase)
            contact = np.zeros((192,5),np.uint8)
            contact[index,0] = 1
            np.savez(root/f'{index}.npz', phase=phase, attrs=np.ones((5,3),np.float32), time=np.arange(193,dtype=np.float32)/30,
                     frame_contact=contact)
            records.append({'sample_id':str(index),'npz_path':f'../{index}.npz'})
            events.insert(0, {'sample_id':str(index),'events':[{'substep':index*8,'pair':['ball:0','wall:left']}]})
        (root/'manifests/dev.jsonl').write_text('\n'.join(map(json.dumps,records)))
        (root/'metadata/events/dev.jsonl').write_text('\n'.join(map(json.dumps,events)))
    out=tmp_path/'episodes.h5'
    export_hdf5('h1',tmp_path,out,splits=('validation',),seeds=(40,41))
    with h5py.File(out) as f:
        np.testing.assert_array_equal(f['validation/phase'][:],expected)
        assert f['validation/contact'][0,1,0] == 1
        assert f['validation/contact'][1,0,0] == 1
    with pytest.raises(FileExistsError):
        export_hdf5('h1',tmp_path,out,splits=('validation',))


def test_h2_exports_full_shard_episode_not_training_crop(tmp_path):
    (tmp_path/'manifests').mkdir()
    arrays={'phase':np.arange(2*193*10*6,dtype=np.float32).reshape(2,193,10,6),
            'attrs':np.ones((2,10,3),np.float32),'time':np.tile(np.arange(193,dtype=np.float32)/30,(2,1)),
            'object_mask':np.ones((2,10),np.uint8),'window_start':np.array([32,48]),
            'gravity':np.zeros((2,3),np.float32),'bounds':np.ones((2,3),np.float32),
            'qa_frame_contact':np.zeros((2,192,10),np.uint8)}
    for k in ('spring_mask','spring_k','spring_rest_length'):
        arrays[k]=np.zeros((2,10,10),np.float32)
    np.savez(tmp_path/'shard.npz',**arrays)
    records=[{'shard_path':'shard.npz','shard_row':i} for i in (1,0)]
    (tmp_path/'manifests/validation.jsonl').write_text('\n'.join(map(json.dumps,records)))
    export_hdf5('h2',tmp_path,tmp_path/'episodes.h5',splits=('validation',))
    with h5py.File(tmp_path/'episodes.h5') as f:
        np.testing.assert_array_equal(f['validation/phase'][:],arrays['phase'][[1,0]])
        np.testing.assert_array_equal(f['validation/window_start'][:],[48,32])


def test_failed_export_does_not_leave_final_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_hdf5('h2',tmp_path,tmp_path/'episodes.h5',splits=('validation',))
    assert not (tmp_path/'episodes.h5').exists()
    assert not list(tmp_path.glob('*.partial'))
