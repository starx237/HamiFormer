import json
from pathlib import Path

import numpy as np
import pytest

from hamiformer.data.contact_labels import replay_contacts


def test_positive_impulse_replay_checks_full_trajectory(tmp_path):
    pytest.importorskip('pymunk')
    from hamiformer.data_generation.config import load_generator_config
    from hamiformer.data_generation.plan import stable_uint64
    from hamiformer.data_generation.scene import sample_initial_scene
    from hamiformer.data_generation.simulator import _run_continuous
    cfg = load_generator_config(Path(__file__).parents[1]/'configs/generator/hamiballs_canonical_v2_seed40.yaml')
    scene_seed = stable_uint64(40,'dev',0,'scene')
    initial = sample_initial_scene(np.random.default_rng(stable_uint64(scene_seed,0,'attempt')),cfg)
    phase, _, _, _, _, _ = _run_continuous(initial,193,cfg)
    (tmp_path/'metadata').mkdir()
    (tmp_path/'metadata/dataset_meta.json').write_text(json.dumps({'semantic_config':cfg.semantic_payload()}))
    contact = replay_contacts(tmp_path,{'sample_id':'dev_00000000_t000000'},{'attempts':1},phase,initial.attrs)
    assert contact.shape == (192,5)
    assert contact.sum() > 0
    phase = phase.copy()
    phase[5,0,0] += .001
    with pytest.raises(ValueError,match='reproduce'):
        replay_contacts(tmp_path,{'sample_id':'dev_00000000_t000000'},{'attempts':1},phase,initial.attrs)
