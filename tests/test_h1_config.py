import copy
import json
from pathlib import Path

import pytest

from hamiformer.training import base


def _config():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'hamiballs1' / 'base.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_h1_training_config_contains_only_active_sections():
    config = _config()
    base._validate_config(config, production_batches=True)
    assert set(config) == {
        'schema', 'seed', 'dataset', 'model', 'hamiltonian',
        'residual', 'gate', 'optimization', 'rectified_flow',
    }
    assert set(config['optimization']) == {
        'batch_size', 'cache_batch_size', 'num_workers',
        'weight_decay', 'grad_clip',
    }


def test_h1_training_config_rejects_unknown_sections():
    config = _config()
    config['unexpected'] = {}
    with pytest.raises(ValueError, match='keys mismatch'):
        base._validate_config(config, production_batches=True)


def test_h1_smoke_config_changes_only_runtime_batching():
    config = _config()
    smoke = base._smoke_config(config)
    base._validate_runtime_config(smoke, smoke=True)
    expected = copy.deepcopy(config)
    expected['optimization'].update(batch_size=8, cache_batch_size=16, num_workers=0)
    assert smoke == expected
