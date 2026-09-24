"""Object-edge positive-impulse labels for generated HamiBalls-1 episodes."""
import json

import numpy as np


def replay_contacts(source_root, record, diagnostics, phase, attrs):
    from hamiformer.data_generation.config import (
        GeneratorConfig, GeneratorOutputConfig, GeneratorRuntimeConfig,
        PhysicsConfig, SamplingConfig, SplitCounts,
    )
    from hamiformer.data_generation.plan import stable_uint64
    from hamiformer.data_generation.scene import sample_initial_scene
    from hamiformer.data_generation import simulator
    raw = json.loads((source_root / 'metadata/dataset_meta.json').read_text(encoding='utf-8'))['semantic_config']
    cfg = GeneratorConfig(format_version=raw['format_version'], seed=raw['seed'],
        output=GeneratorOutputConfig(root='.', compressed=raw['output']['compressed'], reserve_free_gib=0.),
        runtime=GeneratorRuntimeConfig(workers=1, required_pymunk_version=raw['runtime']['required_pymunk_version']),
        physics=PhysicsConfig(**raw['physics']), sampling=SamplingConfig(**raw['sampling']),
        splits=SplitCounts(**raw['splits']))
    cfg.validate()
    source_id = record['sample_id']
    split, index = source_id.rsplit('_t', 1)[0].rsplit('_', 1)
    scene_seed = stable_uint64(cfg.seed, split, int(index), 'scene')
    attempt = int(diagnostics['attempts']) - 1
    if attempt < 0:
        raise ValueError('invalid accepted sampling attempt')
    initial = sample_initial_scene(np.random.default_rng(stable_uint64(scene_seed, attempt, 'attempt')), cfg)
    contact = np.zeros((len(phase)-1, len(attrs)), dtype=np.uint8)
    original = simulator.CollisionTracker

    class Tracker(original):
        def post_solve(self, arbiter, space, data):
            edge = self.substep_index // cfg.physics.substeps
            if float(arbiter.total_impulse.length) / cfg.physics.physics_dt > 1e-12:
                for shape in arbiter.shapes:
                    label = self.shape_labels[id(shape)]
                    if label.startswith('ball:'):
                        contact[edge, int(label.split(':')[1])] = 1
            super().post_solve(arbiter, space, data)

    simulator.CollisionTracker = Tracker
    try:
        replayed, _, _, _, _, _ = simulator._run_continuous(initial, len(phase), cfg)
    finally:
        simulator.CollisionTracker = original
    if not np.array_equal(replayed, phase) or not np.array_equal(initial.attrs, attrs):
        raise ValueError('contact replay does not reproduce the stored physical trajectory')
    return contact
