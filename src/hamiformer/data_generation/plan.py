from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass
from typing import Literal
from .config import GeneratorConfig
SPLIT_ORDER = ('train', 'dev', 'calibration_fit', 'calibration_audit', 'test', 'long_test')
CollisionRequirement = Literal['at_least', 'zero', 'natural']

def stable_uint64(*parts: object) -> int:
    payload = '\x1f'.join((str(part) for part in parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little', signed=False)

@dataclass(frozen=True)
class SceneTask:
    split: str
    index: int
    scene_id: str
    sample_id: str
    frames: int
    scene_seed: int
    collision_requirement: CollisionRequirement

    @property
    def require_collision(self) -> bool:
        return self.collision_requirement == 'at_least'

def build_scene_plan(config: GeneratorConfig) -> list[SceneTask]:
    counts = asdict(config.splits)
    train_required_count = round(counts['train'] * config.sampling.collision_enriched_train_fraction)
    train_rank = {index for _, index in sorted(((stable_uint64(config.seed, 'train', index, 'collision_rank'), index) for index in range(counts['train'])))[:train_required_count]}
    result: list[SceneTask] = []
    for split in SPLIT_ORDER:
        count = counts[split]
        frames = config.sampling.frames_for_split(split)
        for index in range(count):
            scene_id = f'{split}_{index:08d}'
            if config.sampling.collision_mode != 'mixed':
                collision_requirement = config.sampling.collision_mode
            elif split != 'train':
                collision_requirement: CollisionRequirement = 'natural'
            elif index in train_rank:
                collision_requirement = 'at_least'
            else:
                collision_requirement = 'zero'
            result.append(SceneTask(split=split, index=index, scene_id=scene_id, sample_id=f'{scene_id}_t000000', frames=frames, scene_seed=stable_uint64(config.seed, split, index, 'scene'), collision_requirement=collision_requirement))
    return result
