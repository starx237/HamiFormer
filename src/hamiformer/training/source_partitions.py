import json
from pathlib import Path

def load_scene_exclusions(path=None, *, expected_count=None):
    if path is None:
        return set()
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    scenes = record.get('scene_ids')
    if record.get('schema') != 'hamiformer.scene_exclusions.v1' or not isinstance(scenes, list):
        raise ValueError('invalid scene-exclusion specification')
    if expected_count is not None and len(scenes) != expected_count or len(set(map(str, scenes))) != len(scenes):
        raise ValueError('scene-exclusion count or uniqueness mismatch')
    return set(map(str, scenes))

def exclusion_metadata(path=None):
    from hamiformer.utils import sha256_file
    scenes = load_scene_exclusions(path)
    return {'path': str(Path(path).resolve()) if path is not None else None, 'sha256': sha256_file(Path(path)) if path is not None else None, 'scene_count': len(scenes)}
