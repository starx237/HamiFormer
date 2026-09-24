from __future__ import annotations
import argparse
import json
from hamiformer.data_generation3d import generate_dataset, load_config
from hamiformer.data_generation3d.pipeline import audit_dataset, scene_plan
from hamiformer.data_generation3d.preflight import run_preflight

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--estimate', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--generate', action='store_true')
    modes.add_argument('--audit', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--preflight-scenes-per-seed', type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.estimate:
        result = {'samples': len(scene_plan(cfg)), 'semantic_hash': cfg.semantic_hash, 'output_root': cfg.output.root}
    elif args.preflight:
        result = run_preflight(cfg, scenes_per_seed=args.preflight_scenes_per_seed)
    elif args.audit:
        result = audit_dataset(cfg)
    else:
        result = generate_dataset(cfg, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
