from hamiformer.utils.paths import project_root
import argparse
import json
from pathlib import Path
import torch
from hamiformer.utils import sha256_file
ROOT = project_root()

def prepare_residual(main_path, wide_path, h_path):
    template = ROOT / 'configs/hamiballs1/residual.json'
    config = json.loads(template.read_text())
    main = torch.load(main_path, map_location='cpu', weights_only=False)
    h = torch.load(h_path, map_location='cpu', weights_only=False)
    if main.get('role') != 'main' or main.get('step') != 50000:
        raise ValueError('residual training requires the completed 50000-update Main base')
    if h.get('status') != 'COMPLETE' or h.get('updates') != 50000 or (not h.get('joint_training')):
        raise ValueError('residual training requires the completed 50000-update joint continuous H')
    if h.get('parent_checkpoint_sha256') != sha256_file(main_path):
        raise ValueError('continuous H and Main base must have the same parent')
    config['parents'].update(main_checkpoint=str(main_path), wide_checkpoint=str(wide_path), main_checkpoint_sha256=sha256_file(main_path), wide_checkpoint_sha256=sha256_file(wide_path))
    config['continuous_h'].update(path=str(h_path), sha256=sha256_file(h_path), h_digest=h['h_digest'], expected_schema=h['schema'], expected_role=h['role'], expected_updates=50000)
    return config

def prepare_gate(r_config_path, r_path):
    template = ROOT / 'configs/hamiballs1/gate.json'
    config = json.loads(template.read_text())
    residual = json.loads(r_config_path.read_text())
    if not r_path.is_file():
        raise FileNotFoundError(r_path)
    config['parents'] = residual['parents']
    config['r_terminal'].update(path=str(r_path), sha256=sha256_file(r_path), training_registration_sha256=sha256_file(r_config_path))
    return config

def main():
    parser = argparse.ArgumentParser(description='Bind H1 training configurations to model inputs')
    parser.add_argument('stage', choices=('residual', 'gate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--wide-d', type=Path)
    parser.add_argument('--hamiltonian', type=Path)
    parser.add_argument('--r-config', type=Path)
    parser.add_argument('--residual', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    required = (args.main, args.wide_d, args.hamiltonian) if args.stage == 'residual' else (args.r_config, args.residual)
    if any((path is None or not path.is_file() for path in required)):
        parser.error('supply existing --main/--wide-d/--hamiltonian for residual, or --r-config/--residual for gate')
    paths = [path.resolve() for path in required]
    config = prepare_residual(*paths) if args.stage == 'residual' else prepare_gate(*paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + '\n')
if __name__ == '__main__':
    main()
