import argparse
import importlib
import sys
COMMANDS = {'generate-h1': 'hamiformer.cli.generate_data', 'pack-h1': 'hamiformer.data.pack_hamiballs1', 'generate-h2': 'hamiformer.cli.generate_hamiballs2', 'train-joint': 'hamiformer.training.joint', 'prepare-h1-training': 'hamiformer.training.prepare', 'train-h1-r': 'hamiformer.training.residual', 'train-h1-gate': 'hamiformer.training.router', 'train-h1-final': 'hamiformer.training.final', 'train-h2': 'hamiformer.training.hamiballs2.downstream', 'train-dit': 'hamiformer.training.baselines.dit', 'train-transformer-ar-h1': 'hamiformer.training.baselines.transformer_ar1', 'train-transformer-ar-h2': 'hamiformer.training.baselines.transformer_ar2', 'train-hgdpf': 'hamiformer.training.baselines.hgdpf', 'train-physiformer-h1': 'hamiformer.training.base', 'train-physiformer-h2': 'hamiformer.cli.train_hamiballs2_wide_d', 'evaluate': 'hamiformer.evaluation.tables', 'error-figure': 'hamiformer.visualization.error_curves', 'render-h1': 'hamiformer.visualization.hamiballs1_cli', 'render-h2': 'hamiformer.visualization.acrylic_cli'}

COMMANDS['export-hdf5'] = 'hamiformer.data.export_hdf5'

def main():
    parser = argparse.ArgumentParser(description='HamiFormer workflows')
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    command = args.command
    if command == 'train-h1-gate' and '--stop-after' not in rest and (not any((a.startswith('--stop-after=') for a in rest))):
        rest = ['--stop-after', 'scalar1', *rest]
    sys.argv = [COMMANDS[command], *rest]
    importlib.import_module(COMMANDS[command]).main()
if __name__ == '__main__':
    main()
