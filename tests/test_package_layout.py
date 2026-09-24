import ast
import importlib.util
from pathlib import Path


def test_command_modules_resolve():
    from hamiformer.cli.main import COMMANDS
    assert all(importlib.util.find_spec(module) is not None for module in COMMANDS.values())


def test_baseline_layer():
    root = Path(__file__).resolve().parents[1] / 'src' / 'hamiformer' / 'baselines'
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('hamiformer.training')


def test_config_layout():
    root = Path(__file__).resolve().parents[1] / 'configs'
    for dataset in ('hamiballs1', 'hamiballs2'):
        assert (root / dataset).is_dir()
