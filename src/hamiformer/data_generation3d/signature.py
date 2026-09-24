from pathlib import Path
from .config import load_config

def dataset_signature(config):
    return load_config(Path(config['generator_config'])).semantic_hash
