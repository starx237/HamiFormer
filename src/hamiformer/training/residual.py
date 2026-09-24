from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1 import residual_setup as base
SCHEMA = base.SCHEMA
REGISTRATION_SCHEMA = base.REGISTRATION_SCHEMA
ROLE = base.ROLE
PURPOSE = base.PURPOSE
H_SCHEMA = base.H_SCHEMA
H_ROLE = base.H_ROLE
OBJECTIVE = base.OBJECTIVE
PROHIBITIONS = base.PROHIBITIONS
stage_a = base.stage_a
_build_continuous_experts = base._build_continuous_experts
_source_manifest = base._source_manifest

def configure_runner() -> None:
    base.configure_runner()

def _load_registration(path: Path):
    configure_runner()
    return base._load_registration(path)

def main() -> None:
    configure_runner()
    base.runner.main()
if __name__ == '__main__':
    main()
