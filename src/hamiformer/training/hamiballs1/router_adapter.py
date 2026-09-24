from __future__ import annotations
from hamiformer.utils.paths import project_root
from types import SimpleNamespace

def adapt(r_tool):
    return SimpleNamespace(configure_runner=r_tool.configure_runner, _load_registration=r_tool._load_registration, SCHEMA=r_tool.SCHEMA, ROLE=r_tool.ROLE, base=SimpleNamespace(_build_continuous_experts=r_tool._build_continuous_experts, stage_a=r_tool.stage_a))
