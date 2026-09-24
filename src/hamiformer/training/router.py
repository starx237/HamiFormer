from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
from hamiformer.training.hamiballs1 import router_adapter as r_adapter
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training.hamiballs1 import refresh as fresh
from hamiformer.training.hamiballs1 import gate_runtime as gate_runtime
from hamiformer.training import residual as r_tool
SCHEMA = 'hamiformer.hami1.gate_run.v1'
STAGE_SCHEMA = 'hamiformer.hami1.gate_stage.v1'
GATE_PURPOSE = 'hami1_component_gate_training'
SCALAR_OBJECTIVE = packed.QP_NO_HARM_OBJECTIVE
QP_OBJECTIVE = packed.DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE
NO_HARM_WEIGHT = 0.25
QP_WIDTH = 16

def configure_runner() -> None:
    r_tool.configure_runner()
    gate_runtime.r_tool = r_adapter.adapt(r_tool)
    gate_runtime.GATE_PURPOSE = GATE_PURPOSE
    gate_runtime.SCHEMA, gate_runtime.STAGE_SCHEMA = (SCHEMA, STAGE_SCHEMA)
    gate_runtime.RUNNER_PATH = Path(__file__).resolve()
    gate_runtime._install = gate_runtime._ORIGINAL_INSTALL
    gate_runtime.configure_runner()
    packed.COMPONENT_TAIL_COMPONENT_SPECIFIC_Q_NO_TAIL_P_TAIL = True
    fresh.SCHEMA, fresh.STAGE_SCHEMA = (SCHEMA, STAGE_SCHEMA)
    fresh.RUNNER_PATH = Path(__file__).resolve()
    fresh.QP_DELTA_WIDTH = QP_WIDTH
    fresh.OBJECTIVES = {'scalar0': 'global_sequence_projection_regret', 'scalar1': SCALAR_OBJECTIVE, 'qp': QP_OBJECTIVE}
    fresh.STAGE_EXTRA_ARGUMENTS = {'scalar1': ('--qp-no-harm-weight', str(NO_HARM_WEIGHT))}

def main() -> None:
    configure_runner()
    fresh.main()
if __name__ == '__main__':
    main()
