from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
ROOT = project_root()
OUTPUT = ROOT / 'outputs/hami1/router_train_carriers'
GATE_ROOT = ROOT / 'artifacts/hami1_gate_v1'
RUN_SPEC = GATE_ROOT / 'validation_run_spec_COMPONENT_ROUTER_v1/hami1_plas.json'
DATASET_ROOT = ROOT / 'data/hamiballs_canonical_v2/train_adapter48'
ROW_LEDGER = ROOT / 'data/hamiballs_canonical_v2/packs/train48_stride48/rows.jsonl'
EXCLUSION = None
NOISE_SEEDS = (1942634267, 2035767743)
FIELD_INDICES = (0, 5, 11, 17)
KEEP = ('source_index', 'rf_field_order', 'rf_trace_index', 'residual_hidden', 'noisy', 'previous_mixed', 'h_candidate', 'hr_candidate', 'd_candidate', 'mixed', 'target', 'attrs', 'tau', 'physical_time', 'previous_g', 'previous_heun_defect', 'committed_gate')
