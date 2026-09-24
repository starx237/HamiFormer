from __future__ import annotations
import argparse
import json
from hamiformer.data_generation import estimate_generation_budget, load_generator_config
from hamiformer.data_generation.pipeline import generate_dataset

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='估算或显式生成 HamiBalls-2D；生成阶段只使用 CPU，不使用 GPU')
    parser.add_argument('--config', required=True, help='生成器 YAML；与训练配置分离')
    parser.add_argument('--execute', action='store_true', help='显式允许写数据；省略时只打印预算且不创建 output root')
    return parser

def main() -> None:
    args = build_parser().parse_args()
    config = load_generator_config(args.config)
    budget = estimate_generation_budget(config)
    report = {'mode': 'execute' if args.execute else 'estimate_only', 'output_root': config.output.root, 'gpu_required': False, 'semantic_hash': config.semantic_hash, 'budget': budget.as_dict(), 'estimated_disk_mib': budget.estimated_disk_bytes / 1024 ** 2, 'advised_total_ram_allowance_gib': budget.advised_total_ram_allowance_bytes / 1024 ** 3, 'notes': ['physics_substeps 与 closure_substeps 均未计拒绝重试；实际时间以 micro-smoke 吞吐为准', 'estimated_disk 是正常碰撞密度下的保守基线，不是 event 数无界时的严格上界', 'memory allowance 包含 worker 与 coordinator 余量，但仍需用 micro-smoke 实测峰值']}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        print('未提供 --execute：没有导入 Pymunk，也没有创建或写入数据目录。')
        return
    summary = generate_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('数据尚未成为正式实验输入：请配置训练路径后运行全量 hamiformer-validate-data。')
if __name__ == '__main__':
    main()
