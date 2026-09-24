from hamiformer.utils.paths import project_root
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--inputs', type=Path, nargs='+')
    source.add_argument('--curves', type=Path)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.inputs and (not args.labels or len(args.inputs) != len(args.labels)):
        parser.error('One label is required per input')
    styles = {'Ours': '#159447', 'Physiformer': '#2f6db0', 'DiT': '#843c9c', 'Transformer-AR (ctx=1)': '#d17c13'}
    curves = {}
    if args.curves:
        record = json.loads(args.curves.read_text())
        curves = {name: np.asarray(record['per_edge'][name])[:, 0] for name in styles if name in record['per_edge']}
    for path, name in zip(args.inputs or [], args.labels or []):
        with np.load(path, allow_pickle=False) as data:
            delta = (data['prediction'].astype(np.float64) - data['target'].astype(np.float64)[None]) / data['state_scale']
            mask = data['object_mask'].astype(bool)
            curves[name] = (delta ** 2 * mask[None, :, None, :, None]).sum((0, 1, 3, 4)) / (delta.shape[0] * mask.sum() * delta.shape[-1])
        if curves[name].shape != (192,) or not np.isfinite(curves[name]).all():
            raise ValueError('Expected 192 finite per-edge errors')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.linewidth': 1, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (lo, hi, ticks) in zip(axes, [(1, 48, [1, 12, 24, 36, 48]), (49, 96, [49, 60, 72, 84, 96]), (97, 192, [97, 120, 144, 168, 192])]):
        values = []
        for name, all_y in curves.items():
            y = all_y[lo - 1:hi]
            values.extend(y.tolist())
            ax.plot(np.arange(lo, hi + 1), y, color=styles.get(name), lw=1.8, label=name)
        lower, upper = (min(values), max(values))
        pad = 0.07 * (upper - lower)
        ax.set_xlim(lo, hi)
        ax.set_ylim(max(0, lower - pad), upper + pad)
        ax.set_title(f'Edges {lo}–{hi}')
        ax.set_xlabel('Prediction horizon (edges)')
        ax.set_ylabel('Per-edge normalized $z$ MSE')
        ax.set_xticks(ticks)
        for boundary in [48, 96, 144]:
            if lo < boundary < hi:
                ax.axvline(boundary, color='#aaaaaa', ls='--', lw=0.7, zorder=0)
        ax.grid(axis='y', alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)
if __name__ == '__main__':
    main()
