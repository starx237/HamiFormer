from hamiformer.utils.paths import project_root
import argparse
from pathlib import Path
import numpy as np
from hamiformer.visualization.hamiballs_trajectory import TrajectoryBundle, render_trajectory_video

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--noise-index', type=int, default=0)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--tail', type=int, default=16)
    p.add_argument('--max-edges', type=int)
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.inputs) != len(a.labels):
        p.error('One label is required per input')
    predictions = {}
    for file, name in zip(a.inputs, a.labels):
        with np.load(file, allow_pickle=False) as f:
            target = f['target'][a.sample]
            time = f['time'][a.sample]
            attrs = f['attrs'][a.sample]
            if 'initial' not in f:
                raise ValueError('Trajectory input requires its physical initial state')
            initial = f['initial'][a.sample]
            current = np.concatenate((initial[None], target), axis=0)
            if predictions:
                np.testing.assert_array_equal(gt, current)
            gt = current
            predictions[name] = np.concatenate((initial[None], f['prediction'][a.noise_index, a.sample]), axis=0)
    bundle = TrajectoryBundle(gt_phase=gt, attrs=attrs, time=time, predictions=predictions, event_edge=None, metadata={}, input_files=())
    render_trajectory_video(bundle, a.output, layout='comparison', fps=a.fps, tail=a.tail, max_edges=a.max_edges, dpi=a.dpi)
if __name__ == '__main__':
    main()
