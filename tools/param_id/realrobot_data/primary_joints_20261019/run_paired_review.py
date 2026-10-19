"""Reproduce this offline review; outputs refuse overwrite. No hardware imports."""
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from friction_id.paired_dynamic import Options, fit_paired, validate_paired


def main():
    output = BASE/'paired_dynamic_final'
    cases = [('J1', ['J1_damping', 'J1_speed_screen'], 'J1_halfperiod_start'),
             ('J2', ['J2_speed_screen'], 'J2_damping'),
             ('J3', ['J3_train'], 'J3_damping'), ('J4', ['J4_pilot'], 'J4_speed15')]
    summaries = []
    for window in (.20, .16, .24):
        for joint, train, test in cases:
            if window != .20 and joint not in ('J1', 'J2'):
                continue
            target = output/(joint if window == .20 else f'{joint}_window_{window:.2f}')
            paths = [next((BASE/g).glob('*.npz')) for g in train]
            model_path = fit_paired(paths, target/'fit', Options(window=window))
            model = json.loads(model_path.read_text())
            try:
                validation = validate_paired(model_path, next((BASE/test).glob('*.npz')), target/'validation')
                error = None
            except ValueError as exc:
                validation, error = None, str(exc)
                (target/'validation_refusal.json').write_text(json.dumps(dict(reason=error, compensation_allowed=False), indent=2)+'\n')
            rows = []
            for m in model['models']:
                rows.append(dict(kind=m['kind'], period=m['period_rad'],
                    coefficients=dict(zip(m['names'], m['coefficients'])), issues=m['issues'],
                    scaled_condition=m['scaled_condition'], deletion_ranges=m['stability']['coefficient_ranges'],
                    training_metrics=m['training_metrics'],
                    heldout_metrics=validation['results'][m['kind']] if validation else None))
            summaries.append(dict(joint=joint, window=window, training=train, heldout=test,
                excitation_issues=model['excitation_issues'], models=rows, validation_error=error))
            print(joint, window, model['excitation_issues'], error or 'frozen prediction complete', flush=True)
    (output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for row, joint in enumerate(('J1', 'J2', 'J3')):
        with np.load(output/joint/'validation/paired_predictions.npz', allow_pickle=False) as z:
            for col, term in enumerate(('D', 'S')):
                ax = axes[row, col]; ax.plot(z['q'], z[term], color='.6', lw=.8, label='held-out matched pairs')
                for kind in ('velocity', 'inertia', 'spatial'):
                    ax.plot(z['q'], z[term+'_'+kind], label=kind, lw=1)
                ax.set_title(f'{joint} held-out: {term} = half '+('difference' if term == 'D' else 'sum'))
                ax.set_xlabel('q (rad)'); ax.set_ylabel('Unverified protocol torque')
                ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/'heldout_pairs.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
