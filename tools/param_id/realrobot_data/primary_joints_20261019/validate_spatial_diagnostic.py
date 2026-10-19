"""Frozen cross-run ripple prediction. Offline diagnostic; no friction output.

Models per direction: linear q; linear q + spatial sine; linear q + temporal
sine. Frequencies and ALL coefficients are chosen on the training run alone.
The shared angular interval uses run configuration, never held-out torque.
"""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from friction_id.analysis import Gates, prepare
from friction_id.recording import load_run
from friction_id.common import file_hash


def data(path, bounds):
    meta, fb, cmd = load_run(path)
    if meta['run_status'] != 'complete' or not cmd['ok'].all() or fb['status'].any():
        raise ValueError('Incomplete/faulted run')
    a, _ = prepare(meta, fb, cmd, Gates())
    a['elapsed'] = np.full(len(a['t']), np.nan)
    mask = np.zeros(len(a['t']), bool)
    for seg in np.unique(a['segment'][a['segment'] >= 0]):
        ii = np.flatnonzero((a['segment'] == seg) & (a['phase'] == 'constant'))
        if not len(ii):
            continue
        a['elapsed'][ii] = a['t'][ii] - a['t'][ii[0]]
        mask[ii] = (a['elapsed'][ii] >= .25) & (a['t'][ii] <= a['t'][ii[-1]] - .25)
    mask &= a['transport_valid'] & np.isfinite(a['v']) & np.isfinite(a['acceleration'])
    mask &= (a['q'] >= bounds[0]) & (a['q'] <= bounds[1])
    mask &= a['position_residual'] <= .001
    mask &= abs(a['q'] - a['q_ref']) <= .05
    other = np.arange(6) != meta['joint'] - 1
    mask &= np.max(abs(a['pose'][:, other] - np.asarray(meta['home'])[other]), axis=1) <= .02
    # Deliberately include nonsteady data to investigate ripple; never Method D.
    result = {}
    for direction in (1, -1):
        ix = np.flatnonzero(mask & (a['direction'] == direction))
        # Avoid weighting near-simultaneous status reports twice.
        keep = []
        for i in ix:
            if not keep or a['t'][i] - a['t'][keep[-1]] >= .009:
                keep.append(i)
        if len(keep) < 100:
            raise ValueError('Insufficient diagnostic samples')
        result[direction] = {k: a[k][keep] for k in ('q', 'elapsed', 'tau_fb', 'qd_ref')}
    return result


def design(d, kind, frequency=None):
    cols = [np.ones(len(d['q'])), d['q']]
    if kind != 'baseline':
        phase = 2 * np.pi * (d['q'] / frequency if kind == 'spatial' else d['elapsed'] * frequency)
        cols += [np.sin(phase), np.cos(phase)]
    return np.column_stack(cols)


def fit(train, kind, frequency=None):
    coef = {}; scores = []
    for direction, d in train.items():
        X = design(d, kind, frequency)
        beta, _, rank, _ = np.linalg.lstsq(X, d['tau_fb'], rcond=None)
        if rank != X.shape[1]:
            raise ValueError('Rank deficient diagnostic')
        coef[str(direction)] = beta.tolist()
        scores.append(float(np.mean((d['tau_fb'] - X @ beta) ** 2)))
    return dict(kind=kind, frequency=frequency, coefficients=coef, train_rmse=float(np.sqrt(np.mean(scores))))


def main():
    train_group, test_group = sys.argv[1:3]
    p, h = [next((BASE / group).glob('*.npz')) for group in (train_group, test_group)]
    tm, _, _ = load_run(p); hm, _, _ = load_run(h)
    if p.resolve() == h.resolve() or tm['run_id'] == hm['run_id']:
        raise ValueError('Independent raw run required')
    for key in ('joint', 'kp', 'kd', 'home', 'torque_calibration'):
        if tm[key] != hm[key]:
            raise ValueError('Condition mismatch: ' + key)
    for key in ('model_hash', 'load_id'):
        if tm['conditions'][key] != hm['conditions'][key]:
            raise ValueError('Condition mismatch: ' + key)
    bounds = [max(tm['measurement_range'][0], hm['measurement_range'][0]), min(tm['measurement_range'][1], hm['measurement_range'][1])]
    train = data(p, bounds)
    # Freeze all models BEFORE reading held-out target samples.
    models = [fit(train, 'baseline')]
    for kind, grid in [('spatial', np.linspace(.025, .12, 600)), ('temporal', np.linspace(.5, 8, 600))]:
        models.append(min((fit(train, kind, float(f)) for f in grid), key=lambda x: x['train_rmse']))
    test = data(h, bounds)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, direction in zip(axs, (1, -1)):
        d = test[direction]
        ax.plot(d['q'], d['tau_fb'], color='.65', label='held-out measured', lw=.8)
        for model in models:
            prediction = design(d, model['kind'], model['frequency']) @ model['coefficients'][str(direction)]
            model.setdefault('heldout_by_direction', {})[str(direction)] = dict(samples=len(prediction), rmse=float(np.sqrt(np.mean((prediction - d['tau_fb']) ** 2))))
            for speed in np.unique(abs(d['qd_ref'])):
                select = abs(d['qd_ref']) == speed
                model.setdefault('heldout_by_speed_direction', []).append(dict(speed=float(speed), direction=direction,
                    samples=int(select.sum()), rmse=float(np.sqrt(np.mean((prediction[select] - d['tau_fb'][select]) ** 2)))))
            ax.plot(d['q'], prediction, label=model['kind'])
        ax.set_title(f'J{tm["joint"]} direction {direction:+d}: {train_group} -> {test_group}')
        ax.set_xlabel('q (rad)'); ax.set_ylabel('Torque (unverified protocol scale)')
        ax.legend(); ax.grid(alpha=.2)
    for model in models:
        model['heldout_rmse'] = float(np.sqrt(np.mean([v['rmse'] ** 2 for v in model['heldout_by_direction'].values()])))
    out = BASE / (train_group + '_to_' + test_group)
    out.mkdir(exist_ok=True)
    report = dict(status='diagnostic_only', compensation_allowed=False, joint=tm['joint'],
                  training_file=str(p.relative_to(BASE)), heldout_file=str(h.relative_to(BASE)),
                  training_sha256=file_hash(p), heldout_sha256=file_hash(h), script_sha256=file_hash(__file__),
                  common_q_bounds=bounds, models=models,
                  notes=['No held-out coefficient, phase, intercept or frequency refit.',
                         'Nonsteady samples included for ripple diagnostics; no Fc/Fv or inertia identification.',
                         'No gravity subtraction from unverified protocol torque; linear q is a descriptive trend only.',
                         'Same configuration does not establish unchanged physical setup; load ID is user reported.'])
    (out / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fig.savefig(out / 'heldout.png', dpi=140)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
