#!/usr/bin/env python3
"""Reference-only asymmetric excitation preview. Never opens hardware.

Uses the production Sweep and the same local position derivative estimator.
No simulated force data or parameter-fit NPZ is generated.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from friction_id.analysis import Gates, local_motion
from friction_id.common import Sweep, file_hash
from run_phase2_pointmatch import configuration, parser


def features(config, scope='full', window=.20):
    rows = []; reports = []
    j = config['joint']-1
    lower, upper = config['start'][j], config['end'][j]
    bounds = config['dynamic_range'] if scope == 'full' else config['measurement_range']
    grid = np.arange(bounds[0]+.001, bounds[1], .002)
    for sid, speed in enumerate(config['speeds']):
        passes = []
        for direction in (1, -1):
            start, end = (lower, upper) if direction == 1 else (upper, lower)
            sweep = Sweep(start, end, speed, config['trajectory']['ramp_up_time'], config['trajectory']['ramp_down_time'])
            t = np.arange(0, sweep.duration, .005)
            samples = [sweep.evaluate(x) for x in t]
            q = np.array([x[0] for x in samples]); phase = np.array([x[2] for x in samples])
            v, a, _ = local_motion(t, q, Gates(window=window))
            select = np.isfinite(v) & np.isfinite(a) & (v*direction > .01)
            if scope == 'constant':
                ix = np.flatnonzero(phase == 'constant')
                select &= (phase == 'constant') & (t >= t[ix[0]]+.25) & (t <= t[ix[-1]]-.25)
            else:
                select &= (t >= .25) & (t <= t[-1]-.25)
            order = np.argsort(q[select]); passes.append((q[select][order], v[select][order], a[select][order]))
        count = 0
        for q in grid:
            if not all(len(p[0]) > 6 and p[0][0] < q < p[0][-1] for p in passes):
                continue
            vp, vn = [np.interp(q, p[0], p[1]) for p in passes]
            ap, an = [np.interp(q, p[0], p[2]) for p in passes]
            rows.append(dict(q=q, df=(np.tanh(vp/.02)-np.tanh(vn/.02))/2,
                dv=(vp-vn)/2, da=(ap-an)/2, vp=vp, vn=vn, ap=ap, an=an,
                group=f'{config["trajectory"]["ramp_up_time"]}/{config["trajectory"]["ramp_down_time"]}:{sid}'))
            count += 1
        reports.append(dict(speed=speed, bins=count, coverage=count/len(grid)))
    if not rows:
        raise ValueError('Preview has no matched motion coverage')
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}, reports


def identifiability(d, period):
    X = np.column_stack([d['df'], d['dv'], d['da'],
                         np.sin(2*np.pi*d['q']/period), np.cos(2*np.pi*d['q']/period)])
    w = np.zeros(len(d['q']))
    for group in np.unique(d['group']):
        m = d['group'] == group; w[m] = 1/np.sqrt(m.sum())
    A = X*w[:, None]; scale = np.linalg.norm(A, axis=0)
    # A numerically tiny acceleration column must not become full excitation
    # merely because it is normalized to unit length.
    normalized = A/np.maximum(scale, 1e-10)
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-8))
    s = np.linalg.svd(normalized, compute_uv=False)
    condition = float(s[0]/max(s[-1], 1e-15))
    others = A[:, [0, 1, 3, 4]]
    residual = A[:, 2]-others@np.linalg.lstsq(others, A[:, 2], rcond=None)[0]
    rms = float(np.linalg.norm(A[:, 2])/np.linalg.norm(w))
    independent = float(np.linalg.norm(residual)/np.linalg.norm(w))
    supported = rank == 5 and condition < 100 and independent >= .01
    return dict(rank=rank, columns=5, scaled_condition=condition,
                acceleration_difference_rms=rms, independent_acceleration_rms=independent,
                independent_fraction=independent/max(rms, 1e-15),
                reference_design_supported=bool(supported),
                criteria=dict(rank=5,max_scaled_condition=100,min_independent_acceleration_rms=.01))


def build(joint, up, down, speeds=(.06, .09, .12)):
    kp, kd = (80, 10) if joint == 1 else (150, 20)
    argv = ['--joint', str(joint), '--speeds', *map(str, speeds), '--kp', str(kp), '--kd', str(kd),
            '--ramp-time', str(up), '--ramp-down-time', str(down), '--repeats', '1',
            '--load-id', 'unloaded_wrist_fixed_setup_20261019_user_reported']
    config = configuration(parser().parse_args(argv))
    return dict(joint=joint, **config), argv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--joint', type=int, choices=(1, 2), default=2)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.output/'preview.json').exists():
        raise FileExistsError(args.output/'preview.json')
    configs = []; rows = []; full = []
    for up, down in ((1., 1.), (.8, 2.), (2., .8)):
        config, command = build(args.joint, up, down)
        configs.append(dict(config=config, preview_argv=command))
        for scope in ('constant', 'full'):
            d, coverage = features(config, scope)
            rows.append(dict(ramp_up=up, ramp_down=down, scope=scope, coverage=coverage,
                             metrics=identifiability(d, .0498666)))
            if scope == 'full' and up != down:
                full.append(d)
    combined = {k: np.concatenate([d[k] for d in full]) for k in full[0]}
    result = dict(kind='reference_only_excitation_preview', hardware_opened=False, joint=args.joint,
        implementation_sha256=dict(preview=file_hash(__file__),sweep=file_hash(Path(__file__).parent/'friction_id/common.py'),
                                   derivative=file_hash(Path(__file__).parent/'friction_id/analysis.py')),
        configs=configs, individual=rows, combined_asymmetric_full=identifiability(combined, .0498666),
        notes=['Reference-only design matrix, not real feedback or torque identification.',
               'Derivative window .20 s; no assumption that tracking or encoder accuracy is ideal on hardware.',
               'Same chronological up/down durations in both directions; swapping durations produces a separate run.',
               'Original steady-motion Method D is unchanged; independent acceleration uses full-motion pairing.'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'preview.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for entry in configs:
        config = entry['config']; c = dict(config, speeds=[.12])
        d, _ = features(c, 'full')
        label = f'up/down={config["trajectory"]["ramp_up_time"]}/{config["trajectory"]["ramp_down_time"]} s'
        axes[0].plot(d['q'], d['dv'], label=label)
        axes[1].plot(d['q'], d['da'], label=label)
    axes[0].set_ylabel('(v+ - v-)/2 (rad/s)'); axes[1].set_ylabel('(a+ - a-)/2 (rad/s²)')
    for ax in axes:
        ax.set_xlabel('Reference angle (rad)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f'J{args.joint}: reference-only paired excitation at .12 rad/s')
    fig.savefig(args.output/'excitation.png', dpi=140)
    print(json.dumps(dict(combined=result['combined_asymmetric_full'],individual=rows), indent=2))


if __name__ == '__main__':
    main()
