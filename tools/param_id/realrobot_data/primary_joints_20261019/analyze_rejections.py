"""Offline rejection attribution using unchanged production gates. No hardware."""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from friction_id.analysis import Gates, prepare, local_motion
from friction_id.recording import load_run


def longest(t, mask):
    ii = np.flatnonzero(mask)
    chunks = np.split(ii, np.flatnonzero((np.diff(ii) != 1) | (np.diff(t[ii]) > .04)) + 1)
    return float(max((t[x[-1]] - t[x[0]] for x in chunks if len(x)), default=0))


rows = []
groups = ['J1_pilot', 'J1_damping', 'J2_pilot', 'J2_damping', 'J4_pilot', 'J4_speed15']
fig, axes = plt.subplots(6, 3, figsize=(16, 18), constrained_layout=True)
g = Gates()
for row, group in enumerate(groups):
    path = next((BASE / group).glob('*.npz'))
    meta, fb, cmd = load_run(path)
    a, reports = prepare(meta, fb, cmd, g)
    j = int(meta['joint'])
    f = fb[fb['joint'] == j]
    c = cmd[(cmd['joint'] == j) & cmd['ok']]
    for report in reports:
        seg = report['segment']
        constant = (a['segment'] == seg) & (a['phase'] == 'constant')
        ii = np.flatnonzero(constant)
        inside = constant & (a['t'] >= a['t'][ii[0]] + g.guard) & (a['t'] <= a['t'][ii[-1]] - g.guard)
        inside &= (a['q'] >= meta['measurement_range'][0]) & (a['q'] <= meta['measurement_range'][1])
        other = np.arange(6) != j - 1
        criteria = {
            'transport': a['transport_valid'],
            'finite_motion': np.isfinite(a['v']) & np.isfinite(a['acceleration']),
            'speed': abs(a['v'] - a['qd_ref']) <= g.velocity_relative_error * abs(a['qd_ref']) + g.velocity_absolute_error,
            'acceleration': abs(a['acceleration']) <= g.max_acceleration,
            'position_fit': a['position_residual'] <= g.max_position_residual,
            'tracking': abs(a['q'] - a['q_ref']) <= g.max_tracking_error,
            'other_pose': np.max(abs(a['pose'][:, other] - np.asarray(meta['home'])[other]), axis=1) <= g.max_pose_difference,
        }
        good = inside & np.logical_and.reduce(list(criteria.values()))
        n = int(inside.sum())
        assert abs(good.sum() / n - report['steady_fraction']) < 1e-12
        stats = dict(group=group, joint=j, direction=report['direction'], report=report,
                     inside_samples=n, target=float(np.median(abs(a['qd_ref'][inside]))),
                     failure_percent={k: float(100 * (inside & ~v).sum() / n) for k, v in criteria.items()},
                     acceleration_abs_p50_p95_p99=np.percentile(abs(a['acceleration'][inside]), [50, 95, 99]).tolist(),
                     speed_signed_p05_p50_p95=np.percentile(a['v'][inside] * report['direction'], [5, 50, 95]).tolist(),
                     tracking_abs_max=float(abs(a['q'][inside] - a['q_ref'][inside]).max()),
                     torque_min_max=[float(a['tau_fb'][inside].min()), float(a['tau_fb'][inside].max())],
                     ff_min_max=[float(a['tau_ff'][inside].min()), float(a['tau_ff'][inside].max())])
        # Attribution only: these do not change acceptance or fit any parameters.
        for omitted in ('speed', 'acceleration'):
            alternative = inside & np.logical_and.reduce([v for k, v in criteria.items() if k != omitted])
            stats['diagnostic_without_' + omitted] = dict(fraction=float(alternative.sum() / n), longest_s=longest(a['t'], alternative))
        ct = c['tx_end'][(c['segment'] == seg) & (c['phase'] == 'constant')]
        ft = a['t'][constant]
        stats['command_interval_ms_p50_p99_max'] = (np.percentile(np.diff(ct), [50, 99, 100]) * 1000).tolist()
        stats['feedback_interval_ms_p50_p99_max'] = (np.percentile(np.diff(ft), [50, 99, 100]) * 1000).tolist()
        # Ideal constant motion sampled at the SAME receive times and quantized
        # to the decoded 16-bit status-position step. Diagnostic, not a noise model.
        full = (a['segment'] == seg) & np.isin(a['phase'], ['accel', 'constant', 'decel'])
        tt = a['t'][full]
        step = 8 * np.pi / 65535
        ideal = a['q'][full][0] + report['direction'] * stats['target'] * (tt - tt[0])
        ideal = np.round((ideal + 4 * np.pi) / step) * step - 4 * np.pi
        _, ideal_a, _ = local_motion(tt, ideal, g)
        stats['ideal_quantized_constant_acceleration_abs_p99'] = float(np.nanpercentile(abs(ideal_a[inside[full]]), 99))
        # In-sample spatial-period search; no mechanical/electrical attribution.
        qfit = a['q'][inside]
        for name, values in [('velocity', a['v'][inside]), ('torque', a['tau_fb'][inside])]:
            base = np.column_stack([np.ones(n), qfit - qfit.mean()])
            residual = values - base @ np.linalg.lstsq(base, values, rcond=None)[0]
            denominator = float(residual @ residual)
            best = (-np.inf, None)
            for period in np.linspace(.025, .12, 600):
                design = np.column_stack([base, np.sin(2 * np.pi * qfit / period), np.cos(2 * np.pi * qfit / period)])
                error = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
                score = 1 - float(error @ error) / denominator
                if score > best[0]:
                    best = score, float(period)
            stats[name + '_spatial_sine_diagnostic'] = dict(period_rad=best[1], improvement_over_linear_q_r2=best[0])
        rows.append(stats)
        color = 'C0' if report['direction'] > 0 else 'C1'
        label = '+' if report['direction'] > 0 else '-'
        q = a['q'][inside] - meta['home'][j - 1]
        axes[row, 0].plot(q, a['v'][inside] * report['direction'], color=color, label=label, lw=.8)
        axes[row, 1].plot(q, a['acceleration'][inside], color=color, lw=.8)
        axes[row, 2].plot(q, a['tau_fb'][inside], color=color, lw=.8)
    target = stats['target']
    axes[row, 0].axhspan(target - .15 * target - .003, target + .15 * target + .003, alpha=.12, color='green')
    axes[row, 1].axhspan(-.15, .15, alpha=.12, color='green')
    axes[row, 0].set_title(f'{group}: direction-adjusted v (rad/s)')
    axes[row, 1].set_title('Estimated acceleration (rad/s²)')
    axes[row, 2].set_title('Feedback torque (unverified protocol scale)')
    axes[row, 0].legend()
    for ax in axes[row]:
        ax.set_xlabel('q - HOME (rad)')
        ax.grid(alpha=.2)
(BASE / 'rejection_details.json').write_text(json.dumps(rows, indent=2) + '\n')
fig.savefig(BASE / 'rejection_traces.png', dpi=140)
for r in rows:
    print(r['group'], r['direction'], 'fail%', {k: round(v, 1) for k, v in r['failure_percent'].items()},
          'a95', round(r['acceleration_abs_p50_p95_p99'][1], 3),
          'noA', r['diagnostic_without_acceleration'], 'noV', r['diagnostic_without_speed'])
