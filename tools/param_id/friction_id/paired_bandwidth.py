"""One fixed linear position-domain operator on paired targets AND regressors.

Diagnostic only. This does not undo local derivative bias or certify physical
torque units. Split gaps before filtering; use a common endpoint exclusion for
all sensitivity settings so their averages refer to the same angular interval.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

NAMES = ['smooth_fc', 'fv', 'effective_inertia', 'odd_position_sin', 'odd_position_cos']


def equation(d, sigma=.004, endpoint_sigma=.006, period=.0498666, step=.002):
    if not 0 <= sigma <= endpoint_sigma or min(endpoint_sigma, period, step) <= 0:
        raise ValueError('Invalid fixed filtering settings')
    if not np.isfinite([sigma, endpoint_sigma, period, step]).all():
        raise ValueError('Nonfinite filtering settings')
    output = {k: [] for k in ('q', 'group', 'y', 'raw_y', 'X')}
    for group in np.unique(d['group']):
        ix = np.flatnonzero(d['group'] == group)
        ix = ix[np.argsort(d['q'][ix])]
        # Missing bins must never be filled by an implicit filter/interpolation.
        breaks = np.flatnonzero(~np.isclose(np.diff(d['q'][ix]), step, rtol=0, atol=step*1e-4))+1
        for chunk in np.split(ix, breaks):
            radius = int(np.ceil(3*endpoint_sigma/step))
            if len(chunk) < 2*radius+10:
                continue
            q = d['q'][chunk]
            X = np.column_stack([d['df'][chunk], d['dv'][chunk], d['da'][chunk],
                np.sin(2*np.pi*q/period), np.cos(2*np.pi*q/period)])
            block = np.column_stack([d['D'][chunk], X])
            smooth = gaussian_filter1d(block, sigma/step, axis=0, truncate=3.) if sigma else block.copy()
            keep = slice(radius, -radius)
            output['q'].extend(q[keep]); output['group'].extend([str(group)]*len(q[keep]))
            output['y'].extend(smooth[keep, 0]); output['raw_y'].extend(block[keep, 0])
            output['X'].extend(smooth[keep, 1:])
    if not output['q']:
        raise ValueError('No sufficiently long contiguous matched-position intervals')
    return {k: np.array(v) for k, v in output.items()}


def summaries(e):
    rows = []
    for group in np.unique(e['group']):
        mask = e['group'] == group
        rows.append(dict(group=str(group), bins=int(mask.sum()), q_range=[float(e['q'][mask].min()),float(e['q'][mask].max())],
            mean_D=float(np.mean(e['y'][mask])), raw_same_support_mean_D=float(np.mean(e['raw_y'][mask])),
            point_std=float(np.std(e['y'][mask]))))
    means = np.array([r['mean_D'] for r in rows])
    return dict(roundtrips=rows, mean_D=float(means.mean()),
                roundtrip_std=float(means.std(ddof=1)) if len(means)>1 else None,
                roundtrip_cv=float(means.std(ddof=1)/abs(means.mean())) if len(means)>1 and abs(means.mean())>1e-9 else None)
