"""Offline matched-position dynamics diagnostics. Never deployable parameters.

D=(tau+ - tau-)/2, S=(tau+ + tau-)/2. Shared gravity and direction-even
position ripple cancel in D only to the extent the FULL poses match. Fit D
with velocity, acceleration and optional direction-odd spatial terms; fit S
with the same motion coefficients plus a descriptive smooth angle trend and
direction-even spatial terms. No model N*m is subtracted from protocol torque.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis import Gates, prepare
from .common import digest, file_hash
from .recording import load_run

SCHEMA = 'paired_dynamic_diagnostic_v1'


@dataclass
class Options:
    motion_scope: str = 'constant'
    window: float = .20
    guard: float = .25
    grid_step: float = .002
    max_q_gap: float = .004
    max_other_pose_difference: float = .003
    min_pair_bins: int = 30
    min_coverage: float = .60
    period_min: float = .03
    period_max: float = .06
    period_steps: int = 241

    def __post_init__(self):
        if self.motion_scope not in ('constant', 'full'):
            raise ValueError('Invalid motion scope')
        if not all(np.isfinite(v) and v > 0 for k,v in asdict(self).items() if k != 'motion_scope'):
            raise ValueError('Invalid paired diagnostic options')
        if self.period_min >= self.period_max or self.min_coverage > 1:
            raise ValueError('Invalid period/coverage options')
        if self.period_steps < 3 or int(self.period_steps) != self.period_steps:
            raise ValueError('Invalid frequency grid')


def conditions(m):
    # Start/end and measurement windows may change in a held-out experiment.
    return dict(joint=m['joint'], kp=m['kp'], kd=m['kd'], home=m['home'],
                model_hash=m['conditions']['model_hash'], load_id=m['conditions']['load_id'],
                calibration=m['torque_calibration'], rate=m['rate'])


def identity(path, m, f, c):
    return dict(path=str(Path(path)), run_id=m['run_id'], sha256=file_hash(path),
                raw_sha256=hashlib.sha256(f.tobytes()+c.tobytes()).hexdigest())


def motion_bounds(meta, opts):
    if opts.motion_scope == 'constant':
        return meta['measurement_range']
    if ('dynamic_range' not in meta or 'trajectory' not in meta
            or meta['trajectory'].get('profile') != 'quintic_velocity_sweep_v1'):
        raise ValueError('Full-motion pairing requires recorded trajectory and dynamic_range metadata')
    return meta['dynamic_range']


def pairs(path, opts, bounds=None):
    m, f, c = load_run(path)
    if m['run_status'] != 'complete' or m['mode'] not in ('tracking', 'identify'):
        raise ValueError('A complete tracking/identify run is required')
    if not len(c) or not c['ok'].all() or f['status'].any():
        raise ValueError('Failed commands/drive faults invalidate paired diagnostics')
    a, _ = prepare(m, f, c, Gates(window=opts.window))
    own_bounds = motion_bounds(m, opts)
    bounds = list(bounds if bounds is not None else own_bounds)
    grid = np.arange(bounds[0]+opts.grid_step/2, bounds[1], opts.grid_step)
    if len(grid) < opts.min_pair_bins:
        raise ValueError('Measurement range too short')
    other = np.arange(6) != m['joint']-1
    good = a['transport_valid'] & np.isfinite(a['v']) & np.isfinite(a['acceleration'])
    good &= (a['position_residual'] <= .001) & (abs(a['q']-a['q_ref']) <= .05)
    good &= (abs(a['v']) <= .6) & (abs(a['acceleration']) <= 5)
    good &= np.max(abs(a['pose'][:, other]-np.asarray(m['home'])[other]), axis=1) <= .02
    # Any unreliable neighbour invalidates a local derivative window.
    reliable = a['transport_valid']
    window_good = np.zeros(len(good), bool)
    for seg in np.unique(a['segment'][a['segment'] >= 0]):
        ix = np.flatnonzero(a['segment'] == seg); t = a['t'][ix]
        bad = np.r_[0, np.cumsum(~reliable[ix])]
        lo = np.searchsorted(t, t-opts.window/2)
        hi = np.searchsorted(t, t+opts.window/2, side='right')
        window_good[ix] = bad[hi] == bad[lo]
    good &= window_good
    passes = {}
    for seg in np.unique(a['segment'][a['segment'] >= 0]):
        phases = ['constant'] if opts.motion_scope == 'constant' else ['accel', 'constant', 'decel']
        ix = np.flatnonzero((a['segment'] == seg) & np.isin(a['phase'], phases))
        if not len(ix):
            continue
        sel = good[ix] & (a['t'][ix] >= a['t'][ix[0]]+opts.guard) & (a['t'][ix] <= a['t'][ix[-1]]-opts.guard)
        sel &= (a['q'][ix] >= own_bounds[0]) & (a['q'][ix] <= own_bounds[1])
        direction = int(a['direction'][ix[0]])
        sel &= a['v'][ix]*direction > .01
        take = ix[sel]
        key = int(a['repeat'][ix[0]]), int(a['speed_id'][ix[0]])
        if direction in passes.setdefault(key, {}):
            raise ValueError('Ambiguous duplicate direction in a sweep pair')
        passes[key][direction] = take
    rows, reports = [], []

    def at_q(ix, q):
        if len(ix) < 6:
            return None
        # Coalesce repeated encoder codes, no extrapolation across missing data.
        unique = np.unique(a['q'][ix]); hi = np.searchsorted(unique, q, side='right')
        if hi == 0 or hi == len(unique) or unique[hi]-unique[hi-1] > opts.max_q_gap:
            return None
        local = ix[(a['q'][ix] >= unique[hi-1]) & (a['q'][ix] <= unique[hi])]
        if np.ptp(a['t'][local]) > .04:
            return None
        def value(values):
            med = [np.median(values[local][a['q'][local] == u], axis=0) for u in unique[hi-1:hi+1]]
            w = (q-unique[hi-1])/(unique[hi]-unique[hi-1])
            return (1-w)*med[0]+w*med[1]
        return {k: value(a[k]) for k in ('v', 'acceleration', 'tau_fb', 'pose')}

    for (repeat, sid), pp in sorted(passes.items()):
        group = []
        if 1 in pp and -1 in pp:
            for q in grid:
                p, n = at_q(pp[1], q), at_q(pp[-1], q)
                if p is None or n is None:
                    continue
                gap = float(np.max(abs(p['pose'][other]-n['pose'][other])))
                if gap > opts.max_other_pose_difference:
                    continue
                vp, vn, ap, an = p['v'], n['v'], p['acceleration'], n['acceleration']
                group.append(dict(q=float(q), D=float((p['tau_fb']-n['tau_fb'])/2),
                    S=float((p['tau_fb']+n['tau_fb'])/2),
                    dv=float((vp-vn)/2), sv=float((vp+vn)/2),
                    da=float((ap-an)/2), sa=float((ap+an)/2),
                    df=float((np.tanh(vp/.02)-np.tanh(vn/.02))/2),
                    sf=float((np.tanh(vp/.02)+np.tanh(vn/.02))/2),
                    vp=float(vp),vn=float(vn),ap=float(ap),an=float(an),
                    pose_gap=gap, group=f'{m["run_id"]}:{repeat}:{sid}',
                    speed=float(m['speeds'][sid]), repeat=repeat))
        coverage = len(group)/len(grid)
        accepted = len(group) >= opts.min_pair_bins and coverage >= opts.min_coverage
        reports.append(dict(repeat=repeat, speed_id=sid, speed=float(m['speeds'][sid]),
                            bins=len(group), coverage=coverage, accepted=accepted))
        if accepted:
            rows.extend(group)
    if not rows:
        raise ValueError('No sufficiently covered reliable matched-position pairs')
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return d, dict(identity=identity(path, m, f, c), conditions=conditions(m),
                   pair_reports=reports, bounds=bounds,motion_scope=opts.motion_scope,
                   trajectory=m.get('trajectory'),
                   temperature_range=[float(f['temperature'].min()), float(f['temperature'].max())])


def weights(d):
    w = np.zeros(len(d['q']))
    for group in np.unique(d['group']):
        mask = d['group'] == group
        w[mask] = 1/np.sqrt(mask.sum())
    return w


def design(d, kind, period=None, center=0., span=1.):
    X = [d['df'], d['dv']]; names = ['smooth_fc', 'fv']
    if kind != 'velocity':
        X.append(d['da']); names.append('effective_inertia')
    q = (d['q']-center)/span
    Z = [np.ones(len(q)), q, q*q, q*q*q]
    if kind == 'spatial':
        sine, cosine = np.sin(2*np.pi*d['q']/period), np.cos(2*np.pi*d['q']/period)
        X.extend([sine, cosine]); names.extend(['odd_position_sin', 'odd_position_cos'])
        Z.extend([sine, cosine])
    return np.column_stack(X), np.column_stack(Z), names


def solve(X, y, w):
    A = X*w[:, None]; scale = np.linalg.norm(A, axis=0)
    safe = np.maximum(scale, 1e-15)
    beta, _, rank, s = np.linalg.lstsq(A/safe, y*w, rcond=None)
    return beta/safe, int(rank), float(s[0]/max(s[-1], 1e-15))


def predict(d, model):
    X, Z, _ = design(d, model['kind'], model['period_rad'], model['q_center'], model['q_span'])
    beta = np.asarray(model['coefficients']); params = dict(zip(model['names'], beta))
    motion_sum = params['smooth_fc']*d['sf']+params['fv']*d['sv']+params.get('effective_inertia', 0.)*d['sa']
    return X@beta, Z@np.asarray(model['common_coefficients'])+motion_sum


def metrics(d, model):
    pred_d, pred_s = predict(d, model); per = []
    for group in np.unique(d['group']):
        m = d['group'] == group
        per.append(dict(group=str(group), speed=float(d['speed'][m][0]), bins=int(m.sum()),
            difference_rmse=float(np.sqrt(np.mean((d['D'][m]-pred_d[m])**2))),
            common_rmse=float(np.sqrt(np.mean((d['S'][m]-pred_s[m])**2)))))
    dm = float(np.sqrt(np.mean([x['difference_rmse']**2 for x in per])))
    sm = float(np.sqrt(np.mean([x['common_rmse']**2 for x in per])))
    return dict(difference_rmse=dm, common_rmse=sm, reconstructed_torque_rmse=float(np.hypot(dm, sm)), groups=per)


def fit_model(d, kind, period=None):
    center, span = float(np.mean(d['q'])), float(np.ptp(d['q']))
    X, Z, names = design(d, kind, period, center, span); w = weights(d)
    beta, rank, cond = solve(X, d['D'], w)
    params = dict(zip(names, beta))
    motion_sum = params['smooth_fc']*d['sf']+params['fv']*d['sv']+params.get('effective_inertia', 0.)*d['sa']
    gamma, common_rank, common_cond = solve(Z, d['S']-motion_sum, w)
    issues = []
    if rank != X.shape[1] or cond > 100:
        issues.append('difference_terms_not_well_separated')
    if common_rank != Z.shape[1] or common_cond > 100:
        issues.append('common_angle_terms_not_well_separated')
    for k in ('smooth_fc', 'fv', 'effective_inertia'):
        if params.get(k, 0.) < 0:
            issues.append('negative_'+k)
    model = dict(kind=kind, period_rad=period, q_center=center, q_span=span,
                 names=names, coefficients=beta.tolist(), common_coefficients=gamma.tolist(),
                 rank=rank, columns=X.shape[1], scaled_condition=cond,
                 common_scaled_condition=common_cond, issues=issues)
    model['training_metrics'] = metrics(d, model)
    return model


def stability(d, model):
    fits = []
    for group in np.unique(d['group']):
        mask = d['group'] != group
        if mask.sum() < 30:
            continue
        fit = fit_model({k: v[mask] for k, v in d.items()}, model['kind'], model['period_rad'])
        fits.append(dict(omitted=str(group), coefficients=fit['coefficients'], issues=fit['issues']))
    ranges, issues = {}, []
    if not fits:
        issues.append('no_roundtrip_deletion_stability_check')
    else:
        for i, name in enumerate(model['names']):
            if name not in ('smooth_fc', 'fv', 'effective_inertia'):
                continue
            values = np.array([f['coefficients'][i] for f in fits])
            ranges[name] = [float(values.min()), float(values.max())]
            full = model['coefficients'][i]
            if values.min() < 0 < values.max():
                issues.append('roundtrip_deletion_sign_change_'+name)
            if np.max(abs(values-full)) > max(abs(full), .01):
                issues.append('roundtrip_deletion_large_change_'+name)
    return dict(method='leave_one_roundtrip_out; period frozen; descriptive ranges, not confidence intervals',
                fits=fits, coefficient_ranges=ranges, issues=issues)


def implementation():
    return dict(paired_dynamic=file_hash(__file__), alignment=file_hash(Path(__file__).with_name('analysis.py')))


def fit_paired(paths, output, opts=None):
    opts = opts or Options(); paths = list(paths)
    metas = [load_run(p)[0] for p in paths]
    if not metas or any(conditions(m) != conditions(metas[0]) for m in metas):
        raise ValueError('Training conditions differ')
    bounds = [max(motion_bounds(m,opts)[0] for m in metas), min(motion_bounds(m,opts)[1] for m in metas)]
    datasets, infos = [], []
    for p in paths:
        d, info = pairs(p, opts, bounds)
        if any(any(info['identity'][k] == old['identity'][k] for k in ('run_id', 'sha256', 'raw_sha256')) for old in infos):
            raise ValueError('Duplicate training run')
        datasets.append(d); infos.append(info)
    d = {k: np.concatenate([v[k] for v in datasets]) for k in datasets[0]}
    models = [fit_model(d, 'velocity'), fit_model(d, 'inertia')]
    candidates = [fit_model(d, 'spatial', float(p)) for p in np.linspace(opts.period_min, opts.period_max, int(opts.period_steps))]
    full_rank = [m for m in candidates if m['rank'] == m['columns']]
    spatial = min(full_rank or candidates, key=lambda m: m['training_metrics']['reconstructed_torque_rmse'])
    models.append(spatial)
    levels = sorted(set(np.round(d['speed'], 6))); issues = []
    if len(levels) < 3 or levels[-1]-levels[0] < .06-1e-9:
        issues.append('insufficient_commanded_speed_coverage')
    per_speed = {str(float(s)): len(set(d['group'][np.isclose(d['speed'], s)])) for s in levels}
    if min(per_speed.values()) < 3:
        issues.append('insufficient_independent_roundtrips_per_speed')
    for model in models:
        model['stability'] = stability(d, model)
        model['issues'].extend(model['stability']['issues'])
    report = dict(schema=SCHEMA, status='diagnostic_candidate', compensation_allowed=False,
        options=asdict(opts), implementation_sha256=implementation(), conditions=conditions(metas[0]),
        training=infos, bounds=bounds, models=models, excitation_issues=issues, independent_roundtrips_per_speed=per_speed,
        limits=['Effective coefficients are in dataset calibration units, not verified physical parameters.',
                'Same-position difference cancels common gravity/ripple only for matched full poses.',
                'Angle cubic is descriptive, not an identified gravity model; other-axis residual motion/coupling is unmodeled.',
                'Odd/even position terms are phenomenological; they do not identify a physical cause.',
                'Neighbouring bins/derivatives are correlated; bin count is not independent trial count.'])
    report['checksum'] = digest(report)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    target = out/'paired_dynamic_model.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'paired_training.npz', **d)
    return target


def validate_paired(model_path, heldout, output):
    report = json.loads(Path(model_path).read_text()); checksum = report.pop('checksum', None)
    if (digest(report) != checksum or report.get('schema') != SCHEMA
            or report.get('status') != 'diagnostic_candidate' or report.get('compensation_allowed') is not False):
        raise ValueError('Paired diagnostic model integrity failure')
    if report['implementation_sha256'] != implementation():
        raise ValueError('Implementation changed; retrain')
    d, info = pairs(heldout, Options(**report['options']), report['bounds'])
    if any(any(info['identity'][k] == old['identity'][k] for k in ('run_id', 'sha256', 'raw_sha256')) for old in report['training']):
        raise ValueError('Independent raw run required')
    if info['conditions'] != report['conditions']:
        raise ValueError('Held-out conditions differ')
    speeds = sorted(float(k) for k in report['independent_roundtrips_per_speed'])
    if min(d['speed']) < min(speeds)-1e-9 or max(d['speed']) > max(speeds)+1e-9:
        raise ValueError('Held-out commanded speed outside training range')
    low = min(i['temperature_range'][0] for i in report['training'])
    high = max(i['temperature_range'][1] for i in report['training'])
    if info['temperature_range'][0] < low-10 or info['temperature_range'][1] > high+10:
        raise ValueError('Held-out temperature outside training range +/-10 C')
    results = {m['kind']: metrics(d, m) for m in report['models']}
    result = dict(schema=SCHEMA, status='independent_diagnostic_validation', compensation_allowed=False,
                  training_checksum=checksum, heldout=info, results=results,
                  excitation_issues=report['excitation_issues'], frozen_models=report['models'])
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    target = out/'paired_dynamic_validation.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    predictions = {}
    for m in report['models']:
        predictions['D_'+m['kind']], predictions['S_'+m['kind']] = predict(d, m)
    np.savez_compressed(out/'paired_predictions.npz', **d, **predictions)
    return result
