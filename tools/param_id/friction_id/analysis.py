"""Offline quality gates and position-bin pairing; no hardware imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import hashlib

import numpy as np
from scipy.stats import t as student_t

from .common import EFFORT, canonical, digest, file_hash
from .recording import load_run


@dataclass
class Gates:
    max_gap: float = .04
    window: float = .20
    guard: float = .25
    velocity_relative_error: float = .15
    velocity_absolute_error: float = .003
    max_acceleration: float = .15
    max_position_residual: float = .001
    max_tracking_error: float = .05
    min_valid_fraction: float = .70
    max_velocity_cv: float = .15
    min_duration: float = .8
    min_samples: int = 30
    bin_width: float = .01
    min_bins: int = 12
    min_position_coverage: float = .60
    max_pose_difference: float = .02
    min_speed_levels: int = 3
    min_speed_span: float = .06
    min_repeats: int = 3
    max_fit_rmse: float = .10
    max_relative_rmse: float = .25
    max_scaled_condition: float = 50.

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('All quality gate values must be finite and positive')
        if max(self.min_valid_fraction, self.min_position_coverage) > 1:
            raise ValueError('Coverage fractions cannot exceed 1')


def local_motion(t, q, gates):
    """Quadratic local position regression at actual receive times, never index dt."""
    n = len(t)
    v = np.full(n, np.nan)
    a = np.full(n, np.nan)
    rms = np.full(n, np.nan)
    for i in range(n):
        lo = np.searchsorted(t, t[i]-gates.window/2)
        hi = np.searchsorted(t, t[i]+gates.window/2, side='right')
        if hi-lo < 6 or t[hi-1]-t[lo] < gates.window*.7 or np.any(np.diff(t[lo:hi]) > gates.max_gap):
            continue
        dt = t[lo:hi]-t[i]
        b = np.column_stack([np.ones(len(dt)), dt, dt*dt])
        beta = np.linalg.lstsq(b, q[lo:hi], rcond=None)[0]
        v[i], a[i] = beta[1], 2*beta[2]
        rms[i] = np.sqrt(np.mean((q[lo:hi]-b@beta)**2))
    return v, a, rms


def prepare(meta, feedback, commands, gates, gravity=None):
    j = int(meta['joint'])
    f = feedback[feedback['joint'] == j]
    c = commands[(commands['joint'] == j) & commands['ok']]
    if len(f) < gates.min_samples or len(c) < 2:
        raise ValueError('Insufficient raw feedback/commands')
    if np.any(np.diff(f['t']) <= 0) or np.any(np.diff(f['seq']) <= 0) or np.any(np.diff(c['tx_end']) <= 0):
        raise ValueError('Nonmonotonic feedback/command events')
    if not all(np.isfinite(f[name]).all() for name in ('t', 'read_t', 'q', 'qd', 'tau', 'temperature')):
        raise ValueError('Nonfinite raw feedback')
    if not all(np.isfinite(commands[name]).all() for name in ('tx_start','tx_end','q_ref','qd_ref','kp','kd','tau_ff')):
        raise ValueError('Nonfinite command events')
    if np.any(commands['tx_end'] < commands['tx_start']): raise ValueError('Invalid command time interval')
    ix = np.searchsorted(c['tx_end'], f['t'], side='right')-1
    in_command = ix >= 0
    ix = np.maximum(ix, 0)
    aligned_c = c[ix]
    valid = in_command & (f['t']-aligned_c['tx_end'] <= gates.max_gap)
    valid &= f['timestamp_valid'] & (f['status'] == 0) & (f['temperature'] <= 65)
    valid &= (f['read_t']-f['t'] >= -.001) & (f['read_t']-f['t'] <= gates.max_gap)
    valid &= f['clock_error'] <= .001
    # Full actual pose from bounded interpolation. No extrapolation over stale axes.
    pose = np.full((len(f), 6), np.nan)
    for axis in range(1, 7):
        fa = feedback[feedback['joint'] == axis]
        if len(fa) < 2 or np.any(np.diff(fa['t']) <= 0):
            valid[:] = False
            continue
        hi = np.searchsorted(fa['t'], f['t'], side='right')
        lo = np.clip(hi-1, 0, len(fa)-1)
        hi = np.clip(hi, 0, len(fa)-1)
        bounded = (f['t'] >= fa['t'][0]) & (f['t'] <= fa['t'][-1])
        bounded &= (fa['t'][hi]-fa['t'][lo] <= gates.max_gap)
        bounded &= fa['timestamp_valid'][lo] & fa['timestamp_valid'][hi]
        bounded &= (fa['status'][lo] == 0) & (fa['status'][hi] == 0)
        for edge in (lo,hi):
            age=fa['read_t'][edge]-fa['t'][edge]
            bounded &= (age>=-.001) & (age<=gates.max_gap)
            bounded &= (fa['clock_error'][edge]<=.001) & (fa['temperature'][edge]<=65)
        pose[:, axis-1] = np.interp(f['t'], fa['t'], fa['q'])
        valid &= bounded
    scale = meta['torque_calibration']['scale'][j-1]
    offset = meta['torque_calibration']['offset'][j-1]
    tau = f['tau']*scale+offset
    valid &= np.abs(f['tau']) < EFFORT[j-1]*.9
    motion_v = np.full(len(f), np.nan)
    motion_a = np.full(len(f), np.nan)
    residual = np.full(len(f), np.nan)
    sample_valid = np.zeros(len(f), bool)
    segment_reports = []
    for seg in np.unique(aligned_c['segment']):
        # Estimate actual derivatives over the complete pass, including ramps.
        # Method D still selects only quality-approved constant-speed samples.
        full = np.flatnonzero((aligned_c['segment'] == seg) & np.isin(aligned_c['phase'], ['accel','constant','decel']))
        if seg >= 0 and len(full) >= gates.min_samples:
            vf, af, rf = local_motion(f['t'][full], f['q'][full], gates)
            motion_v[full], motion_a[full], residual[full] = vf, af, rf
        mask = (aligned_c['segment'] == seg) & (aligned_c['phase'] == 'constant')
        ii = np.flatnonzero(mask)
        if seg < 0 or len(ii) < gates.min_samples:
            continue
        v, a, rms = motion_v[ii], motion_a[ii], residual[ii]
        target = aligned_c['qd_ref'][ii]
        inside = (f['t'][ii] >= f['t'][ii[0]]+gates.guard) & (f['t'][ii] <= f['t'][ii[-1]]-gates.guard)
        pos_window = meta['measurement_range']
        inside &= (f['q'][ii] >= pos_window[0]) & (f['q'][ii] <= pos_window[1])
        good = valid[ii] & np.isfinite(v) & inside
        good &= (np.abs(v-target) <= gates.velocity_relative_error*np.abs(target)+gates.velocity_absolute_error)
        good &= np.abs(a) <= gates.max_acceleration
        good &= rms <= gates.max_position_residual
        good &= np.abs(f['q'][ii]-aligned_c['q_ref'][ii]) <= gates.max_tracking_error
        other = np.arange(6) != j-1
        good &= np.max(np.abs(pose[ii][:, other]-np.asarray(meta['home'])[other]), axis=1) <= gates.max_pose_difference
        eligible = valid[ii] & inside & np.isfinite(v)
        fraction = float(good.sum()/max(1, inside.sum()))
        ve = v[eligible]
        cv = float(np.std(ve)/max(abs(np.mean(ve)), 1e-9)) if len(ve) else None
        reasons = []
        if good.sum() < gates.min_samples: reasons.append('insufficient_steady_samples')
        if fraction < gates.min_valid_fraction: reasons.append('low_steady_fraction')
        if cv is None or cv > gates.max_velocity_cv: reasons.append('nonuniform_motion')
        # Require a continuous valid window, not scattered good points in stick-slip.
        accepted = ii[good]
        chunks = np.split(accepted, np.flatnonzero((np.diff(accepted) != 1) | (np.diff(f['t'][accepted]) > gates.max_gap))+1)
        longest = max((f['t'][x[-1]]-f['t'][x[0]] for x in chunks if len(x)), default=0.)
        if longest < gates.min_duration: reasons.append('no_continuous_steady_window')
        if not reasons: sample_valid[ii] = good
        segment_reports.append(dict(segment=int(seg), repeat=int(aligned_c['repeat'][ii[0]]),
            speed_id=int(aligned_c['speed_id'][ii[0]]), direction=int(aligned_c['direction'][ii[0]]),
            status='invalid' if reasons else 'valid', reasons=reasons, samples=int(good.sum()),
            steady_fraction=fraction, velocity_cv=cv, longest_steady_s=float(longest)))
    pd = aligned_c['kp']*(aligned_c['q_ref']-f['q'])+aligned_c['kd']*(aligned_c['qd_ref']-f['qd'])
    g = np.full(len(f), np.nan)
    if gravity:
        for i in np.flatnonzero(sample_valid): g[i] = gravity(pose[i])[j-1]
    return dict(t=f['t'], seq=f['seq'], q=f['q'], qd_feedback=f['qd'], v=motion_v, acceleration=motion_a,
        q_ref=aligned_c['q_ref'], qd_ref=aligned_c['qd_ref'], command_id=aligned_c['command_id'],
        command_tx_start=aligned_c['tx_start'], command_tx_end=aligned_c['tx_end'], kp=aligned_c['kp'], kd=aligned_c['kd'],
        position_residual=residual, tau_fb=tau, tau_pd_host=pd, tau_ff=aligned_c['tau_ff'],
        tau_minus_g=tau-g, pose=pose, segment=aligned_c['segment'], repeat=aligned_c['repeat'],
        speed_id=aligned_c['speed_id'], direction=aligned_c['direction'], valid=sample_valid,
        transport_valid=valid, phase=aligned_c['phase']), segment_reports


def pair_samples(aligned, meta, gates):
    pairs = []
    lo, hi = meta['measurement_range']
    n_bins = int(np.floor((hi-lo)/gates.bin_width))
    if n_bins < gates.min_bins: raise ValueError('Measurement interval too short')
    # Position bins handle repeated encoder positions without interpolating a nonmonotonic axis.
    bins = np.floor((aligned['q']-lo)/gates.bin_width).astype(int)
    for repeat in np.unique(aligned['repeat'][aligned['valid']]):
        for sid in np.unique(aligned['speed_id'][aligned['valid']]):
            selected = aligned['valid'] & (aligned['repeat'] == repeat) & (aligned['speed_id'] == sid)
            group = []
            for b in range(n_bins):
                out = selected & (bins == b) & (aligned['direction'] == 1)
                ret = selected & (bins == b) & (aligned['direction'] == -1)
                if out.sum() < 2 or ret.sum() < 2: continue
                lower=max(np.min(aligned['q'][out]),np.min(aligned['q'][ret]))
                upper=min(np.max(aligned['q'][out]),np.max(aligned['q'][ret]))
                if upper <= lower: continue
                target=(lower+upper)/2
                # Within a quality-approved single pass/bin, coalesce repeated
                # encoder positions, then interpolate BOTH directions to exactly
                # the same actual q, without extrapolation.
                def at_q(key, mask):
                    q=aligned['q'][mask]; values=aligned[key][mask]
                    if not np.isfinite(values).all(): return None
                    unique=np.unique(q)
                    medians=np.array([np.median(values[q==p]) for p in unique])
                    return float(np.interp(target,unique,medians))
                po, pr = np.median(aligned['pose'][out], axis=0), np.median(aligned['pose'][ret], axis=0)
                po[meta['joint']-1]=pr[meta['joint']-1]=target
                other = np.arange(6) != meta['joint']-1
                if np.max(np.abs(po[other]-pr[other])) > gates.max_pose_difference: continue
                def med(key, mask):
                    return at_q(key, mask)
                vp, vm = med('v', out), med('v', ret)
                if vp <= 0 or vm >= 0: continue
                group.append(dict(repeat=int(repeat), speed_id=int(sid), q=target, position_bin=b,
                    v_plus=vp, v_minus=vm, v=(vp-vm)/2,
                    tau_plus=med('tau_fb', out), tau_minus=med('tau_fb', ret),
                    time_plus=med('t', out), time_minus=med('t', ret),
                    D=(med('tau_fb', out)-med('tau_fb', ret))/2,
                    E=(med('tau_pd_host', out)-med('tau_pd_host', ret))/2,
                    C_plus=med('tau_minus_g', out), C_minus=med('tau_minus_g', ret),
                    pose_plus=po.tolist(), pose_minus=pr.tolist()))
            if len(group) >= gates.min_bins and len(group)/n_bins >= gates.min_position_coverage:
                pairs.extend(group)
    return pairs


def regression(x, y, speed_ids, repeats, gates, with_bias=False):
    result = dict(status='invalid', reasons=[], fc=None, fv=None, bias=0.)
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        result['reasons'].append('insufficient_finite_observations'); return result
    if len(set(speed_ids)) < gates.min_speed_levels or np.ptp(np.abs(x)) < gates.min_speed_span:
        result['reasons'].append('insufficient_speed_coverage')
    for s in set(speed_ids):
        if len({r for r, sid in zip(repeats, speed_ids) if sid == s}) < gates.min_repeats:
            result['reasons'].append('insufficient_independent_repeats'); break
    B = np.column_stack([np.sign(x), x]) if with_bias else np.column_stack([np.ones(len(x)), x])
    if with_bias: B = np.column_stack([np.ones(len(x)), B])
    scales = np.linalg.norm(B, axis=0)
    beta, _, rank, singular = np.linalg.lstsq(B/np.maximum(scales, 1e-12), y, rcond=None)
    beta = beta/np.maximum(scales, 1e-12)
    cond = float(singular[0]/max(singular[-1], 1e-15))
    result.update(rank=int(rank), scaled_condition=cond)
    if rank < B.shape[1] or cond > gates.max_scaled_condition:
        result['reasons'].append('ill_conditioned'); return result
    if with_bias: result.update(bias=float(beta[0]), fc=float(beta[1]), fv=float(beta[2]))
    else: result.update(fc=float(beta[0]), fv=float(beta[1]))
    residual = y-B@beta
    rmse = float(np.sqrt(np.mean(residual**2)))
    result.update(rmse=rmse, relative_rmse=rmse/max(np.mean(np.abs(y)), .01), observations=len(x))
    if rmse > gates.max_fit_rmse or result['relative_rmse'] > gates.max_relative_rmse:
        result['reasons'].append('large_residual')
    if result['fc'] < 0 or result['fv'] < 0: result['reasons'].append('negative_coefficient_in_coulomb_viscous_model')
    dof = len(y)-B.shape[1]
    covariance = np.linalg.inv(B.T@B)*(residual@residual)/max(1, dof)
    half = student_t.ppf(.975, max(1, dof))*np.sqrt(np.maximum(0, np.diag(covariance)))
    result['ols95'] = [(beta-half).tolist(), (beta+half).tolist()]
    # Pass summaries, not individual position bins, are regression observations.
    if dof > 0 and 2*half[-1] > max(abs(result['fv']), .2): result['reasons'].append('uncertain_viscous_slope')
    result['status'] = 'valid' if not result['reasons'] else 'invalid'
    return result


def fit_pairs(pairs, gates):
    summaries = []
    for key in sorted({(p['repeat'], p['speed_id']) for p in pairs}):
        rows = [p for p in pairs if (p['repeat'], p['speed_id']) == key]
        summaries.append(dict(repeat=key[0], speed_id=key[1], **{k: (float(np.mean([p[k] for p in rows])) if all(p[k] is not None for p in rows) else None)
            for k in ('v', 'D', 'E', 'v_plus', 'v_minus', 'C_plus', 'C_minus')}))
    fits = {}
    for method in ('D', 'E', 'C'):
        if method == 'C' and any(s['C_plus'] is None or s['C_minus'] is None for s in summaries):
            fits['C'] = dict(status='invalid', reasons=['matching_gravity_model_not_supplied'], fc=None, fv=None)
            continue
        x, y, ids, reps = [], [], [], []
        for s in summaries:
            if method == 'C':
                x.extend([s['v_plus'], s['v_minus']]); y.extend([s['C_plus'], s['C_minus']])
                ids.extend([s['speed_id']]*2); reps.extend([s['repeat']]*2)
            else:
                x.append(s['v']); y.append(s[method]); ids.append(s['speed_id']); reps.append(s['repeat'])
        fits[method] = regression(x, y, ids, reps, gates, method == 'C')
    return fits, summaries


def analyze(path, output, gates=None, gravity=None):
    gates = gates or Gates()
    meta, feedback, commands = load_run(path)
    if meta.get('mode') not in ('tracking','identify'):
        raise ValueError('Not a friction acquisition run; use compare for compensation logs')
    aligned, segments = prepare(meta, feedback, commands, gates, gravity)
    pairs = pair_samples(aligned, meta, gates)
    fits, summaries = fit_pairs(pairs, gates)
    # A remains a diagnostic baseline, now restricted to accepted steady data.
    av, at, ids, repeats = [], [], [], []
    for seg in np.unique(aligned['segment'][aligned['valid']]):
        keep = aligned['valid'] & (aligned['segment'] == seg)
        av.append(float(np.mean(aligned['v'][keep])))
        at.append(float(np.mean(aligned['tau_pd_host'][keep])))
        ids.append(int(aligned['speed_id'][keep][0]))
        repeats.append(int(aligned['repeat'][keep][0]))
    fits['A'] = regression(av, at, ids, repeats, gates, with_bias=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'aligned.npz', **aligned)
    # C is diagnostic only when a matching gravity model is supplied.
    report = dict(schema_version=1, run_id=meta['run_id'], input_sha256=file_hash(path), metadata=meta,
                  raw_sha256=hashlib.sha256(feedback.tobytes()+commands.tobytes()).hexdigest(),
                  gates=asdict(gates), segments=segments, pairs=pairs, summaries=summaries, fits=fits,
                  eligible_for_validation=(meta['run_status'] == 'complete' and meta['mode'] == 'identify'
                      and meta['torque_calibration']['verified'] and fits['D']['status'] == 'valid'
                      and bool(np.all(commands['ok']))))
    # Missing gravity is explicit, never silently reported as measured zero.
    if not gravity: report['fits']['C']['reasons'] = ['matching_gravity_model_not_supplied']
    (output/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    candidate = dict(schema_version=1, status='candidate', joint=meta['joint'], fc=fits['D']['fc'], fv=fits['D']['fv'],
                     method='D', units=meta['torque_calibration']['units'], training_run_ids=[meta['run_id']],
                     training_sha256=report['input_sha256'], conditions=meta['conditions'],
                     training_raw_sha256=report['raw_sha256'],
                     eligible=report['eligible_for_validation'], gates=asdict(gates),
                     speed_range=[min((p['v'] for p in pairs), default=0), max((p['v'] for p in pairs), default=0)])
    tf = feedback[(feedback['joint'] == meta['joint']) & np.isfinite(feedback['temperature'])]['temperature']
    candidate['temperature_range'] = [float(min(tf)), float(max(tf))] if len(tf) else [0.,0.]
    candidate['checksum'] = digest(candidate)
    (output/'candidate.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False))
    return report, candidate


def read_parameters(path, required_status):
    value = json.loads(Path(path).read_text())
    checksum = value.pop('checksum', '')
    if digest(value) != checksum or value.get('status') != required_status:
        raise ValueError(f'Parameter integrity/status failure: requires {required_status}')
    value['checksum'] = checksum
    return value


def validate(candidate_path, heldout_path, output, gravity=None):
    target = Path(output)/'validated_parameters.json'
    if target.exists(): raise FileExistsError(f'Validation output already exists: {target}')
    candidate = read_parameters(candidate_path, 'candidate')
    if not candidate['eligible']: raise ValueError('Training run was not eligible for validation')
    report, _ = analyze(heldout_path, output, Gates(**candidate['gates']), gravity)
    meta = report['metadata']
    if (meta['run_id'] in candidate['training_run_ids'] or report['input_sha256'] == candidate['training_sha256']
            or report['raw_sha256'] == candidate['training_raw_sha256']
            or meta['joint'] != candidate['joint'] or meta['conditions'] != candidate['conditions']):
        raise ValueError('Validation requires an independent run with matching joint/model/calibration/load/gains')
    if not report['eligible_for_validation']: raise ValueError('Held-out run failed acquisition/quality gates')
    meta_fb = load_run(heldout_path)[1]
    temps = meta_fb[meta_fb['joint'] == meta['joint']]['temperature']
    if min(temps) < candidate['temperature_range'][0]-10 or max(temps) > candidate['temperature_range'][1]+10:
        raise ValueError('Held-out temperature differs by more than 10 C')
    samples = report['summaries']
    x, y = np.array([s['v'] for s in samples]), np.array([s['D'] for s in samples])
    if min(x) < candidate['speed_range'][0]-.01 or max(x) > candidate['speed_range'][1]+.01:
        raise ValueError('Validation speeds outside training coverage')
    rmse = float(np.sqrt(np.mean((y-candidate['fc']-candidate['fv']*x)**2)))
    gates = Gates(**candidate['gates'])
    if rmse > gates.max_fit_rmse or rmse/max(np.mean(np.abs(y)), .01) > gates.max_relative_rmse:
        raise ValueError(f'Held-out prediction failed: RMSE={rmse}')
    candidate.pop('checksum')
    candidate.update(status='validated', validation_run_ids=[meta['run_id']], validation_sha256=report['input_sha256'],
                     validation_rmse=rmse)
    candidate['temperature_range'] = [min(float(min(temps)),candidate['temperature_range'][0]),max(float(max(temps)),candidate['temperature_range'][1])]
    candidate['checksum'] = digest(candidate)
    target.write_text(json.dumps(candidate, ensure_ascii=False, indent=2))
    return target


def plot_report(report, aligned_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(aligned_path, allow_pickle=False) as d:
        moving = d['segment'] >= 0
        t = d['t'][moving]
        if not len(t): return
        t = t-t[0]
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout='constrained')
        axes[0].plot(t, d['q'][moving]); axes[0].set_ylabel('q (rad)')
        axes[1].plot(t, d['qd_feedback'][moving], alpha=.4, label='drive velocity')
        axes[1].plot(t, d['v'][moving], label='local position slope'); axes[1].legend()
        axes[1].set_ylabel('rad/s')
        axes[2].plot(t, d['tau_fb'][moving]); axes[2].set_ylabel(report['metadata']['torque_calibration']['units'])
        axes[3].plot(t, d['valid'][moving].astype(int)); axes[3].set_ylabel('accepted'); axes[3].set_xlabel('s')
        fig.suptitle(f"J{report['metadata']['joint']} - {report['run_id']}")
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(Path(output)/'tracking.png', dpi=150); plt.close(fig)


def compare_compensation(baseline_path, trial_path, output):
    runs=[]
    for path in (baseline_path,trial_path):
        meta,f,c=load_run(path)
        if meta.get('mode')!='compensation' or meta.get('run_status')!='complete':
            raise ValueError('Comparison requires completed compensation runs')
        axes=[]
        for j in range(1,7):
            fb=f[f['joint']==j];cmd=c[(c['joint']==j)&c['ok']]
            if len(fb)<10 or len(cmd)<10 or np.any(np.diff(cmd['tx_end'])<=0):
                raise ValueError('Insufficient/invalid compensation commands')
            ii=np.searchsorted(cmd['tx_end'],fb['t'],side='right')-1
            valid=ii>=0;ii=np.maximum(ii,0)
            valid&=fb['timestamp_valid']&(fb['status']==0)&(fb['t']-cmd['tx_end'][ii]<.04)
            if valid.sum()<10: raise ValueError('Insufficient fresh compensation samples')
            err=cmd['q_ref'][ii[valid]]-fb['q'][valid]
            axes.append(dict(joint=j,samples=int(valid.sum()),position_rmse_rad=float(np.sqrt(np.mean(err**2))),
                position_peak_rad=float(np.max(np.abs(err))),protocol_torque_peak=float(np.max(np.abs(fb['tau'][valid]))),
                temperature_range=[float(np.min(fb['temperature'])),float(np.max(fb['temperature']))],
                rejected_feedback=int((~valid).sum()),failed_commands=int((~c[c['joint']==j]['ok']).sum())))
        runs.append(dict(metadata=meta,axes=axes,input_sha256=file_hash(path)))
    a,b=(r['metadata'] for r in runs)
    for key in ('conditions','motion','rate','gravity_scale'):
        if key not in a or a[key]!=b.get(key): raise ValueError(f'Comparison conditions differ: {key}')
    if not a['motion'].get('joint') or a['run_id']==b['run_id']:
        raise ValueError('Use independent repeatable --tracking-joint runs for comparison')
    report=dict(status='descriptive_comparison_only',baseline=runs[0],trial=runs[1],
                note='Inspect raw traces, saturation and thermal repeatability; these metrics do not certify stability')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report
