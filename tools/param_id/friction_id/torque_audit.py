"""Receive-chain audit and fixed-pose/motion diagnostics, never torque calibration.

Frequency/lag refer to host reception streams. No motor sampling timestamps
are available. Correlations never change timestamps or deployment parameters.
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .analysis import Gates, prepare, local_motion
from .common import EFFORT, file_hash
from .recording import load_run


def correlation(x, y):
    if len(x) < 30 or min(np.std(x), np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spectrum(t, y):
    if len(t) < 40 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .04:
        raise ValueError('Insufficient or gapped stream for spectral diagnostics')
    rate = min(100., float(1/np.median(np.diff(t))))
    tt = np.arange(t[0], t[-1], 1/rate)
    yy = np.interp(tt, t, y)
    freq, power = welch(yy, fs=rate, nperseg=min(128, len(yy)), detrend='linear')
    total = float(power.sum())
    return dict(resampled_rate_hz=rate, frequency_resolution_hz=float(freq[1]-freq[0]),
                power_fraction_ge_10hz=float(power[freq >= 10].sum()/total) if total > 1e-20 else None,
                largest_bin_hz=float(freq[np.argmax(power)]) if total > 1e-20 else None), (freq, power)


def lag_correlations(t, tau_residual, velocity_residual, lags=None):
    lags = np.arange(-4, 5)*.01 if lags is None else lags
    rows = []
    for lag in lags:
        # Positive lag means earlier reported velocity correlates with torque.
        tt = t-lag; keep = (tt >= t[0]) & (tt <= t[-1])
        x = -np.interp(tt[keep], t, velocity_residual)
        rows.append(dict(lag_s=float(lag), correlation=correlation(x, tau_residual[keep])))
    return rows


def residuals(t, q, y, hold):
    if hold:
        X = np.column_stack([np.ones(len(t)), t-t.mean()])
    else:
        z = q-q.mean()
        X = np.column_stack([np.ones(len(q)), z, z*z, z*z*z,
                             np.sin(2*np.pi*q/.05), np.cos(2*np.pi*q/.05)])
    return y-X@np.linalg.lstsq(X, y, rcond=None)[0]


def analyze_torque(path, output):
    m, feedback, commands = load_run(path)
    if m['run_status'] != 'complete' or not commands['ok'].all() or feedback['status'].any():
        raise ValueError('Complete fault-free acquisition required')
    if m['joint'] != 2:
        raise ValueError('This preregistered angle-trend audit currently targets J2')
    a, _ = prepare(m, feedback, commands, Gates())
    f = feedback[feedback['joint'] == m['joint']]
    c = commands[commands['joint'] == m['joint']]
    decoded = np.array([(int(p[8:12], 16)/32767.-1)*EFFORT[m['joint']-1] for p in f['payload']])
    payload_error = float(np.max(abs(decoded-f['tau'])))
    types = ((f['can_id'] >> 24) & 31).astype(int)
    labels = np.array([f'motion:{seg}:{direction}' if phase == 'constant' else ('hold' if phase == 'hold' else 'other')
                       for seg, direction, phase in zip(c['segment'], c['direction'], c['phase'])])
    cuts = np.flatnonzero((labels[1:] != labels[:-1]) | (np.diff(c['tx_end']) > .04))+1
    reports, signals = [], {}
    skipped = []
    for block, ix in enumerate(np.split(np.arange(len(c)), cuts)):
        label = labels[ix[0]]
        if label == 'other':
            continue
        begin, end = c['tx_end'][ix[0]]+.25, c['tx_end'][ix[-1]]-.25
        if end-begin < 1.5:
            skipped.append(dict(block=block,label=label,reason='less_than_1.5s_after_0.25s_edge_trims'))
            continue
        mask = (f['t'] >= begin) & (f['t'] <= end) & a['transport_valid']
        for typ in (2, 24):
            ii = np.flatnonzero(mask & (types == typ))
            if len(ii) < 40:
                continue
            t, q, v, tau = [f[k][ii] for k in ('t', 'q', 'qd', 'tau')]
            hold = label == 'hold'
            tr, vr = residuals(t, q, tau, hold), residuals(t, q, v, hold)
            try:
                psd, (freq, power) = spectrum(t, tr)
            except ValueError as exc:
                skipped.append(dict(block=block,label=label,frame_type=typ,reason=str(exc)))
                continue
            actual_v, _, _ = local_motion(t, q, Gates())
            key = f'b{block}_type{typ}'
            report = dict(key=key,block=block,label=label,frame_type=typ,duration=float(t[-1]-t[0]),samples=len(t),
                target_q=float(np.median(c['q_ref'][ix])),q_span=float(np.ptp(q)),
                local_velocity_rms=float(np.sqrt(np.nanmean(actual_v**2))),
                torque_mean=float(np.mean(tau)),torque_std=float(np.std(tau)),residual_std=float(np.std(tr)),
                torque_spectrum=psd,feedback_velocity_residual_std=float(np.std(vr)),
                receive_interval_ms_p50_p99_max=(np.percentile(np.diff(t),[50,99,100])*1000).tolist(),
                lag_scan=lag_correlations(t,tr,vr))
            reports.append(report)
            signals[key+'_t'] = t; signals[key+'_q'] = q
            signals[key+'_tau'] = tau; signals[key+'_tau_residual'] = tr
            signals[key+'_v_residual'] = vr; signals[key+'_local_v'] = actual_v
            signals[key+'_frequency'] = freq; signals[key+'_power'] = power
    # First sufficiently long hold is training; freeze its lag for later holds
    # at the same target. Correlation of detrended signals, not mean prediction.
    validation = []
    for typ in (2, 24):
        holds = [r for r in reports if r['label'] == 'hold' and r['frame_type'] == typ]
        if not holds:
            continue
        train = holds[0]; candidates = [r for r in train['lag_scan'] if r['correlation'] is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda r:r['correlation'])
        for test in holds[1:]:
            if abs(test['target_q']-train['target_q']) > .01:
                continue
            key = test['key']
            result = lag_correlations(signals[key+'_t'], signals[key+'_tau_residual'], signals[key+'_v_residual'], [best['lag_s']])[0]
            validation.append(dict(frame_type=typ,training=train['key'],heldout=key,
                training_lag_s=best['lag_s'],training_correlation=best['correlation'],heldout_correlation=result['correlation']))
    report = dict(kind='J2_torque_receive_chain_diagnostic',compensation_allowed=False,
        input=str(path),input_sha256=file_hash(path),implementation_sha256=file_hash(__file__),
        units=m['units'],run_id=m['run_id'],payload_decode_max_error=payload_error,
        read_delay_ms_p50_p99_max=(np.percentile(f['read_t']-f['t'],[50,99,100])*1000).tolist(),
        invalid_transport_samples=int((~a['transport_valid']).sum()),sections=reports,skipped=skipped,
        frozen_hold_lag_checks=validation,
        limits=['No device sample time; reception lag is not a calibrated motor/control delay.',
                'Spectra use separate frame types, host-time interpolation and at most 100 Hz resampling; aliases remain possible.',
                'Lag selected only on first long hold; later same-target holds use the frozen lag.',
                'Signals are locally detrended for correlation; this is not prediction of the DC torque or calibration.',
                'Motion trend uses fixed .05 rad descriptive basis, not a newly identified physical period.'])
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if (out/'torque_audit.json').exists():
        raise FileExistsError(out/'torque_audit.json')
    (out/'torque_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'torque_audit_signals.npz', **signals)
    return report


def plot_audit(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output); report = json.loads((out/'torque_audit.json').read_text())
    selected = [next((s for s in report['sections'] if s['label'] == label and s['frame_type'] == typ), None)
                for label in ('hold','motion:0:1') for typ in (2,24)]
    selected = [s for s in selected if s]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected),2,figsize=(13,3*len(selected)),squeeze=False,layout='constrained')
    with np.load(out/'torque_audit_signals.npz',allow_pickle=False) as z:
        for row,s in enumerate(selected):
            k=s['key'];t=z[k+'_t'];freq=z[k+'_frequency']
            axes[row,0].plot(t-t[0],z[k+'_tau_residual'],lw=.8)
            axes[row,0].set_title(f'{s["label"]} frame type {s["frame_type"]}');axes[row,0].set_xlabel('Host receive time (s)')
            axes[row,0].set_ylabel('Torque residual (protocol scale)')
            axes[row,1].semilogy(freq,z[k+'_power']);axes[row,1].set_xlabel('Receive-stream frequency (Hz)')
            axes[row,1].set_ylabel('Residual PSD');axes[row,1].axvline(10,color='.5',ls='--')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.savefig(out/'torque_audit.png',dpi=140);plt.close(fig)
