"""Offline A/B/C comparison; run only after motion and recovery complete."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from friction_id.recording import load_run
from friction_id.analysis import prepare,Gates
BASE=Path(__file__).resolve().parent
PERIOD=.04487989505128276

def harmonic(q,y,period):
    X=np.column_stack([np.ones(len(q)),q-q.mean(),np.sin(2*np.pi*q/period),np.cos(2*np.pi*q/period)])
    b=np.linalg.lstsq(X,y,rcond=None)[0];err=y-X@b
    return dict(period_rad=float(period),amplitude=float(np.hypot(*b[-2:])),rmse=float(np.sqrt(np.mean(err**2))),r2=float(1-np.sum(err**2)/max(np.sum((y-y.mean())**2),1e-30)))

runs=[];curves={};reference=None
for group in ['A','B','C','C_repeat','A_repeat']:
    files=list((BASE/group).glob('*.npz'))
    if len(files)!=1:raise ValueError(f'{group}: expected one complete raw pilot')
    p=files[0];m,f,c=load_run(p)
    assert m['run_status']=='complete' and c['ok'].all() and not f['status'].any()
    with np.load(p,allow_pickle=False) as z:annotations=[json.loads(str(s)) for s in z['annotations_json']]
    fold=next(x for x in annotations if x.get('kind')=='fold_verified')
    disabled=next(x for x in annotations if x.get('kind')=='disabled_after_fold')
    assert fold['t']<disabled['t'] and max(abs(x) for x in fold['q'])<=.02
    condition=dict(m['conditions']);kp=condition.pop('kp');kd=condition.pop('kd')
    condition.update(other_kp=kp[:5],other_kd=kd[:5],rate=m['rate'],start=m['start'],end=m['end'],speeds=m['speeds'],repeats=m['repeats'])
    if reference is None:reference=condition
    assert condition==reference,'Comparison differs outside J6 gains'
    a,quality=prepare(m,f,c,Gates())
    run=dict(group=group,file=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kp=kp[5],kd=kd[5],run_id=m['run_id'],fold_verified=fold,disabled_after_fold=disabled,peak_temperature=[float(f[f['joint']==j]['temperature'].max()) for j in range(1,7)],peak_j6_torque=float(abs(f[f['joint']==6]['tau']).max()),quality=quality,segments=[])
    for seg in np.unique(a['segment'][a['segment']>=0]):
        lo,hi=m['measurement_range']
        mask=(a['segment']==seg)&(a['phase']=='constant')&a['transport_valid']&np.isfinite(a['v'])&(a['q']>=lo)&(a['q']<=hi)
        assert mask.sum()>100
        q=a['q'][mask];v=a['v'][mask];e=a['q_ref'][mask]-q;tau=a['tau_fb'][mask]
        periods=np.linspace(.03,.06,301)
        best=min((harmonic(q,tau,period) for period in periods),key=lambda h:h['rmse'])
        direction=int(a['direction'][mask][0])
        row=dict(segment=int(seg),direction=direction,samples=int(mask.sum()),mean_speed=float(np.mean(v)),velocity_cv=float(np.std(v)/abs(np.mean(v))),tracking_rms_rad=float(np.sqrt(np.mean(e**2))),tracking_peak_rad=float(abs(e).max()),tau_rms=float(np.sqrt(np.mean(tau**2))),torque_spatial_fit=best,tracking_electrical_harmonic=harmonic(q,e,PERIOD))
        run['segments'].append(row)
        for k,x in dict(q=q,v=v,e=e,tau=tau,t=a['t'][mask]).items():curves[f'{group}_{direction}_{k}']=x
    for k in ['velocity_cv','tracking_rms_rad','tracking_peak_rad']:
        run[k]=float(np.mean([s[k] for s in run['segments']]))
    runs.append(run)
report=dict(status='pilot_pd_comparison',compensation_allowed=False,conditions=reference,metric_policy='Common measured angle interval; identical local polynomial position derivative window 0.20s; equal weight by direction. Quality gates unchanged. Spatial frequency fit is descriptive, not held-out validation.',limits=['One round trip per run, A/B/C followed by independent C/A repeats; order not randomized; no independent physical angle measurement','Torque is uncalibrated protocol scale','Changing gains changes controller response; cannot identify a component fault from these runs alone'],runs=runs)
(BASE/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
np.savez_compressed(BASE/'comparison_curves.npz',**curves)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
for col,d in enumerate([1,-1]):
    for r in runs:
        group=r['group'];label=f"{group}: kp={r['kp']:g}, kd={r['kd']:g}";q=curves[f'{group}_{d}_q']
        axes[0,col].plot(q,curves[f'{group}_{d}_v'],label=label,alpha=.85)
        axes[1,col].plot(q,curves[f'{group}_{d}_e']*1000,label=label,alpha=.85)
        axes[2,col].plot(q,curves[f'{group}_{d}_tau'],label=label,alpha=.8)
    axes[0,col].axhline(d*.1,color='k',linestyle='--',linewidth=.8)
    axes[0,col].set_title(f'J6 direction {d:+d}')
    for row,ylabel in enumerate(['Position-derived speed (rad/s)','Tracking error (mrad)','Feedback torque (unverified)']):
        axes[row,col].set_ylabel(ylabel);axes[row,col].set_xlabel('Measured joint angle (rad)');axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=8)
fig.savefig(BASE/'comparison.png',dpi=150);plt.close(fig)
for r in runs:
    print(r['group'],r['kp'],r['kd'],'CV',r['velocity_cv'],'error RMS rad',r['tracking_rms_rad'],'peak torque',r['peak_j6_torque'],'quality',[(x['status'],x['reasons']) for x in r['quality']],'periods',[x['torque_spatial_fit']['period_rad'] for x in r['segments']])
