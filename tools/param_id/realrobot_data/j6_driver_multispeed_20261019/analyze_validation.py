"""Offline control-quality validation; does not fit or deploy friction parameters."""
import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from friction_id.analysis import prepare,Gates
from friction_id.recording import load_run
B=Path(__file__).resolve().parent;g=Gates();runs=[];curves={};ref=None
for name in ['train','holdout']:
    paths=list((B/name).glob('*.npz'));assert len(paths)==1
    p=paths[0];m,f,c=load_run(p)
    assert m['run_status']=='complete' and c['ok'].all() and not f['status'].any()
    with np.load(p,allow_pickle=False) as z:ann=[json.loads(str(x)) for x in z['annotations_json']]
    fold=next(x for x in ann if x.get('kind')=='fold_verified');disabled=next(x for x in ann if x.get('kind')=='disabled_after_fold')
    assert fold['t']<disabled['t'] and max(abs(x) for x in fold['q'])<=.02
    cond=dict(conditions=m['conditions'],rate=m['rate'],speeds=m['speeds'],start=m['start'],end=m['end'],repeats=m['repeats'])
    if ref is None:ref=cond
    else:assert cond==ref
    a,quality=prepare(m,f,c,g);other=np.arange(6)!=5
    run=dict(name=name,file=str(p.relative_to(B)),run_id=m['run_id'],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seed=m['seed'],segments=[],peak_temperature=[float(f[f['joint']==j]['temperature'].max()) for j in range(1,7)],fold_verified=fold,disabled_after_fold=disabled,peak_j6_torque=float(abs(f[f['joint']==6]['tau']).max()))
    for row in quality:
        seg=row['segment'];ix=np.flatnonzero((a['segment']==seg)&(a['phase']=='constant'))
        inside=(a['t'][ix]>=a['t'][ix[0]]+g.guard)&(a['t'][ix]<=a['t'][ix[-1]]-g.guard)
        inside&=(a['q'][ix]>=m['measurement_range'][0])&(a['q'][ix]<=m['measurement_range'][1])
        target=a['qd_ref'][ix]
        criteria=dict(transport=a['transport_valid'][ix],finite_velocity=np.isfinite(a['v'][ix]),speed=np.abs(a['v'][ix]-target)<=g.velocity_relative_error*abs(target)+g.velocity_absolute_error,acceleration=np.abs(a['acceleration'][ix])<=g.max_acceleration,position_fit=a['position_residual'][ix]<=g.max_position_residual,tracking=np.abs(a['q'][ix]-a['q_ref'][ix])<=g.max_tracking_error,other_axes=np.max(abs(a['pose'][ix][:,other]-np.asarray(m['home'])[other]),axis=1)<=g.max_pose_difference)
        failure={key:int(np.count_nonzero(inside&~value)) for key,value in criteria.items()}
        mask=inside&a['transport_valid'][ix]&np.isfinite(a['v'][ix]);jj=ix[mask]
        assert len(jj)>100
        v=a['v'][jj];err=a['q_ref'][jj]-a['q'][jj]
        s=dict(row,speed=m['speeds'][row['speed_id']],eligible_samples=int(inside.sum()),failure_counts=failure,mean_velocity=float(np.mean(v)),tracking_rms_rad=float(np.sqrt(np.mean(err**2))),tracking_peak_rad=float(abs(err).max()),abs_acceleration_p99=float(np.percentile(abs(a['acceleration'][jj]),99)))
        run['segments'].append(s)
        for k in ['q','v','acceleration','tau_fb','q_ref','t']:curves[f'{name}_{seg}_{k}']=a[k][jj]
    runs.append(run)
assert runs[0]['run_id']!=runs[1]['run_id'] and runs[0]['sha256']!=runs[1]['sha256']
speeds=[]
for speed in ref['speeds']:
    rows=[x for r in runs for x in r['segments'] if x['speed']==speed]
    speeds.append(dict(speed=speed,segments=len(rows),passed=sum(x['status']=='valid' for x in rows),velocity_cv_mean=float(np.mean([x['velocity_cv'] for x in rows])),tracking_rms_rad_mean=float(np.mean([x['tracking_rms_rad'] for x in rows])),failure_counts={k:sum(x['failure_counts'][k] for x in rows) for k in rows[0]['failure_counts']}))
report=dict(status='independent_multispeed_control_validation',compensation_allowed=False,conditions=ref,gates=asdict(g),runs=runs,by_speed=speeds,limits=['One training and one independent validation run, two round trips per speed in each; changed order by seed','Same encoder supplies position and derivatives; no independent physical angle reference','Torque remains uncalibrated protocol scale; no deployable friction fit','Existing quality gates unchanged; failure counts are overlapping individual criteria within the eligible interval'])
(B/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False));np.savez_compressed(B/'validation_curves.npz',**curves)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
for col,r in enumerate(runs):
    for row,speed in enumerate(ref['speeds']):
        ax=axes[row,col]
        for s in r['segments']:
            if s['speed']!=speed:continue
            tag=f"{r['name']}_{s['segment']}";label=f"repeat {s['repeat']+1}, dir {s['direction']:+d}, {s['status']}"
            ax.plot(curves[tag+'_q'],abs(curves[tag+'_v']),label=label,alpha=.8)
        ax.axhline(speed,color='k',linestyle='--',linewidth=.8);ax.set_title(f"{r['name']}: {speed:.2f} rad/s")
        ax.set_xlabel('Measured angle (rad)');ax.set_ylabel('Position-derived speed magnitude (rad/s)');ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.savefig(B/'validation.png',dpi=150);plt.close(fig)
print(json.dumps(speeds,indent=2))
for r in runs:print(r['name'],'peakT',r['peak_temperature'],'peak J6 torque',r['peak_j6_torque'])
