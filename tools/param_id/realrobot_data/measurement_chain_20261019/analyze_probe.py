"""Compare internal float and status position at native and matched times."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from friction_id.analysis import prepare,Gates,local_motion
from friction_id.recording import load_run
B=Path(__file__).resolve().parent;group=sys.argv[1] if len(sys.argv)>1 else 'motion';prefix='' if group=='motion' else group+'_';p=next((B/group).glob('*.npz'));m,f,c=load_run(p)
with np.load(p,allow_pickle=False) as z:ann=[json.loads(str(x)) for x in z['annotations_json']]
assert m['run_status']=='complete' and c['ok'].all() and not f['status'].any()
assert not any(x.get('kind')=='position_probe_failed' for x in ann)
fold=next(x for x in ann if x.get('kind')=='fold_verified');disabled=next(x for x in ann if x.get('kind')=='disabled_after_fold');assert fold['t']<disabled['t']
samples=[x for x in ann if x.get('kind')=='position_probe_sample'];t=np.array([x['t'] for x in samples]);q=np.array([x['q'] for x in samples]);rtt=np.array([x['t']-x['tx_end'] for x in samples]);read_age=np.array([x['read_t']-x['t'] for x in samples])
assert np.all(np.diff(t)>0) and len(samples)>500 and np.max(read_age)<.04
ff=f[f['joint']==6];cc=c[c['joint']==6];ix=np.searchsorted(cc['tx_end'],t,side='right')-1;ix=np.clip(ix,0,len(cc)-1);cmd=cc[ix]
status_at_probe=np.interp(t,ff['t'],ff['q']);g=Gates();a,quality=prepare(m,f,c,g);rows=[];curves={}
def longest(ts,good):
 ids=np.flatnonzero(good);chunks=np.split(ids,np.flatnonzero((np.diff(ids)!=1)|(np.diff(ts[ids])>g.max_gap))+1)
 return max((ts[x[-1]]-ts[x[0]] for x in chunks if len(x)),default=0.)
for seg in np.unique(cmd['segment'][cmd['segment']>=0]):
 ii=np.flatnonzero((cmd['segment']==seg)&np.isin(cmd['phase'],['accel','constant','decel']));tt=t[ii];high=q[ii];coarse=status_at_probe[ii]
 hv,ha,hr=local_motion(tt,high,g);cv,ca,cr=local_motion(tt,coarse,g)
 inside=(cmd['phase'][ii]=='constant')&(high>=m['measurement_range'][0])&(high<=m['measurement_range'][1])
 ci=np.flatnonzero(cmd['phase'][ii]=='constant');inside&=(tt>=tt[ci[0]]+g.guard)&(tt<=tt[ci[-1]]-g.guard)
 assert np.count_nonzero(inside)>50
 speed=float(np.max(abs(cmd['qd_ref'][ii])));direction=int(cmd['direction'][ii][ci[0]])
 row=dict(segment=int(seg),speed=speed,direction=direction,channels={})
 for name,v,acc,res in [('float_native_50hz',hv,ha,hr),('status_interpolated_same_times',cv,ca,cr)]:
  good=inside&np.isfinite(v)&(abs(v-cmd['qd_ref'][ii])<=g.velocity_relative_error*abs(cmd['qd_ref'][ii])+g.velocity_absolute_error)&(abs(acc)<=g.max_acceleration)&(res<=g.max_position_residual)
  row['channels'][name]=dict(samples=int(inside.sum()),velocity_cv=float(np.std(v[inside])/abs(np.mean(v[inside]))),abs_acceleration_p99=float(np.percentile(abs(acc[inside]),99)),acceleration_over_fraction=float(np.mean(abs(acc[inside])>g.max_acceleration)),longest_steady_s=float(longest(tt,good)))
 row['position_difference']=dict(mean=float(np.mean((coarse-high)[inside])),rms=float(np.sqrt(np.mean((coarse-high)[inside]**2))),abs_p99=float(np.percentile(abs((coarse-high)[inside]),99)))
 row['matched_velocity_correlation']=float(np.corrcoef(hv[inside],cv[inside])[0,1])
 row['matched_acceleration_correlation']=float(np.corrcoef(ha[inside],ca[inside])[0,1])
 rows.append(row)
 for key,value in dict(t=tt,inside=inside,q_float=high,q_status=coarse,v_float=hv,v_status=cv,a_float=ha,a_status=ca).items():curves[f'{seg}_{key}']=value
report=dict(status='internal_measurement_chain_diagnostic',compensation_allowed=False,input=str(p.relative_to(B)),kp=m['kp'][5],kd=m['kd'][5],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),samples=len(samples),reply_interval_ms=dict(median=float(np.median(np.diff(t))*1000),max=float(np.max(np.diff(t))*1000)),rtt_ms=dict(median=float(np.median(rtt)*1000),p99=float(np.percentile(rtt,99)*1000),max=float(rtt.max()*1000)),read_age_ms_max=float(read_age.max()*1000),segments=rows,original_status_quality=quality,fold_verified=fold,disabled_after_fold=disabled,limits=['Both channels are internal and may share an encoder or filters; no external angle truth','Sample timestamps are host kernel reception; motor sampling age unknown','Interpolating status onto probe timestamps changes its observation operator; matched-time diagnostics do not replace native quality gates','Only one run, one round trip per speed; no parameter deployment'])
(B/(prefix+'probe_analysis.json')).write_text(json.dumps(report,indent=2,allow_nan=False));np.savez_compressed(B/(prefix+'probe_comparison.npz'),**curves)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
for col,row in enumerate(rows):
 seg=row['segment'];dd={k:curves[f'{seg}_{k}'] for k in ['t','inside','q_float','q_status','v_float','v_status','a_float','a_status']};inside=dd['inside'];x=dd['q_float'][inside]
 axes[0,col].plot(x,dd['v_status'][inside],label='status at same times');axes[0,col].plot(x,dd['v_float'][inside],label='float register')
 axes[1,col].plot(x,dd['a_status'][inside],label='status at same times');axes[1,col].plot(x,dd['a_float'][inside],label='float register')
 for threshold in [-g.max_acceleration,g.max_acceleration]:axes[1,col].axhline(threshold,color='k',linestyle='--',linewidth=.8)
 axes[2,col].plot(x,(dd['q_status']-dd['q_float'])[inside]*1000)
 axes[0,col].set_title(f"{row['speed']:.2f} rad/s, dir {row['direction']:+d}")
 for ri,label in enumerate(['Velocity (rad/s)','Acceleration (rad/s^2)','Status minus float (mrad)']):
  axes[ri,col].set_ylabel(label);axes[ri,col].set_xlabel('Float register angle (rad)');axes[ri,col].grid(alpha=.25)
  if ri<2:axes[ri,col].legend(fontsize=7)
fig.savefig(B/(prefix+'probe_comparison.png'),dpi=140);plt.close(fig)
print('samples',len(samples),'RTT',report['rtt_ms'])
for r in rows:print(json.dumps(r))
