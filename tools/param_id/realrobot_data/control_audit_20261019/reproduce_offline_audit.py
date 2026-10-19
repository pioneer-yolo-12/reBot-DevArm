"""Offline audit only: no motorbridge imports and no CAN sockets."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/a12/reBot-DevArm/tools/param_id/realrobot_data')
OUT=ROOT/'control_audit_20261019'
PERIOD=.04477639408192329

def stats(x):
 x=np.asarray(x);return dict(median=float(np.median(x)),p99=float(np.percentile(x,99)),max=float(x.max()))
def harmonic(q,y):
 X=np.column_stack([np.ones(len(q)),q-q.mean(),np.sin(2*np.pi*q/PERIOD),np.cos(2*np.pi*q/PERIOD)])
 coef=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@coef
 return dict(amplitude=float(np.hypot(*coef[-2:])),phase_rad=float(np.arctan2(coef[-1],coef[-2])),r2=float(1-np.sum((y-pred)**2)/max(np.sum((y-y.mean())**2),1e-30)))
reports=[];plots=[]
for name in ['friction_20261016_172447_b4729f07.npz','friction_20261016_172702_03e23883.npz']:
 with np.load(ROOT/name,allow_pickle=False) as z:
  f=z['feedback'];c=z['commands'];meta=json.loads(str(z['metadata_json']))
 report=dict(file=name,run_status=meta['run_status'],joint=meta['joint'],commands=[],passes=[],stationary_axes=[],feedback=[])
 for j in range(1,7):
  cj=c[(c['joint']==j)&(c['segment']>=0)];fj=f[f['joint']==j]
  same_segment=np.diff(cj['segment'])==0; intervals=np.diff(cj['tx_start'])[same_segment]
  report['commands'].append(dict(joint=j,interval_ms=stats(intervals*1e3),send_duration_ms=stats((cj['tx_end']-cj['tx_start'])*1e3),gaps_gt_20ms=int((intervals>.02).sum()),policy='Only within one motion segment; intentional 0.5s holds between segments excluded'))
  report['feedback'].append(dict(joint=j,read_latency_ms=stats((fj['read_t']-fj['t'])*1e3),types={str(typ):int(((fj['can_id']>>24)==typ).sum()) for typ in [2,24]}))
 for seg in np.unique(c['segment'][c['segment']>=0]):
  cs=c[(c['joint']==6)&(c['segment']==seg)];constant=cs[cs['phase']=='constant'];lo,hi=constant['tx_end'][0]+.15,constant['tx_start'][-1]-.15
  ts=np.arange(lo,hi,.01);qs={};vels={};taus={};rawvels={};fs={}
  for j in range(1,7):
   fj=f[(f['joint']==j)&(f['t']>=lo-.2)&(f['t']<=hi+.2)]
   q=np.interp(ts,fj['t'],fj['q']);v=savgol_filter(q,15,3,deriv=1,delta=.01)
   qs[j]=q;vels[j]=v;taus[j]=np.interp(ts,fj['t'],fj['tau']);rawvels[j]=np.interp(ts,fj['t'],fj['qd'])
   if j!=6:
    report['stationary_axes'].append(dict(segment=int(seg),joint=j,q_span=float(np.ptp(q)),position_velocity_rms=float(np.sqrt(np.mean(v*v))),reported_velocity_rms=float(np.sqrt(np.mean(rawvels[j]**2))),tau_rms=float(np.sqrt(np.mean(taus[j]**2)))))
   fs[j]=fj
  q=qs[6];v=vels[6];tau=taus[6];fj=fs[6]
  qref=np.interp(ts,cs['tx_end'],cs['q_ref']);ff=np.interp(ts,cs['tx_end'],cs['tau_ff']);e=qref-q
  phase=2*np.pi*q/PERIOD
  # Match actual encoder quantization used in current SDK, for a command-side diagnostic.
  encoded=(np.floor((cs['q_ref'].astype('f4')/np.float32(4*np.pi)+np.float32(1))*np.float32(32767))/32767-1)*4*np.pi
  qe=np.interp(ts,cs['tx_end'],encoded-cs['q_ref'])
  row=dict(segment=int(seg),direction=int(cs['direction'][0]),speed=float(max(abs(cs['qd_ref']))),duration=float(ts[-1]-ts[0]),samples=len(ts),mean_position_velocity=float(np.mean(v)),position_velocity_std=float(np.std(v)),raw_velocity_std=float(np.std(rawvels[6])),torque_harmonic=harmonic(q,tau),feedforward_harmonic=harmonic(q,ff),tracking_error_harmonic=harmonic(q,e),position_velocity_harmonic=harmonic(q,v),raw_velocity_harmonic=harmonic(q,rawvels[6]),command_quant_error_harmonic=harmonic(q,qe),ff_span=float(np.ptp(ff)),tau_span=float(np.ptp(tau)),type_specific={})
  for typ in [2,24]:
   ft=fj[(fj['can_id']>>24)==typ]
   row['type_specific'][str(typ)]=dict(q_span=float(np.ptp(ft['q'])),torque_harmonic=harmonic(ft['q'],ft['tau']),zero_position_step_fraction=float(np.mean(np.diff(ft['q'])==0)),unique_positive_position_steps=np.unique(np.round(abs(np.diff(ft['q']))[abs(np.diff(ft['q']))>0],8))[:5].tolist())
  report['passes'].append(row)
  if abs(row['speed']-.1)<1e-6 and row['direction']==1 and not any(p[0]==name for p in plots):
   plots.append((name,ts-ts[0],q,qref,v,rawvels[6],tau,ff,cs))
 reports.append(report)
(OUT/'raw_control_audit.json').write_text(json.dumps(dict(spatial_period_rad=PERIOD,derivative_policy='15-point order-3 Savitzky-Golay derivative on 100Hz interpolated position; 0.15s trimmed constant segments',limits=['Same encoder underlies actual position and derivative; does not independently verify physical shaft motion','Command timestamps bracket host send API; not wire TX timestamp or motor execution time','Existing datasets move J6 only; J1-J5 are holding and are not matched moving-axis controls'],runs=reports),indent=2,allow_nan=False))
fig,ax=plt.subplots(4,2,figsize=(14,11),layout='constrained')
for col,(name,t,q,qr,v,rv,tau,ff,cs) in enumerate(plots):
 ax[0,col].plot(t,(qr-q)*1e3);ax[0,col].set_title(name);ax[0,col].set_ylabel('q_ref - q (mrad)')
 ax[1,col].plot(t,rv,alpha=.45,label='reported velocity');ax[1,col].plot(t,v,label='position derivative');ax[1,col].set_ylabel('rad/s');ax[1,col].legend()
 ax[2,col].plot(q,tau,label='feedback');ax[2,col].plot(q,ff,label='gravity FF');ax[2,col].set_ylabel('Torque (unverified scale)');ax[2,col].legend();ax[2,col].set_xlabel('Actual encoder angle (rad)')
 dt=np.diff(cs['tx_start'])*1000;ax[3,col].plot(cs['tx_start'][1:]-cs['tx_start'][0],dt);ax[3,col].set_ylabel('Host send interval (ms)');ax[3,col].set_xlabel('Pass elapsed time (s)')
 for row in range(4):ax[row,col].grid(alpha=.25)
fig.savefig(OUT/'control_ripple_audit.png',dpi=150);plt.close(fig)
for r in reports:
 print(r['file'])
 print('cadence J6',r['commands'][-1]);print('RX J6',r['feedback'][-1])
 for sp in [.07,.1,.15]:
  rows=[x for x in r['passes'] if abs(x['speed']-sp)<1e-6]
  print('speed',sp,{k:round(float(np.median([x[k]['amplitude'] for x in rows])),8) for k in ['torque_harmonic','feedforward_harmonic','tracking_error_harmonic','position_velocity_harmonic','command_quant_error_harmonic']})
 for j in range(1,6):
  rows=[x for x in r['stationary_axes'] if x['joint']==j]
  print('hold',j,'median rawvel RMS',np.median([x['reported_velocity_rms'] for x in rows]),'posvelRMS',np.median([x['position_velocity_rms'] for x in rows]),'qspan',np.median([x['q_span'] for x in rows]))
