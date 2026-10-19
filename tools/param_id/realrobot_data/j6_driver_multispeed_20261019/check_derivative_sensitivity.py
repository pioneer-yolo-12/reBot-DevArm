"""Offline sensitivity only; does not change primary acceptance or deploy parameters."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from friction_id.analysis import prepare,Gates,local_motion
from friction_id.recording import load_run
B=Path(__file__).resolve().parent;rows=[];quantum=4*np.pi/32767
for name in ['train','holdout']:
 p=next((B/name).glob('*.npz'));m,f,c=load_run(p)
 # Alternate windows are declared sensitivity cases; primary stays 0.20 s.
 for window in [.12,.20,.28]:
  g=Gates(window=window);a,quality=prepare(m,f,c,g)
  for s in quality:
   mask=(a['segment']==s['segment'])&(a['phase']=='constant')&a['transport_valid']&(a['q']>=m['measurement_range'][0])&(a['q']<=m['measurement_range'][1])
   rows.append(dict(run=name,window=window,speed=m['speeds'][s['speed_id']],segment=s['segment'],status=s['status'],longest_steady_s=s['longest_steady_s'],reasons=s['reasons'],abs_acceleration_p99=float(np.nanpercentile(abs(a['acceleration'][mask]),99))))
 # Ideal constant speed at actual received timestamps, quantized at several
 # fixed sub-LSB phases. This tests simple quantization, not internal encoder
 # update latency, filtering, systematic encoder error, or physical vibration.
 g=Gates();a,_=prepare(m,f,c,g)
 for seg in np.unique(a['segment'][a['segment']>=0]):
  ix=np.flatnonzero(a['segment']==seg);t=a['t'][ix];speed=float(a['qd_ref'][ix][np.argmax(abs(a['qd_ref'][ix]))]);q0=a['q'][ix[0]]
  select=(a['phase'][ix]=='constant')&(a['q'][ix]>=m['measurement_range'][0])&(a['q'][ix]<=m['measurement_range'][1])
  vals=[]
  for phase in [0.,.2,.4,.6,.8]:
   ideal=q0+speed*(t-t[0]);quant=np.floor(ideal/quantum+phase)*quantum
   _,acc,_=local_motion(t,quant,g)
   vals.append(dict(phase_lsb=phase,p99=float(np.nanpercentile(abs(acc[select]),99)),max=float(np.nanmax(abs(acc[select]))),over_fraction=float(np.mean(abs(acc[select])>g.max_acceleration))))
  rows.append(dict(run=name,kind='ideal_constant_speed_quantization',segment=int(seg),speed=abs(speed),samples=vals))
(B/'derivative_sensitivity.json').write_text(json.dumps(dict(primary_window=.20,alternate_windows=[.12,.28],quantum_rad=quantum,policy='Sensitivity only; no acceptance tuning; ideal replay cannot exclude firmware or systematic measurement effects',results=rows),indent=2,allow_nan=False))
for w in [.12,.2,.28]:
 for speed in [.07,.1,.15]:
  group=[r for r in rows if r.get('window')==w and r['speed']==speed]
  print(w,speed,'passed',sum(r['status']=='valid' for r in group),'/',len(group),'median p99acc',np.median([r['abs_acceleration_p99'] for r in group]))
print('ideal quantized p99 max',max(s['p99'] for r in rows if r.get('kind')=='ideal_constant_speed_quantization' for s in r['samples']))
print('ideal quantized acceleration max',max(s['max'] for r in rows if r.get('kind')=='ideal_constant_speed_quantization' for s in r['samples']))
