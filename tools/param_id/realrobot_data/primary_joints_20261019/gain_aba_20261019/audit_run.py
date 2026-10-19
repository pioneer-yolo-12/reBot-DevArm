"""Offline analysis only after the selected A/B/A run has folded and disabled."""
import json
from pathlib import Path
import sys
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parents[2]))
from friction_id.recording import load_run
from friction_id.analysis import prepare,Gates
from friction_id.paired_dynamic import pairs,Options
from friction_id.paired_bandwidth import equation,summaries
from friction_id.torque_audit import analyze_torque,plot_audit

name=sys.argv[1];p=next((BASE/name).glob('*.npz'));m,f,c=load_run(p)
with np.load(p,allow_pickle=False) as z:annotations=[json.loads(str(x)) for x in z['annotations_json']]
fold=next(x for x in annotations if x.get('kind')=='fold_verified');off=next(x for x in annotations if x.get('kind')=='disabled_after_fold')
assert m['run_status']=='complete' and c['ok'].all() and not f['status'].any() and fold['t']<off['t']
out=BASE/(name+'_audit')
audit=analyze_torque(p,out);plot_audit(out)
d,info=pairs(p,Options(motion_scope='full'));processed=[]
for sigma in [0,.002,.004,.006]:
 e=equation(d,sigma=sigma);processed.append(dict(sigma=sigma,**summaries(e)))
 if sigma==.004:np.savez_compressed(out/'paired_equation.npz',**e)
a,segments=prepare(m,f,c,Gates());motion=(a['segment']>=0)&np.isin(a['phase'],['accel','constant','decel'])
r=dict(file=str(p),kp=m['kp'],kd=m['kd'],pairing=info,processed=processed,original_method_D_segments=segments,
 max_tracking_error=float(abs(a['q'][motion]-a['q_ref'][motion]).max()),mean_motion_pose=a['pose'][motion].mean(axis=0).tolist(),
 peak_temperature=[float(f[f['joint']==j]['temperature'].max()) for j in range(1,7)],fold=fold,disabled=off,
 force_sections=audit['sections'],payload_decode_max_error=audit['payload_decode_max_error'],compensation_allowed=False)
(out/'run_summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(group=name,processed=processed,max_tracking_error=r['max_tracking_error'],peak_temperature=r['peak_temperature'],
 quality=segments,force_std=[dict(label=s['label'],type=s['frame_type'],std=s['residual_std']) for s in audit['sections']]),indent=2))
