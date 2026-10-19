"""Offline per-axis check after folded shutdown, not during motion."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from friction_id.recording import load_run
from friction_id.analysis import prepare,Gates
B=Path(__file__).resolve().parent;j=int(sys.argv[1]);group=sys.argv[2] if len(sys.argv)>2 else f'J{j}_pilot';p=next((B/group).glob('*.npz'));m,f,c=load_run(p)
with np.load(p,allow_pickle=False) as z:ann=[json.loads(str(x)) for x in z['annotations_json']]
assert m['run_status']=='complete' and c['ok'].all() and not f['status'].any()
fold=next(x for x in ann if x.get('kind')=='fold_verified');off=next(x for x in ann if x.get('kind')=='disabled_after_fold');assert fold['t']<off['t']
a,segments=prepare(m,f,c,Gates());out=dict(joint=j,file=str(p.relative_to(B)),kp=m['kp'],kd=m['kd'],segments=segments,fold=fold,disabled=off,peak_temperature=[float(f[f['joint']==k]['temperature'].max()) for k in range(1,7)],peak_axis_torque=float(abs(f[f['joint']==j]['tau']).max()),units=m['units'])
(B/f'{group}_quality.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
