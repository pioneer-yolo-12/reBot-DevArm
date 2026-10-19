"""Summarize the frozen A-B-A protocol without hardware access."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
rows=json.loads((BASE/'comparison.json').read_text())
configs=[];audits=[];initial=[]
for name in ['A1','B','A2']:
 p=next((BASE/name).glob('*.npz'))
 with np.load(p,allow_pickle=False) as z:configs.append(json.loads(str(z['metadata_json'])))
 audits.append(json.loads((BASE/(name+'_audit/run_summary.json')).read_text()))
 initial.append(json.loads((BASE/('preflight_'+name+'.json')).read_text())['feedback'])
for key in ['joint','kp','home','start','end','speeds','repeats','trajectory','hold_time','rate','torque_calibration','seed']:
 assert configs[0][key]==configs[1][key]==configs[2][key],key
for key in ['load_id','model_hash','calibration']:
 assert configs[0]['conditions'][key]==configs[1]['conditions'][key]==configs[2]['conditions'][key],key
assert configs[0]['kd']==configs[2]['kd']
diff=np.array(configs[0]['kd'])-np.array(configs[1]['kd']);assert np.array_equal(diff,[0,10,0,0,0,0])
for row,a,start in zip(rows,audits,initial):
 row['initial_temperature']=[r['temperature'] for r in start]
 row['quality']=[s['status'] for s in a['original_method_D_segments']]
 row['payload_error']=a['payload_decode_max_error']
 assert a['fold']['t']<a['disabled']['t']
with np.load(BASE/'A1_audit/paired_equation.npz') as z:q=z['q']
for name in ['B','A2']:
 with np.load(BASE/(name+'_audit/paired_equation.npz')) as z:assert np.array_equal(q,z['q'])
mean_A_hold=(rows[0]['hold_std']+rows[2]['hold_std'])/2
mean_A_motion=(rows[0]['motion_std']+rows[2]['motion_std'])/2
result=dict(rows=rows,primary=dict(window=.2,sigma=.004,endpoint_sigma=.006,period=.0498666),
 hold_std_reduction_B_vs_A=1-rows[1]['hold_std']/mean_A_hold,
 motion_std_reduction_B_vs_A=1-rows[1]['motion_std']/mean_A_motion,
 A2_vs_A1_mean_D_relative_difference=abs(rows[2]['mean_D']-rows[0]['mean_D'])/abs(rows[0]['mean_D']),
 stop_multispeed=True,reason='A1/A2 mean half-difference mismatch exceeds the preregistered 10% screening line',
 compensation_allowed=False)
(BASE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');x=np.arange(3)
axes[0].bar(x-.18,[r['hold_std'] for r in rows],.36,label='hold')
axes[0].bar(x+.18,[r['motion_std'] for r in rows],.36,label='motion');axes[0].legend()
axes[0].set_title('Torque residual standard deviation');axes[0].set_ylabel('Unverified protocol torque')
axes[1].plot(x,[100*r['cv'][0] for r in rows],'o-',label='forward')
axes[1].plot(x,[100*r['cv'][1] for r in rows],'o-',label='reverse');axes[1].legend()
axes[1].set_title('Velocity CV');axes[1].set_ylabel('%')
axes[2].plot(x,[r['mean_D'] for r in rows],'o-');axes[2].set_title('Mean paired half-difference')
axes[2].set_ylabel('Unverified protocol torque; not Fc')
for ax in axes:ax.set_xticks(x,['A1: kd20','B: kd10','A2: kd20']);ax.grid(axis='y',alpha=.2)
fig.savefig(BASE/'comparison.png',dpi=140);plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
