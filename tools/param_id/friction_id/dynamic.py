"""Single-axis OFFLINE diagnostics, never deployable friction parameters.

Training selects all frequencies; held-out prediction never refits coefficients,
phase, intercept, or frequencies. Raw protocol torque cannot be added to model
gravity in N*m until torque calibration has been verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .analysis import Gates, prepare
from .common import digest, file_hash
from .recording import load_run

SCHEMA = 'single_axis_dynamic_diagnostic_v1'
OPTIONS = dict(window=.20, max_residual=.002, max_tracking=.08,
               max_acceleration=5., min_fraction=.70, min_samples=100,
               period_min=.03, period_max=.06, frequency_min=.5, frequency_max=8.)


def save_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def conditions(meta):
    return dict(joint=meta['joint'],conditions=meta['conditions'],rate=meta.get('rate'))


def dataset(path, gravity, options):
    meta, fb, cmd = load_run(path)
    if meta.get('run_status')!='complete' or meta.get('mode') not in ('tracking','identify'):
        raise ValueError('Dynamic diagnostics require a complete tracking/identify run')
    if not len(cmd) or not cmd['ok'].all() or np.any(fb['status']):
        raise ValueError('Failed command/drive fault invalidates dynamic run')
    if meta['conditions']['model_hash']!=gravity.model_hash:
        raise ValueError('Gravity model hash differs from acquisition')
    gates=Gates(window=options['window'])
    a,_=prepare(meta,fb,cmd,gates)
    j=meta['joint']-1
    other=np.arange(6)!=j
    pose_good=np.isfinite(a['pose']).all(axis=1)
    pose_good &= np.max(np.abs(a['pose'][:,other]-np.asarray(meta['home'])[other]),axis=1)<=gates.max_pose_difference
    raw=a['transport_valid'] & pose_good
    # An invalid neighbor must not contaminate a local derivative window.
    window_good=np.zeros(len(raw),bool)
    for seg in np.unique(a['segment'][a['segment']>=0]):
        ix=np.flatnonzero(a['segment']==seg)
        ts=a['t'][ix]; bad=np.r_[0,np.cumsum(~raw[ix])]
        lo=np.searchsorted(ts,ts-options['window']/2)
        hi=np.searchsorted(ts,ts+options['window']/2,side='right')
        window_good[ix]=(bad[hi]-bad[lo])==0
    selected=raw & window_good & (a['segment']>=0)
    selected &= np.isfinite(a['v']) & np.isfinite(a['acceleration'])
    selected &= (a['position_residual']<=options['max_residual'])
    selected &= (np.abs(a['q']-a['q_ref'])<=options['max_tracking'])
    selected &= (np.abs(a['v'])<=.6) & (np.abs(a['acceleration'])<=options['max_acceleration'])
    elapsed=np.zeros(len(raw)); speed=np.zeros(len(raw)); segment_reports=[]
    jc=cmd[cmd['joint']==j+1]
    for seg in np.unique(a['segment'][a['segment']>=0]):
        ix=np.flatnonzero(a['segment']==seg)
        cs=jc[jc['segment']==seg]
        target_speed=float(np.max(np.abs(cs['qd_ref'])))
        elapsed[ix]=a['t'][ix]-float(cs['tx_end'][0])
        speed[ix]=target_speed
        inside=(a['t'][ix]>=a['t'][ix[0]]+options['window']/2)
        inside &= a['t'][ix]<=a['t'][ix[-1]]-options['window']/2
        selected[ix] &= inside
        count=int(selected[ix].sum());fraction=count/max(1,int(inside.sum()))
        if count<options['min_samples'] or fraction<options['min_fraction']:
            raise ValueError(f'Dynamic segment {seg} has insufficient reliable motion samples ({count}, {fraction:.1%})')
        segment_reports.append(dict(segment=int(seg),direction=int(a['direction'][ix[0]]),
            speed=target_speed,samples=count,reliable_fraction=fraction))
    if not segment_reports:
        raise ValueError('No reliable motion segments')
    d={k:v[selected] for k,v in a.items()}
    d['elapsed']=elapsed[selected];d['speed']=speed[selected]
    # At most 100 Hz per pass: do not overweight command replies plus active
    # reports that carry nearly simultaneous encoder values.
    keep=np.zeros(len(d['t']),bool)
    for seg in np.unique(d['segment']):
        ix=np.flatnonzero(d['segment']==seg);last=-np.inf
        for i in ix:
            if d['t'][i]-last>=.009:
                keep[i]=True;last=d['t'][i]
    d={k:v[keep] for k,v in d.items()}
    low,high=meta['measurement_range']
    d['in_measurement_range']=(d['q']>=low)&(d['q']<=high)
    d['gravity_reference_nm']=np.array([gravity(q)[j] for q in d['pose']])
    calibrated=meta['torque_calibration'].get('verified') is True and meta['torque_calibration']['units']=='N*m'
    d['gravity_applied']=d['gravity_reference_nm'] if calibrated else np.zeros(len(d['t']))
    d['target']=d['tau_fb']-d['gravity_applied']
    fingerprint=hashlib.sha256(fb.tobytes()+cmd.tobytes()).hexdigest()
    info=dict(run_id=meta['run_id'],file_sha256=file_hash(path),raw_sha256=fingerprint,
        conditions=conditions(meta),calibrated=calibrated,segments=segment_reports,
        torque_units=meta['torque_calibration']['units'],
        feedback_frames=int(len(fb)),fit_samples=int(len(d['t'])),
        gravity_policy='subtract model N*m' if calibrated else 'reference only; no N*m subtraction from unverified protocol torque',
        temperature_range=[float(fb[fb['joint']==j+1]['temperature'].min()),float(fb[fb['joint']==j+1]['temperature'].max())])
    return d,info


def weights(d):
    w=np.zeros(len(d['t']))
    for seg in np.unique(d['segment']):
        mask=d['segment']==seg;w[mask]=1/np.sqrt(mask.sum())
    return w


def design(d, kind, frequency=None):
    cols=[np.ones(len(d['t'])),d['acceleration'],np.tanh(d['v']/.02),d['v']]
    names=['bias','effective_inertia','smooth_fc','fv']
    if kind=='spatial':
        phase=2*np.pi*d['q']/frequency  # frequency is spatial PERIOD in rad
        cols += [np.sin(phase),np.cos(phase)];names += ['position_sin','position_cos']
    elif kind=='temporal':
        phase=2*np.pi*frequency*d['elapsed']
        # Separate direction phases; all coefficients are still shared across
        # repeats/speeds and fixed before validation.
        for direction in (1,-1):
            cols += [(d['direction']==direction)*np.sin(phase),(d['direction']==direction)*np.cos(phase)]
            names += [f'time_sin_{direction}',f'time_cos_{direction}']
    return np.column_stack(cols),names


def regress(d, kind, frequency=None):
    X,names=design(d,kind,frequency);w=weights(d)
    A=X*w[:,None];y=d['target']*w
    scale=np.linalg.norm(A,axis=0)
    if np.any(scale<1e-10):
        return None
    coef,_,rank,s=np.linalg.lstsq(A/scale,y,rcond=None);coef/=scale
    if rank<X.shape[1]:return None
    error=d['target']-X@coef
    score=float(np.sqrt(np.sum((error*w)**2)/np.sum(w*w)))
    physical=[]
    for i in (1,2,3):
        if coef[i]<0:physical.append(f'{names[i]} is negative; no physical parameter interpretation')
    cond=float(s[0]/s[-1])
    if cond>100:physical.append('scaled condition number exceeds 100; weak parameter separation')
    return dict(kind=kind,frequency=frequency,coefficients=coef.tolist(),names=names,
        frequency_units='rad (spatial period)' if kind=='spatial' else ('Hz' if kind=='temporal' else None),
        training_balanced_rmse=score,scaled_condition=cond,rank=int(rank),physical_issues=physical)


def scan_model(d,kind,grid):
    candidates=[regress(d,kind,float(x)) for x in grid]
    valid=[m for m in candidates if m is not None]
    if not valid:raise ValueError(f'Rank deficient {kind} diagnostic model')
    best=min(valid,key=lambda m:m['training_balanced_rmse'])
    k=int(np.argmin(np.abs(grid-best['frequency'])))
    lo,hi=float(grid[max(0,k-1)]),float(grid[min(len(grid)-1,k+1)])
    def objective(x):
        m=regress(d,kind,float(x))
        return m['training_balanced_rmse'] if m else 1e10
    if hi>lo:
        refined=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-8})
        model=regress(d,kind,float(refined.x))
        if model and model['training_balanced_rmse']<best['training_balanced_rmse']:best=model
    return best


def wave_scan(x,y,grid,spatial):
    # Detrended one-pass sine scan, for descriptive period/frequency only.
    best=None
    if len(x)<30:return None
    variance=float(np.sum((y-y.mean())**2))
    if variance<1e-8:return None
    for value in grid:
        phase=2*np.pi*x/value if spatial else 2*np.pi*value*x
        A=np.column_stack([np.ones(len(x)),x-x.mean(),np.sin(phase),np.cos(phase)])
        beta=np.linalg.lstsq(A,y,rcond=None)[0]
        r2=float(1-np.sum((y-A@beta)**2)/variance)
        if best is None or r2>best['in_sample_r2']:
            best=dict(value=float(value),in_sample_r2=r2)
    return best


def periods(d, options):
    reports=[]
    for seg in np.unique(d['segment']):
        mask=(d['segment']==seg)&(d['phase']=='constant')&d['in_measurement_range']
        ix=np.flatnonzero(mask)
        if len(ix)<60:continue
        q=d['q'][ix];tau=d['tau_fb'][ix]
        edges=np.arange(float(q.min()),float(q.max())+.002,.002)
        bins=np.digitize(q,edges)-1;xs=[];ys=[]
        for b in np.unique(bins):
            select=bins==b
            if select.sum()>=1:
                xs.append(float(np.median(q[select])));ys.append(float(np.median(tau[select])))
        spatial=wave_scan(np.array(xs),np.array(ys),np.linspace(options['period_min'],options['period_max'],241),True)
        temporal=wave_scan(d['elapsed'][ix],tau,np.linspace(options['frequency_min'],options['frequency_max'],301),False)
        reports.append(dict(segment=int(seg),direction=int(d['direction'][ix[0]]),speed=float(d['speed'][ix[0]]),
            mean_actual_speed=float(np.mean(np.abs(d['v'][ix]))),spatial_period=spatial,temporal_frequency=temporal))
    return reports


def fit_dynamic(path, output, gravity, options=None):
    opts=dict(OPTIONS,**(options or {}))
    if set(opts)!=set(OPTIONS) or not all(np.isfinite(x) and x>0 for x in opts.values()):
        raise ValueError('Invalid dynamic diagnostic options')
    if not opts['period_min']<opts['period_max'] or not opts['frequency_min']<opts['frequency_max']:
        raise ValueError('Invalid frequency search bounds')
    if opts['min_fraction']>1:raise ValueError('Invalid reliable fraction')
    d,info=dataset(path,gravity,opts)
    if len(np.unique(np.round(d['speed'],4)))<3:
        raise ValueError('At least three training speed levels required to separate position/velocity effects')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'dynamic_diagnostic_model.json'
    if target.exists():raise FileExistsError(target)
    base=regress(d,'baseline')
    if base is None:raise ValueError('Rank deficient dynamic baseline')
    models=[base,scan_model(d,'spatial',np.linspace(opts['period_min'],opts['period_max'],241)),
            scan_model(d,'temporal',np.linspace(opts['frequency_min'],opts['frequency_max'],151))]
    value=dict(schema=SCHEMA,status='diagnostic_candidate',compensation_allowed=False,
        options=opts,training=info,models=models,period_diagnostics=periods(d,opts),
        interpretation='Lumped single-axis regression; bias absorbs offsets. Coefficients are not physical N*m parameters without calibration; position derivatives may contain encoder error.',
        implementation_sha256=dict(dynamic=file_hash(__file__),alignment=file_hash(Path(__file__).with_name('analysis.py'))))
    value['checksum']=digest(value);save_json(target,value)
    np.savez_compressed(output/'training_aligned.npz',**d)
    return target


def components(d,model):
    X,_=design(d,model['kind'],model['frequency']);coef=np.asarray(model['coefficients'])
    return dict(bias=X[:,0]*coef[0],inertia=X[:,1]*coef[1],
        friction=X[:,2:4]@coef[2:4],gravity=d['gravity_applied'],
        position=X[:,4:]@coef[4:] if model['kind']=='spatial' else np.zeros(len(X)),
        time=X[:,4:]@coef[4:] if model['kind']=='temporal' else np.zeros(len(X)))


def metrics(d, model):
    X,_=design(d,model['kind'],model['frequency'])
    prediction=X@np.asarray(model['coefficients'])+d['gravity_applied']
    error=d['tau_fb']-prediction;w=weights(d)
    per=[]
    for seg in np.unique(d['segment']):
        mask=d['segment']==seg
        per.append(dict(segment=int(seg),direction=int(d['direction'][mask][0]),speed=float(d['speed'][mask][0]),
            rmse=float(np.sqrt(np.mean(error[mask]**2)))))
    return dict(balanced_rmse=float(np.sqrt(np.sum((error*w)**2)/np.sum(w*w))),
        max_error=float(np.max(np.abs(error))),segments=per,
        component_rms={k:float(np.sqrt(np.mean(v*v))) for k,v in components(d,model).items()}),prediction


def validate_dynamic(model_path, heldout, output, gravity):
    model=json.loads(Path(model_path).read_text());checksum=model.pop('checksum',None)
    if digest(model)!=checksum or model.get('schema')!=SCHEMA or model.get('status')!='diagnostic_candidate':
        raise ValueError('Dynamic diagnostic model integrity/status failure')
    implementation=dict(dynamic=file_hash(__file__),alignment=file_hash(Path(__file__).with_name('analysis.py')))
    if model.get('implementation_sha256')!=implementation:
        raise ValueError('Diagnostic implementation changed since fitting; retrain before validation')
    d,info=dataset(heldout,gravity,model['options']);train=model['training']
    if any(info[k]==train[k] for k in ('run_id','file_sha256','raw_sha256')):
        raise ValueError('Independent raw run required; relabeling training data is not validation')
    if info['conditions']!=train['conditions']:
        raise ValueError('Held-out joint/model/load/calibration/gains/rate conditions differ')
    if info['temperature_range'][0]<train['temperature_range'][0]-10 or info['temperature_range'][1]>train['temperature_range'][1]+10:
        raise ValueError('Held-out temperature outside training range +/-10 C')
    train_speeds={round(s['speed'],4) for s in train['segments']}
    if {round(s['speed'],4) for s in info['segments']}!=train_speeds:
        raise ValueError('Held-out speed coverage differs')
    results={};predictions={}
    for m in model['models']:
        results[m['kind']],predictions['prediction_'+m['kind']]=metrics(d,m)
        if m['kind']=='spatial':
            predictions.update({'component_'+k:v for k,v in components(d,m).items()})
    baseline=results['baseline']['balanced_rmse']
    for k,r in results.items():r['improvement_over_baseline']=1-r['balanced_rmse']/max(baseline,1e-12)
    spatial=results['spatial'];temporal=results['temporal']
    per_improvements=[]
    for speed in sorted(train_speeds):
        b=np.mean([x['rmse'] for x in results['baseline']['segments'] if round(x['speed'],4)==speed])
        s=np.mean([x['rmse'] for x in spatial['segments'] if round(x['speed'],4)==speed])
        per_improvements.append(dict(speed=speed,improvement=float(1-s/max(b,1e-12))))
    # Preregistered descriptive prediction gates, not a stability certificate.
    support=(spatial['improvement_over_baseline']>=.20
        and spatial['balanced_rmse']<=.8*temporal['balanced_rmse']
        and all(x['improvement']>=.10 for x in per_improvements))
    report=dict(schema=SCHEMA,status='independent_diagnostic_validation',compensation_allowed=False,
        model_checksum=checksum,training=train,heldout=info,results=results,
        per_speed_spatial_improvement=per_improvements,
        position_term_prediction_supported=bool(support),
        conclusion='Position-dependent term generalizes better than baseline and fixed-time model' if support else 'Insufficient held-out evidence for the position term',
        limits=['Not a causal diagnosis: encoder/drive/transmission effects remain possible',
                'No compensation approval; torque calibration, identifiability and separate intervention tests are required'],
        frozen_models=model['models'])
    report['validation_gates']=dict(min_global_improvement=.20,min_each_speed_improvement=.10,
        max_rmse_vs_fixed_time_model=.80,selected_before_heldout=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'dynamic_validation.json').exists():raise FileExistsError(output/'dynamic_validation.json')
    save_json(output/'dynamic_validation.json',report)
    np.savez_compressed(output/'validation_predictions.npz',**d,**predictions)
    return report


def plot_dynamic(path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    seg=np.unique(d['segment'])[0];mask=d['segment']==seg;t=d['t'][mask]-d['t'][mask][0]
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained')
    axes[0].plot(t,d['v'][mask]);axes[0].set_ylabel('Actual velocity (rad/s)')
    axes[1].plot(t,d['tau_fb'][mask],label='feedback',color='black',linewidth=1)
    for kind in ('baseline','spatial','temporal'):
        axes[1].plot(t,d['prediction_'+kind][mask],label=kind,alpha=.8)
        axes[2].plot(t,d['tau_fb'][mask]-d['prediction_'+kind][mask],label=kind)
    axes[1].set_ylabel('Torque: dataset calibration units');axes[1].legend()
    axes[2].set_ylabel('Prediction residual');axes[2].legend()
    for kind in ('inertia','friction','position','bias','gravity'):
        axes[3].plot(t,d['component_'+kind][mask],label=kind)
    axes[3].set_ylabel('Spatial model components');axes[3].set_xlabel('Held-out pass time (s)');axes[3].legend(ncol=5)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Independent run; all coefficients and frequencies frozen from training')
    fig.savefig(Path(output)/'dynamic_validation.png',dpi=150);plt.close(fig)
