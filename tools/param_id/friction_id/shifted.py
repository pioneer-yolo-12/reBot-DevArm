"""Changed-start experiment: compare fixed absolute angle vs start-relative phase.

No frequency, amplitude, offset or phase is fitted to the changed-start run.
This module exports diagnostic reports only, never compensation parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dynamic
from .common import digest,file_hash
from .recording import load_run


def assert_pair(reference_meta,trial_meta):
    for key in ('joint','rate'):
        if reference_meta.get(key)!=trial_meta.get(key):raise ValueError(f'Shift experiment differs in {key}')
    a=dict(reference_meta['conditions']);b=dict(trial_meta['conditions'])
    a.pop('measurement_range',None);b.pop('measurement_range',None)
    if a!=b:raise ValueError('Shift experiment model/load/calibration/gains/HOME differ')
    joint=reference_meta['joint']-1
    start_a=np.array(reference_meta.get('start',reference_meta['home']),float)
    start_b=np.array(trial_meta.get('start',trial_meta['home']),float)
    delta=start_b-start_a;other=np.arange(6)!=joint
    if np.any(np.abs(delta[other])>1e-8) or not .005<=abs(delta[joint])<=.10:
        raise ValueError('Only a 0.005..0.10 rad moving-axis start shift is allowed')
    excursion_a=np.array(reference_meta['end'])-start_a
    excursion_b=np.array(trial_meta['end'])-start_b
    if not np.allclose(excursion_a,excursion_b,atol=1e-8,rtol=0):
        raise ValueError('Shift experiment excursion differs')
    if set(np.round(reference_meta['speeds'],5))!=set(np.round(trial_meta['speeds'],5)):
        raise ValueError('Shift experiment speeds differ')
    return float(delta[joint])


def profile(d,direction):
    mask=(d['direction']==direction)&(d['phase']=='constant')&d['in_measurement_range']
    q=d['q'][mask];tau=d['tau_fb'][mask]
    if len(q)<60:raise ValueError('Insufficient constant-reference data for changed-start profile')
    bins=np.floor(q/.002).astype(int);xs=[];ys=[]
    for b in np.unique(bins):
        select=bins==b;xs.append(np.median(q[select]));ys.append(np.median(tau[select]))
    if len(xs)<40:raise ValueError('Insufficient actual-angle coverage')
    return np.asarray(xs),np.asarray(ys)


def compare_shift(model_path,reference_path,trial_path,output,gravity):
    model=json.loads(Path(model_path).read_text());checksum=model.pop('checksum',None)
    if digest(model)!=checksum or model.get('schema')!=dynamic.SCHEMA or model.get('status')!='diagnostic_candidate':
        raise ValueError('Dynamic diagnostic model integrity/status failure')
    hashes=dict(dynamic=file_hash(dynamic.__file__),alignment=file_hash(Path(dynamic.__file__).with_name('analysis.py')))
    if model['implementation_sha256']!=hashes:raise ValueError('Dynamic implementation differs from frozen model')
    ma,_,_=load_run(reference_path);mb,_,_=load_run(trial_path)
    shift=assert_pair(ma,mb)
    a,ia=dynamic.dataset(reference_path,gravity,model['options'])
    b,ib=dynamic.dataset(trial_path,gravity,model['options'])
    if ia['run_id']==ib['run_id'] or ia['raw_sha256']==ib['raw_sha256']:
        raise ValueError('Changed-start comparison requires independent raw runs')
    # Reference may have a narrower speed selection/measurement interval than
    # the original multi-speed training run. All physical conditions stay fixed.
    trained=dict(model['training']['conditions']['conditions'])
    compared=dict(ma['conditions'])
    trained.pop('measurement_range',None);compared.pop('measurement_range',None)
    if trained!=compared or model['training']['conditions']['joint']!=ma['joint']:
        raise ValueError('Frozen model physical conditions differ from reference')
    trange=model['training']['temperature_range']
    for info in (ia,ib):
        if info['temperature_range'][0]<trange[0]-10 or info['temperature_range'][1]>trange[1]+10:
            raise ValueError('Changed-start temperature differs by more than 10 C')
    curves={};directions=[]
    for direction in (1,-1):
        x0,y0=profile(a,direction);x1,y1=profile(b,direction)
        lo=max(x0.min(),x1.min(),x0.min()+shift)
        hi=min(x0.max(),x1.max(),x0.max()+shift)
        if hi-lo<.15:raise ValueError('Insufficient common absolute/relative angle interval')
        x=np.arange(lo,hi,.002)
        actual=np.interp(x,x1,y1)
        absolute=np.interp(x,x0,y0)
        relative=np.interp(x-shift,x0,y0)
        abs_rmse=float(np.sqrt(np.mean((actual-absolute)**2)))
        rel_rmse=float(np.sqrt(np.mean((actual-relative)**2)))
        directions.append(dict(direction=direction,absolute_angle_rmse=abs_rmse,
            start_relative_rmse=rel_rmse,common_angle_range=[float(x.min()),float(x.max())]))
        for name,value in dict(q=x,trial=actual,reference_absolute=absolute,reference_relative=relative).items():
            curves[f'{name}_{direction}']=value
    # Freeze the previously learned dynamic model, restrict trial predictions
    # to the reference run's measured angle range; no extrapolated score.
    amin,amax=float(a['q'].min()),float(a['q'].max())
    mask=(b['q']>=amin)&(b['q']<=amax)
    restricted={k:v[mask] for k,v in b.items()}
    scores={m['kind']:dynamic.metrics(restricted,m)[0] for m in model['models']}
    supported=all(d['absolute_angle_rmse']<.6*d['start_relative_rmse'] for d in directions)
    report=dict(status='changed_start_diagnostic',compensation_allowed=False,
        start_shift_rad=shift,reference=ia,trial=ib,frozen_model_checksum=checksum,
        directions=directions,frozen_dynamic_scores=scores,
        absolute_angle_alignment_supported=bool(supported),
        policy='No shifted-run phase/amplitude/bias/frequency refit; profiles use common actual-angle coverage',
        limits=['Not a diagnosis of a particular motor/encoder/transmission fault',
            'Unverified protocol torque remains uncalibrated; no online compensation approval'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'changed_start.json').exists():raise FileExistsError(out/'changed_start.json')
    dynamic.save_json(out/'changed_start.json',report)
    np.savez_compressed(out/'changed_start_profiles.npz',**curves)
    return report


def plot_shift(path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    fig,axes=plt.subplots(2,2,figsize=(13,7),sharex='col',layout='constrained')
    for col,direction in enumerate((1,-1)):
        x=d[f'q_{direction}'];trial=d[f'trial_{direction}']
        absolute=d[f'reference_absolute_{direction}'];relative=d[f'reference_relative_{direction}']
        axes[0,col].plot(x,trial,label='changed-start measured')
        axes[0,col].plot(x,absolute,label='reference at same actual angle')
        axes[0,col].plot(x,relative,'--',label='reference anchored to shifted start',alpha=.7)
        axes[1,col].plot(x,trial-absolute,label='absolute-angle residual')
        axes[1,col].plot(x,trial-relative,label='start-relative residual',alpha=.7)
        axes[0,col].set_title(f'Direction {direction:+d}')
        axes[0,col].set_ylabel('Feedback torque (protocol scale, unverified)')
        axes[1,col].set_ylabel('Difference');axes[1,col].set_xlabel('Actual joint angle (rad)')
        for row in (0,1):axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=8)
    fig.suptitle('Changed start; no phase fitting and no encoder zero change')
    fig.savefig(Path(output)/'changed_start.png',dpi=160);plt.close(fig)
