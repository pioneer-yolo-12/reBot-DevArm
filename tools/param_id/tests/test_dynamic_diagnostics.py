"""Known dynamic plants, spatial vs temporal disturbances, held-out leakage."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from friction_id.dynamic import OPTIONS,dataset,fit_dynamic,validate_dynamic
from friction_id.analysis import read_parameters
from friction_id.recording import load_run
from test_friction_pipeline import make_run


class Gravity:
    model_hash='synthetic'
    def __call__(self,q):return np.zeros(6)


def plant(path,run_id,offset,kind='spatial'):
    meta,f,c=make_run(path,run_id=run_id,start_time=offset,mode='tracking')
    # One independent pass per speed and direction is enough for a deterministic
    # test; real acquisitions use two repeats in each independent run.
    c=c[c['repeat']==0]
    last=c['tx_end'].max()+.01;f=f[f['t']<=last]
    j=np.flatnonzero(f['joint']==6)
    for seg in np.unique(c['segment']):
        cs=c[(c['joint']==6)&(c['segment']==seg)]
        ix=j[(f['t'][j]>=cs['tx_end'][0])&(f['t'][j]<=cs['tx_end'][-1]+.005)]
        t=f['t'][ix];q=f['q'][ix];v=f['qd'][ix]
        acc=np.gradient(v,t)
        disturbance=.15*np.sin(2*np.pi*q/.045)+.05*np.cos(2*np.pi*q/.045)
        if kind=='temporal':disturbance=.16*np.sin(2*np.pi*2.25*(t-t[0]))
        f['tau'][ix]=.03+.03*acc+.2*np.tanh(v/.02)+.4*v+disturbance
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)
    return meta,f,c


@pytest.mark.parametrize('kind',['spatial','temporal'])
def test_known_plant_and_frozen_independent_prediction(tmp_path,kind):
    train=tmp_path/'train.npz';test=tmp_path/'heldout.npz'
    plant(train,'train',10.,kind);plant(test,'heldout',500.,kind)
    model_path=fit_dynamic(train,tmp_path/'fit',Gravity())
    model=json.loads(model_path.read_text())
    assert model['compensation_allowed'] is False
    report=validate_dynamic(model_path,test,tmp_path/'validation',Gravity())
    assert report['status']=='independent_diagnostic_validation'
    assert report['compensation_allowed'] is False
    with np.load(tmp_path/'validation/validation_predictions.npz') as z:
        total=sum(z['component_'+k] for k in ('bias','inertia','friction','position','time','gravity'))
        assert np.allclose(total,z['prediction_spatial'])
    with pytest.raises(ValueError,match='integrity/status'):
        read_parameters(model_path,'validated')
    winner=min(report['results'],key=lambda k:report['results'][k]['balanced_rmse'])
    assert winner==kind
    assert report['results'][kind]['balanced_rmse']<.01
    if kind=='spatial':
        fit=next(m for m in model['models'] if m['kind']=='spatial')
        assert fit['frequency']==pytest.approx(.045,abs=.0001)
        coefficients=dict(zip(fit['names'],fit['coefficients']))
        assert coefficients['effective_inertia']==pytest.approx(.03,abs=.005)
        assert coefficients['smooth_fc']==pytest.approx(.2,abs=.01)
        assert coefficients['fv']==pytest.approx(.4,abs=.03)
        assert report['position_term_prediction_supported']
    else:assert not report['position_term_prediction_supported']
    # The validator must not quietly relearn a held-out offset or phase.
    biased_meta,biased_f,biased_c=load_run(test)
    biased_meta['run_id']='heldout-with-new-offset'
    biased_f['tau'][biased_f['joint']==6]+=.15
    np.savez_compressed(tmp_path/'biased.npz',metadata_json=json.dumps(biased_meta),feedback=biased_f,commands=biased_c)
    biased_report=validate_dynamic(model_path,tmp_path/'biased.npz',tmp_path/'biased_validation',Gravity())
    assert biased_report['results'][kind]['balanced_rmse']>.14
    with pytest.raises(ValueError,match='Independent raw run'):
        validate_dynamic(model_path,train,tmp_path/'leaked',Gravity())
    meta,f,c=load_run(train);meta['run_id']='relabeled'
    np.savez_compressed(tmp_path/'renamed.npz',metadata_json=json.dumps(meta),feedback=f,commands=c)
    with pytest.raises(ValueError,match='Independent raw run'):
        validate_dynamic(model_path,tmp_path/'renamed.npz',tmp_path/'relabeled',Gravity())
    meta,f,c=load_run(test);meta['conditions']['kd'][5]+=1
    np.savez_compressed(tmp_path/'different_gains.npz',metadata_json=json.dumps(meta),feedback=f,commands=c)
    with pytest.raises(ValueError,match='conditions differ'):
        validate_dynamic(model_path,tmp_path/'different_gains.npz',tmp_path/'changed_gains',Gravity())
    model['models'][0]['coefficients'][0]+=1
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError,match='integrity'):
        validate_dynamic(model_path,test,tmp_path/'tampered',Gravity())


def test_unverified_protocol_torque_is_never_mixed_with_model_nm(tmp_path):
    p=tmp_path/'raw.npz';meta,f,c=plant(p,'raw',10.)
    meta['torque_calibration'].update(verified=False,units='protocol_nm_unverified')
    np.savez_compressed(p,metadata_json=json.dumps(meta),feedback=f,commands=c)
    class ReferenceGravity(Gravity):
        def __call__(self,q):return np.ones(6)*2
    d,info=dataset(p,ReferenceGravity(),OPTIONS)
    assert not info['calibrated']
    assert np.all(d['gravity_applied']==0.)
    assert np.all(d['gravity_reference_nm']==2.)
    assert np.array_equal(d['target'],d['tau_fb'])


@pytest.mark.parametrize('fault',['failed_command','missing_axis','invalid_time','stale_read','stale_peer','aborted','model_hash'])
def test_bad_acquisition_cannot_enter_dynamic_regression(tmp_path,fault):
    p=tmp_path/'bad.npz';meta,f,c=plant(p,'bad',10.)
    if fault=='failed_command':c['ok'][0]=False
    if fault=='missing_axis':f=f[f['joint']!=2]
    if fault=='invalid_time':f['timestamp_valid']=False
    if fault=='stale_read':f['read_t']=f['t']+.2
    if fault=='stale_peer':f['read_t'][f['joint']==2]=f['t'][f['joint']==2]+.2
    if fault=='aborted':meta['run_status']='aborted'
    if fault=='model_hash':meta['conditions']['model_hash']='different'
    np.savez_compressed(p,metadata_json=json.dumps(meta),feedback=f,commands=c)
    with pytest.raises(ValueError):dataset(p,Gravity(),OPTIONS)
