"""Host-stream spectra and lag diagnostics with known synthetic signals."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from friction_id.torque_audit import spectrum,lag_correlations,analyze_torque
from friction_id.recording import FEEDBACK_DTYPE,COMMAND_DTYPE
from friction_id.common import HOME,KP,KD
from run_phase2_pointmatch import parser,configuration


def test_spectrum_separates_high_frequency_power_and_rejects_gaps():
    t=np.arange(0,4,.01)
    low=np.sin(2*np.pi*2*t);high=np.sin(2*np.pi*20*t)
    assert spectrum(t,low)[0]['power_fraction_ge_10hz']<.001
    assert spectrum(t,high)[0]['power_fraction_ge_10hz']>.999
    t[200:]+=.1
    with pytest.raises(ValueError,match='gapped'):spectrum(t,high)


def test_positive_lag_means_reported_velocity_precedes_torque():
    t=np.arange(0,3,.01);v=-np.sin(2*np.pi*12*t)
    tau=.2*np.sin(2*np.pi*12*(t-.02))
    rows=lag_correlations(t,tau,v)
    best=max(rows,key=lambda r:r['correlation'])
    assert best['lag_s']==pytest.approx(.02)
    assert best['correlation']>.999


def test_raw_audit_and_same_target_frozen_hold_lag(tmp_path):
    frows=[];crows=[];seq=0
    for start in (10.,14.):
        for step in range(600):
            t=start+step*.005;seq+=1;typ=2 if step%2==0 else 24
            for j in range(1,7):
                value=5+.2*np.sin(2*np.pi*12*(t-.02)) if j==2 else 0.
                code=int(round((value/36+1)*32767));tau=(code/32767.-1)*36
                payload=f'80008000{code:04x}015e'
                frows.append((t,t+.0001,int((1e9+t)*1e9),0.,seq,j,HOME[j-1],
                    -np.sin(2*np.pi*12*t) if j==2 else 0.,tau,35.,0,True,(typ<<24)|(j<<8)|0xFD,payload))
                if step%2==0:
                    crows.append((t-.003,t-.002,j,seq,-1,-1,-1,0,'hold',HOME[j-1],0.,KP[j-1],KD[j-1],0.,True))
    meta=dict(schema_version=3,run_id='synthetic-holds',run_status='complete',joint=2,home=HOME.tolist(),
        measurement_range=[1.28,1.67],torque_calibration=dict(scale=[1.]*6,offset=[0.]*6),
        units=dict(tau='protocol_nm_unverified'))
    p=tmp_path/'raw.npz';np.savez_compressed(p,metadata_json=json.dumps(meta),feedback=np.array(frows,dtype=FEEDBACK_DTYPE),commands=np.array(crows,dtype=COMMAND_DTYPE))
    report=analyze_torque(p,tmp_path/'audit')
    assert report['payload_decode_max_error']==0
    assert report['compensation_allowed'] is False
    assert len(report['sections'])==4
    assert len(report['frozen_hold_lag_checks'])==2
    for check in report['frozen_hold_lag_checks']:
        assert check['training_lag_s']==pytest.approx(.02)
        assert check['heldout_correlation']>.99
    assert all(s['local_velocity_rms']<1e-8 for s in report['sections'])


def test_hold_config_is_bounded_and_keeps_default():
    assert configuration(parser().parse_args([]))['hold_time']==.5
    assert configuration(parser().parse_args(['--hold-time','3']))['hold_time']==3
    for value in ('nan','0.1','6'):
        with pytest.raises(ValueError):configuration(parser().parse_args(['--hold-time',value]))
