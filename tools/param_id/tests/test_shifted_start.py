"""Differentiate angle-fixed and experiment-start-fixed waveforms."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from friction_id.common import HOME
from friction_id.dynamic import fit_dynamic
from friction_id.shifted import compare_shift
from run_phase2_pointmatch import parser,configuration
from test_dynamic_diagnostics import plant,Gravity


def experiment(path,run_id,clock,shift=0.,position_fixed=True):
    meta,f,c=plant(path,run_id,clock)
    j=f['joint']==6;old=f['q'][j].copy()
    if shift and position_fixed:
        ripple=lambda q:.15*np.sin(2*np.pi*q/.045)+.05*np.cos(2*np.pi*q/.045)
        f['tau'][j]+=ripple(old+shift)-ripple(old)
    f['q'][j]+=shift;c['q_ref'][c['joint']==6]+=shift
    start=HOME.copy();start[5]+=shift;end=start.copy();end[5]+=.55
    meta.update(start=start.tolist(),end=end.tolist(),speeds=[.05,.12,.20],start_offset=shift)
    meta['measurement_range']=(np.array(meta['measurement_range'])+shift).tolist()
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)


def test_start_shift_retains_home_and_explicit_bounds():
    config=configuration(parser().parse_args(['--joint','6','--start-offset','.02']))
    assert config['home']==HOME.tolist()
    assert config['start'][5]==pytest.approx(.02)
    assert config['end'][5]==pytest.approx(.57)
    assert config['measurement_range']==pytest.approx([.10,.49])
    for value in ['.11','nan','inf']:
        with pytest.raises(ValueError):configuration(parser().parse_args(['--start-offset',value]))


@pytest.mark.parametrize('position_fixed',[True,False])
def test_changed_start_recognizes_absolute_vs_relative_without_phase_refit(tmp_path,position_fixed):
    reference=tmp_path/'ref.npz';trial=tmp_path/'trial.npz'
    experiment(reference,'reference',10.)
    model=fit_dynamic(reference,tmp_path/'fit',Gravity())
    experiment(trial,'trial',500.,.02,position_fixed)
    report=compare_shift(model,reference,trial,tmp_path/'out',Gravity())
    assert report['compensation_allowed'] is False
    assert report['absolute_angle_alignment_supported']==position_fixed
    for row in report['directions']:
        winning='absolute_angle_rmse' if position_fixed else 'start_relative_rmse'
        assert row[winning]<.03
    with pytest.raises(ValueError,match='start shift'):
        compare_shift(model,reference,reference,tmp_path/'same',Gravity())


def test_changed_start_rejects_altered_gains_before_fitting(tmp_path):
    from friction_id.shifted import assert_pair
    a=dict(joint=6,rate=100.,conditions={'kp':[25]*6},home=HOME.tolist(),start=HOME.tolist(),
        end=(HOME+np.array([0,0,0,0,0,.55])).tolist(),speeds=[.1])
    b=json.loads(json.dumps(a));b['start'][5]=.02;b['end'][5]=.57;b['conditions']['kp'][5]=30
    with pytest.raises(ValueError,match='gains'):
        assert_pair(a,b)
