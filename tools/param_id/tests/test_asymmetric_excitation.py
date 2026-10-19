"""Actual production trajectory, full-motion pairing and excitation observability."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_phase2_pointmatch import parser, configuration
from preview_dynamic_excitation import build, features, identifiability
from friction_id.common import Sweep, HOME, KP, KD
from friction_id.paired_dynamic import Options, pairs, fit_model
from friction_id.recording import FEEDBACK_DTYPE, COMMAND_DTYPE


@pytest.mark.parametrize('direction', [1, -1])
def test_asymmetric_sweep_has_continuous_bounded_motion(direction):
    s = Sweep(0 if direction == 1 else .55, .55 if direction == 1 else 0, .12, .8, 2.)
    samples = [s.evaluate(t) for t in np.linspace(0, s.duration, 2001)]
    q, v = np.array([x[:2] for x in samples]).T
    assert q.min() >= -1e-12 and q.max() <= .55+1e-12
    assert np.all(np.diff(q)*direction >= -1e-12)
    assert abs(v).max() <= .12+1e-12
    assert s.evaluate(s.duration)[0] == pytest.approx(s.end)
    assert abs(s.evaluate(0)[1])+abs(s.evaluate(s.duration)[1]) < 1e-12
    for t in (.8, .8+s.constant_time):
        left, right = s.evaluate(t-1e-6), s.evaluate(t+1e-6)
        assert abs(left[0]-right[0]) < 1e-6
        assert abs(left[1]-right[1]) < 1e-6


def test_ramp_metadata_and_default_compatibility():
    original = configuration(parser().parse_args([]))
    assert original['trajectory']['ramp_up_time'] == original['trajectory']['ramp_down_time'] == 1.
    c, _ = build(2, .8, 2.)
    assert c['trajectory']['ramp_down_time'] == 2.
    assert c['end'][1] == pytest.approx(HOME[1]+.55)
    assert c['measurement_range'][0] == pytest.approx(HOME[1]+.12+.03)
    for value in ('nan', '0.1', 'inf'):
        with pytest.raises(ValueError):
            configuration(parser().parse_args(['--ramp-down-time', value]))


def test_only_full_asymmetric_motion_supplies_independent_acceleration():
    symmetric, _ = build(2, 1., 1.)
    asymmetric, _ = build(2, .8, 2.)
    normal, _ = features(symmetric)
    plateau, _ = features(asymmetric, 'constant')
    full, _ = features(asymmetric)
    assert not identifiability(normal, .0498666)['reference_design_supported']
    assert not identifiability(plateau, .0498666)['reference_design_supported']
    assert identifiability(full, .0498666)['reference_design_supported']


def asymmetric_raw(path):
    cfig, _ = build(2, .8, 2., speeds=(.06, .09, .12))
    meta = dict(cfig, schema_version=3, run_id='synthetic-asymmetric', run_status='complete',
        mode='tracking', rate=100., torque_calibration=dict(scale=[1.]*6,offset=[0.]*6,verified=False,units='protocol_nm_unverified'),
        conditions=dict(model_hash='synthetic',load_id='fixture'))
    frows, crows = [], []; clock = 10.; seq = 0; seg = 0
    for sid, speed in enumerate(cfig['speeds']):
        for direction in (1, -1):
            start, end = (1.2, 1.75) if direction == 1 else (1.75, 1.2)
            sweep = Sweep(start, end, speed, .8, 2.)
            for t in np.arange(0, sweep.duration, .01):
                q, v, phase = sweep.evaluate(t)
                h = 1e-4
                acc = (sweep.evaluate(t+h)[1]-sweep.evaluate(max(0, t-h))[1])/(h if t == 0 else 2*h)
                # Varying common gravity and an even position ripple.
                tau = 4+2*q+.1*np.sin(2*np.pi*q/.05)+.3*np.tanh(v/.02)+.8*v+.04*acc
                seq += 1
                for j in range(1, 7):
                    frows.append((clock,clock+.001,int((clock+1e9)*1e9),0.,seq,j,
                        q if j == 2 else HOME[j-1],v if j == 2 else 0.,tau if j == 2 else 0.,35.,0,True,0,'00'*8))
                    crows.append((clock-.003,clock-.002,j,seq,seg,0,sid,direction,phase,
                        q if j == 2 else HOME[j-1],v if j == 2 else 0.,cfig['kp'][j-1],cfig['kd'][j-1],0.,True))
                clock += .01
            seg += 1; clock += .1
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=np.array(frows,dtype=FEEDBACK_DTYPE),commands=np.array(crows,dtype=COMMAND_DTYPE))


def test_full_motion_pairs_recover_inertia_and_preserve_constant_scope(tmp_path):
    path = tmp_path/'asymmetric.npz'; asymmetric_raw(path)
    full, _ = pairs(path, Options(motion_scope='full'))
    constant, _ = pairs(path, Options())
    assert np.std(full['da']) > .01
    assert np.max(abs(constant['da'])) < .001
    model = fit_model(full, 'inertia')
    assert model['coefficients'] == pytest.approx([.3, .8, .04], abs=.015)
    assert not model['issues']
    with np.load(path) as z:
        meta=json.loads(str(z['metadata_json'])); f=z['feedback']; c=z['commands']
    del meta['trajectory']
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)
    with pytest.raises(ValueError,match='recorded trajectory'):
        pairs(path, Options(motion_scope='full'))
