"""Matched gravity cancellation, excitation failures and frozen raw validation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from friction_id.paired_dynamic import fit_model, predict, metrics, fit_paired, validate_paired, Options, pairs
from friction_id.analysis import read_parameters
from test_dynamic_diagnostics import plant as base_plant
from friction_id.recording import load_run
from friction_id.common import KP, KD


def plant(path, run_id, start_time):
    m, f, c = base_plant(path, run_id, start_time)
    m.update(kp=KP.tolist(), kd=KD.tolist(), rate=50., speeds=[.05, .12, .20])
    np.savez_compressed(path, metadata_json=json.dumps(m), feedback=f, commands=c)
    return m, f, c


def known_pairs():
    q = np.tile(np.linspace(.1, .45, 180), 9)
    speed = np.repeat(np.repeat([.06, .10, .15], 3), 180)
    group = np.repeat(np.arange(9).astype(str), 180)
    vp = speed+.006*np.sin(2*np.pi*q/.045)
    vn = -speed+.004*np.cos(2*np.pi*q/.045)
    ap = .2*np.sin(2*np.pi*q/.071)+speed*.4
    an = .13*np.cos(2*np.pi*q/.057)-speed*.3
    # Strong varying gravity/even ripple does not enter D.
    common = 4+3*q-2*q*q+.15*np.sin(2*np.pi*q/.045)
    tp = common+.04*ap+.3*np.tanh(vp/.02)+.8*vp
    tn = common+.04*an+.3*np.tanh(vn/.02)+.8*vn
    return dict(q=q, D=(tp-tn)/2, S=(tp+tn)/2,
                dv=(vp-vn)/2, sv=(vp+vn)/2, da=(ap-an)/2, sa=(ap+an)/2,
                df=(np.tanh(vp/.02)-np.tanh(vn/.02))/2,
                sf=(np.tanh(vp/.02)+np.tanh(vn/.02))/2, group=group, speed=speed)


def test_varying_gravity_and_even_ripple_cancel_without_nm_subtraction():
    d = known_pairs(); m = fit_model(d, 'spatial', .045)
    assert not m['issues']
    assert m['coefficients'][:3] == pytest.approx([.3, .8, .04], abs=1e-9)
    assert np.linalg.norm(m['coefficients'][3:]) < 1e-9
    assert metrics(d, m)['reconstructed_torque_rmse'] < 1e-10
    biased = {k: v.copy() for k, v in d.items()}; biased['D'] += .2
    # Prediction is frozen: an unseen directional offset must remain an error.
    assert metrics(biased, m)['difference_rmse'] == pytest.approx(.2)


def test_periodic_acceleration_cannot_be_claimed_separate_from_position():
    d = known_pairs(); d['da'] = np.sin(2*np.pi*d['q']/.045)
    m = fit_model(d, 'spatial', .045)
    assert m['rank'] < m['columns']
    assert 'difference_terms_not_well_separated' in m['issues']


def test_negative_coefficients_are_retained_and_flagged():
    d = known_pairs(); d['D'] = .3*d['df']-.8*d['dv']+.04*d['da']
    m = fit_model(d, 'inertia')
    assert m['coefficients'][1] == pytest.approx(-.8)
    assert 'negative_fv' in m['issues']


def test_raw_pairing_frozen_validation_and_integrity(tmp_path):
    train = tmp_path/'train.npz'; heldout = tmp_path/'test.npz'
    plant(train, 'train-paired', 10.); plant(heldout, 'test-paired', 500.)
    opts = Options(period_steps=21)
    d, info = pairs(train, opts)
    assert len(d['q']) > 100 and max(d['pose_gap']) < opts.max_other_pose_difference
    path = fit_paired([train], tmp_path/'model', opts)
    model = json.loads(path.read_text())
    assert model['compensation_allowed'] is False
    assert 'insufficient_independent_roundtrips_per_speed' in model['excitation_issues']
    result = validate_paired(path, heldout, tmp_path/'validation')
    assert result['compensation_allowed'] is False
    assert result['frozen_models'] == model['models']
    with pytest.raises(ValueError, match='integrity/status'):
        read_parameters(path, 'validated')
    with pytest.raises(ValueError, match='Independent raw run'):
        validate_paired(path, train, tmp_path/'leak')
    meta, f, c = load_run(train); meta['run_id'] = 'relabel'
    np.savez_compressed(tmp_path/'relabeled.npz', metadata_json=json.dumps(meta), feedback=f, commands=c)
    with pytest.raises(ValueError, match='Independent raw run'):
        validate_paired(path, tmp_path/'relabeled.npz', tmp_path/'relabel_validation')
    meta, f, c = load_run(heldout); meta['kp'][5] += 1
    np.savez_compressed(tmp_path/'changed.npz', metadata_json=json.dumps(meta), feedback=f, commands=c)
    with pytest.raises(ValueError, match='conditions differ'):
        validate_paired(path, tmp_path/'changed.npz', tmp_path/'changed_validation')
    model['models'][0]['coefficients'][0] += .1
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match='integrity'):
        validate_paired(path, heldout, tmp_path/'tampered')


@pytest.mark.parametrize('fault', ['stale_peer', 'pose_mismatch'])
def test_stale_peer_and_pose_mismatch_reject_pairs(tmp_path, fault):
    p = tmp_path/'bad.npz'; m, f, c = plant(p, 'bad-pairs', 10.)
    if fault == 'stale_peer':
        f['read_t'][f['joint'] == 2] += .2
    else:
        positive = np.zeros(len(f), bool)
        for seg in np.unique(c['segment'][c['direction'] == 1]):
            cs = c[c['segment'] == seg]
            positive |= (f['t'] >= cs['tx_end'].min()) & (f['t'] <= cs['tx_end'].max()+.005)
        f['q'][positive & (f['joint'] == 2)] += .008
    np.savez_compressed(p, metadata_json=json.dumps(m), feedback=f, commands=c)
    with pytest.raises(ValueError, match='No sufficiently covered'):
        pairs(p, Options())
