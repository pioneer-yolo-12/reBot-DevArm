from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HOME = np.array([0., 1.2, 1., 0., 0., 0.])
FOLDED = np.zeros(6)
ZERO_TOLERANCE = np.array([0., .02, .02, 0., 0., 0.])
LIMITS = np.array([[-2.8, 2.8], [0, 3.14], [0, 3.14], [-1.57, 1.57], [-1.57, 1.57], [-3.14, 3.14]])
KP = np.array([80., 150., 150., 25., 12., 25.])
KD = np.array([5., 10., 10., 1.5, .8, 1.5])
EFFORT = np.array([36., 36., 36., 14., 14., 14.])
MODELS = ['rs-06'] * 3 + ['rs-00'] * 3
SPEEDS = [.05, .07, .09, .11, .13, .15, .17, .20]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, name, n=6, minimum=None):
    a = np.broadcast_to(np.asarray(value, float), (n,)).copy()
    if not np.all(np.isfinite(a)) or (minimum is not None and np.any(a < minimum)):
        raise ValueError(f'{name}: invalid values')
    return a


def check_position(q, lower_tolerance=0.):
    q = vector(q, 'position')
    tolerance = vector(lower_tolerance, 'lower position tolerance', minimum=0.)
    if np.any(tolerance > .02): raise ValueError('Startup tolerance cannot exceed 0.02 rad')
    if np.any(q < LIMITS[:, 0]-tolerance) or np.any(q > LIMITS[:, 1]):
        raise RuntimeError(f'Joint limit exceeded: {q}')
    return q


def control_module(name):
    """Lazy import; offline tools never load or connect motorbridge."""
    try:
        return importlib.import_module('reBotArm_control_py.' + name)
    except ModuleNotFoundError as exc:
        if not exc.name.startswith('reBotArm_control_py'):
            raise
    candidates = [os.environ.get('REBOT_CONTROL_ROOT', ''), str(ROOT.parent / 'reBotArm_control_py-1'), str(ROOT.parent / 'reBotArm_control_py')]
    for root in candidates:
        if root and (Path(root) / 'reBotArm_control_py').is_dir():
            sys.path.insert(0, root)
            return importlib.import_module('reBotArm_control_py.' + name)
    raise ImportError('Set REBOT_CONTROL_ROOT to the reBotArm_control_py project')


class Gravity:
    def __init__(self):
        loader = control_module('kinematics.robot_model')
        self.model = loader.load_robot_model()
        if list(self.model.names)[1:7] != [f'joint{i}' for i in range(1, 7)]:
            raise ValueError('Unexpected model joint order')
        self.data = self.model.createData()
        self.compute = control_module('dynamics').compute_generalized_gravity
        urdf, _ = loader._resolve_urdf()
        self.model_hash = file_hash(urdf)

    def __call__(self, q):
        full = np.zeros(self.model.nq)
        full[:6] = vector(q, 'gravity position')
        return np.asarray(self.compute(self.model, full, self.data)[:6]).copy()


def smoothstep(s):
    s = np.clip(s, 0., 1.)
    return 10*s**3 - 15*s**4 + 6*s**5


def ramp(t, duration, start, end):
    s = np.clip(t/duration, 0., 1.)
    delta = np.asarray(end)-start
    return start+delta*smoothstep(s), delta*(30*s**2-60*s**3+30*s**4)/duration


class Sweep:
    """Fixed endpoints for every speed, signed continuous acceleration and deceleration."""
    def __init__(self, start, end, speed, ramp_time=1., ramp_down_time=None):
        down = ramp_time if ramp_down_time is None else ramp_down_time
        if not np.isfinite([start, end, speed, ramp_time, down]).all() or speed <= 0 or min(ramp_time, down) <= 0:
            raise ValueError('Invalid sweep')
        self.start, self.end, self.ramp_time = start, end, ramp_time
        self.down_time = down
        self.v = float(np.sign(end-start)*speed)
        self.constant_time = abs(end-start)/speed-(ramp_time+down)/2
        if self.constant_time <= 0:
            raise ValueError('Excursion too short for this speed/ramp')
        self.duration = ramp_time+down+self.constant_time

    def evaluate(self, t):
        T, v = self.ramp_time, self.v
        if t <= T:
            s = np.clip(t/T, 0., 1.)
            return self.start+v*T*(s**6-3*s**5+2.5*s**4), v*smoothstep(s), 'accel'
        if t < T+self.constant_time:
            return self.start+v*T/2+v*(t-T), v, 'constant'
        s = np.clip((t-T-self.constant_time)/self.down_time, 0., 1.)
        q = self.start+v*T/2+v*self.constant_time+v*self.down_time*(s-s**6+3*s**5-2.5*s**4)
        return q, v*(1-smoothstep(s)), 'decel'


class ReferenceTrial:
    """Repeatable single-owner trajectory for baseline/friction comparisons."""
    def __init__(self, joint=6, speed=.1, excursion=.55, repeats=1):
        if joint not in range(1,7) or not .01 <= speed <= .20 or not 1 <= repeats <= 10:
            raise ValueError('Invalid comparison trial')
        self.joint, self.repeats = joint, repeats
        self.end = HOME.copy()
        self.end[joint-1] += excursion
        check_position(self.end)
        self.out = Sweep(HOME[joint-1], self.end[joint-1], speed)
        self.ret = Sweep(self.end[joint-1], HOME[joint-1], speed)

    def __call__(self, elapsed, initial):
        homing = max(3., 1.875*np.max(np.abs(HOME-initial))/.3)
        if elapsed < homing:
            q, qd = ramp(elapsed, homing, initial, HOME)
            return q, qd, False
        t = elapsed-homing
        if t < 1.: return HOME.copy(), np.zeros(6), False
        t -= 1.
        cycle = self.out.duration+self.ret.duration+1.
        if t >= self.repeats*cycle:
            return HOME.copy(), np.zeros(6), True
        t %= cycle
        q, qd = HOME.copy(), np.zeros(6)
        if t <= self.out.duration:
            q[self.joint-1],qd[self.joint-1],_=self.out.evaluate(t)
        elif t < self.out.duration+.5:
            q = self.end.copy()
        elif t <= self.out.duration+.5+self.ret.duration:
            q[self.joint-1],qd[self.joint-1],_=self.ret.evaluate(t-self.out.duration-.5)
        return q, qd, False


def load_calibration(path):
    if path is None:
        return dict(verified=False, scale=[1.]*6, offset=[0.]*6, calibration_id='unverified', units='protocol_nm_unverified')
    value = json.loads(Path(path).read_text())
    if value.get('verified') is not True or value.get('units') != 'N*m' or not value.get('calibration_id') or not value.get('evidence'):
        raise ValueError('Calibration requires verified, N*m units, calibration_id and evidence')
    value['scale'] = vector(value['scale'], 'torque scale').tolist()
    if np.any(np.asarray(value['scale']) == 0):
        raise ValueError('Torque scale cannot be zero')
    value['offset'] = vector(value['offset'], 'torque offset').tolist()
    value['sha256'] = file_hash(path)
    return value
