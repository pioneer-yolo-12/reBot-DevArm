#!/usr/bin/env python3
"""RS single-axis tracking and friction acquisition. Defaults to preview, no CAN IO."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np

from friction_id.common import (HOME, KP, KD, MODELS, SPEEDS, ROOT, Gravity,
                                Sweep, check_position, file_hash, load_calibration, ramp, vector)
from friction_id.recording import Journal


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['tracking', 'identify'], default='tracking')
    ap.add_argument('--joint', type=int, choices=range(1, 7), default=6)
    ap.add_argument('--speeds', type=float, nargs='+')
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--kp', type=float, help='Override only the moving axis; tune one setting per run')
    ap.add_argument('--kd', type=float)
    ap.add_argument('--excursion', type=float, default=.55)
    ap.add_argument('--start-offset',type=float,default=0.,help='Moving-axis start offset from HOME, rad; bounded to +/-0.10. Does not change encoder zero.')
    ap.add_argument('--ramp-time', type=float, default=1.)
    ap.add_argument('--rate', type=float, default=100.)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--channel', default='can0')
    ap.add_argument('--load-id', default='unrecorded', help='Fixed gripper, payload and cable configuration ID')
    ap.add_argument('--torque-calibration', type=Path)
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'realrobot_data')
    ap.add_argument('--execute', action='store_true', help='Connect and move the robot')
    ap.add_argument('--dry-run', action='store_true', help='Preview only; overrides --execute')
    return ap


def configuration(args):
    speeds = sorted(set(args.speeds or ([.10] if args.mode == 'tracking' else SPEEDS)))
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == 'tracking' else 3)
    if not speeds or not np.isfinite(speeds).all() or min(speeds) < .01 or max(speeds) > .20:
        raise ValueError('Experiment speeds must be within [0.01, 0.20] rad/s')
    if not 1 <= repeats <= 10 or not np.isfinite([args.rate,args.ramp_time,args.excursion]).all() or not 50 <= args.rate <= 200 or args.ramp_time < .5:
        raise ValueError('Invalid repeat count/rate/ramp time')
    if not np.isfinite(args.start_offset) or abs(args.start_offset)>.10:
        raise ValueError('Start offset must be finite and within +/-0.10 rad')
    start=HOME.copy();start[args.joint-1]+=args.start_offset
    check_position(start)
    end=start.copy()
    end[args.joint-1] += args.excursion
    check_position(end)
    if args.excursion <= 0: raise ValueError('Excursion must be positive')
    # A common position interval for every speed and direction.
    margin = max(speeds)*args.ramp_time/2+.03
    measurement = [start[args.joint-1]+margin, end[args.joint-1]-margin]
    if measurement[1]-measurement[0] < .15: raise ValueError('Common measurement interval too short')
    kp, kd = KP.copy(), KD.copy()
    if args.kp is not None: kp[args.joint-1] = args.kp
    if args.kd is not None: kd[args.joint-1] = args.kd
    vector(kp,'kp',minimum=0); vector(kd,'kd',minimum=0)
    if np.any(kp > KP*2) or np.any(kd > KD*2): raise ValueError('Gain override exceeds 2x experiment baseline')
    for speed in speeds: Sweep(start[args.joint-1], end[args.joint-1], speed, args.ramp_time)
    return dict(speeds=speeds, repeats=repeats, kp=kp.tolist(), kd=kd.tolist(), home=HOME.tolist(),
                start=start.tolist(),start_offset=args.start_offset,end=end.tolist(), measurement_range=measurement)


def provenance():
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = 'unknown'
    files = sorted(Path(__file__).parent.glob('*.py')) + sorted((Path(__file__).parent/'friction_id').glob('*.py'))
    return dict(git_commit=commit, source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in files})


def run(args, config):
    from friction_id.hardware import FeedbackReceiver, MotorSession
    gravity = Gravity()
    calibration = load_calibration(args.torque_calibration)
    run_id = str(uuid.uuid4())
    path = args.output / ('friction_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+run_id[:8]+'.npz')
    conditions = dict(model_hash=gravity.model_hash, load_id=args.load_id,
                      calibration=calibration, kp=config['kp'], kd=config['kd'], home=config['home'],
                      measurement_range=config['measurement_range'], temperature_limit_c=65.)
    metadata = dict(run_id=run_id, mode=args.mode, joint=args.joint, **config,
                    motor_models=MODELS, firmware_versions='not_read; record in calibration evidence',
                    conditions=conditions, torque_calibration=calibration, rate=args.rate,
                    timestamp_source='SO_TIMESTAMPNS kernel realtime mapped to monotonic; not device sample time',
                    sequence_source='local per-axis receive counter; not a device packet counter',
                    units=dict(q='rad',qd='rad/s',tau=calibration['units'],time='monotonic seconds'),
                    provenance=provenance(), seed=args.seed, channel=args.channel,
                    motorbridge_version=importlib.metadata.version('motorbridge'))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    journal = Journal(path, metadata)
    old_handler = signal.signal(signal.SIGINT, request_stop)
    receiver = FeedbackReceiver(args.channel, journal)
    status, reason = 'aborted', 'initialization incomplete'
    period = 1/args.rate
    try:
        with MotorSession(args.channel, receiver, journal, gravity, np.array(config['kp']), np.array(config['kd'])) as session:
            def execute(duration, reference, **tags):
                began = time.monotonic()
                deadline = began
                while True:
                    if stop: raise KeyboardInterrupt
                    if journal.error: raise RuntimeError(f'Checkpoint writer failed: {journal.error}')
                    now = time.monotonic()
                    if now-deadline > .05: raise RuntimeError('Control deadline missed by more than 50 ms')
                    elapsed = min(now-began,duration)
                    q, qd, phase = reference(elapsed)
                    session.send(q, qd, phase=phase, **tags)
                    if elapsed >= duration: break
                    deadline += period
                    time.sleep(max(0.,deadline-time.monotonic()))
            if stop: raise KeyboardInterrupt
            session.enable()
            q0 = np.array([r['q'] for r in receiver.snapshot()])
            execute(1.,lambda t:(q0,np.zeros(6),'startup_hold'))
            home, end = np.array(config['home']), np.array(config['end'])
            # Quintic duration bounds peak velocity (1.875*distance/duration).
            duration = max(3.,1.875*np.max(np.abs(home-q0))/.30)
            execute(duration,lambda t:(*ramp(t,duration,q0,home),'home'))
            execute(.5,lambda t:(home,np.zeros(6),'hold'))
            if np.max(np.abs(np.array([r['q'] for r in receiver.snapshot()])-home)) > .05:
                raise RuntimeError('HOME verification failed')
            session.finish_startup()
            sweep_start=np.array(config['start'])
            if args.start_offset:
                # Shift only the experiment start; retain the folded -> HOME
                # startup and mandatory HOME -> folded shutdown paths.
                duration=max(1.,1.875*abs(args.start_offset)/.10)
                execute(duration,lambda t:(*ramp(t,duration,home,sweep_start),'preposition'))
                execute(.5,lambda t:(sweep_start,np.zeros(6),'hold'))
                if np.max(np.abs(np.array([r['q'] for r in receiver.snapshot()])-sweep_start))>.05:
                    raise RuntimeError('Shifted start verification failed')
            rng = np.random.default_rng(args.seed)
            segment = 0
            for repeat in range(config['repeats']):
                for sid in rng.permutation(len(config['speeds'])):
                    speed = config['speeds'][sid]
                    for direction in (1,-1):
                        start, finish = (sweep_start,end) if direction == 1 else (end,sweep_start)
                        sweep = Sweep(start[args.joint-1],finish[args.joint-1],speed,args.ramp_time)
                        def reference(t):
                            q,qd = home.copy(),np.zeros(6)
                            q[args.joint-1],qd[args.joint-1],phase = sweep.evaluate(t)
                            return q,qd,phase
                        print(f'J{args.joint} repeat={repeat+1} speed={speed:.3f} direction={direction:+d}')
                        execute(sweep.duration,reference,segment=segment,repeat=repeat,speed_id=int(sid),direction=direction)
                        segment += 1
                        execute(.5,lambda t:(finish,np.zeros(6),'hold'))
                        journal.request_checkpoint()
            status,reason = 'complete','planned trajectories finished; quality not yet assessed'
    except KeyboardInterrupt:
        status = 'aborted'
        reason = 'user interrupt'
        print('Stopped; preserving raw data.')
    except Exception as exc:
        status = 'aborted'
        reason = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            journal.checkpoint(status,reason)
        finally:
            try: journal.close()
            finally: signal.signal(signal.SIGINT,old_handler)
        print(f'Raw run: {path} ({status})')
    return path


def main(argv=None):
    args = parser().parse_args(argv)
    config = configuration(args)
    print(json.dumps(dict(mode=args.mode,joint=args.joint,**config),ensure_ascii=False,indent=2))
    if not args.execute or args.dry_run:
        print('Preview only; no hardware opened. --execute starts motion. Exit: HOME → folded zero → verify → disable.')
        return
    if args.load_id == 'unrecorded': raise ValueError('Provide --load-id describing fixed payload/gripper/cables')
    run(args,config)


if __name__ == '__main__':
    main()
