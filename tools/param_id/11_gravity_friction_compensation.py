#!/usr/bin/env python3
"""Stage 4: bounded gravity/friction validation. Preview unless --execute is set."""
import argparse
import signal
import time
from datetime import datetime
import uuid
from pathlib import Path
from friction_id.analysis import read_parameters
from friction_id.common import control_module, ReferenceTrial
from gravity_friction_compensation import GravityFrictionCompensation
from friction_id.recording import Journal
from run_phase2_pointmatch import provenance


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parameters',type=Path,nargs='*',default=[])
    ap.add_argument('--load-id',default=None)
    ap.add_argument('--friction-scale',type=float,default=0.,help='Start with 0, then single-axis low fractions')
    ap.add_argument('--return-home',action='store_true',help='Compatibility flag; exit always returns through HOME to folded zero before disabling')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'realrobot_data')
    ap.add_argument('--tracking-joint',type=int,choices=range(1,7),help='Repeatable single-axis comparison instead of compliant hold')
    ap.add_argument('--speed',type=float,default=.10)
    ap.add_argument('--excursion',type=float,default=.55)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--rate',type=float,default=100.)
    args=ap.parse_args(argv)
    if not 0<=args.friction_scale<=1: ap.error('friction-scale must be within [0,1]')
    if not 50<=args.rate<=200: ap.error('rate must be within [50,200] Hz')
    for path in args.parameters: read_parameters(path,'validated')
    if args.friction_scale and not args.parameters: ap.error('Friction requires validated parameter files')
    trial=ReferenceTrial(args.tracking_joint,args.speed,args.excursion,args.repeats) if args.tracking_joint else None
    if not args.execute:
        print('Preview: gravity_scale=[1,.98,.98,1,1,1], kp=15, kd=2.5, friction_scale=',args.friction_scale)
        print('No hardware opened. --execute starts control. Exit: HOME → folded zero → verify → disable.')
        return
    if args.load_id is None: ap.error('--execute requires a reproducible --load-id')
    robot=control_module('actuator').RebotArm()
    ctrl=GravityFrictionCompensation(robot,kp=15.,kd=2.5,parameter_files=args.parameters,
        load_id=args.load_id,friction_scale=args.friction_scale,gravity_scale=[1,.98,.98,1,1,1],log_every=100,reference_source=trial,rate=args.rate)
    path=args.output/('compensation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.npz')
    journal=Journal(path,dict(run_id=str(uuid.uuid4()),mode='compensation',load_id=args.load_id,
        parameter_files=[str(p.resolve()) for p in args.parameters],parameters=[read_parameters(p,'validated') for p in args.parameters],
        motion=dict(joint=args.tracking_joint,speed=args.speed,excursion=args.excursion,repeats=args.repeats),
        provenance=provenance(),rate=args.rate,
        friction_scale=args.friction_scale,gravity_scale=[1,.98,.98,1,1,1],kp=15.,kd=2.5))
    ctrl.journal=journal
    running=True
    def stop(*_):
        nonlocal running
        running=False
    old=signal.signal(signal.SIGINT,stop)
    ctrl.stop_requested=lambda:not running
    status,reason='aborted','startup incomplete'
    try:
        ctrl.start()
        while running:
            if ctrl.failure: raise RuntimeError(f'Controller failed: {ctrl.failure}') from ctrl.failure
            if ctrl.finished: break
            time.sleep(.05)
        status,reason='complete','comparison trajectory finished' if ctrl.finished else 'user requested stop'
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            ctrl.close()
        except BaseException as exc:
            status,reason='aborted',f'cleanup: {type(exc).__name__}: {exc}'
            raise
        finally:
            try: journal.checkpoint(status,reason)
            finally:
                try: journal.close()
                finally: signal.signal(signal.SIGINT,old)
                print(f'Runtime log: {path}')


if __name__=='__main__':
    main()
