"""Production-pipeline tests with raw events, not copied fitting formulas."""
import json
from pathlib import Path
import struct
import socket
import signal
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from friction_id.analysis import Gates, analyze, read_parameters, validate, compare_compensation
from friction_id.common import HOME, KP, KD, Sweep, ReferenceTrial, check_position
from friction_id.hardware import EFF_FLAG, FeedbackReceiver, MotorSession, decode_status, fold_before_disable
from friction_id.recording import COMMAND_DTYPE, FEEDBACK_DTYPE, Journal, load_run, recover
from gravity_friction_compensation import Feedforward, GravityFrictionCompensation
from run_phase2_pointmatch import configuration, parser
import run_phase2_pointmatch as acquisition


def make_run(path, run_id='training', stick_slip=False, start_time=10., mode='identify', calibrated=True):
    """A smooth plant with known friction; full 6-axis reception/command events."""
    frows,crows=[],[]
    dt=.02; clock=start_time; seq=0; seg=0
    for repeat in range(3):
        for sid,speed in enumerate([.05,.12,.20]):
            for direction in (1,-1):
                sweep=Sweep(0. if direction==1 else .55,.55 if direction==1 else 0.,speed)
                for t in np.arange(0,sweep.duration,dt):
                    pos,vel,phase=sweep.evaluate(t)
                    if stick_slip:
                        # Continuous periodic stall/burst with monotone position.
                        pos += .012*np.sin(2*np.pi*3*t)
                        vel += .012*2*np.pi*3*np.cos(2*np.pi*3*t)
                    seq+=1
                    for j in range(1,7):
                        q=pos if j==6 else HOME[j-1]
                        v=vel if j==6 else 0.
                        tau=.2*np.sign(v)+.4*v if j==6 else 0.
                        frows.append((clock,clock+.001,int((clock+1e9)*1e9),0.,seq,j,q,v,tau,35.,0,True,0,'00'*8))
                        crows.append((clock-.003,clock-.002,j,seq,seg,repeat,sid,direction,phase,
                                      pos if j==6 else HOME[j-1],sweep.evaluate(t)[1] if j==6 else 0.,KP[j-1],KD[j-1],0.,True))
                    clock+=dt
                clock+=.1;seg+=1
    calibration=dict(verified=calibrated,scale=[1.]*6,offset=[0.]*6,units='N*m' if calibrated else 'protocol_nm_unverified',calibration_id='synthetic-fixture')
    conditions=dict(model_hash='synthetic',load_id='fixture',calibration=calibration,kp=KP.tolist(),kd=KD.tolist())
    meta=dict(schema_version=3,run_id=run_id,run_status='complete',mode=mode,joint=6,home=HOME.tolist(),
              measurement_range=[.13,.42],torque_calibration=calibration,conditions=conditions)
    f=np.array(frows,dtype=FEEDBACK_DTYPE);c=np.array(crows,dtype=COMMAND_DTYPE)
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)
    return meta,f,c


def test_fixed_path_signed_sweep_boundaries():
    for start,end in [(0.,.55),(.55,0.)]:
        for speed in [.05,.1,.2]:
            sweep=Sweep(start,end,speed,1.,1.2)
            q,v,_=sweep.evaluate(sweep.duration)
            assert q==pytest.approx(end)
            assert v==pytest.approx(0.)
            for boundary in [sweep.ramp_time,sweep.ramp_time+sweep.constant_time]:
                left=sweep.evaluate(boundary-1e-7)
                right=sweep.evaluate(boundary+1e-7)
                assert abs(left[0]-right[0])<1e-6
                assert abs(left[1]-right[1])<1e-6


def test_real_pipeline_identifies_and_independently_validates(tmp_path):
    path=tmp_path/'training.npz';make_run(path)
    report,candidate=analyze(path,tmp_path/'fit',gravity=lambda q:np.zeros(6))
    assert report['eligible_for_validation'],report['segments']
    assert report['fits']['D']['fc']==pytest.approx(.2,abs=.002)
    assert report['fits']['D']['fv']==pytest.approx(.4,abs=.01)
    assert candidate['status']=='candidate'
    with pytest.raises(ValueError,match='independent'):
        validate(tmp_path/'fit/candidate.json',path,tmp_path/'bad',lambda q:np.zeros(6))
    heldout=tmp_path/'heldout.npz';make_run(heldout,'heldout',start_time=500.)
    validated=validate(tmp_path/'fit/candidate.json',heldout,tmp_path/'validated',lambda q:np.zeros(6))
    p=read_parameters(validated,'validated')
    assert p['validation_rmse']<.002
    p['fc']=20.
    validated.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='integrity'):
        read_parameters(validated,'validated')


def test_stick_slip_is_rejected_before_fitting(tmp_path):
    path=tmp_path/'slip.npz';make_run(path,stick_slip=True)
    report,candidate=analyze(path,tmp_path/'slip')
    assert not report['eligible_for_validation']
    assert candidate['fc'] is None
    assert all(s['status']=='invalid' for s in report['segments'])


@pytest.mark.parametrize('fault',['gap','timestamp','missing_axis','stale_read','command_failure'])
def test_invalid_raw_data_cannot_become_deployable(tmp_path,fault):
    path=tmp_path/'bad.npz';meta,f,c=make_run(path)
    if fault=='gap':
        f=f[(f['seq']%20)<2]
    elif fault=='timestamp': f['timestamp_valid']=False
    elif fault=='missing_axis': f=f[f['joint']!=4]
    elif fault=='stale_read': f['read_t']=f['t']+.2
    else: c['ok']=False
    np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)
    try:
        report,candidate=analyze(path,tmp_path/'out')
    except ValueError as exc:
        assert 'Insufficient' in str(exc)
    else:
        assert not report['eligible_for_validation']
        assert candidate['status']=='candidate'


def test_unknown_calibration_and_tracking_are_not_eligible(tmp_path):
    path=tmp_path/'tracking.npz';make_run(path,mode='tracking',calibrated=False)
    report,_=analyze(path,tmp_path/'out')
    assert report['fits']['D']['status']=='valid'
    assert not report['eligible_for_validation']
    assert report['fits']['C']['status']=='invalid'


def test_legacy_summary_is_explicitly_rejected(tmp_path):
    p=tmp_path/'legacy.npz';np.savez(p,joint6_fc_E=.2)
    with pytest.raises(ValueError,match='Legacy summary'):
        load_run(p)


def test_journal_recovers_partial_final_line(tmp_path):
    path=tmp_path/'run.npz'
    j=Journal(path,dict(run_id='interrupted'))
    j.append('annotation',dict(kind='stage',value='startup'))
    j.checkpoint('aborted','keyboard interrupt');j.close()
    jp=path.with_suffix('.jsonl')
    with jp.open('a') as f:f.write('{"kind":')
    recover(jp,tmp_path/'recovered.npz')
    meta,fb,cmd=load_run(tmp_path/'recovered.npz')
    assert meta['run_status']=='recovered'
    assert len(fb)==len(cmd)==0


def test_slow_disk_never_blocks_event_submission_or_starts_live_npz(tmp_path,monkeypatch):
    import friction_id.recording as recording
    blocked=threading.Event();release=threading.Event();submitted=threading.Event()
    fsync_threads=[];exports=[];errors=[]
    actual_write_npz=recording.write_npz
    def slow_sync(fd):
        fsync_threads.append(threading.get_ident())
        blocked.set()
        if not release.wait(3.):raise TimeoutError('test did not release fake disk')
    monkeypatch.setattr(recording.os,'fsync',slow_sync)
    monkeypatch.setattr(recording,'write_npz',lambda *args:(exports.append(True),actual_write_npz(*args)))
    j=Journal(tmp_path/'slow_disk.npz',dict(run_id='slow-disk'))
    j._enqueue(('flush',threading.Event()))
    producer=None
    try:
        assert blocked.wait(1.)
        def submit():
            try:
                for i in range(1000):j.append('annotation',dict(kind='motion_sample',i=i))
                j.request_checkpoint()
            except BaseException as exc:errors.append(exc)
            finally:submitted.set()
        producer=threading.Thread(target=submit);producer.start()
        assert submitted.wait(.5),'control producer waited for blocked fsync'
        assert not errors and not exports
        assert set(fsync_threads)=={j.worker.ident}
    finally:
        release.set()
        if producer:producer.join(timeout=2.)
        try:j.checkpoint('complete','offline slow-disk test')
        finally:j.close()
    assert exports==[True]
    recovered=tmp_path/'recovered.npz'
    recover(j.path.with_suffix('.jsonl'),recovered)
    with np.load(recovered,allow_pickle=False) as z:
        annotations=[json.loads(str(x)) for x in z['annotations_json']]
    assert [x['i'] for x in annotations if x['kind']=='motion_sample']==list(range(1000))


def test_journal_writer_failure_is_reported_without_waiting_forever(tmp_path,monkeypatch):
    import friction_id.recording as recording
    failed=threading.Event()
    def fail_sync(fd):
        failed.set();raise OSError('disk failed')
    monkeypatch.setattr(recording.os,'fsync',fail_sync)
    j=Journal(tmp_path/'disk_failed.npz',dict(run_id='disk-failed'))
    with pytest.raises(RuntimeError,match='disk failed'):j.checkpoint('aborted','test')
    assert failed.is_set()
    with pytest.raises(RuntimeError,match='disk failed'):j.close()


def test_journal_queue_overflow_is_explicit_and_never_silently_drops(tmp_path,monkeypatch):
    import friction_id.recording as recording
    blocked=threading.Event();release=threading.Event()
    def slow_sync(fd):
        blocked.set()
        if not release.wait(3.):raise TimeoutError('test fake disk was not released')
    monkeypatch.setattr(recording.os,'fsync',slow_sync)
    j=Journal(tmp_path/'full_queue.npz',dict(run_id='full-queue'),queue_capacity=2)
    j._enqueue(('flush',threading.Event()))
    try:
        assert blocked.wait(1.)
        j.append('annotation',dict(kind='queued',i=0))
        j.append('annotation',dict(kind='queued',i=1))
        with pytest.raises(RuntimeError,match='queue full'):
            j.append('annotation',dict(kind='unqueued',i=2))
        assert isinstance(j.error,BufferError)
    finally:
        release.set()
        with pytest.raises(RuntimeError,match='queue full'):j.close()
    lines=[json.loads(s) for s in j.path.with_suffix('.jsonl').read_text().splitlines()]
    assert [s['value']['i'] for s in lines if s['value'].get('kind')=='queued']==[0,1]
    assert not any(s['value'].get('kind')=='unqueued' for s in lines)


def test_rs_decoder_rejects_nonstatus_and_version_replies():
    arb=EFF_FLAG|(2<<24)|(6<<8)|0xFD
    row=decode_status(arb,struct.pack('>4H',32767,32767,32767,350))
    assert row['joint']==6 and row['tau']==0 and row['temperature']==35.
    assert decode_status(arb,b'\x00\xc4\x56'+b'\x00'*5) is None
    assert decode_status(arb&~EFF_FLAG,b'\x00'*8) is None
    active=0x980006fd
    assert decode_status(active,struct.pack('>4H',32767,32767,32767,350))['joint']==6
    assert decode_status(EFF_FLAG|(21<<24)|(6<<8)|0xfd,b'\x00'*8) is None


def test_receiver_filters_accept_status_without_selecting_can_error_filters(monkeypatch):
    filters=[]
    receiver=FeedbackReceiver()
    class Socket:
        def setsockopt(self,level,option,value):
            if level==socket.SOL_CAN_RAW and option==socket.CAN_RAW_FILTER:filters.append(value)
        def settimeout(self,*a):pass
        def bind(self,*a):pass
        def close(self):pass
    monkeypatch.setattr('friction_id.hardware.socket.socket',lambda *a:Socket())
    monkeypatch.setattr(receiver,'_run',lambda:None)
    receiver.start();receiver.stop()
    rows=list(struct.iter_unpack('=II',filters[0]))
    assert len(rows)==2
    for frame_type in (2,24):
        frame=EFF_FLAG|(frame_type<<24)|(6<<8)|0xfd
        assert any((frame&mask)==(ident&mask) for ident,mask in rows)
    assert all(mask&0x20000000==0 for _,mask in rows)


def test_snapshot_requires_new_enough_coherent_fault_free_axes(monkeypatch):
    receiver=FeedbackReceiver()
    monkeypatch.setattr('friction_id.hardware.time.monotonic',lambda:10.)
    receiver.latest={j:dict(t=9.99,q=HOME[j-1],status=0,temperature=35,timestamp_valid=True) for j in range(1,7)}
    assert len(receiver.snapshot())==6
    receiver.latest[4]['t']=9.5
    with pytest.raises(RuntimeError,match='Stale'):
        receiver.snapshot()


def test_partial_enable_failure_verifies_fold_before_disabling(tmp_path,monkeypatch):
    calls=[]
    clock=[0.]
    monkeypatch.setattr('friction_id.hardware.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('friction_id.hardware.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    class Motor:
        def __init__(self,j):self.j=j
        def robstride_get_param_f32(self,p):return 0.
        def clear_error(self):pass
        def ensure_mode(self,*a):pass
        def robstride_set_active_report(self,*a):pass
        def send_mit(self,*a):pass
        def enable(self):
            calls.append(('enable',self.j))
            if self.j==3:raise RuntimeError('enable failed')
        def disable(self):calls.append(('disable',self.j))
    class Controller:
        def __init__(self,*a):pass
        def add_robstride_motor(self,j,*a):return Motor(j)
        def close(self):calls.append(('close',0))
        def close_bus(self):calls.append(('close_bus',0))
    monkeypatch.setitem(sys.modules,'motorbridge',SimpleNamespace(Controller=Controller,Mode=SimpleNamespace(MIT=1)))
    receiver=SimpleNamespace(start=lambda:None,stop=lambda:None,snapshot=lambda:[dict(q=0.,tau=0.,qd=0.) for _ in range(6)])
    journal=SimpleNamespace(append=lambda kind,row:calls.append((row.get('kind',kind),0)))
    with pytest.raises(RuntimeError,match='enable failed'):
        with MotorSession('unused',receiver,journal,lambda q:np.zeros(6),KP,KD) as session:
            session.enable()
    assert {j for action,j in calls if action=='disable'}==set(range(1,7))
    assert calls.index(('fold_verified',0))<calls.index(('disable',1))
    assert ('close',0) in calls


def test_fold_trajectory_passes_home_and_settles_at_zero(monkeypatch):
    clock=[10.];actual=HOME.copy();actual[5]=.4
    commands=[];events=[]
    monkeypatch.setattr('friction_id.hardware.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('friction_id.hardware.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    receiver=SimpleNamespace(snapshot=lambda:[dict(q=q,qd=0.) for q in actual])
    def send(q,qd,phase):
        commands.append((clock[0],q.copy(),qd.copy(),phase));actual[:]=q
    fold_before_disable(receiver,send,SimpleNamespace(append=lambda k,r:events.append(r)))
    assert set(c[3] for c in commands)=={'exit_home','exit_fold'}
    assert max(np.max(np.abs(c[2])) for c in commands)<=.3+1e-9
    assert np.allclose(commands[-1][1],0)
    assert events[-1]['kind']=='fold_verified'
    assert commands[-1][0]-next(c[0] for c in commands if c[3]=='exit_fold' and np.max(np.abs(c[1]))<1e-8)>=.49


def test_fold_still_blocks_disable_after_actual_control_stall(monkeypatch):
    clock=[10.];commands=[];events=[]
    monkeypatch.setattr('friction_id.hardware.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('friction_id.hardware.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    receiver=SimpleNamespace(snapshot=lambda:[dict(q=q,qd=0.) for q in HOME])
    def slow_send(q,qd,phase):
        commands.append(phase);clock[0]+=.075
    with pytest.raises(RuntimeError,match=r'deadline missed by 65.0 ms'):
        fold_before_disable(receiver,slow_send,SimpleNamespace(append=lambda k,r:events.append(r)))
    assert commands==['exit_home']
    assert events[-1]['kind']=='fold_deadline_missed'
    assert events[-1]['lateness_s']==pytest.approx(.065)
    assert not any(e['kind']=='fold_verified' for e in events)


@pytest.mark.parametrize('owner',['acquisition','compensation'])
def test_failed_fold_never_disables_or_calls_shutdown(owner,monkeypatch):
    calls=[]
    def no_feedback():raise RuntimeError('feedback lost')
    receiver=SimpleNamespace(snapshot=no_feedback,stop=lambda:calls.append('receiver_stop'))
    motor=SimpleNamespace(disable=lambda:calls.append('disable'))
    controller=SimpleNamespace(close_bus=lambda:calls.append('close_bus'),close=lambda:calls.append('close'),
                               shutdown=lambda:calls.append('shutdown'))
    if owner=='acquisition':
        journal=SimpleNamespace(append=lambda *args:None)
        ctrl=MotorSession('unused',receiver,journal,lambda q:np.zeros(6),KP,KD)
        ctrl.ctrl=controller;ctrl.motors=[motor]*6;ctrl.enabled=True
    else:
        robot=SimpleNamespace(_motor_map={'joint1':motor},_ctrl_map={'robstride':controller},
            stop_control_loop=lambda:None,disconnect=lambda:calls.append('disconnect'))
        ctrl=GravityFrictionCompensation(robot,feedback_source=receiver)
        ctrl._connected=True;ctrl._enabled=True
    with pytest.raises(RuntimeError,match='feedback lost'):ctrl.close()
    assert 'disable' not in calls and 'shutdown' not in calls and 'disconnect' not in calls
    assert calls==['close_bus','close','receiver_stop']
    ctrl.close()  # idempotent, no late implicit disable
    assert calls==['close_bus','close','receiver_stop']


def test_compensation_exit_always_folds_even_when_legacy_flag_false(monkeypatch):
    calls=[];clock=[1.];actual=HOME.copy()
    monkeypatch.setattr('friction_id.hardware.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('friction_id.hardware.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    class Motor:
        def __init__(self,j):self.j=j
        def send_mit(self,q,*args):actual[self.j]=q
        def disable(self):
            assert np.max(np.abs(actual))<=.02
            calls.append('disable')
    transport=SimpleNamespace(close_bus=lambda:calls.append('close_bus'),close=lambda:calls.append('close'))
    robot=SimpleNamespace(_motor_map={f'joint{j+1}':Motor(j) for j in range(6)},
        _ctrl_map={'rs':transport},arm=SimpleNamespace(joint_names=[f'joint{j+1}' for j in range(6)]),
        stop_control_loop=lambda:None)
    receiver=SimpleNamespace(snapshot=lambda:[dict(q=q,qd=.21,tau=0.) for q in actual],stop=lambda:None)
    ctrl=GravityFrictionCompensation(robot,feedback_source=receiver)
    ctrl._connected=True;ctrl._enabled=True;ctrl.hold_kp=KP;ctrl.hold_kd=KD
    ctrl.ff=SimpleNamespace(components=lambda *a:(np.zeros(6),np.zeros(6)))
    ctrl.close(return_home=False)
    assert calls==['disable']*6+['close_bus','close']
    assert np.allclose(actual,0.)
    ctrl.close()
    assert calls==['disable']*6+['close_bus','close']


def test_feedforward_uses_actual_kd_and_validated_velocity_envelope():
    ff=Feedforward(lambda q:np.ones(6),np.ones(6)*.2,np.ones(6)*20,np.tile([.05,.2],(6,1)),
                   np.ones(6),np.ones(6),np.ones(6),np.zeros(6))
    g,f=ff.components(np.zeros(6),np.ones(6)*.1,np.ones(6)*2.5)
    assert np.allclose(g,1.)
    assert np.all(f<=.4)  # Fc .2 + enabled Fv <= .8*2.5, times velocity .1
    _,outside=ff.components(np.zeros(6),np.ones(6)*.3,np.ones(6)*2.5)
    assert np.all(outside==0)


def test_speed_filter_does_not_reprocess_cached_state():
    ctrl=GravityFrictionCompensation(SimpleNamespace())
    ctrl._seq_prev=np.ones(6);ctrl._q_prev=np.zeros(6);ctrl._t_prev=np.ones(6);ctrl._qd_filt=np.zeros(6)
    cached=[dict(seq=1,q=.1,t=1.) for _ in range(6)]
    ctrl._update_velocity(cached)
    assert np.all(ctrl._qd_filt==0)
    fresh=[dict(seq=2,q=.001,t=1.01) for _ in range(6)]
    ctrl._update_velocity(fresh);old=ctrl._qd_filt.copy()
    ctrl._update_velocity(fresh)
    assert np.array_equal(old,ctrl._qd_filt)


def test_preview_has_no_hardware_import_or_open():
    config=configuration(parser().parse_args(['--joint','5','--dry-run']))
    assert config['kp'][4]==12
    with pytest.raises(ValueError):configuration(parser().parse_args(['--excursion','0.01']))


@pytest.mark.parametrize('joint,kp,kd',[(4,50,5),(5,50,4),(6,50,4)])
def test_original_driver_wrist_comparison_is_explicit_and_bounded(joint,kp,kd):
    argv=['--joint',str(joint),'--kp',str(kp),'--kd',str(kd)]
    with pytest.raises(ValueError):configuration(parser().parse_args(argv))
    cfg=configuration(parser().parse_args(argv+['--gain-policy','driver-reference']))
    other=np.arange(6)!=joint-1
    assert np.array_equal(np.asarray(cfg['kp'])[other],KP[other])
    assert np.array_equal(np.asarray(cfg['kd'])[other],KD[other])
    assert cfg['kp'][joint-1]==kp and cfg['kd'][joint-1]==kd
    with pytest.raises(ValueError):
        configuration(parser().parse_args(argv+['--gain-policy','driver-reference','--kd',str(kd+.1)]))


@pytest.mark.parametrize('argv',[
    ['--joint','3','--gain-policy','driver-reference'],
    ['--mode','identify','--joint','6','--gain-policy','driver-reference'],
])
def test_driver_reference_policy_cannot_extend_other_experiments(argv):
    with pytest.raises(ValueError):configuration(parser().parse_args(argv))


@pytest.mark.parametrize('phase',['initialization','startup_hold'])
def test_interrupt_during_startup_hold_preserves_aborted_log(tmp_path,monkeypatch,phase):
    import friction_id.hardware as hardware
    exited=[]
    class Receiver:
        def __init__(self,*a):pass
        def snapshot(self):return [dict(q=q) for q in HOME]
    class Session:
        def __init__(self,*a):pass
        def __enter__(self):
            if phase=='initialization':signal.getsignal(signal.SIGINT)(None,None)
            return self
        def __exit__(self,*a):exited.append(True)
        def enable(self):assert phase!='initialization'
        def send(self,*a,**k):raise KeyboardInterrupt
    monkeypatch.setattr(hardware,'FeedbackReceiver',Receiver)
    monkeypatch.setattr(hardware,'MotorSession',Session)
    monkeypatch.setattr(acquisition,'Gravity',lambda:SimpleNamespace(model_hash='synthetic'))
    monkeypatch.setattr(acquisition.importlib.metadata,'version',lambda name:'synthetic-test-sdk')
    args=parser().parse_args(['--output',str(tmp_path),'--load-id','fixture'])
    path=acquisition.run(args,configuration(args))
    meta,_,_=load_run(path)
    assert meta['run_status']=='aborted' and meta['exit_reason']=='user interrupt'
    assert exited


def test_repeated_numeric_values_are_distinct_received_frames(monkeypatch):
    from friction_id.hardware import SO_TIMESTAMPNS
    import friction_id.hardware as hardware
    receiver=FeedbackReceiver()
    captured=[]
    receiver.journal=SimpleNamespace(append=lambda kind,row:captured.append(row))
    class Socket:
        n=0
        def recvmsg(self,*a):
            self.n+=1
            if self.n==3:
                receiver.stop_event.set();raise socket.timeout
            payload=struct.pack('>4H',32767,32767,32767,350)
            frame=struct.pack('=IB3x8s',EFF_FLAG|(2<<24)|(6<<8)|0xFD,8,payload)
            stamp=struct.pack('@ll',1000,self.n*10_000_000)
            return frame,[(socket.SOL_SOCKET,SO_TIMESTAMPNS,stamp)],0,None
    sock=Socket();receiver.sock=sock
    monkeypatch.setattr(hardware.time,'monotonic_ns',lambda:10_000_000_000+sock.n*10_000_000)
    monkeypatch.setattr(hardware.time,'time_ns',lambda:1_000_000_000_000+sock.n*10_000_000)
    receiver._run()
    assert receiver.error is None
    assert [r['seq'] for r in captured]==[1,2]
    assert captured[0]['q']==captured[1]['q']
    assert captured[0]['t']<captured[1]['t']


def test_callback_fault_is_visible_to_main_thread():
    ctrl=GravityFrictionCompensation(SimpleNamespace())
    ctrl._loop_t=0.
    with pytest.raises(RuntimeError,match='timing'):
        ctrl._loop_cb(None, .002)
    assert ctrl.failure is not None


def test_home_timeout_does_not_send_terminal_jump():
    receiver=SimpleNamespace(snapshot=lambda:[dict(q=q) for q in HOME+np.array([.3,0,0,0,0,0])])
    ctrl=GravityFrictionCompensation(SimpleNamespace(),feedback_source=receiver)
    with pytest.raises(TimeoutError):
        ctrl.safe_home(max_vel=.1,timeout=.01)
    assert np.all(ctrl.last_tau==0.)


def test_torque_slew_and_amplitude_are_bounded():
    ctrl=GravityFrictionCompensation(SimpleNamespace(),max_torque=1.,max_torque_rate=2.)
    tau=ctrl._bounded_torque(np.ones(6)*10,np.zeros(6),.1)
    assert np.allclose(tau,.2)
    assert ctrl.last_components['limited']
    for _ in range(20):tau=ctrl._bounded_torque(np.ones(6)*10,np.zeros(6),.1)
    assert np.allclose(tau,1.)


def test_comparison_trajectory_finishes_at_home():
    trial=ReferenceTrial(6,.1,.55,2)
    q,qd,done=trial(0.,HOME.copy())
    assert np.allclose(q,HOME) and not done
    for boundary in [3.,4.,4.+trial.out.duration,4.+trial.out.duration+.5]:
        left=trial(boundary-1e-7,HOME.copy())
        right=trial(boundary+1e-7,HOME.copy())
        assert np.max(np.abs(left[0]-right[0]))<1e-6
        assert np.max(np.abs(left[1]-right[1]))<1e-6
    q,qd,done=trial(1000.,HOME.copy())
    assert done and np.allclose(q,HOME) and np.allclose(qd,0.)


def test_compensation_comparison_requires_matching_motion(tmp_path):
    paths=[]
    for i in range(2):
        path=tmp_path/f'comp{i}.npz'
        meta,f,c=make_run(path,run_id=f'comp{i}',start_time=10.+i*500.)
        meta.update(mode='compensation',motion=dict(joint=6,speed=.1),rate=100.,gravity_scale=[1.]*6)
        np.savez_compressed(path,metadata_json=json.dumps(meta),feedback=f,commands=c)
        paths.append(path)
    report=compare_compensation(*paths,tmp_path/'comparison.json')
    assert report['status']=='descriptive_comparison_only'
    assert report['trial']['axes'][5]['position_rmse_rad']<1e-9
    meta['motion']['speed']=.2
    np.savez_compressed(paths[1],metadata_json=json.dumps(meta),feedback=f,commands=c)
    with pytest.raises(ValueError,match='motion'):
        compare_compensation(*paths,tmp_path/'different.json')


def test_folded_start_tolerance_is_explicit_and_bounded():
    q=HOME.copy();q[1]=-.00886
    with pytest.raises(RuntimeError):check_position(q)
    assert np.array_equal(check_position(q,[0,.02,.02,0,0,0]),q)
    q[1]=-.021
    with pytest.raises(RuntimeError):check_position(q,[0,.02,.02,0,0,0])
    q[1]=3.15
    with pytest.raises(RuntimeError):check_position(q,[0,.02,.02,0,0,0])
