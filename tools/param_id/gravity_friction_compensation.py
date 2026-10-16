"""RS compliant hold with validated friction parameters and bounded feedforward.

A receive-only SocketCAN observer supplies fresh state. Parameter files must
pass independent validation; no historical Method E constants are embedded.
"""
from __future__ import annotations

import time
import numpy as np

from friction_id.analysis import read_parameters
from friction_id.common import EFFORT, HOME, ZERO_TOLERANCE, Gravity, check_position, ramp, vector
from friction_id.hardware import FeedbackReceiver, fold_before_disable


class Feedforward:
    """Pure torque computation; state filtering and command limits live outside."""
    def __init__(self, gravity, fc, fv, speed_ranges, gravity_scale, friction_scale,
                 direction, offset, smoothing=.02):
        self.gravity=gravity
        self.fc,self.fv=np.asarray(fc),np.asarray(fv)
        self.ranges=np.asarray(speed_ranges)
        self.gravity_scale,self.friction_scale=gravity_scale,friction_scale
        self.direction,self.offset=direction,offset
        self.smoothing=smoothing

    def components(self,q,qd,kd):
        g=self.direction*self.gravity(self.direction*(q-self.offset))*self.gravity_scale
        # Smooth activation only inside the independently validated speed envelope.
        speed=np.abs(qd)
        width=np.minimum(.02,np.maximum((self.ranges[:,1]-self.ranges[:,0])/4,1e-6))
        envelope=np.clip((speed-self.ranges[:,0])/width,0,1)*np.clip((self.ranges[:,1]-speed)/width,0,1)
        envelope=envelope*envelope*(3-2*envelope)
        scaled_fc=self.fc*self.friction_scale
        # Limit the ENABLED viscous gain against the kd actually sent this cycle.
        scaled_fv=np.minimum(self.fv*self.friction_scale,.8*kd)
        f=(scaled_fc*np.tanh(qd/self.smoothing)+scaled_fv*qd)*envelope
        return g,f


class GravityFrictionCompensation:
    def __init__(self,rebotarm,*,kp=15.,kd=2.5,parameter_files=(),load_id=None,
                 gravity_scale=1.,friction_scale=0.,tau_scale=None,
                 transition_duration=1.,joint_direction=1.,joint_offset=0.,
                 hold_kp=None,hold_kd=None,enabled_joints=None,log_every=0,
                 max_torque=None,max_torque_rate=20.,velocity_time_constant=.10,
                 feedback_source=None,gravity=None,reference_source=None,rate=100.,stop_requested=None,fc=0.,fv=0.):
        if np.any(np.asarray(fc)!=0) or np.any(np.asarray(fv)!=0):
            raise ValueError('Raw Fc/Fv are not deployable: use independently validated parameter_files')
        self.rebotarm=rebotarm
        self.kp_raw,self.kd_raw,self.hold_kp_raw,self.hold_kd_raw=kp,kd,hold_kp,hold_kd
        self.gravity_scale_raw=gravity_scale if tau_scale is None else tau_scale
        self.friction_scale_raw=friction_scale
        self.direction_raw,self.offset_raw=joint_direction,joint_offset
        self.parameter_files,self.load_id=list(parameter_files),load_id
        self.enabled_joints=list(enabled_joints or [])
        self.transition_duration=float(transition_duration)
        self.max_torque=vector(EFFORT*.5 if max_torque is None else max_torque,'max_torque',minimum=0)
        self.max_torque_rate=vector(max_torque_rate,'max_torque_rate',minimum=0)
        self.velocity_time_constant=float(velocity_time_constant)
        if not np.isfinite([self.transition_duration,self.velocity_time_constant]).all() or min(self.transition_duration,self.velocity_time_constant)<=0:
            raise ValueError('Transition and velocity time constants must be positive')
        self.receiver=feedback_source
        self.gravity=gravity
        self.log_every=log_every
        self.failure=None
        self._running=False
        self._connected=False
        self._enabled=False
        self._closed=False
        self._returning=False
        self._startup=True
        self._last_tau_total=np.zeros(6)
        self.last_components={}
        self._gripper=[]
        self.journal=None
        self._command_id=0
        self.reference_source=reference_source
        self.finished=False
        self.stop_requested=stop_requested or (lambda:False)
        self.rate=float(rate)
        if not 50<=self.rate<=200: raise ValueError('Control rate must be within [50,200] Hz')

    @property
    def last_tau(self):
        return self._last_tau_total.copy()

    def _resolve_params(self):
        robot=self.rebotarm
        if robot.arm.joint_names != [f'joint{i}' for i in range(1,7)]:
            raise ValueError('Controller supports the six-axis RS joint order only')
        self.kp=vector(self.kp_raw,'kp',minimum=0)
        self.kd=vector(self.kd_raw,'kd',minimum=0)
        self.hold_kp=vector(robot.arm._mit_kp if self.hold_kp_raw is None else self.hold_kp_raw,'hold_kp',minimum=0)
        self.hold_kd=vector(robot.arm._mit_kd if self.hold_kd_raw is None else self.hold_kd_raw,'hold_kd',minimum=0)
        self.direction=vector(self.direction_raw,'direction')
        if not np.all(np.abs(self.direction)==1): raise ValueError('Direction must be +/-1')
        self.offset=vector(self.offset_raw,'offset')
        self.gravity=self.gravity or Gravity()
        fc,fv=np.zeros(6),np.zeros(6)
        ranges=np.zeros((6,2))
        self.temperature_ranges=np.tile([0.,65.],(6,1))
        used=set()
        for path in self.parameter_files:
            p=read_parameters(path,'validated');j=p['joint']-1
            if j not in range(6) or j in used: raise ValueError('Duplicate/invalid parameter axis')
            if p['method']!='D' or p['units']!='N*m' or not p.get('eligible'):
                raise ValueError('Only calibrated, validated Method D parameters can be enabled')
            if not p.get('validation_run_ids') or not p.get('validation_sha256'):
                raise ValueError('Independent validation provenance missing')
            if p['conditions']['model_hash']!=self.gravity.model_hash or p['conditions']['load_id']!=self.load_id:
                raise ValueError('Parameter model/load differs from runtime configuration')
            if np.any(self.direction!=1) or np.any(self.offset!=0):
                raise ValueError('These parameters were validated in direct drive coordinates')
            fc[j],fv[j]=p['fc'],p['fv'];ranges[j]=p['speed_range'];used.add(j)
            self.temperature_ranges[j]=p['temperature_range']
            if not np.isfinite(ranges[j]).all() or not 0 < ranges[j,0] < ranges[j,1]:
                raise ValueError('Invalid validated speed range')
            if not np.isfinite(self.temperature_ranges[j]).all(): raise ValueError('Invalid temperature range')
        vector(fc,'fc',minimum=0);vector(fv,'fv',minimum=0)
        scale=vector(self.friction_scale_raw,'friction_scale',minimum=0)
        if np.any(scale>1): raise ValueError('friction_scale must be within [0,1]')
        if np.any(scale) and not used: raise ValueError('Friction requires validated parameter files')
        self.ff=Feedforward(self.gravity,fc,fv,ranges,vector(self.gravity_scale_raw,'gravity_scale',minimum=0),
                            scale,self.direction,self.offset)
        if self.journal:
            self.journal.metadata['conditions']=dict(model_hash=self.gravity.model_hash,load_id=self.load_id,
                kp=self.kp.tolist(),kd=self.kd.tolist(),direction=self.direction.tolist(),offset=self.offset.tolist())
            self.journal.append('annotation',dict(kind='metadata',value=self.journal.metadata.copy()))

    def _snapshot(self):
        return self.receiver.snapshot()

    def _send(self,q,qd,kp,kd,tau):
        q=check_position(q,ZERO_TOLERANCE if self._startup or self._returning else 0.)
        if not all(np.isfinite(x).all() for x in (qd,kp,kd,tau)):
            raise RuntimeError('Nonfinite MIT command')
        for i,name in enumerate(self.rebotarm.arm.joint_names):
            started=time.monotonic();ok=False
            try:
                self.rebotarm._motor_map[name].send_mit(float(q[i]),float(qd[i]),float(kp[i]),float(kd[i]),float(tau[i]))
                ok=True
            finally:
                if self.journal:
                    self.journal.append('command',dict(tx_start=started,tx_end=time.monotonic(),joint=i+1,
                        command_id=self._command_id,segment=-1,repeat=-1,speed_id=-1,direction=0,
                        phase=getattr(self,'_phase','compensation'),
                        q_ref=float(q[i]),qd_ref=float(qd[i]),kp=float(kp[i]),kd=float(kd[i]),tau_ff=float(tau[i]),ok=ok))
        self._command_id+=1
        for motor,pos,kp_g,kd_g in self._gripper:
            motor.send_mit(pos,0.,kp_g,kd_g,0.)

    def start(self):
        if self._running: raise RuntimeError('Already running')
        if self._closed: raise RuntimeError('Closed controller cannot restart; create a new instance')
        robot=self.rebotarm
        try:
            robot.connect();self._connected=True
            self._resolve_params()
            self.receiver=self.receiver or FeedbackReceiver(robot._channel,self.journal)
            self.receiver.start()
            names=robot.arm.joint_names
            q=check_position([robot._motor_map[name].robstride_get_param_f32(0x7019) for name in names],ZERO_TOLERANCE)
            if np.max(np.abs(q))>.05:
                raise RuntimeError('Start must be folded; refusing mode changes outside folded pose')
            self.receiver.lower_position_tolerance=ZERO_TOLERANCE.copy()
            if not robot.arm.mode_mit(): raise RuntimeError('Arm MIT mode initialization failed')
            if robot.has_gripper:
                if not robot.gripper.mode_mit(): raise RuntimeError('Gripper MIT mode initialization failed')
                for i,name in enumerate(robot.gripper.joint_names):
                    m=robot._motor_map[name]
                    p=float(m.robstride_get_param_f32(0x7019))
                    if not np.isfinite(p): raise RuntimeError('Invalid gripper position')
                    self._gripper.append((m,p,float(robot.gripper._mit_kp[i]),float(robot.gripper._mit_kd[i])))
            for name in names: robot._motor_map[name].robstride_set_active_report(True)
            deadline=time.monotonic()+2.
            while True:
                try: rows=self._snapshot();break
                except RuntimeError:
                    if time.monotonic()>deadline: raise
                    time.sleep(.01)
            allowed=set(robot.joint_names)
            if any(n not in allowed for n in self.enabled_joints): raise ValueError('Unknown enabled joint')
            if self.stop_requested(): raise KeyboardInterrupt
            q=np.array([r['q'] for r in rows])
            self._send(q,np.zeros(6),self.hold_kp,self.hold_kd,np.zeros(6))
            self._enabled=True  # partial enable failure also requires fold confirmation
            if self.enabled_joints:
                for name in self.enabled_joints: robot._motor_map[name].enable()
            else: robot.enable_all()
            rows=self._snapshot()
            self._q_prev=np.array([r['q'] for r in rows]);self._t_prev=np.array([r['t'] for r in rows])
            self._seq_prev=np.array([r['seq'] for r in rows]);self._qd_filt=np.zeros(6)
            self._q_frozen=q.copy();self._started=time.monotonic();self._loop_t=self._started
            self._initial_q=q.copy();self.finished=False
            self._last_tau_total=np.zeros(6);self.failure=None;self._counter=0;self._running=True
            robot.start_control_loop(self._loop_cb,rate=self.rate)
        except BaseException:
            self.close()
            raise

    def _update_velocity(self,rows):
        for j,row in enumerate(rows):
            if row['seq']==self._seq_prev[j]: continue
            dt=row['t']-self._t_prev[j]
            if not .0001<=dt<=.1: raise RuntimeError('Invalid fresh-feedback interval')
            v=(row['q']-self._q_prev[j])/dt
            alpha=1-np.exp(-dt/self.velocity_time_constant)
            self._qd_filt[j]+=alpha*(v-self._qd_filt[j])
            self._q_prev[j],self._t_prev[j],self._seq_prev[j]=row['q'],row['t'],row['seq']

    def _bounded_torque(self,g,f,dt,blend=1.):
        requested=(g+f)*blend
        target=np.clip(requested,-self.max_torque,self.max_torque)
        tau=self._last_tau_total+np.clip(target-self._last_tau_total,-self.max_torque_rate*dt,self.max_torque_rate*dt)
        self.last_components=dict(gravity=g.copy(),friction=f.copy(),requested=requested.copy(),sent=tau.copy(),
                                  limited=bool(np.any(np.abs(tau-requested)>1e-9)))
        self._last_tau_total=tau
        return tau

    def _loop_cb(self,robot,dt):
        try:
            now=time.monotonic();actual_dt=now-self._loop_t;self._loop_t=now
            if not 0<actual_dt<=.1: raise RuntimeError('Control loop timing invalid')
            rows=self._snapshot();q=np.array([r['q'] for r in rows])
            if np.any(np.abs([r['tau'] for r in rows])>EFFORT*.9): raise RuntimeError('Effort threshold exceeded')
            self._update_velocity(rows)
            ratio=np.clip((now-self._started)/self.transition_duration,0,1)
            blend=ratio*ratio*(3-2*ratio)
            kp=self.hold_kp+blend*(self.kp-self.hold_kp)
            kd=self.hold_kd+blend*(self.kd-self.hold_kd)
            qd_target=np.zeros(6)
            if self.reference_source:
                target,qd_target,self.finished=self.reference_source(now-self._started,self._initial_q)
                if np.max(np.abs(target-q))>.15 or np.max(np.abs(qd_target))>.6:
                    raise RuntimeError('Comparison trajectory tracking/velocity bound exceeded')
            else:
                if np.linalg.norm(q-self._q_frozen)>.005:
                    self._q_frozen+=(1-np.exp(-actual_dt/.13))*(q-self._q_frozen)
                target=self._q_frozen
            g,f=self.ff.components(q,qd_target if self.reference_source else self._qd_filt,kd)
            temps=np.array([r['temperature'] for r in rows])
            f=np.where((temps>=self.temperature_ranges[:,0]-5)&(temps<=self.temperature_ranges[:,1]+5),f,0.)
            tau=self._bounded_torque(g,f,actual_dt,blend)
            estimate=kp*(target-q)+kd*(qd_target-np.array([r['qd'] for r in rows]))+tau
            if np.any(np.abs(estimate)>EFFORT*.8): raise RuntimeError('Estimated MIT effort exceeds experiment bound')
            self._send(target,qd_target,kp,kd,tau)
            self._counter+=1
            if self.log_every and self._counter%self.log_every==0:
                print(f'tau={np.round(tau,3)} limited={self.last_components["limited"]}')
            if self.journal and self._counter%50==0:
                self.journal.append('annotation',dict(kind='feedforward',t=now,
                    gravity=g.tolist(),friction=f.tolist(),sent=tau.tolist(),limited=self.last_components['limited']))
            if self.journal and self._counter%1000==0:
                if self.journal.error: raise RuntimeError(f'Runtime logger failed: {self.journal.error}')
                self.journal.request_checkpoint()
        except Exception as exc:
            self.failure=exc
            # Main thread observes failure and closes; do not attempt blind homing.
            raise

    def end(self):
        if self._connected: self.rebotarm.stop_control_loop()
        self._running=False

    def safe_home(self,max_vel=.3,send_freq=50.,settle=.5,timeout=20.):
        if self._running: raise RuntimeError('Stop the control loop before homing')
        if not np.isfinite([max_vel,send_freq,settle,timeout]).all() or min(max_vel,send_freq,timeout)<=0 or settle<0:
            raise ValueError('Invalid homing limits')
        rows=self._snapshot();q0=np.array([r['q'] for r in rows])
        duration=max(1.,1.875*np.max(np.abs(HOME-q0))/max_vel)
        if duration+settle>timeout: raise TimeoutError('Insufficient timeout for bounded HOME trajectory')
        started=time.monotonic();last=started
        while True:
            now=time.monotonic()
            if now-started>timeout: raise TimeoutError('HOME timeout')
            rows=self._snapshot();actual=np.array([r['q'] for r in rows])
            q,qd=ramp(min(now-started,duration),duration,q0,HOME)
            if np.max(np.abs(q-actual))>.15: raise RuntimeError('HOME tracking failed')
            g,_=self.ff.components(actual,np.zeros(6),self.hold_kd)
            tau=self._bounded_torque(g,np.zeros(6),max(now-last,1e-4));last=now
            self._send(q,qd,self.hold_kp,self.hold_kd,tau)
            if now-started>=duration+settle: break
            time.sleep(1/send_freq)
        if np.max(np.abs(np.array([r['q'] for r in self._snapshot()])-HOME))>.05:
            raise RuntimeError('HOME final position not reached')

    def close(self,return_home=None):
        # return_home remains a compatibility argument; folding is mandatory.
        if self._closed: return
        try:
            self.end()
            if self._enabled:
                self._returning=True
                last=time.monotonic()
                def send_fold(q,qd,phase):
                    nonlocal last
                    rows=self._snapshot();actual=np.array([r['q'] for r in rows])
                    if np.any(np.abs([r['tau'] for r in rows])>EFFORT*.9):
                        raise RuntimeError('Fold effort threshold exceeded')
                    g,_=self.ff.components(actual,np.zeros(6),self.hold_kd)
                    now=time.monotonic()
                    tau=self._bounded_torque(g,np.zeros(6),min(max(now-last,1e-4),.1));last=now
                    estimate=self.hold_kp*(q-actual)+self.hold_kd*(qd-np.array([r['qd'] for r in rows]))+tau
                    if np.any(np.abs(estimate)>EFFORT*.8):
                        raise RuntimeError('Fold estimated effort exceeds bound')
                    self._phase=phase
                    self._send(q,qd,self.hold_kp,self.hold_kd,tau)
                fold_before_disable(self.receiver,send_fold,self.journal,rate=self.rate)
                # Do not call disconnect/shutdown: they also disable implicitly.
                # All explicit disable commands are below the fold confirmation.
                errors=[]
                for name,motor in self.rebotarm._motor_map.items():
                    try: motor.disable()
                    except Exception as exc: errors.append(f'{name}: {exc}')
                if errors: raise RuntimeError(f'Fold verified but disable failed: {errors}')
                self._enabled=False
                if self.journal:
                    self.journal.append('annotation',dict(kind='disabled_after_fold',t=time.monotonic()))
        except BaseException as exc:
            if self.journal:
                self.journal.append('annotation',dict(kind='cleanup_failed',error=str(exc),
                    policy='No automatic disable without verified folded pose'))
            print(f'退出失败：{exc}。未确认折叠到位时不发送失能；连接将关闭，需现场处理。',flush=True)
            raise
        finally:
            self._closed=True
            try:
                # Covers partially opened connections without implicit disable.
                for controller in getattr(self.rebotarm,'_ctrl_map',{}).values():
                    try: controller.close_bus()
                    finally: controller.close()
                self._connected=False
                if hasattr(self.rebotarm,'_connected'): self.rebotarm._connected=False
            finally:
                if self.receiver: self.receiver.stop()

    def __enter__(self):
        self.start();return self

    def __exit__(self,*args):
        self.close()
