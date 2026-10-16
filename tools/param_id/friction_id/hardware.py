"""Receive-only SocketCAN observer. Each socket has its own kernel RX queue.

See https://docs.kernel.org/networking/can.html and timestamping.html.
Timestamps represent host kernel reception, NOT motor sampling time.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

from .common import EFFORT, FOLDED, HOME, LIMITS, ZERO_TOLERANCE, check_position, ramp

EFF_FLAG = 0x80000000
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)  # Linux OLD timespec ABI
SO_RXQ_OVFL = 40


def decode_status(can_id, payload, host_id=0xFD):
    if not can_id & EFF_FLAG or can_id & 0x60000000 or len(payload) != 8:
        return None
    arb = can_id & 0x1fffffff
    # Operation replies (2) and periodic active reports (24) carry the same
    # measured-state payload on the installed RS firmware.
    if (arb >> 24) & 31 not in (2, 24) or arb & 255 != host_id:
        return None
    # Firmware version replies share communication type 2.
    if payload[:3] == bytes([0, 0xc4, 0x56]):
        return None
    extra = (arb >> 8) & 65535
    joint = extra & 255
    if joint not in range(1, 7):
        return None
    p, v, tau, temp = struct.unpack('>4H', payload)
    return dict(joint=joint, q=(p/32767.-1)*4*np.pi,
                qd=(v/32767.-1)*(50. if joint <= 3 else 33.),
                tau=(tau/32767.-1)*EFFORT[joint-1], temperature=temp*.1,
                status=(extra >> 8) & 63, can_id=arb, payload=payload.hex())


class FeedbackReceiver:
    def __init__(self, channel='can0', journal=None, host_id=0xFD):
        self.channel, self.journal, self.host_id = channel, journal, host_id
        self.lock = threading.Lock()
        self.latest = {}
        self.sequences = np.zeros(6, dtype=int)
        self.error = None
        self.stop_event = threading.Event()
        self.sock = None
        self.thread = None
        self.clock_offset = None
        self.lower_position_tolerance = np.zeros(6)

    def start(self):
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock = s
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            s.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
            # Accept only extended operation-status frames for this host.
            s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b''.join(
                struct.pack('=II', EFF_FLAG | (typ << 24) | self.host_id,
                            # CAN_ERR_FLAG in a Linux RX mask selects error
                            # filters; it must not be used to reject errors here.
                            EFF_FLAG | 0x1f0000ff | 0x40000000) for typ in (2, 24)))
            s.settimeout(.1)
            s.bind((self.channel,))
        except BaseException:
            s.close()
            self.sock = None
            raise
        self.thread = threading.Thread(target=self._run, daemon=True, name='friction-can-observer')
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame, anc, flags, _ = self.sock.recvmsg(16, 128)
                except socket.timeout:
                    continue
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(frame) != 16:
                    raise RuntimeError('Truncated CAN frame/timestamp')
                a = time.monotonic_ns()
                wall = time.time_ns()
                b = time.monotonic_ns()
                offset = wall-(a+b)//2
                if self.clock_offset is None:
                    self.clock_offset = offset
                drift = abs(offset-self.clock_offset)/1e9
                if drift > .001:
                    raise RuntimeError('Realtime clock stepped; kernel timestamp mapping is invalid')
                kernel_ns = None
                for level, typ, data in anc:
                    if level == socket.SOL_SOCKET and typ == SO_TIMESTAMPNS:
                        sec, nsec = struct.unpack('@ll', data[:struct.calcsize('@ll')])
                        kernel_ns = sec*10**9+nsec
                    if level == socket.SOL_SOCKET and typ == SO_RXQ_OVFL and struct.unpack('=I', data[:4])[0]:
                        raise RuntimeError('CAN observer RX queue overflow; run aborted')
                if kernel_ns is None:
                    raise RuntimeError('Kernel reception timestamp missing')
                can_id, length, payload = struct.unpack('=IB3x8s', frame)
                row = decode_status(can_id, payload[:length], self.host_id)
                if row is None:
                    continue
                row.update(t=(kernel_ns-offset)/1e9, read_t=b/1e9, kernel_ns=kernel_ns,
                           clock_error=drift+(b-a)/2e9, timestamp_valid=True)
                with self.lock:
                    j = row['joint']-1
                    self.sequences[j] += 1
                    row['seq'] = int(self.sequences[j])  # local receive seq, NOT a device packet counter
                    old = self.latest.get(j+1)
                    if old and row['t'] <= old['t']:
                        raise RuntimeError('Nonmonotonic feedback timestamp')
                    self.latest[j+1] = row
                if self.journal:
                    self.journal.append('feedback', row)
        except BaseException as exc:
            self.error = exc

    def snapshot(self, max_age=.1, max_skew=.03):
        if self.error:
            raise RuntimeError(f'Feedback receiver failed: {self.error}') from self.error
        with self.lock:
            rows = [self.latest.get(j) for j in range(1, 7)]
        if any(r is None for r in rows):
            raise RuntimeError('Missing feedback on one or more axes')
        now = time.monotonic()
        times = np.array([r['t'] for r in rows])
        if np.any(now-times > max_age) or np.any(times > now+.001) or np.ptp(times) > max_skew:
            raise RuntimeError('Stale or incoherent feedback snapshot')
        if any(r['status'] or r['temperature'] > 65 or not r['timestamp_valid'] for r in rows):
            raise RuntimeError('Drive fault, overtemperature or invalid reception time')
        check_position([r['q'] for r in rows], self.lower_position_tolerance)
        return rows

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.)
            if self.thread.is_alive():
                raise RuntimeError('CAN observer failed to stop')
        if self.sock:
            self.sock.close()
            self.sock = None


def fold_before_disable(receiver, send, journal=None, *, rate=100., max_vel=.3,
                        settle=.5, timeout=35.):
    """Return through HOME to zero and confirm a fresh, stationary folded pose.

    Failure deliberately propagates before the caller can send any disable.
    `send(q, qd, phase)` must retain gravity support and enforce command bounds.
    Ctrl+C in acquisition requests this recovery; it must not skip this step.
    """
    if not np.isfinite([rate,max_vel,settle,timeout]).all() or not (50<=rate<=200 and 0<max_vel<=.3 and settle>=.5 and timeout>0):
        raise ValueError('Invalid fold recovery limits')
    receiver.lower_position_tolerance = ZERO_TOLERANCE.copy()
    started = time.monotonic()
    rows = receiver.snapshot()
    initial = check_position([r['q'] for r in rows], ZERO_TOLERANCE)
    if journal:
        journal.append('annotation',dict(kind='fold_started',t=started,q=initial.tolist()))
    print('退出回位：HOME → 折叠零位；确认到位后才失能。', flush=True)
    # Already near the folded pose: do not unfold merely to shut down.
    targets = [FOLDED] if np.max(np.abs(initial))<=.05 else [HOME,FOLDED]
    for target in targets:
        rows = receiver.snapshot()
        q0 = np.array([r['q'] for r in rows])
        duration = max(1.,1.875*np.max(np.abs(target-q0))/max_vel)
        began = time.monotonic()
        if began-started+duration+settle>timeout:
            raise TimeoutError('Insufficient time for bounded fold recovery; disable blocked')
        stable_since = None
        settled_samples = []
        deadline = began
        while True:
            now = time.monotonic()
            if now-started>timeout:
                raise TimeoutError('Fold position/velocity not confirmed; disable blocked')
            lateness = now-deadline
            if lateness>.05:
                if journal:
                    journal.append('annotation',dict(kind='fold_deadline_missed',t=now,
                        lateness_s=lateness,elapsed_s=now-began,
                        phase='exit_fold' if np.array_equal(target,FOLDED) else 'exit_home'))
                raise RuntimeError(f'Fold control deadline missed by {lateness*1000:.1f} ms; disable blocked')
            rows = receiver.snapshot()
            actual = np.array([r['q'] for r in rows])
            velocity = np.array([r['qd'] for r in rows])
            q,qd = ramp(now-began,duration,q0,target)
            if np.max(np.abs(q-actual))>.15 or np.max(np.abs(velocity))>1.:
                raise RuntimeError('Fold tracking/velocity bound exceeded; disable blocked')
            phase = 'exit_fold' if np.array_equal(target,FOLDED) else 'exit_home'
            send(q,qd,phase)
            tolerance = .02 if phase=='exit_fold' else .05
            arrived = now-began>=duration and np.max(np.abs(actual-target))<=tolerance
            stable_since = (now if stable_since is None else stable_since) if arrived else None
            if arrived:
                settled_samples.append((now,actual.copy()))
                settled_samples=[sample for sample in settled_samples if sample[0]>=now-settle-.02]
            else:
                settled_samples=[]
            if stable_since is not None and now-stable_since>=settle:
                ts=np.array([sample[0] for sample in settled_samples])
                positions=np.array([sample[1] for sample in settled_samples])
                centered=ts-ts.mean()
                position_velocity=centered @ positions / np.sum(centered**2)
                # RS velocity reports have ~0.2 rad/s noise even when the
                # encoders are stationary. Verify motion using the position
                # window, rather than requiring noisy instantaneous qd == 0.
                if (np.max(np.ptp(positions,axis=0))<=.002
                        and np.max(np.abs(position_velocity))<=.02):
                    break
            deadline += 1/rate
            time.sleep(max(0.,deadline-time.monotonic()))
    # Final observation, immediately before granting permission to disable.
    rows = receiver.snapshot()
    if (np.max(np.abs([r['q'] for r in rows]))>.02
            or np.max(np.abs([r['qd'] for r in rows]))>1.
            or np.max(np.abs(np.array([r['q'] for r in rows])-positions[-1]))>.002):
        raise RuntimeError('Fold confirmation lost; disable blocked')
    if journal:
        journal.append('annotation',dict(kind='fold_verified',t=time.monotonic(),
            q=[r['q'] for r in rows],qd=[r['qd'] for r in rows],
            position_velocity=position_velocity.tolist(),settle=settle))
    print('折叠零位已确认，允许失能。', flush=True)


class MotorSession:
    """One command owner; cleanup covers partial initialization and Ctrl+C."""
    def __init__(self, channel, receiver, journal, gravity, kp, kd, max_ff=None):
        self.channel, self.receiver, self.journal, self.gravity = channel, receiver, journal, gravity
        self.kp, self.kd = kp, kd
        self.max_ff = np.asarray(max_ff if max_ff is not None else EFFORT*.5)
        self.motors = []
        self.ctrl = None
        self.command_id = 0
        self.enabled = False
        self.enabled_at = None
        self.previous_ff = np.zeros(6)
        self.last_send = time.monotonic()
        self.startup = True
        self.startup_tolerance = ZERO_TOLERANCE.copy()
        self.returning = False
        self.closed = False

    def __enter__(self):
        try:
            from motorbridge import Controller, Mode
            from .common import MODELS
            self.receiver.start()
            self.ctrl = Controller(self.channel)
            for j, model in enumerate(MODELS, 1):
                self.motors.append(self.ctrl.add_robstride_motor(j, 0xFD, model))
            # Typed register reads fail explicitly; no zero replacement.
            self.initial = check_position([m.robstride_get_param_f32(0x7019) for m in self.motors], self.startup_tolerance)
            if np.max(np.abs(self.initial))>.05:
                raise RuntimeError('Start must be folded; refusing mode changes/clear_error outside folded pose')
            self.receiver.lower_position_tolerance = self.startup_tolerance.copy()
            self.reference_floor = np.minimum(LIMITS[:,0],self.initial)
            self.reference_floor = np.where(self.initial < LIMITS[:,0],self.reference_floor-.001,self.reference_floor)
            self.journal.append('annotation',dict(kind='startup_position',initial=self.initial.tolist(),
                lower_tolerance=self.startup_tolerance.tolist(),policy='inward recovery only; strict after HOME'))
            for m in self.motors:
                m.clear_error()
                m.ensure_mode(Mode.MIT, 2000)
                m.robstride_set_active_report(True)
            deadline = time.monotonic()+2.
            while True:
                try:
                    self.receiver.snapshot()
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.01)
            return self
        except BaseException:
            self.close()
            raise

    def enable(self):
        rows = self.receiver.snapshot()
        q = check_position([r['q'] for r in rows], self.startup_tolerance if self.startup else 0.)
        # Seed a hold command before enabling; mode changes must precede this.
        self.send(q, np.zeros(6), phase='enable_hold', ff_scale=0.)
        self.enabled_at = time.monotonic()
        self.enabled = True  # includes partial enable failures in cleanup
        for m in self.motors:
            m.enable()
        self.send(q, np.zeros(6), phase='enable_hold')

    def send(self, q_ref, qd_ref, segment=-1, repeat=-1, speed_id=-1, direction=0, phase='hold', ff_scale=None):
        q_ref = check_position(q_ref, self.startup_tolerance if self.startup or self.returning else 0.)
        if self.startup and not self.returning and np.any(q_ref < self.reference_floor):
            raise RuntimeError('Startup target moves further outside the joint limit')
        rows = self.receiver.snapshot()
        q = np.array([r['q'] for r in rows])
        qd_ref = np.asarray(qd_ref, float)
        if not np.isfinite(qd_ref).all() or np.max(np.abs(qd_ref)) > .6:
            raise RuntimeError('Reference velocity invalid or exceeds experiment limit')
        if self.enabled and np.max(np.abs(q_ref-q)) > .15:
            raise RuntimeError('Tracking error exceeded 0.15 rad')
        if np.any(np.abs([r['tau'] for r in rows]) > EFFORT*.9):
            raise RuntimeError('Drive effort exceeded experiment threshold')
        if np.any(np.abs([r['qd'] for r in rows]) > 1.):
            raise RuntimeError('Actual velocity exceeded experiment threshold')
        ff = np.asarray(self.gravity(q), float)
        if not np.isfinite(ff).all() or np.any(np.abs(ff) > self.max_ff):
            raise RuntimeError('Gravity feedforward exceeds configured limit')
        if ff_scale is None:
            ff_scale = min(1., max(0., time.monotonic()-(self.enabled_at or time.monotonic())))
        ff = ff*ff_scale
        now = time.monotonic()
        dt = max(0., min(now-self.last_send, .1))
        ff = self.previous_ff+np.clip(ff-self.previous_ff,-20.*dt,20.*dt)
        estimated_pd = self.kp*(q_ref-q)+self.kd*(qd_ref-np.array([r['qd'] for r in rows]))
        if np.any(np.abs(estimated_pd+ff) > EFFORT*.8):
            raise RuntimeError('Estimated MIT effort exceeds experiment bound')
        self.command_id += 1
        for j, m in enumerate(self.motors):
            start = time.monotonic()
            ok = False
            try:
                m.send_mit(float(q_ref[j]), float(qd_ref[j]), float(self.kp[j]), float(self.kd[j]), float(ff[j]))
                ok = True
            finally:
                self.journal.append('command', dict(tx_start=start, tx_end=time.monotonic(), joint=j+1,
                    command_id=self.command_id, segment=segment, repeat=repeat, speed_id=speed_id,
                    direction=direction, phase=phase, q_ref=float(q_ref[j]), qd_ref=float(qd_ref[j]),
                    kp=float(self.kp[j]), kd=float(self.kd[j]), tau_ff=float(ff[j]), ok=ok))
        self.previous_ff, self.last_send = ff, now

    def finish_startup(self):
        check_position([r['q'] for r in self.receiver.snapshot()])
        self.startup = False
        self.receiver.lower_position_tolerance = np.zeros(6)
        self.journal.append('annotation',dict(kind='startup_complete',policy='strict URDF limits restored'))

    def close(self):
        if self.closed:
            return
        try:
            if self.enabled:
                self.returning = True
                fold_before_disable(self.receiver,lambda q,v,p:self.send(q,v,phase=p),self.journal)
                errors = []
                for m in self.motors:
                    try:
                        m.disable()
                    except Exception as exc:
                        errors.append(str(exc))
                if errors:
                    raise RuntimeError(f'Fold verified but disable failed: {errors}')
                self.enabled = False
                self.journal.append('annotation',dict(kind='disabled_after_fold',t=time.monotonic()))
        except BaseException as exc:
            self.journal.append('annotation',dict(kind='cleanup_failed',error=str(exc),
                policy='No automatic disable without verified folded pose'))
            print(f'退出失败：{exc}。未确认折叠到位时不发送失能；连接将关闭，需现场处理。',flush=True)
            raise
        finally:
            self.closed = True
            try:
                # close_bus/free do not disable; shutdown/disconnect do.
                if self.ctrl:
                    try: self.ctrl.close_bus()
                    finally: self.ctrl.close()
            finally:
                self.receiver.stop()

    def __exit__(self, *args):
        self.close()
