"""Optional read-only 0x7019 probe, isolated from the control SDK socket.

One outstanding request, kernel RX timestamps, no parameter writes or motion.
This is a higher precision INTERNAL channel, not an independent angle sensor.
"""
from __future__ import annotations

import select
import socket
import struct
import threading
import time
import numpy as np

from .hardware import EFF_FLAG, SO_TIMESTAMPNS, SO_RXQ_OVFL


def request_frame(joint, host=0xFD):
    if joint not in range(1,7) or host not in range(256):
        raise ValueError('Invalid probe joint/host')
    return struct.pack('=IB3x8s', EFF_FLAG | (17<<24) | (host<<8) | joint,
                       8, bytes.fromhex('1970000000000000'))


def decode_reply(can_id,payload,joint,host=0xFD):
    expected=EFF_FLAG | (17<<24) | (joint<<8) | host
    if can_id!=expected or len(payload)!=8 or payload[:2]!=bytes.fromhex('1970'):
        return None
    q=struct.unpack('<f',payload[4:])[0]
    if not np.isfinite(q) or abs(q)>4*np.pi:
        raise RuntimeError('Invalid float position probe reply')
    return float(q)


class PositionProbe:
    def __init__(self,channel,joint,journal,rate=50.):
        self.frame=request_frame(joint)
        if not 10<=rate<=100:raise ValueError('Probe rate outside 10..100 Hz')
        self.channel,self.joint,self.journal,self.rate=channel,joint,journal,rate
        self.error=None;self.sock=None;self.thread=None
        self.stop_event=threading.Event()

    def start(self):
        s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
        self.sock=s
        try:
            s.setsockopt(socket.SOL_SOCKET,SO_TIMESTAMPNS,1)
            s.setsockopt(socket.SOL_SOCKET,SO_RXQ_OVFL,1)
            reply=EFF_FLAG | (17<<24) | (self.joint<<8) | 0xFD
            # Do not include CAN_ERR_FLAG in a normal Linux CAN RX mask.
            s.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,
                         struct.pack('=II',reply,EFF_FLAG|0x40000000|0x1fffffff))
            s.bind((self.channel,));s.setblocking(False)
        except BaseException:
            s.close();self.sock=None;raise
        self.journal.append('annotation',dict(kind='position_probe_started',joint=self.joint,
            rate=self.rate,register='0x7019',policy='Read-only internal float position; not independent angle truth'))
        self.thread=threading.Thread(target=self._run,name='position-register-probe',daemon=True)
        self.thread.start()

    def _run(self):
        try:
            deadline=time.monotonic();seq=0;clock_offset=None;last_t=-np.inf
            while not self.stop_event.is_set():
                tx_start=time.monotonic()
                if self.sock.send(self.frame)!=len(self.frame):raise RuntimeError('Partial probe request')
                tx_end=time.monotonic();expires=tx_end+.05
                while True:
                    remaining=expires-time.monotonic()
                    if remaining<=0:raise RuntimeError('Position probe response timeout')
                    ready,_,_=select.select([self.sock],[],[],remaining)
                    if not ready:raise RuntimeError('Position probe response timeout')
                    frame,anc,flags,_=self.sock.recvmsg(16,128)
                    a=time.monotonic_ns();wall=time.time_ns();b=time.monotonic_ns()
                    if flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC) or len(frame)!=16:
                        raise RuntimeError('Truncated position probe frame/timestamp')
                    cid,length,data=struct.unpack('=IB3x8s',frame)
                    q=decode_reply(cid,data[:length],self.joint)
                    if q is None:continue
                    kernel=None
                    for level,typ,value in anc:
                        if level==socket.SOL_SOCKET and typ==SO_TIMESTAMPNS:
                            sec,nsec=struct.unpack('@ll',value[:struct.calcsize('@ll')]);kernel=sec*10**9+nsec
                        if level==socket.SOL_SOCKET and typ==SO_RXQ_OVFL and struct.unpack('=I',value[:4])[0]:
                            raise RuntimeError('Position probe RX queue overflow')
                    if kernel is None:raise RuntimeError('Missing probe kernel timestamp')
                    offset=wall-(a+b)//2
                    if clock_offset is None:clock_offset=offset
                    drift=abs(offset-clock_offset)/1e9
                    if drift>.001:raise RuntimeError('Probe clock mapping stepped')
                    t=(kernel-clock_offset)/1e9
                    # Old replies cannot be attributed to the new request.
                    if t<tx_start-.001 or t>expires:raise RuntimeError('Probe reply outside request interval')
                    if t<=last_t:raise RuntimeError('Nonmonotonic probe reply timestamp')
                    last_t=t
                    seq+=1
                    self.journal.append('annotation',dict(kind='position_probe_sample',joint=self.joint,
                        seq=seq,register='0x7019',q=q,t=t,read_t=b/1e9,kernel_ns=kernel,
                        clock_error=drift+(b-a)/2e9,
                        tx_start=tx_start,tx_end=tx_end,can_id=cid&0x1fffffff,payload=data.hex()))
                    break
                deadline+=1/self.rate
                self.stop_event.wait(max(0.,deadline-time.monotonic()))
        except BaseException as exc:
            self.error=exc
            try:self.journal.append('annotation',dict(kind='position_probe_failed',reason=str(exc),t=time.monotonic()))
            except Exception:pass

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=1.)
        if self.thread and self.thread.is_alive():raise RuntimeError('Position probe did not stop')
        if self.sock:self.sock.close();self.sock=None
