import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest

from friction_id import probe
from run_phase2_pointmatch import parser,configuration


def test_probe_frame_is_only_a_read_request():
    raw=probe.request_frame(6)
    cid,length,payload=struct.unpack('=IB3x8s',raw)
    assert cid==probe.EFF_FLAG|(17<<24)|(0xFD<<8)|6
    assert length==8 and payload==bytes.fromhex('1970000000000000')
    with pytest.raises(ValueError):probe.request_frame(7)


def test_probe_decoding_rejects_other_sources_and_registers():
    cid=probe.EFF_FLAG|(17<<24)|(6<<8)|0xFD
    data=bytes.fromhex('19700000')+struct.pack('<f',.123456)
    assert probe.decode_reply(cid,data,6)==pytest.approx(.123456)
    assert probe.decode_reply(cid,data,5) is None
    assert probe.decode_reply(cid^(1<<24),data,6) is None
    assert probe.decode_reply(cid,bytes.fromhex('1b700000')+data[4:],6) is None
    for q in [np.nan,np.inf,20.]:
        with pytest.raises(RuntimeError):probe.decode_reply(cid,data[:4]+struct.pack('<f',q),6)


def test_probe_logs_real_reply_and_request_interval(monkeypatch):
    rows=[]
    p=probe.PositionProbe('mock',6,SimpleNamespace(append=lambda kind,row:record(row)))
    def record(row):
        rows.append(row)
        if row['kind']=='position_probe_sample':p.stop_event.set()
    sent=[]
    class FakeSocket:
        def send(self,frame):sent.append(frame);return len(frame)
        def recvmsg(self,*args):
            now=time.time_ns();sec,ns=divmod(now,10**9)
            cid=probe.EFF_FLAG|(17<<24)|(6<<8)|0xFD
            frame=struct.pack('=IB3x8s',cid,8,bytes.fromhex('19700000')+struct.pack('<f',.01234))
            return frame,[(probe.socket.SOL_SOCKET,probe.SO_TIMESTAMPNS,struct.pack('@ll',sec,ns))],0,None
    p.sock=FakeSocket()
    monkeypatch.setattr(probe.select,'select',lambda *a:([p.sock],[],[]))
    p._run()
    assert p.error is None and sent==[probe.request_frame(6)]
    assert len(rows)==1 and rows[0]['q']==pytest.approx(.01234)
    assert rows[0]['tx_start']<=rows[0]['tx_end']<=rows[0]['read_t']
    assert rows[0]['seq']==1 and rows[0]['kernel_ns']>0


def test_probe_timeout_is_explicit_and_does_not_send_motion(monkeypatch):
    rows=[];sent=[]
    p=probe.PositionProbe('mock',6,SimpleNamespace(append=lambda kind,row:rows.append(row)))
    p.sock=SimpleNamespace(send=lambda frame:(sent.append(frame) or len(frame)))
    monkeypatch.setattr(probe.select,'select',lambda *a:([],[],[]))
    p._run()
    assert isinstance(p.error,RuntimeError)
    assert rows[0]['kind']=='position_probe_failed' and sent==[probe.request_frame(6)]


def test_probe_is_explicit_tracking_diagnostic():
    assert configuration(parser().parse_args(['--position-probe']))['position_probe'] is True
    assert configuration(parser().parse_args([]))['position_probe'] is False
    with pytest.raises(ValueError):configuration(parser().parse_args(['--position-probe','--mode','identify']))


def test_probe_failure_runs_session_exit_before_probe_cleanup(tmp_path,monkeypatch):
    import run_phase2_pointmatch as acquisition
    import friction_id.hardware as hardware
    from friction_id.common import HOME
    from friction_id.recording import load_run
    order=[]
    class Receiver:
        def __init__(self,*a):pass
        def snapshot(self):return [dict(q=q) for q in HOME]
    class Session:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def enable(self):pass
        def __exit__(self,*a):order.append('session_exit')
    class FailedProbe:
        def __init__(self,*a):self.error=None
        def start(self):self.error=RuntimeError('synthetic missing probe reply')
        def stop(self):order.append('probe_stop')
    monkeypatch.setattr(hardware,'FeedbackReceiver',Receiver)
    monkeypatch.setattr(hardware,'MotorSession',Session)
    monkeypatch.setattr(probe,'PositionProbe',FailedProbe)
    monkeypatch.setattr(acquisition,'Gravity',lambda:SimpleNamespace(model_hash='synthetic'))
    monkeypatch.setattr(acquisition.importlib.metadata,'version',lambda name:'synthetic')
    args=parser().parse_args(['--position-probe','--output',str(tmp_path),'--load-id','fixture'])
    with pytest.raises(RuntimeError,match='Position probe failed'):
        acquisition.run(args,configuration(args))
    assert order==['session_exit','probe_stop']
    meta,_,_=load_run(next(tmp_path.glob('*.npz')))
    assert meta['run_status']=='aborted'
