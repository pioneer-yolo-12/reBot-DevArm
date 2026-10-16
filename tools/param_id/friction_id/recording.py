from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .common import canonical

SCHEMA = 3
FEEDBACK_DTYPE = np.dtype([
    ('t', 'f8'), ('read_t', 'f8'), ('kernel_ns', 'i8'), ('clock_error', 'f8'),
    ('seq', 'i8'), ('joint', 'i4'), ('q', 'f8'), ('qd', 'f8'), ('tau', 'f8'),
    ('temperature', 'f8'), ('status', 'i4'), ('timestamp_valid', '?'),
    ('can_id', 'u4'), ('payload', 'U16'),
])
COMMAND_DTYPE = np.dtype([
    ('tx_start', 'f8'), ('tx_end', 'f8'), ('joint', 'i4'), ('command_id', 'i8'),
    ('segment', 'i4'), ('repeat', 'i4'), ('speed_id', 'i4'), ('direction', 'i4'), ('phase', 'U16'),
    ('q_ref', 'f8'), ('qd_ref', 'f8'), ('kp', 'f8'), ('kd', 'f8'), ('tau_ff', 'f8'), ('ok', '?'),
])


def rows_array(rows, dtype):
    return np.array([tuple(r[name] for name in dtype.names) for r in rows], dtype=dtype)


class Journal:
    """Nonblocking event submission; JSONL IO is owned by one writer thread.

    NPZ conversion/compression is deliberately deferred until checkpoint() is
    called after motion/cleanup. A Python background NPZ thread still competes
    for the GIL and can stall the command and CAN receiver threads together.
    """
    def __init__(self, path, metadata, *, queue_capacity=8192):
        if not isinstance(queue_capacity,int) or queue_capacity<1:
            raise ValueError('Journal queue capacity must be a positive integer')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata, schema_version=SCHEMA, run_status='running')
        self.metadata['recording_policy'] = dict(jsonl='bounded queue to IO thread',
            queue_capacity=queue_capacity,npz='blocking export only after motion and cleanup')
        self.lock = threading.RLock()
        self.writer_lock = threading.Lock()
        self.error = None
        self.closed = False
        self.rows = {'feedback': [], 'command': [], 'annotation': []}
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.file = self.path.with_suffix('.jsonl').open('x', buffering=64*1024)
        self.last_sync = time.monotonic()
        self.worker = threading.Thread(target=self._write_events, name='friction-jsonl-writer', daemon=True)
        self.worker.start()
        self.append('annotation', dict(kind='metadata', value=self.metadata.copy()))

    def append(self, kind, row):
        if kind not in self.rows:
            raise ValueError(f'Unknown journal event: {kind}')
        self._enqueue((kind, dict(row)))

    def _enqueue(self, item):
        with self.lock:
            if self.error:
                raise RuntimeError(f'Journal writer failed: {self.error}') from self.error
            if self.closed:
                raise RuntimeError('Journal is closed')
            try:
                self.queue.put_nowait(item)
            except queue.Full as exc:
                self.error = BufferError('Journal queue full; event could not be recorded')
                raise RuntimeError(str(self.error)) from exc

    def _sync(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.last_sync = time.monotonic()

    def _write_events(self):
        try:
            while True:
                try:
                    kind, value = self.queue.get(timeout=.1)
                except queue.Empty:
                    if time.monotonic()-self.last_sync>.5:
                        self._sync()
                    continue
                if kind == 'stop':
                    self._sync()
                    return
                if kind == 'flush':
                    self._sync()
                    value.set()
                else:
                    self.file.write(canonical(dict(kind=kind,value=value))+'\n')
                    with self.lock:
                        self.rows[kind].append(value)
                if time.monotonic()-self.last_sync>.5:
                    self._sync()
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.file.close()
            except BaseException as exc:
                self.error = self.error or exc

    def _flush(self):
        done = threading.Event()
        self._enqueue(('flush',done))
        deadline = time.monotonic()+10.
        while not done.wait(.02):
            if self.error:
                raise RuntimeError(f'Journal writer failed: {self.error}') from self.error
            if time.monotonic()>deadline:
                raise TimeoutError('Journal writer flush timed out')
        if self.error:
            raise RuntimeError(f'Journal writer failed: {self.error}') from self.error

    def checkpoint(self, status=None, reason=''):
        """Blocking NPZ export. Call only after stopping motion and CAN reception."""
        with self.writer_lock:
            if status:
                self.metadata.update(run_status=status, exit_reason=reason)
                self.append('annotation', dict(kind='metadata', value=self.metadata.copy()))
            self._flush()
            with self.lock:
                metadata = self.metadata.copy()
                rows = {k: list(v) for k, v in self.rows.items()}
            write_npz(self.path, metadata, rows)

    def request_checkpoint(self):
        # A live durability marker, not a concurrent full-array conversion.
        # JSONL continues syncing in the IO thread at <= .5 s intervals when IO
        # is responsive. Export the final NPZ only once recovery has finished.
        self.append('annotation',dict(kind='checkpoint_deferred',t=time.monotonic(),
            policy='JSONL during motion; NPZ after cleanup'))

    def close(self):
        with self.lock:
            if self.closed: return
            self.closed = True
        # The producer queue may be full after an abort; shutdown can wait now
        # that control has stopped. Writer errors are never silently ignored.
        if self.worker.is_alive():
            try: self.queue.put(('stop',None),timeout=10.)
            except queue.Full as exc: raise TimeoutError('Journal writer shutdown timed out') from exc
            self.worker.join(timeout=10.)
        if self.worker.is_alive():
            raise TimeoutError('Journal writer failed to stop')
        if self.error:
            raise RuntimeError(f'Journal writer failed: {self.error}') from self.error


def write_npz(path, metadata, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, metadata_json=canonical(metadata),
                            feedback=rows_array(rows['feedback'], FEEDBACK_DTYPE),
                            commands=rows_array(rows['command'], COMMAND_DTYPE),
                            annotations_json=np.array([canonical(r) for r in rows['annotation']], dtype=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def recover(journal_path, output):
    rows = {'feedback': [], 'command': [], 'annotation': []}
    metadata = None
    lines = Path(journal_path).read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise ValueError('Corrupted journal before final line')
            break  # a process can die in the middle of its last write
        rows[event['kind']].append(event['value'])
        if event['kind'] == 'annotation' and event['value'].get('kind') == 'metadata':
            metadata = event['value']['value']
    if metadata is None:
        raise ValueError('Journal has no metadata')
    metadata.update(run_status='recovered', exit_reason='journal recovery; experiment incomplete')
    write_npz(output, metadata, rows)


def load_run(path):
    with np.load(path, allow_pickle=False) as d:
        if 'metadata_json' not in d or 'feedback' not in d or 'commands' not in d:
            raise ValueError('Legacy summary NPZ: raw feedback/command times are missing; cannot re-identify')
        meta = json.loads(str(d['metadata_json']))
        if meta.get('schema_version') != SCHEMA:
            raise ValueError('Unsupported acquisition schema')
        return meta, d['feedback'].copy(), d['commands'].copy()
