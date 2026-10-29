"""Capture CLI output before application imports; no application logger is modified.

Linux/macOS are the supported evaluation hosts (including Docker). A parent
process drains both child pipes, preserving their terminal destinations. The
private rotating handler is shared safely by independent evaluation processes.
"""

from __future__ import annotations

import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from uuid import uuid4


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "evaluation.log"
MAX_BYTES = 50 * 1024 * 1024
BACKUP_COUNT = 3
_CHILD_ENV = "AGENTRAG_EVALUATION_LOG_CHILD"


class EvaluationLog(RotatingFileHandler):
    """Byte-preserving, size-bounded rotating sink with a stable lock file.

    Handler.handle supplies the thread lock; flock covers size checks, rollover
    and writes across processes. Reopening after another process rotates avoids
    continuing to write to a renamed backup. Never register this on a logger.
    """

    def __init__(self, path=LOG_PATH, *, max_bytes=MAX_BYTES):
        if max_bytes < 4:
            raise ValueError("max_bytes must be at least 4")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(str(path) + ".lock", "ab")
        try:
            super().__init__(path, maxBytes=max_bytes, backupCount=BACKUP_COUNT, delay=True)
            # Create even for an otherwise silent run, without truncating.
            self.stream = self._open()
        except BaseException:
            self._lock_file.close()
            raise

    def _open(self):
        return open(self.baseFilename, "ab")

    def emit(self, record):
        data = record.msg
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        try:
            if self.stream is not None:
                try:
                    same_file = os.path.samestat(os.fstat(self.stream.fileno()), os.stat(self.baseFilename))
                except FileNotFoundError:
                    same_file = False
                if not same_file:
                    self.stream.close()
                    self.stream = None
            if self.stream is None:
                self.stream = self._open()
            offset = 0
            while offset < len(data):
                size = os.fstat(self.stream.fileno()).st_size
                if size >= self.maxBytes:
                    self.doRollover()
                    self.stream = self._open()
                    size = 0
                end = min(len(data), offset + self.maxBytes - size)
                self.stream.write(data[offset:end])
                self.stream.flush()
                offset = end
        finally:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)

    def write(self, data: bytes):
        if data:
            self.handle(logging.LogRecord("evaluation.output", logging.INFO, "", 0, data, (), None))

    def close(self):
        try:
            super().close()
        finally:
            self._lock_file.close()


def is_logging_child(script: str) -> bool:
    """Only the supervised CLI skips bootstrapping; imports never start runs."""
    if os.environ.get(_CHILD_ENV) == str(Path(script).resolve()):
        os.environ.pop(_CHILD_ENV)
        return True
    return False


def _terminal_write(stream, data):
    binary = getattr(stream, "buffer", None)
    if binary is not None:
        binary.write(data)
    else:
        stream.write(data.decode("utf-8", errors="replace"))
    stream.flush()


def run_logged_script(script: str, *, log_path=None, max_bytes=MAX_BYTES) -> int:
    """Supervise the unchanged CLI, including imports, tracebacks and FD output.

    Child processes inheriting stdout/stderr are captured too. Detached children
    or output redirected to other files are outside this capture boundary.
    """
    run_id = uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    exit_code = 1
    child = None
    previous_sigterm = None
    sink = EvaluationLog(log_path or LOG_PATH, max_bytes=max_bytes)

    def marker(text):
        sink.write(("\n" + "=" * 24 + f" RUN {run_id} " + text + " " + "=" * 24 + "\n").encode())

    try:
        marker(f"START started_at={started} script={Path(script).name} argv={sys.argv[1:]!r}")
        env = {**os.environ, _CHILD_ENV: str(Path(script).resolve()), "PYTHONUNBUFFERED": "1"}
        child = subprocess.Popen(
            [sys.executable, "-u", str(script), *sys.argv[1:]],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            start_new_session=True,
        )
        previous_sigterm = signal.signal(signal.SIGTERM, lambda signum, frame: child.send_signal(signum))
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, sys.stdout)
            selector.register(child.stderr, selectors.EVENT_READ, sys.stderr)
            while selector.get_map():
                try:
                    for key, _ in selector.select(timeout=0.1):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        _terminal_write(key.data, data)
                        sink.write(data)
                except KeyboardInterrupt:
                    child.send_signal(signal.SIGINT)
        return_code = child.wait()
        exit_code = return_code if return_code >= 0 else 128 - return_code
        return exit_code
    except BaseException:
        detail = traceback.format_exc().encode()
        _terminal_write(sys.stderr, detail)
        sink.write(detail)
        return 1
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if child is not None:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            child.stderr.close()
        try:
            marker(f"END started_at={started} ended_at={datetime.now(timezone.utc).isoformat()} "
                   f"duration_seconds={time.monotonic() - clock:.3f} "
                   f"status={'SUCCESS' if exit_code == 0 else 'FAILED'} exit_code={exit_code}")
        finally:
            sink.close()
