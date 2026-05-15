"""Run a kohya training script as a subprocess and stream events.

The runner owns the child process and two pump threads that read stdout/stderr
line-by-line, hand each line to the parser, and forward the resulting events
to the listener supplied by the caller.

Stopping is platform-aware: on Windows we send CTRL_BREAK_EVENT to the process
group (Popen with CREATE_NEW_PROCESS_GROUP), then escalate to terminate() and
finally kill() if the child does not exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from lorahub.core.backends.kohya.parser import parse_line
from lorahub.core.events import EventType, TrainingEvent

EventListener = Callable[[TrainingEvent], None]


@dataclass(slots=True)
class RunResult:
    returncode: int
    duration_s: float


class KohyaRunner:
    """Owns one kohya training subprocess and pumps its stdout into events."""

    def __init__(
        self,
        python: Path,
        script: Path,
        argv: list[str],
        workspace: Path,
        on_event: EventListener,
        *,
        job_id: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._python = python
        self._script = script
        self._argv = argv
        self._workspace = workspace
        self._on_event = on_event
        self._job_id = job_id
        self._env = env

        self._proc: subprocess.Popen[str] | None = None
        self._pump_threads: list[threading.Thread] = []
        self._reaper_thread: threading.Thread | None = None
        self._done_emitted: bool = False
        self._started_at: float | None = None
        self._lock = threading.RLock()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                msg = "runner already started"
                raise RuntimeError(msg)

            self._workspace.mkdir(parents=True, exist_ok=True)
            cmd = [str(self._python), str(self._script), *self._argv]
            full_env = {**os.environ, **(self._env or {})}
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            start_new_session = sys.platform != "win32"

            self._started_at = time.time()
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(self._script.parent),
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=full_env,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            self._spawn_pumps()
            self._spawn_reaper()

    def _spawn_pumps(self) -> None:
        assert self._proc is not None
        for stream, source in ((self._proc.stdout, "stdout"), (self._proc.stderr, "stderr")):
            if stream is None:
                continue
            t = threading.Thread(
                target=self._pump,
                args=(stream, source),
                name=f"kohya-{source}",
                daemon=True,
            )
            t.start()
            self._pump_threads.append(t)

    def _spawn_reaper(self) -> None:
        """Background thread that waits for the child and emits the `done` event.

        Without this, callers that don't invoke `wait()` (e.g. an HTTP API that
        fires-and-forgets) would never see job completion.
        """
        thread = threading.Thread(
            target=self._reap,
            name="kohya-reaper",
            daemon=True,
        )
        thread.start()
        self._reaper_thread = thread

    def _reap(self) -> None:
        assert self._proc is not None
        assert self._started_at is not None
        rc = self._proc.wait()
        for t in self._pump_threads:
            t.join(timeout=5.0)
        with self._lock:
            if self._done_emitted:
                return
            self._done_emitted = True
        duration = time.time() - self._started_at
        self._safe_emit(
            TrainingEvent(
                type=EventType.done,
                payload={"returncode": rc, "duration_s": duration},
                job_id=self._job_id,
            )
        )

    def _pump(self, stream: IO[str], source: str) -> None:
        try:
            for raw in stream:
                event = parse_line(raw, job_id=self._job_id)
                if event is None:
                    continue
                if source == "stderr" and event.type is EventType.log:
                    event.payload.setdefault("source", "stderr")
                self._safe_emit(event)
        finally:
            stream.close()

    def _safe_emit(self, event: TrainingEvent) -> None:
        try:
            self._on_event(event)
        except Exception as exc:  # noqa: BLE001
            self._on_event(
                TrainingEvent(
                    type=EventType.error,
                    payload={"source": "listener", "error": repr(exc)},
                    job_id=self._job_id,
                )
            )

    def wait(self, timeout: float | None = None) -> RunResult:
        if self._proc is None or self._started_at is None:
            msg = "runner has not been started"
            raise RuntimeError(msg)
        rc = self._proc.wait(timeout=timeout)
        for t in self._pump_threads:
            t.join(timeout=5.0)
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5.0)
        duration = time.time() - self._started_at
        return RunResult(returncode=rc, duration_s=duration)

    def stop(self, *, graceful: bool = True, timeout: float = 10.0) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            if graceful:
                self._signal_graceful()
                try:
                    self._proc.wait(timeout=timeout)
                    return
                except subprocess.TimeoutExpired:
                    pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _signal_graceful(self) -> None:
        assert self._proc is not None
        if sys.platform == "win32":
            self._proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self._proc.send_signal(signal.SIGINT)
