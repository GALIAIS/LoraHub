"""Generic line-buffered subprocess runner shared by every backend.

Both kohya and diffusion-pipe spawn a single training process and pump its
stdout/stderr through a line parser to derive `TrainingEvent`s. The shape is
identical: a Popen pipe, two daemon pump threads, a reaper thread that emits
the terminal `done` event, and platform-aware graceful stop.

Backends only differ in *how* a line gets translated into an event, so this
module keeps the runner generic and takes a `parse_line` callable as a
constructor argument.
"""

from __future__ import annotations

import contextlib
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

from lorahub.core.events import EventType, TrainingEvent

EventListener = Callable[[TrainingEvent], None]
LineParser = Callable[..., TrainingEvent | None]


@dataclass(slots=True)
class RunResult:
    returncode: int
    duration_s: float


class SubprocessRunner:
    """Owns one training subprocess and pumps its stdout into events."""

    def __init__(
        self,
        argv: list[str],
        workspace: Path,
        on_event: EventListener,
        parse_line: LineParser,
        *,
        cwd: Path | None = None,
        job_id: str | None = None,
        env: dict[str, str] | None = None,
        thread_label: str = "trainer",
    ) -> None:
        self._argv = list(argv)
        self._workspace = workspace
        self._on_event = on_event
        self._parse_line = parse_line
        self._cwd = cwd
        self._job_id = job_id
        self._env = env
        self._thread_label = thread_label

        self._proc: subprocess.Popen[str] | None = None
        self._pump_threads: list[threading.Thread] = []
        self._reaper_thread: threading.Thread | None = None
        self._done_emitted: bool = False
        self._started_at: float | None = None
        self._lock = threading.RLock()

        # Real-time pattern matcher. Imported lazily so this module
        # stays importable in environments where lorahub.api isn't on
        # the path (CLI-only invocations, doctor checks). The watcher
        # only emits when a known failure-mode regex hits, so attaching
        # it unconditionally is cheap.
        from lorahub.api.streaming_diagnostics import (  # noqa: PLC0415
            StreamingDiagnosticWatcher,
        )

        self._diag_watcher = StreamingDiagnosticWatcher(
            on_event=self._safe_emit,
            job_id=job_id,
        )

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                msg = "runner already started"
                raise RuntimeError(msg)

            self._workspace.mkdir(parents=True, exist_ok=True)
            full_env = {**os.environ, **(self._env or {})}
            # Force the Python child to write stdout/stderr as UTF-8.
            #
            # On Windows the default child interpreter encodes via the
            # system ANSI codepage (cp936 in zh-CN, cp1252 in en-US,
            # ...). We always read the pipe back as UTF-8, so a tqdm
            # bar like ``Resizing: |█████| 1/1`` shows up as ``����``
            # the moment the child emits a non-ASCII byte. PEP 540
            # UTF-8 mode + PYTHONIOENCODING flips the child to plain
            # UTF-8 unconditionally, matching what hatch / tox / uv /
            # pdm all do for the same reason. ``setdefault`` lets
            # callers still pin a specific encoding (e.g. for tests).
            full_env.setdefault("PYTHONIOENCODING", "utf-8")
            full_env.setdefault("PYTHONUTF8", "1")
            # Strip stale Visual Studio env vars that confuse triton's
            # MSVC discovery on Windows.
            #
            # Symptom: when ``torch.compile`` triggers Inductor codegen
            # and triton tries to invoke ``cl.exe`` for the kernel JIT,
            # ``triton/windows_utils.py::find_msvc_env`` reads
            # ``VCINSTALLDIR`` and ``VCToolsVersion`` from os.environ.
            # Many Windows boxes have a half-finished VS install that
            # left ``VCINSTALLDIR`` set but ``VCToolsVersion`` blank;
            # triton then does ``Path(...) / None`` and crashes the
            # whole training run with:
            #
            #     TypeError: unsupported operand type(s) for /:
            #     'WindowsPath' and 'NoneType'
            #
            # Clearing both lets triton fall through to its vswhere /
            # PATH-based discovery (or fail with a *clear* "MSVC not
            # found" error), which is what users without VS Build
            # Tools should see anyway. If a real VS install is
            # actually wanted, callers can re-export the pair via
            # ``self._env`` — that takes precedence below.
            if sys.platform == "win32":
                stale = full_env.get("VCToolsVersion")
                if stale is None or not stale.strip():
                    full_env.pop("VCINSTALLDIR", None)
                    full_env.pop("VCToolsVersion", None)
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            start_new_session = sys.platform != "win32"

            self._started_at = time.time()
            self._proc = subprocess.Popen(
                self._argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(self._cwd) if self._cwd else None,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=full_env,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            # Log the actual argv we spawned so post-mortems can answer
            # "which interpreter ran my job?" without re-running the job.
            self._safe_emit(
                TrainingEvent(
                    type=EventType.log,
                    payload={
                        "level": "info",
                        "source": "runner",
                        "message": "spawn: " + " ".join(self._argv),
                    },
                    job_id=self._job_id,
                )
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
                name=f"{self._thread_label}-{source}",
                daemon=True,
            )
            t.start()
            self._pump_threads.append(t)

    def _spawn_reaper(self) -> None:
        """Background thread that waits for the child and emits the `done` event.

        Without this, callers that don't invoke `wait()` (e.g. an HTTP API
        that fires-and-forgets) would never see job completion.
        """
        thread = threading.Thread(
            target=self._reap,
            name=f"{self._thread_label}-reaper",
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
                # Hand every raw line to the diagnostic watcher *first*
                # so that even lines the backend's parser would discard
                # (Python tracebacks, library warnings) still get a
                # shot at matching a failure-mode regex.
                self._diag_watcher.feed(raw, source=source)
                event = self._parse_line(raw, job_id=self._job_id)
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
        """Stop the training subprocess **and all of its descendants**.

        anima / kohya / diffusion-pipe trainers can fork further workers
        (accelerate dataloader workers, deepspeed launchers, torch.compile
        background processes). Killing only ``self._proc`` leaves those
        children orphaned with ``PPid=1`` — they keep holding multi-GB GPU
        allocations until the box reboots. We sidestep that by signalling
        the entire process group (POSIX) or process tree (Windows).
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            if graceful:
                self._signal_group_graceful()
                try:
                    self._proc.wait(timeout=timeout)
                    return
                except subprocess.TimeoutExpired:
                    pass
            self._signal_group_terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._signal_group_kill()
                # Reap so wait() returns even if the kernel hasn't fully
                # cleaned up the descendants yet — the immediate child is
                # what we own, and SIGKILL on it always works.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=5.0)

    def _signal_group_graceful(self) -> None:
        assert self._proc is not None
        if sys.platform == "win32":
            # CTRL_BREAK_EVENT is delivered to every process in the
            # CREATE_NEW_PROCESS_GROUP we created at spawn time.
            self._proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # start_new_session=True at spawn means the child became its
            # own session leader (pgid == pid). killpg fans the signal
            # out to every descendant in that group.
            self._killpg(signal.SIGINT)

    def _signal_group_terminate(self) -> None:
        assert self._proc is not None
        if sys.platform == "win32":
            # Windows has no group SIGTERM equivalent; rely on the tree
            # walker (taskkill /T) to wipe descendants.
            self._taskkill(force=False)
        else:
            self._killpg(signal.SIGTERM)

    def _signal_group_kill(self) -> None:
        assert self._proc is not None
        if sys.platform == "win32":
            self._taskkill(force=True)
        else:
            self._killpg(signal.SIGKILL)

    def _killpg(self, sig: int) -> None:
        """POSIX: signal the spawned child's process group.

        Falls back to ``self._proc.send_signal`` if the pgid lookup races
        a process exit — that scenario is benign (the child already died).
        """
        assert self._proc is not None
        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass

    def _taskkill(self, *, force: bool) -> None:
        """Windows: walk the process tree via ``taskkill /T``.

        Subprocess.terminate on Windows only signals the immediate child,
        so accelerate / deepspeed grandchildren survive a Ctrl-C cancel.
        ``taskkill /T`` recursively walks the parent-child tree and
        terminates every descendant, which is what we actually want.
        """
        assert self._proc is not None
        args = ["taskkill", "/PID", str(self._proc.pid), "/T"]
        if force:
            args.append("/F")
        try:
            subprocess.run(args, capture_output=True, check=False, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Last-ditch: at least try to flag the immediate child.
            self._proc.terminate() if not force else self._proc.kill()


__all__ = [
    "EventListener",
    "LineParser",
    "RunResult",
    "SubprocessRunner",
]
