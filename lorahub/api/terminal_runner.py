"""Subprocess plumbing for the in-app terminal.

Split from :mod:`lorahub.api.routers.terminal` so the route handler stays
small and testable. The router is responsible for HTTP shape; everything
about *resolving the backend's venv*, *building the env block*, and
*streaming the subprocess's stdout/stderr line by line* lives here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from lorahub.api.settings import Settings
from lorahub.core.backends._common.bootstrap import venv_python

_log = logging.getLogger(__name__)


class TerminalDenied(Exception):
    """Raised when a backend / command is not eligible for terminal use."""


# --------------------------------------------------------------------------- #
# Backend → terminal session resolver
# --------------------------------------------------------------------------- #

_BACKEND_DISPLAY = {
    "kohya": "Kohya (sd-scripts)",
    "diffusion-pipe": "Diffusion-pipe",
    "anima_lora": "Anima LoRA",
}


@dataclass(slots=True)
class TerminalSession:
    backend_id: str
    display_name: str
    repo_path: Path
    python_path: Path | None
    venv_dir: Path | None
    ready: bool
    """True iff the backend probe says everything's installed."""

    @property
    def prompt(self) -> str:
        """Synthetic shell prompt the UI splices before each command line."""
        # Mirror the ``(env) path$`` convention so the page reads as a real
        # terminal. Use forward slashes for prompt readability — the
        # subprocess still gets the platform-correct cwd.
        normalised = str(self.repo_path).replace("\\", "/")
        return f"({self.backend_id}) {normalised}$"

    def process_env(self) -> dict[str, str]:
        """Build an env mapping the subprocess inherits.

        Keeps the parent process's environment (so DNS / TLS / CA bundles
        keep working) but injects:

        * ``VIRTUAL_ENV`` pointing at the venv root so tools that check
          for it (e.g. ``uv``) target the right environment.
        * ``PATH`` with the venv's ``Scripts/`` (or ``bin/``) prepended.
        * ``PYTHONNOUSERSITE=1`` so ``python -m pip install --user`` can't
          accidentally drop wheels under ``%APPDATA%/Python``.
        * ``PIP_DISABLE_PIP_VERSION_CHECK=1`` to suppress the noisy
          version banner at the top of every pip command.
        * ``LORAHUB_TERMINAL=1`` as a hint for any tools that want to
          adjust their output (none today, but cheap to set).
        """
        env = os.environ.copy()
        if self.venv_dir is not None:
            env["VIRTUAL_ENV"] = str(self.venv_dir)
            scripts_dir = self.venv_dir / ("Scripts" if os.name == "nt" else "bin")
            sep = ";" if os.name == "nt" else ":"
            env["PATH"] = f"{scripts_dir}{sep}{env.get('PATH', '')}"
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["LORAHUB_TERMINAL"] = "1"
        # Force unbuffered Python output so streamed lines show up live —
        # without this pip can sit silent for ages while it resolves.
        env["PYTHONUNBUFFERED"] = "1"
        return env


def resolve_backend_session(backend_id: str, settings: Settings) -> TerminalSession:
    """Map a backend id to a :class:`TerminalSession`.

    Pulls the repo path + interpreter the same way the backend probes
    do, but skips the requirements-completeness check — terminal use
    cases include "fix the missing dep" so we'd be locking the user
    out of the very tool they want to use to fix it.
    """
    if backend_id == "kohya":
        return _resolve_kohya(settings)
    if backend_id == "diffusion-pipe":
        return _resolve_diffusion_pipe(settings)
    if backend_id == "anima_lora":
        return _resolve_anima_lora(settings)
    raise TerminalDenied(f"unknown backend {backend_id!r}")


def _resolve_kohya(settings: Settings) -> TerminalSession:
    from lorahub.core.backends.kohya.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_SD_SCRIPTS,
        default_sd_scripts_path,
    )

    sd_raw = (
        os.environ.get(_ENV_SD_SCRIPTS)
        or settings.sd_scripts_path
        or str(default_sd_scripts_path())
    )
    repo = Path(sd_raw).expanduser()
    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.python_executable
        or (str(venv_python(repo)) if venv_python(repo) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    if py_path is not None and not py_path.is_file():
        py_path = None
    venv_dir = py_path.parent.parent if py_path is not None else None
    return TerminalSession(
        backend_id="kohya",
        display_name=_BACKEND_DISPLAY["kohya"],
        repo_path=repo,
        python_path=py_path,
        venv_dir=venv_dir,
        ready=repo.is_dir() and py_path is not None,
    )


def _resolve_diffusion_pipe(settings: Settings) -> TerminalSession:
    from lorahub.core.backends.diffusion_pipe.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        default_repo_path,
    )

    raw = (
        os.environ.get(_ENV_REPO)
        or settings.diffusion_pipe_repo_path
        or str(default_repo_path())
    )
    repo = Path(raw).expanduser()
    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.diffusion_pipe_python
        or (str(venv_python(repo)) if venv_python(repo) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    if py_path is not None and not py_path.is_file():
        py_path = None
    venv_dir = py_path.parent.parent if py_path is not None else None
    return TerminalSession(
        backend_id="diffusion-pipe",
        display_name=_BACKEND_DISPLAY["diffusion-pipe"],
        repo_path=repo,
        python_path=py_path,
        venv_dir=venv_dir,
        ready=repo.is_dir() and py_path is not None,
    )


def _resolve_anima_lora(settings: Settings) -> TerminalSession:
    from lorahub.core.backends.anima_lora.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        default_repo_path,
    )

    raw = (
        os.environ.get(_ENV_REPO)
        or settings.anima_lora_repo_path
        or str(default_repo_path())
    )
    repo = Path(raw).expanduser()
    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.anima_lora_python
        or (str(venv_python(repo)) if venv_python(repo) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    if py_path is not None and not py_path.is_file():
        py_path = None
    venv_dir = py_path.parent.parent if py_path is not None else None
    return TerminalSession(
        backend_id="anima_lora",
        display_name=_BACKEND_DISPLAY["anima_lora"],
        repo_path=repo,
        python_path=py_path,
        venv_dir=venv_dir,
        ready=repo.is_dir() and py_path is not None,
    )


# --------------------------------------------------------------------------- #
# Subprocess streamer
# --------------------------------------------------------------------------- #


def stream_command(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> Iterator[dict[str, Any]]:
    """Spawn ``argv`` and yield SSE-shaped event dicts.

    Output streams concurrently from stdout and stderr — without two
    threads, ``readline()`` would block on whichever pipe is quiet and
    starve the other. Each yielded value is a dict the route serializes
    straight into the SSE frame.
    """
    creationflags = 0
    if os.name == "nt":
        # CREATE_NO_WINDOW so spawning ``python -m pip`` from the API
        # process doesn't briefly flash a console window.
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    _log.info(
        "terminal exec: argv=%s cwd=%s python=%s",
        argv,
        cwd,
        env.get("VIRTUAL_ENV", "(none)"),
    )

    try:
        proc = subprocess.Popen(  # noqa: S603
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        msg = f"无法启动子进程: {exc}\nargv = {argv}\ncwd = {cwd}"
        _log.warning("terminal exec FileNotFoundError: %s", msg)
        yield {"type": "error", "data": msg}
        yield {"type": "exit", "code": -1}
        return
    except OSError as exc:
        msg = f"OSError 启动子进程: {exc}\nargv = {argv}\ncwd = {cwd}"
        _log.warning("terminal exec OSError: %s", msg)
        yield {"type": "error", "data": msg}
        yield {"type": "exit", "code": -1}
        return

    queue: Queue[tuple[str, str | None]] = Queue()

    def pump(stream, label: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                queue.put((label, line))
        finally:
            queue.put((label, None))

    threads = [
        threading.Thread(
            target=pump,
            args=(proc.stdout, "stdout"),
            daemon=True,
            name="terminal-stdout",
        ),
        threading.Thread(
            target=pump,
            args=(proc.stderr, "stderr"),
            daemon=True,
            name="terminal-stderr",
        ),
    ]
    for t in threads:
        t.start()

    closed = 0
    deadline = _monotonic() + timeout_s
    timed_out = False
    while closed < 2:
        try:
            label, data = queue.get(timeout=0.5)
        except Empty:
            if _monotonic() > deadline:
                timed_out = True
                break
            continue
        if data is None:
            closed += 1
            continue
        # Strip the trailing newline so the client controls line breaks
        # itself. Ship the rest verbatim — pip / uv emit ANSI for colour
        # which the UI decides to render or strip.
        text = data.rstrip("\r\n")
        yield {"type": label, "data": text}

    if timed_out:
        try:
            proc.kill()
        finally:
            yield {
                "type": "error",
                "data": f"命令运行超过 {timeout_s} 秒，已强制结束。",
            }

    rc = proc.wait()
    _log.info("terminal exec finished: rc=%d argv=%s", rc, argv)
    yield {"type": "exit", "code": int(rc)}


def _monotonic() -> float:
    """Indirected for testability; ``time.monotonic`` is fine in prod."""
    import time as _time  # noqa: PLC0415

    return _time.monotonic()


# Re-export for typing / import-by-name.
__all__ = [
    "TerminalDenied",
    "TerminalSession",
    "resolve_backend_session",
    "stream_command",
]
