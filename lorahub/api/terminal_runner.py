"""Subprocess plumbing for the in-app terminal.

Split from :mod:`lorahub.api.routers.terminal` so the route handler stays
small and testable. The router is responsible for HTTP shape; everything
about *resolving the backend's venv*, *building the env block*, and
*streaming the subprocess's stdout/stderr line by line* lives here.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, TextIO, cast

from lorahub.api.settings import Settings
from lorahub.core.backends._common.bootstrap import venv_python
from lorahub.core.paths import project_root
from lorahub.core.redaction import redact_argv, redact_command_text

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
    "ai_toolkit": "AI Toolkit",
    # Synthetic id — the LoraHub server itself, the venv that runs
    # FastAPI. Useful for installing optional packages (wandb, ruff,
    # huggingface_hub) into the API environment without leaving the
    # in-app terminal. See ``_resolve_lorahub`` for the wiring.
    "lorahub": "LoraHub (本程序)",
}


# IDs accepted by the terminal that are NOT real training backends.
# Kept separate so we don't pollute ``backends.registry`` for a
# UI-only concept.
_TERMINAL_ONLY_IDS: frozenset[str] = frozenset({"lorahub"})


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
        * ``PATH`` with the venv's ``Scripts/`` (or ``bin/``) prepended,
          plus optional ``./tools/`` and ``./.lorahub/tools/`` directories
          under the repo root so user-dropped binaries (ffmpeg, custom
          annotators, ...) become available without per-process shims.
        * ``PYTHONNOUSERSITE=1`` so ``python -m pip install --user`` can't
          accidentally drop wheels under ``%APPDATA%/Python``.
        * ``PIP_DISABLE_PIP_VERSION_CHECK=1`` to suppress the noisy
          version banner at the top of every pip command.
        * ``LORAHUB_TERMINAL=1`` as a hint for any tools that want to
          adjust their output (none today, but cheap to set).
        """
        env = os.environ.copy()
        sep = ";" if os.name == "nt" else ":"
        path_parts: list[str] = []
        # Venv bin dir wins (highest priority on PATH).
        if self.venv_dir is not None:
            env["VIRTUAL_ENV"] = str(self.venv_dir)
            scripts_dir = self.venv_dir / ("Scripts" if os.name == "nt" else "bin")
            path_parts.append(str(scripts_dir))
        # Repo-local tools dirs — prepended next so they shadow system
        # tools, but lose to the venv's own scripts. Both shapes are
        # supported because the LoraHub installer drops uv into
        # ``.lorahub/`` and users sometimes hand-curate ``./tools/``.
        for tools_dir in (
            self.repo_path / "tools",
            self.repo_path / ".lorahub" / "tools",
        ):
            if tools_dir.is_dir():
                path_parts.append(str(tools_dir))
        if path_parts:
            env["PATH"] = sep.join([*path_parts, env.get("PATH", "")])
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["LORAHUB_TERMINAL"] = "1"
        if self.backend_id == "ai_toolkit":
            env.setdefault("MODELS_PATH", str(project_root() / "models"))
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
    if backend_id == "ai_toolkit":
        return _resolve_ai_toolkit(settings)
    if backend_id == "lorahub":
        return _resolve_lorahub(settings)
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


def _resolve_ai_toolkit(settings: Settings) -> TerminalSession:
    from lorahub.core.backends.ai_toolkit.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        default_repo_path,
    )

    raw = (
        os.environ.get(_ENV_REPO)
        or settings.ai_toolkit_repo_path
        or str(default_repo_path())
    )
    repo = Path(raw).expanduser()
    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.ai_toolkit_python
        or (str(venv_python(repo)) if venv_python(repo) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    if py_path is not None and not py_path.is_file():
        py_path = None
    venv_dir = py_path.parent.parent if py_path is not None else None
    return TerminalSession(
        backend_id="ai_toolkit",
        display_name=_BACKEND_DISPLAY["ai_toolkit"],
        repo_path=repo,
        python_path=py_path,
        venv_dir=venv_dir,
        ready=repo.is_dir() and py_path is not None,
    )


def _resolve_lorahub(settings: Settings) -> TerminalSession:  # noqa: ARG001
    """Terminal session for the LoraHub server's own venv.

    Resolution priority (first hit wins):

      1. ``$LORAHUB_PYTHON`` — user override, takes precedence
         over everything. Useful when the API was launched from one
         interpreter but the user wants the in-app terminal to drive
         a different one.
      2. **Project-root venv**: ``./.venv/`` or ``./venv/``. This is
         the convention LoraHub installs follow (``uv venv`` /
         ``python -m venv .venv``), so when the user does
         ``pip install foo`` here it lands where everything else
         expects it. Picks .venv before venv to match modern uv /
         poetry conventions.
      3. **Bundled embedded python**: ``./.lorahub/python/.../python``
         — populated by the LoraHub installer on Windows.
      4. **Fallback**: ``sys.executable`` — whatever launched the
         API server. Always works but may be base conda / system
         python which isn't where the user expects packages to land.

    The session also prepends ``./tools/`` and ``./.lorahub/tools/``
    to PATH (when they exist) so user-dropped binaries (ffmpeg, custom
    annotators, ...) become available without a per-process shim.
    """
    import sys  # noqa: PLC0415

    repo = Path(__file__).resolve().parent.parent.parent

    # 1. Env override.
    py_path: Path | None = None
    venv_root: Path | None = None
    env_override = os.environ.get("LORAHUB_PYTHON", "").strip()
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if candidate.is_file():
            py_path = candidate

    # 2. Project-root venv. ``venv_python()`` already handles the
    # platform / dirname matrix (Windows Scripts/.venv, POSIX bin/venv,
    # both ordered to prefer ``.venv`` last). Reuse it instead of
    # re-rolling our own four-way path probe.
    if py_path is None:
        venv_candidate = venv_python(repo)
        if venv_candidate is not None:
            py_path = venv_candidate
            # Walk back up to the venv root for VIRTUAL_ENV. The helper
            # returns the python binary under either Scripts/ or bin/,
            # so the venv root is python.parent.parent.
            venv_root = venv_candidate.parent.parent

    # 3. Bundled embedded python (Windows installer drops it here).
    if py_path is None:
        bundled_root = repo / ".lorahub" / "python"
        if bundled_root.is_dir():
            for sub in bundled_root.iterdir():
                if not sub.is_dir():
                    continue
                bundled_py = (
                    sub / ("python.exe" if os.name == "nt" else "bin/python")
                )
                if bundled_py.is_file():
                    py_path = bundled_py.resolve()
                    break

    # 4. Fallback to whatever launched the API.
    if py_path is None:
        py_path = Path(sys.executable).resolve()

    if not py_path.is_file():
        return TerminalSession(
            backend_id="lorahub",
            display_name=_BACKEND_DISPLAY["lorahub"],
            repo_path=repo,
            python_path=None,
            venv_dir=None,
            ready=False,
        )

    # Re-detect venv_root if we didn't already pin one (env override or
    # bundled python). pyvenv.cfg sits at the venv root, one level
    # above bin/python on POSIX and two above Scripts/python.exe on
    # Windows. We probe both shapes to stay platform-agnostic.
    if venv_root is None:
        for candidate in (py_path.parent.parent, py_path.parent):
            if (candidate / "pyvenv.cfg").is_file():
                venv_root = candidate
                break

    return TerminalSession(
        backend_id="lorahub",
        display_name=_BACKEND_DISPLAY["lorahub"],
        repo_path=repo,
        python_path=py_path,
        venv_dir=venv_root,
        ready=True,
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
    shell_cmd: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Spawn ``argv`` and yield SSE-shaped event dicts.

    Output streams concurrently from stdout and stderr — without two
    threads, ``readline()`` would block on whichever pipe is quiet and
    starve the other. Each yielded value is a dict the route serializes
    straight into the SSE frame.

    When ``shell_cmd`` is set, the subprocess is started with
    ``shell=True`` and the raw string as its first arg. This is the
    unrestricted-mode path: the user's command goes straight to bash
    / cmd so shell syntax (``$(...)``, ``|``, ``&&``, redirects, glob
    expansion) all works. ``argv`` is still kept for the SSE start
    event so the UI can echo what's about to run. Restricted mode
    leaves ``shell_cmd`` ``None`` and uses the safer pure-argv path.
    """
    creationflags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        # CREATE_NO_WINDOW so spawning ``python -m pip`` from the API
        # process doesn't briefly flash a console window.
        creationflags = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    display_argv = redact_argv(argv)

    _log.info(
        "terminal exec: argv=%s cwd=%s python=%s shell=%s",
        display_argv,
        cwd,
        env.get("VIRTUAL_ENV", "(none)"),
        shell_cmd is not None,
    )

    try:
        if shell_cmd is not None:
            # On Windows, subprocess with shell=True spawns cmd.exe;
            # on POSIX it spawns /bin/sh. We pass the raw string so
            # the shell handles quoting / globbing / pipes itself.
            proc = subprocess.Popen(  # noqa: S602
                shell_cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=True,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        else:
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
                start_new_session=start_new_session,
            )
    except FileNotFoundError as exc:
        msg = (
            f"无法启动子进程: {redact_command_text(str(exc))}\n"
            f"argv = {display_argv}\ncwd = {cwd}"
        )
        _log.warning("terminal exec FileNotFoundError: %s", msg)
        yield {"type": "error", "data": msg}
        yield {"type": "exit", "code": -1}
        return
    except OSError as exc:
        msg = (
            f"OSError 启动子进程: {redact_command_text(str(exc))}\n"
            f"argv = {display_argv}\ncwd = {cwd}"
        )
        _log.warning("terminal exec OSError: %s", msg)
        yield {"type": "error", "data": msg}
        yield {"type": "exit", "code": -1}
        return

    queue: Queue[tuple[str, str | None]] = Queue()
    stdout = proc.stdout
    stderr = proc.stderr
    assert stdout is not None  # noqa: S101
    assert stderr is not None  # noqa: S101

    def pump(stream: TextIO, label: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                queue.put((label, line))
        finally:
            queue.put((label, None))

    threads = [
        threading.Thread(
            target=pump,
            args=(stdout, "stdout"),
            daemon=True,
            name="terminal-stdout",
        ),
        threading.Thread(
            target=pump,
            args=(stderr, "stderr"),
            daemon=True,
            name="terminal-stderr",
        ),
    ]
    for t in threads:
        t.start()

    try:
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
            # Keep terminal output exportable without persisting credentials
            # echoed by package managers or remote endpoints.
            text = redact_command_text(data.rstrip("\r\n"))
            yield {"type": label, "data": text}

        if timed_out:
            _terminate_process_tree(proc)
            yield {
                "type": "error",
                "data": f"命令运行超过 {timeout_s} 秒，已强制结束。",
            }

        rc = proc.wait()
        _log.info("terminal exec finished: rc=%d argv=%s", rc, display_argv)
        yield {"type": "exit", "code": int(rc)}
    finally:
        # Closing the browser aborts the streaming response. Ensure the
        # subprocess and any package-manager children do not survive it.
        if proc.poll() is None:
            _terminate_process_tree(proc)


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    """Best-effort cleanup for a timed-out or disconnected terminal command."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0 and proc.poll() is None:
                proc.kill()
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        return

    get_process_group = cast(Callable[[int], int], getattr(os, "getpgid"))
    kill_process_group = cast(
        Callable[[int, int], None], getattr(os, "killpg")
    )
    try:
        kill_process_group(get_process_group(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except ProcessLookupError:
        return
    except (PermissionError, subprocess.TimeoutExpired):
        try:
            sigkill = cast(int, getattr(signal, "SIGKILL", signal.SIGTERM))
            kill_process_group(get_process_group(proc.pid), sigkill)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _monotonic() -> float:
    """Indirected for testability; ``time.monotonic`` is fine in prod."""
    import time as _time  # noqa: PLC0415

    return _time.monotonic()


# Re-export for typing / import-by-name.
__all__ = [
    "TerminalDenied",
    "TerminalSession",
    "_TERMINAL_ONLY_IDS",
    "resolve_backend_session",
    "stream_command",
]
