"""In-app terminal: stream a single command's output via SSE.

Scope is intentionally narrow:

* The request lands on a path-resolved backend (``kohya``, ``diffusion-pipe``,
  ``anima_lora``) so the backend interpreter + repo dir are server-side facts;
  the client never gets to influence either.
* By default the command's argv[0] must be one of ``pip``, ``uv``, ``python``,
  with ``pip`` rerouted to the venv's interpreter as either
  ``<venv-python> -m pip ...`` (when the venv ships pip) or
  ``uv pip --python <venv-python> ...`` (uv-managed venvs without pip).
  Bare ``python`` is also redirected at the venv interpreter. A
  ``terminal_unrestricted`` flag in Settings opens the door to anything
  else, but only if the user explicitly flips it.
* We stream stdout/stderr line-by-line via Server-Sent Events. WebSocket
  was tempting but SSE is enough for unidirectional output and removes a
  whole reconnect-state machine on the client.
* No PTY, no interactive prompts. Each request runs a single command to
  completion and returns; long-running shells are out of scope (and would
  need PTY+winpty on Windows, which is a different project).

The companion module :mod:`lorahub.api.terminal_runner` owns the actual
subprocess plumbing so the route handler stays thin and unit-testable.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.settings import VALID_BACKEND_IDS
from lorahub.api.terminal_runner import (
    TerminalDenied,
    TerminalSession,
    resolve_backend_session,
    stream_command,
)

router = APIRouter(prefix="/api/terminal")


# --------------------------------------------------------------------------- #
# /sessions — describe each backend's terminal-ready environment
# --------------------------------------------------------------------------- #


class TerminalEnvironment(BaseModel):
    """A snapshot of where a terminal command for ``backend_id`` would run."""

    backend_id: str
    name: str
    """User-facing display label, e.g. "Kohya (sd-scripts)"."""

    repo_path: str
    """Working directory the subprocess will be spawned in."""

    python_path: str | None
    """Resolved interpreter — None if the backend isn't installed yet."""

    venv_dir: str | None
    """Parent of the python binary (its venv root). None when no venv."""

    venv_detected: bool
    ready: bool
    """``ready`` here mirrors the backend probe: repo + interpreter + reqs OK."""

    prompt: str
    """A short string the UI uses as the synthetic shell prompt."""


class TerminalSessionsResponse(BaseModel):
    backends: list[TerminalEnvironment]
    default_backend: str
    unrestricted: bool
    """Mirrors Settings.terminal_unrestricted so the client can show a chip."""

    command_timeout_s: int


@router.get("/sessions", response_model=TerminalSessionsResponse)
def list_sessions() -> TerminalSessionsResponse:
    """Return the venv environment summary for every known backend."""
    settings = app_module._settings_store.load()
    out: list[TerminalEnvironment] = []
    for backend_id in VALID_BACKEND_IDS:
        try:
            session = resolve_backend_session(backend_id, settings)
        except TerminalDenied as exc:
            # Pretend the env entry exists but is "not ready"; surface the
            # error message in the prompt so the UI can show why.
            out.append(
                TerminalEnvironment(
                    backend_id=backend_id,
                    name=backend_id,
                    repo_path="",
                    python_path=None,
                    venv_dir=None,
                    venv_detected=False,
                    ready=False,
                    prompt=f"({backend_id}) [unavailable: {exc}]",
                )
            )
            continue
        out.append(_session_to_env(session))
    return TerminalSessionsResponse(
        backends=out,
        default_backend=settings.default_backend,
        unrestricted=settings.terminal_unrestricted,
        command_timeout_s=int(settings.terminal_command_timeout_s),
    )


def _session_to_env(s: TerminalSession) -> TerminalEnvironment:
    return TerminalEnvironment(
        backend_id=s.backend_id,
        name=s.display_name,
        repo_path=str(s.repo_path),
        python_path=str(s.python_path) if s.python_path else None,
        venv_dir=str(s.venv_dir) if s.venv_dir else None,
        venv_detected=s.python_path is not None,
        ready=s.ready,
        prompt=s.prompt,
    )


# --------------------------------------------------------------------------- #
# /exec — stream a command's stdout/stderr line by line
# --------------------------------------------------------------------------- #


class TerminalExecRequest(BaseModel):
    backend_id: str = Field(..., description="One of: kohya, diffusion-pipe, anima_lora")
    command: str = Field(..., description="Shell-style command line")


class TerminalEvent(BaseModel):
    """Shape of every SSE frame we emit (encoded as JSON in `data:`)."""

    type: Literal["start", "stdout", "stderr", "exit", "error"]
    data: str | None = None
    code: int | None = None
    cwd: str | None = None
    argv: list[str] | None = None


def _sse(event: dict[str, Any]) -> str:
    """Serialize one event into the SSE wire format."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/exec")
def exec_command(req: TerminalExecRequest) -> StreamingResponse:
    """Stream a command in the resolved backend's venv environment.

    Wire format: each line of ``data:`` is a JSON object of shape
    ``{"type": "stdout"|..., "data": "..."}``. ``type=start`` fires once
    before any output (carrying the resolved ``argv`` + ``cwd``), and
    ``type=exit`` fires last with an integer ``code``.
    """
    settings = app_module._settings_store.load()
    if req.backend_id not in VALID_BACKEND_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"backend_id must be one of {sorted(VALID_BACKEND_IDS)}",
        )

    try:
        session = resolve_backend_session(req.backend_id, settings)
    except TerminalDenied as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not session.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Backend {req.backend_id} is not installed yet — set up the "
                "venv from the 设置 → 后端管理 page first."
            ),
        )

    # Parse + validate the command upfront so we can reject 422 before
    # opening the SSE stream (errors inside an SSE stream are awkward to
    # surface in DevTools).
    try:
        argv = shlex.split(req.command, posix=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse command: {exc}") from exc
    if not argv:
        raise HTTPException(status_code=422, detail="Empty command.")

    try:
        argv = _enforce_command_policy(
            argv,
            python_path=session.python_path,
            unrestricted=settings.terminal_unrestricted,
        )
    except TerminalDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    timeout_s = max(5, int(settings.terminal_command_timeout_s))
    cwd = session.repo_path

    def event_stream():
        # Announce the resolved process up-front so the UI can render the
        # canonical `(backend) cwd$ argv` line before any output lands.
        yield _sse(
            {
                "type": "start",
                "argv": argv,
                "cwd": str(cwd),
            }
        )
        try:
            for chunk in stream_command(
                argv=argv,
                cwd=cwd,
                env=session.process_env(),
                timeout_s=timeout_s,
            ):
                yield _sse(chunk)
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "data": str(exc)})
            yield _sse({"type": "exit", "code": -1})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering so chunks land in real time when the
            # API is fronted by nginx / Vite proxy.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------- #
# Command policy
# --------------------------------------------------------------------------- #

# Top-level entry points the user is allowed to invoke (and aliases).
# `pip` / `python` get routed at the venv interpreter; `uv` runs as-is
# but with `--python <venv-python>` injected when missing so it always
# targets the right environment.
_ALLOWED_TOP_LEVEL = {"pip", "pip3", "python", "python3", "py", "uv"}
_PIP_ALIASES = {"pip", "pip3"}
_PYTHON_ALIASES = {"python", "python3", "py"}


def _enforce_command_policy(
    argv: list[str], *, python_path: Path | None, unrestricted: bool
) -> list[str]:
    """Validate + rewrite the parsed argv per the workbench policy.

    In restricted mode (default), the only entry points the user can call
    are ``pip``, ``uv`` and ``python`` (plus their aliases). In every
    case we redirect the command at the resolved venv interpreter so it
    actually targets the backend's environment, not whatever happens to
    be on ``PATH``.

    Pip rewriting strategy: many of our backends ship uv-managed venvs
    that do **not** include pip. So before we settle on
    ``<venv-python> -m pip ...`` we probe the venv for pip; if it's
    missing we fall back to ``uv pip --python <venv-python> ...`` which
    works even on a barebones ``uv venv`` install. ``python`` (no -m
    pip) goes through unchanged at the venv interpreter.

    In unrestricted mode we leave argv exactly as the user typed it.
    """
    head = argv[0]
    if not unrestricted:
        base = head.split("/")[-1].split("\\")[-1]
        if base not in _ALLOWED_TOP_LEVEL:
            raise TerminalDenied(
                f"命令 {head!r} 不在白名单中（pip / uv / python）。"
                "如需自由模式，请到 设置 → 终端 中开启。"
            )
        head = base

    if head in _PYTHON_ALIASES:
        if python_path is None:
            raise TerminalDenied(
                "未找到该后端的 venv python 解释器，无法运行 python 命令。"
            )
        return [str(python_path), *argv[1:]]

    if head in _PIP_ALIASES:
        if python_path is None:
            raise TerminalDenied(
                "未找到该后端的 venv python 解释器，无法运行 pip 命令。"
            )
        return _route_pip(python_path, argv[1:])

    if head == "uv":
        return _route_uv(python_path, argv[1:])

    # Unrestricted fall-through.
    return argv


def _route_pip(python_path: Path, pip_args: list[str]) -> list[str]:
    """Route a `pip ...` invocation at the resolved venv interpreter.

    First attempts ``<venv-python> -m pip``. If pip is not installed in
    the venv (typical for uv-managed envs) and a ``uv`` binary is on
    PATH, swap to ``uv pip <subcommand> --python <venv-python> ...``.
    Note ``--python`` is a flag on each ``uv pip`` *subcommand*, not on
    ``uv pip`` itself, so it has to come after argv[0].
    """
    if _venv_has_pip(python_path):
        return [str(python_path), "-m", "pip", *pip_args]
    uv_binary = _find_uv()
    if uv_binary is None:
        raise TerminalDenied(
            "该 venv 未安装 pip，且找不到 uv。请先在该 venv 中执行 "
            "`python -m ensurepip` 或安装 uv。"
        )
    if not pip_args:
        # Bare `pip` with no subcommand — uv refuses; surface uv's own
        # help to keep behaviour predictable.
        return [uv_binary, "pip", "--help"]
    sub, rest = pip_args[0], pip_args[1:]
    return [uv_binary, "pip", sub, "--python", str(python_path), *rest]


def _route_uv(python_path: Path | None, uv_args: list[str]) -> list[str]:
    """Inject ``--python <venv>`` for ``uv pip <subcommand>`` invocations.

    Without this, ``uv pip install foo`` runs against whatever interpreter
    uv discovers (often the system one), which silently misses the user's
    venv. We only inject when the user types ``uv pip <sub> ...`` (other
    uv subcommands like ``uv venv`` / ``uv lock`` are left alone), and
    place ``--python`` after the subcommand because that's where uv
    actually expects it.
    """
    uv_binary = _find_uv()
    if uv_binary is None:
        raise TerminalDenied("找不到 uv 可执行文件。")
    if (
        python_path is not None
        and len(uv_args) >= 2
        and uv_args[0] == "pip"
        and "--python" not in uv_args
        and "-p" not in uv_args
    ):
        sub, rest = uv_args[1], uv_args[2:]
        return [uv_binary, "pip", sub, "--python", str(python_path), *rest]
    return [uv_binary, *uv_args]


def _find_uv() -> str | None:
    """Locate the uv executable, preferring the project-local copy.

    Hub's installer drops uv at ``.tools/uv/uv.exe`` (Windows) or
    ``.tools/uv/uv``. Prefer that over PATH so a globally-installed
    older uv doesn't shadow the bundled one. Falls back to ``shutil.which``
    when the bundled copy isn't there (dev checkouts, fresh installs).
    """
    name = "uv.exe" if os.name == "nt" else "uv"
    candidate = Path.cwd() / ".tools" / "uv" / name
    if candidate.is_file():
        return str(candidate)
    return shutil.which("uv")


def _venv_has_pip(python_path: Path) -> bool:
    """Return True iff ``<venv-python> -m pip --version`` succeeds.

    Cheap (~50ms) probe with a hard timeout. We deliberately don't cache:
    the user might run ``ensurepip`` between two commands and expect the
    next pip invocation to switch over without restarting the server.
    """
    try:
        result = subprocess.run(  # noqa: S603
            [str(python_path), "-m", "pip", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
