"""In-app terminal: stream a single command's output via SSE.

Scope is intentionally narrow:

* The request lands on a path-resolved backend (``kohya``, ``diffusion-pipe``,
  ``anima_lora``) so the backend interpreter + repo dir are server-side facts;
  the client never gets to influence either.
* By default the command's argv[0] must be one of ``pip``, ``uv``, ``python``,
  with ``pip``/``uv`` further auto-rewritten so they invoke the venv's
  interpreter (``python -m pip ...``). A ``terminal_unrestricted`` flag in
  Settings opens the door to anything else, but only if the user explicitly
  flips it.
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
import shlex
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

# argv[0] -> a normaliser that decides what argv we actually run. When
# the user types ``pip install x`` we rewrite it to
# ``<venv-python> -m pip install x`` so it always targets the venv even
# if the OS PATH would otherwise pick up a different pip.
_PYTHON_TOOL_REWRITES = {
    "pip": ("python", "-m", "pip"),
    "pip3": ("python", "-m", "pip"),
    "python": ("python",),
    "python3": ("python",),
    "py": ("python",),
}

_ALLOWED_TOP_LEVEL = {"pip", "pip3", "python", "python3", "py", "uv"}


def _enforce_command_policy(
    argv: list[str], *, python_path: Path | None, unrestricted: bool
) -> list[str]:
    """Validate + rewrite the parsed argv per the workbench policy.

    In restricted mode (default), the only entry points the user can call
    are ``pip``, ``uv`` and ``python`` (plus their aliases). In every
    case we redirect ``pip`` and bare ``python`` invocations through the
    venv's interpreter so the command actually targets the backend's
    environment, not whatever happens to be on ``PATH``.

    ``uv`` runs as-is (uv resolves the venv via ``--python`` / ``VIRTUAL_ENV``;
    the latter is set by ``TerminalSession.process_env()``).

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

    rewrite = _PYTHON_TOOL_REWRITES.get(head)
    if rewrite is None:
        # Either uv (allowed verbatim) or, in unrestricted mode, anything
        # else — both fall through unchanged.
        return argv

    if python_path is None:
        raise TerminalDenied(
            "未找到该后端的 venv python 解释器，无法运行 pip/python 命令。"
        )
    rest = list(rewrite[1:]) + argv[1:]
    return [str(python_path), *rest]
