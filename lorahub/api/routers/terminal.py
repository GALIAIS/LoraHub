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
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.dataset_files import is_link_like
from lorahub.api.settings import VALID_BACKEND_IDS
from lorahub.api.terminal_runner import (
    _TERMINAL_ONLY_IDS,
    TerminalDenied,
    TerminalSession,
    resolve_backend_session,
    stream_command,
)
from lorahub.core.redaction import redact_argv

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
    """Return the venv environment summary for every known backend.

    The response also includes a synthetic ``lorahub`` entry for the
    API server's own venv — useful for installing optional packages
    (wandb, ruff, ...) into the LoraHub environment without leaving
    the in-app terminal.
    """
    settings = app_module._settings_store.load()
    out: list[TerminalEnvironment] = []
    # Real training backends first (kohya / dp / anima_lora), then the
    # LoraHub-self entry. Listing order matches the BackendPicker tab
    # order in the UI.
    for backend_id in (*sorted(VALID_BACKEND_IDS), *sorted(_TERMINAL_ONLY_IDS)):
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
    if req.backend_id not in VALID_BACKEND_IDS and req.backend_id not in _TERMINAL_ONLY_IDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"backend_id must be one of "
                f"{sorted(VALID_BACKEND_IDS | _TERMINAL_ONLY_IDS)}"
            ),
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

    # Unrestricted mode bypasses argv parsing entirely and hands the
    # raw command line to the platform shell, so shell syntax — pipes,
    # redirects, ``$(...)`` substitution, ``&&``, glob expansion — all
    # work as the user expects. Restricted mode keeps the safe
    # argv-only path with the venv-router policy applied.
    use_shell = settings.terminal_unrestricted
    cwd = session.repo_path
    argv: list[str]
    shell_cmd: str | None = None
    if use_shell:
        if not req.command.strip():
            raise HTTPException(status_code=422, detail="Empty command.")
        shell_cmd = req.command
        # argv is only used for the SSE start event so the UI can echo
        # the canonical "what's about to run" line — show the literal
        # shell invocation under the hood.
        if os.name == "nt":
            argv = ["cmd", "/c", req.command]
        else:
            argv = ["/bin/bash", "-lc", req.command]
    else:
        # Parse + validate the command upfront so we can reject 422
        # before opening the SSE stream (errors inside an SSE stream
        # are awkward to surface in DevTools).
        try:
            argv = shlex.split(req.command, posix=(os.name != "nt"))
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not parse command: {exc}",
            ) from exc
        if not argv:
            raise HTTPException(status_code=422, detail="Empty command.")
        try:
            argv = _enforce_command_policy(
                argv,
                python_path=session.python_path,
                unrestricted=False,
                cwd=cwd,
            )
        except TerminalDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    timeout_s = max(5, int(settings.terminal_command_timeout_s))
    display_argv = redact_argv(argv)

    def event_stream() -> Iterator[str]:
        # Announce the resolved process up-front so the UI can render the
        # canonical `(backend) cwd$ argv` line before any output lands.
        yield _sse(
            {
                "type": "start",
                "argv": display_argv,
                "cwd": str(cwd),
            }
        )
        try:
            for chunk in stream_command(
                argv=argv,
                cwd=cwd,
                env=session.process_env(),
                timeout_s=timeout_s,
                shell_cmd=shell_cmd,
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
# targets the right environment. Everything else in the list runs
# verbatim — these are read-only-ish diagnostic / inspection commands
# that are safe to expose without unrestricted mode.
_ALLOWED_TOP_LEVEL = {
    # Package / interpreter (rewritten through the venv resolver below)
    "pip", "pip3", "python", "python3", "py", "uv",
    # LoraHub self — picks up the in-tree CLI through PATH for read-only
    # diagnostics such as doctor/version.
    "lorahub",
    # Diagnostic / inspection commands. These pass through verbatim,
    # they don't get the venv-router rewrite.
    "which", "where", "whereis", "command", "type",
    "ls", "pwd",
    "cat", "head", "tail",
    "echo", "printf",
    "git",
    "nvidia-smi", "nvcc",
    "df", "du", "free", "uname",
    # Useful for sanity-checking model / cache locations.
    "find", "tree", "stat", "wc", "grep", "rg", "fgrep", "egrep",
}
_PIP_ALIASES = {"pip", "pip3"}
_PYTHON_ALIASES = {"python", "python3", "py"}
# These get routed to the venv interpreter; the rest of
# _ALLOWED_TOP_LEVEL passes through to PATH unchanged.
_VENV_ROUTED = _PIP_ALIASES | _PYTHON_ALIASES | {"uv"}

_PYTHON_SAFE_MODULES = frozenset({"pip"})
_GIT_READ_ONLY_COMMANDS = frozenset(
    {
        "blame",
        "cat-file",
        "describe",
        "diff",
        "log",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "shortlog",
        "show",
        "status",
    }
)
_GIT_BRANCH_FLAGS = frozenset(
    {
        "-a",
        "--all",
        "--color",
        "--column",
        "--ignore-case",
        "-r",
        "--remotes",
        "--show-current",
        "-v",
        "-vv",
        "--verbose",
    }
)
_GIT_BRANCH_VALUE_FLAGS = frozenset(
    {
        "--contains",
        "--format",
        "--merged",
        "--no-contains",
        "--no-merged",
        "--points-at",
        "--sort",
    }
)
_GIT_MUTATING_TAG_FLAGS = frozenset(
    {"-a", "--annotate", "-d", "--delete", "-f", "--force", "-s", "--sign", "-u", "--local-user"}
)
_GIT_OUTPUT_FLAGS = frozenset({"-o", "--output"})
_GIT_UNSAFE_READ_FLAGS = frozenset(
    {"--ext-diff", "--filters", "--no-index", "--textconv"}
)
_GIT_EXTERNAL_INPUT_FLAGS = frozenset(
    {
        "--contents",
        "--exclude-from",
        "--exclude-per-directory",
        "--pathspec-from-file",
    }
)
_LORAHUB_READ_ONLY_COMMANDS = frozenset({"doctor", "info", "validate", "version"})
_FIND_MUTATING_ACTIONS = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-fls",
        "-fprint",
        "-fprintf",
        "-ok",
        "-okdir",
    }
)
_PATH_SCOPED_COMMANDS = frozenset(
    {
        "cat", "df", "du", "find", "grep", "head", "ls", "rg", "stat",
        "tail", "tree", "wc", "fgrep", "egrep",
    }
)
_NVIDIA_SMI_FLAG_ONLY = frozenset(
    {"-B", "-h", "--help", "-L", "--list-gpus", "-q", "--query", "--version"}
)
_NVIDIA_SMI_VALUE_FLAGS = frozenset(
    {"-d", "--display", "-i", "--id", "-l", "--loop", "-lms", "--loop-ms", "--format"}
)
_NVIDIA_SMI_VALUE_PREFIXES = (
    "--display=",
    "--format=",
    "--id=",
    "--loop=",
    "--loop-ms=",
    "--query-accounted-apps=",
    "--query-compute-apps=",
    "--query-gpu=",
    "--query-supported-clocks=",
)
_PIP_ALLOWED_COMMANDS = frozenset(
    {
        "--help",
        "--version",
        "check",
        "debug",
        "freeze",
        "help",
        "index",
        "inspect",
        "list",
        "show",
    }
)
_PIP_EXTERNAL_TARGET_FLAGS = frozenset(
    {
        "--cache-dir",
        "--log",
        "--prefix",
        "--python",
        "--report",
        "--root",
        "--target",
        "--user",
        "-p",
        "-t",
    }
)


def _enforce_command_policy(
    argv: list[str],
    *,
    python_path: Path | None,
    unrestricted: bool,
    cwd: Path | None = None,
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
        base = head.split("/")[-1].split("\\")[-1].lower()
        if base not in _ALLOWED_TOP_LEVEL:
            raise TerminalDenied(
                f"命令 {head!r} 不在白名单中。受限模式只允许 pip / uv / python / "
                "lorahub / 常见诊断命令 (which, ls, cat, head, git, nvidia-smi 等)。"
                "如需任意命令 + shell 语法 ($() / | / && / >),请到 设置 → 终端 "
                "中开启「自由命令模式」。"
            )
        if head.lower() != base:
            raise TerminalDenied("受限模式禁止从自定义路径执行白名单同名程序。")
        head = base
        _validate_restricted_command(head, argv[1:], cwd=cwd)

    if head == "git" and not unrestricted:
        return _restricted_git_argv(argv[1:])

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

    # Whitelisted diagnostic / inspection command — pass through to PATH.
    # These don't get rewritten because their semantics depend on PATH
    # discovery (e.g. ``which lorahub`` would be useless if we hard-coded
    # the venv python here).
    return argv


def _validate_restricted_command(
    head: str,
    args: list[str],
    *,
    cwd: Path | None,
) -> None:
    """Reject whitelist entries whose subcommands can escape argv-only mode."""
    if head in _PIP_ALIASES:
        _validate_pip_command(args, cwd=cwd)
        return

    if head in _PYTHON_ALIASES:
        if not args or args[0] in {"-h", "--help", "-V", "--version"}:
            return
        if len(args) >= 2 and args[0] == "-m" and args[1] in _PYTHON_SAFE_MODULES:
            if args[1] == "pip":
                _validate_pip_command(args[2:], cwd=cwd)
            return
        raise TerminalDenied(
            "受限模式只允许查看 Python 版本或运行 python -m pip 查询；"
            "脚本、-c 和其他模块需要开启自由命令模式。"
        )

    if head == "uv":
        if not args or args[0] in {"-h", "--help", "-V", "--version", "help"}:
            return
        if args[0] == "pip":
            _validate_pip_command(args[1:], cwd=cwd)
            return
        raise TerminalDenied(
            "受限模式只允许 uv pip 与 uv 的帮助/版本命令；"
            "uv run/tool 等执行入口需要开启自由命令模式。"
        )

    if head == "git":
        _validate_git_command(args)
        return

    if head == "nvidia-smi":
        _validate_nvidia_smi(args)
        return

    if head == "nvcc":
        if any(value not in {"-h", "--help", "-V", "--version"} for value in args):
            raise TerminalDenied("受限模式下 nvcc 仅允许查看帮助或版本。")
        return

    if head == "find":
        lowered = {value.lower() for value in args}
        blocked = sorted(lowered & _FIND_MUTATING_ACTIONS)
        if blocked:
            raise TerminalDenied(
                f"受限模式禁止 find 的执行或写文件动作：{', '.join(blocked)}。"
            )
        _validate_scoped_arguments(head, args, cwd=cwd)
        return

    if head == "rg" and any(
        value == "--pre" or value.startswith("--pre=") for value in args
    ):
        raise TerminalDenied("受限模式禁止 rg --pre 执行外部预处理程序。")

    if head == "tree" and any(
        value in {"-o", "--output"} or value.startswith("--output=")
        for value in args
    ):
        raise TerminalDenied("受限模式禁止 tree 将结果写入文件。")

    if head in _PATH_SCOPED_COMMANDS:
        _validate_scoped_arguments(head, args, cwd=cwd)
        return

    if head == "lorahub":
        _validate_lorahub_command(args)


def _validate_pip_command(args: list[str], *, cwd: Path | None) -> None:
    if not args:
        return
    command = args[0].lower()
    if command not in _PIP_ALLOWED_COMMANDS:
        raise TerminalDenied(
            "受限模式只允许 pip 查询与依赖检查；install/uninstall/download/"
            "wheel/config/cache 等变更命令需要开启自由命令模式。"
        )

    for index, value in enumerate(args[1:], start=1):
        option, separator, option_value = value.partition("=")
        lowered = option.lower()
        if lowered in _PIP_EXTERNAL_TARGET_FLAGS:
            raise TerminalDenied(
                f"受限模式禁止 pip 使用 {option} 改写目标环境或输出位置。"
            )
        if lowered.startswith(("-p", "-t")) and len(lowered) > 2:
            raise TerminalDenied("受限模式禁止 pip 改写目标环境或输出位置。")
        if lowered.startswith(("-c", "-r")) and len(value) > 2:
            _validate_path_value(value[2:], cwd=cwd)
        if any(
            lowered.startswith(f"{flag}=") for flag in _PIP_EXTERNAL_TARGET_FLAGS
        ):
            raise TerminalDenied("受限模式禁止 pip 改写目标环境或输出位置。")
        if separator:
            _validate_path_value(option_value, cwd=cwd)
        elif index > 0:
            _validate_path_value(value, cwd=cwd)


def _validate_scoped_arguments(
    command: str,
    args: list[str],
    *,
    cwd: Path | None,
) -> None:
    for value in args:
        if command in {"grep", "rg", "fgrep", "egrep"} and value.startswith("-f") and len(value) > 2:
            _validate_path_value(value[2:], cwd=cwd)
        candidate = value.partition("=")[2] if "=" in value else value
        _validate_path_value(candidate, cwd=cwd)
    if cwd is None:
        return
    try:
        cwd.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TerminalDenied(f"{command} 的工作目录不可用。") from exc


def _validate_path_value(value: str, *, cwd: Path | None) -> None:
    """Reject absolute/escaping file operands and linked paths outside cwd."""
    raw = value.strip().strip('"\'')
    if not raw or raw in {"-", "--"}:
        return
    lowered = raw.lower()
    if "file://" in lowered or lowered.startswith("file:"):
        raise TerminalDenied("受限模式禁止通过 file: URI 访问后端目录外的文件。")
    if "://" in raw and not lowered.startswith("file:"):
        return

    normalised = raw.replace("\\", "/")
    parts = [part for part in normalised.split("/") if part not in {"", "."}]
    windows_absolute = len(normalised) >= 3 and normalised[1:3] == ":/"
    if (
        normalised.startswith("/")
        or windows_absolute
        or ".." in parts
        or "../" in normalised
        or normalised.endswith("/..")
    ):
        raise TerminalDenied("受限模式只允许访问当前后端目录内的相对路径。")
    if cwd is None or raw.startswith("-"):
        return

    candidate = cwd / raw
    if not (candidate.exists() or is_link_like(candidate)):
        return
    try:
        candidate.resolve().relative_to(cwd.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise TerminalDenied("受限模式拒绝指向后端目录外的链接路径。") from exc


def _validate_git_command(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help", "--version"}:
        return
    command = args[0].lower()
    if command in _GIT_READ_ONLY_COMMANDS:
        options = args[1:]
        if any(
            value in _GIT_UNSAFE_READ_FLAGS
            or value.startswith(("--filters=", "--textconv="))
            for value in options
        ):
            raise TerminalDenied("受限模式禁止 Git 调用外部程序或读取仓库外文件。")
        if any(
            value in _GIT_EXTERNAL_INPUT_FLAGS
            or any(value.startswith(f"{flag}=") for flag in _GIT_EXTERNAL_INPUT_FLAGS)
            for value in options
        ):
            raise TerminalDenied("受限模式禁止 Git 从后端目录外的文件读取参数或内容。")
        if any(
            value in _GIT_OUTPUT_FLAGS
            or value.startswith("--output=")
            or (value.startswith("-o") and value != "-o")
            for value in options
        ):
            raise TerminalDenied("受限模式禁止 Git 将查询结果写入文件。")
        return
    if command == "branch" and _git_branch_is_read_only(args[1:]):
        return
    if command == "remote" and args[1:] in ([], ["-v"], ["--verbose"]):
        return
    if command == "tag":
        lowered = {value.lower() for value in args[1:]}
        if not (lowered & _GIT_MUTATING_TAG_FLAGS) and (
            not args[1:] or "-l" in lowered or "--list" in lowered
        ):
            return
    raise TerminalDenied(
        "受限模式下 git 仅允许 status/log/diff/show 等只读查询；"
        "checkout/reset/clean/merge/push 等操作需要开启自由命令模式。"
    )


def _restricted_git_argv(args: list[str]) -> list[str]:
    """Disable repository-configured helpers for accepted Git diagnostics."""
    prefix = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.pager=cat",
        "-c",
        "pager.branch=false",
    ]
    if args and args[0].lower() in {"diff", "log", "show"}:
        return [*prefix, args[0], "--no-ext-diff", "--no-textconv", *args[1:]]
    return [*prefix, *args]


def _validate_nvidia_smi(args: list[str]) -> None:
    """Allow telemetry queries while rejecting reset/configuration commands."""
    index = 0
    while index < len(args):
        value = args[index]
        if value in _NVIDIA_SMI_FLAG_ONLY or value.startswith(_NVIDIA_SMI_VALUE_PREFIXES):
            index += 1
            continue
        if value in _NVIDIA_SMI_VALUE_FLAGS:
            if index + 1 >= len(args) or args[index + 1].startswith("-"):
                raise TerminalDenied(f"nvidia-smi {value} 缺少查询参数。")
            index += 2
            continue
        raise TerminalDenied(
            "受限模式下 nvidia-smi 仅允许状态查询；功耗、时钟、MIG、重置和输出文件参数均被拒绝。"
        )


def _git_branch_is_read_only(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        value = args[index]
        if value in _GIT_BRANCH_FLAGS or value.startswith(("--color=", "--column=")):
            index += 1
            continue
        if value in _GIT_BRANCH_VALUE_FLAGS:
            if index + 1 >= len(args):
                return False
            index += 2
            continue
        if any(value.startswith(f"{flag}=") for flag in _GIT_BRANCH_VALUE_FLAGS):
            index += 1
            continue
        return False
    return True


def _validate_lorahub_command(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help", "--version"}:
        return
    command = args[0].lower()
    if command in _LORAHUB_READ_ONLY_COMMANDS:
        return
    if command == "jobs" and len(args) >= 2 and args[1] in {"ls", "show"}:
        return
    if command == "service" and len(args) >= 2 and args[1] in {"logs", "status"}:
        return
    if command == "system" and len(args) >= 2 and args[1] in {
        "errors",
        "errors-show",
        "gpu",
        "info",
    }:
        return
    raise TerminalDenied(
        "受限模式下 lorahub 仅允许诊断和查询；"
        "训练、更新、服务控制及任务变更需要开启自由命令模式。"
    )


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

    Hub's installer drops uv at ``.lorahub/uv/uv.exe`` (Windows) or
    ``.lorahub/uv/uv``. Prefer that over PATH so a globally-installed
    older uv doesn't shadow the bundled one. Falls back to ``shutil.which``
    when the bundled copy isn't there (dev checkouts, fresh installs).
    """
    name = "uv.exe" if os.name == "nt" else "uv"
    candidate = Path.cwd() / ".lorahub" / "uv" / name
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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32"
            else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
