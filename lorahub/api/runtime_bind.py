"""Runtime binding state for the local API process.

The service CLI and in-app updater both need to agree on the address the
current uvicorn process is using.  Keep that tiny bit of state in the API
package so the web updater can reuse it without importing CLI modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_state_path


@dataclass(frozen=True)
class RuntimeBind:
    host: str
    port: int
    pid: int | None = None


def state_dir() -> Path:
    p = user_state_path("lorahub", "lorahub")
    p.mkdir(parents=True, exist_ok=True)
    return p


def pid_file() -> Path:
    return state_dir() / "uvicorn.pid"


def port_file() -> Path:
    return state_dir() / "uvicorn.port"


def bind_file() -> Path:
    return state_dir() / "uvicorn.bind.json"


def log_file() -> Path:
    return state_dir() / "uvicorn.log"


def write_runtime_bind(host: str, port: int, *, pid: int | None = None) -> None:
    """Persist the current uvicorn bind in both legacy and structured files."""
    normalized = RuntimeBind(host=host.strip() or "127.0.0.1", port=int(port), pid=pid)
    bind_file().write_text(json.dumps(asdict(normalized), indent=2), encoding="utf-8")
    port_file().write_text(f"{normalized.port}\n", encoding="utf-8")
    if pid is not None:
        pid_file().write_text(f"{pid}\n", encoding="utf-8")


def read_runtime_bind() -> RuntimeBind | None:
    """Read the last known bind, falling back to the legacy port file."""
    try:
        raw = json.loads(bind_file().read_text(encoding="utf-8"))
        host = str(raw.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(raw.get("port") or 0)
        pid_raw = raw.get("pid")
        pid = int(pid_raw) if pid_raw is not None else None
        if port > 0:
            return RuntimeBind(host=host, port=port, pid=pid)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    try:
        port = int(port_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if port <= 0:
        return None
    pid: int | None = None
    try:
        pid = int(pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    return RuntimeBind(host="127.0.0.1", port=port, pid=pid)


def clear_runtime_bind(*, keep_bind: bool = False) -> None:
    """Remove PID/runtime files.

    ``keep_bind=True`` is used by restart paths: the process is gone, but the
    last host/port is still useful for the next start.
    """
    pid_file().unlink(missing_ok=True)
    if not keep_bind:
        port_file().unlink(missing_ok=True)
        bind_file().unlink(missing_ok=True)
        return
    current = read_runtime_bind()
    if current is not None:
        write_runtime_bind(current.host, current.port, pid=None)
    else:
        port_file().unlink(missing_ok=True)
        bind_file().unlink(missing_ok=True)


def record_current_process_bind(host: str, port: int) -> None:
    """Record a foreground/in-process uvicorn bind for self-restart."""
    write_runtime_bind(host, port, pid=os.getpid())


def restart_args(executable: str, argv: list[str]) -> list[str]:
    """Return restart argv, preserving the last known bind when needed."""
    args = [executable, *argv]
    bind = read_runtime_bind()
    if bind is None:
        return args
    return ensure_uvicorn_bind_args(args, host=bind.host, port=bind.port)


def spawn_service_restart(bind: RuntimeBind, *, cwd: Path | None = None) -> bool:
    """Launch ``lorahub service restart`` detached on the saved bind."""
    cmd = [
        sys.executable,
        "-m",
        "lorahub.cli.main",
        "--no-tui",
        "service",
        "restart",
        "--host",
        bind.host,
        "--port",
        str(bind.port),
    ]
    log = log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("ab") as fh:
            kwargs: dict[str, object] = {
                "stdin": subprocess.DEVNULL,
                "stdout": fh,
                "stderr": fh,
                "cwd": str(cwd or Path.cwd()),
                "close_fds": True,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = 0x00000008 | 0x08000000
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen(cmd, **kwargs)  # noqa: S603
    except OSError:
        return False
    return True


def refresh_current_uvicorn_bind(argv: list[str] | None = None, *, pid: int | None = None) -> RuntimeBind | None:
    """Refresh pid for the current uvicorn process when it matches the saved bind."""
    bind = read_runtime_bind()
    if bind is None:
        return None
    args = list(sys.argv if argv is None else argv)
    if not _looks_like_uvicorn_args(args):
        return None
    arg_port = _get_option(args, "--port")
    if arg_port is not None:
        try:
            if int(arg_port) != bind.port:
                return None
        except ValueError:
            return None
    host = _get_option(args, "--host") or bind.host
    actual_pid = os.getpid() if pid is None else pid
    write_runtime_bind(host, bind.port, pid=actual_pid)
    return RuntimeBind(host=host, port=bind.port, pid=actual_pid)


def ensure_uvicorn_bind_args(args: list[str], *, host: str, port: int) -> list[str]:
    """Patch argv so uvicorn restarts on the known host/port.

    Existing ``--host`` / ``--port`` values are replaced in place. If the
    process was launched as ``python -m uvicorn lorahub.api.app:app`` without
    explicit bind flags, they are appended. Non-uvicorn commands are left
    untouched; for ``lorahub serve`` the CLI arguments are already present.
    """
    if not _looks_like_uvicorn_args(args):
        return args
    patched = list(args)
    _set_option(patched, "--host", host)
    _set_option(patched, "--port", str(port))
    return patched


def _looks_like_uvicorn_args(args: list[str]) -> bool:
    return (
        "uvicorn" in args
        or "uvicorn.exe" in args
        or "lorahub.api.app:app" in args
    )


def _set_option(args: list[str], option: str, value: str) -> None:
    prefix = f"{option}="
    for i, tok in enumerate(args):
        if tok == option:
            if i + 1 < len(args):
                args[i + 1] = value
            else:
                args.append(value)
            return
        if tok.startswith(prefix):
            args[i] = f"{option}={value}"
            return
    args.extend([option, value])


def _get_option(args: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for i, tok in enumerate(args):
        if tok == option and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


__all__ = [
    "RuntimeBind",
    "bind_file",
    "clear_runtime_bind",
    "ensure_uvicorn_bind_args",
    "log_file",
    "pid_file",
    "port_file",
    "read_runtime_bind",
    "record_current_process_bind",
    "refresh_current_uvicorn_bind",
    "restart_args",
    "spawn_service_restart",
    "state_dir",
    "write_runtime_bind",
]
