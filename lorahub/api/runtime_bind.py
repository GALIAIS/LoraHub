"""Runtime binding state for the local API process.

The service CLI and in-app updater both need to agree on the address the
current uvicorn process is using.  Keep that tiny bit of state in the API
package so the web updater can reuse it without importing CLI modules.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
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
    try:
        p.chmod(0o700)
    except OSError:
        pass
    return p


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _state_path(name: str) -> Path:
    path = state_dir() / name
    if _is_link_like(path):
        raise RuntimeError(f"refusing linked runtime state file: {path}")
    return path


def _read_state_text(path: Path, *, limit: int = 64 * 1024) -> str:
    if _is_link_like(path):
        raise RuntimeError(f"refusing linked runtime state file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"runtime state path is not a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            value = handle.read(limit + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(value) > limit:
        raise RuntimeError(f"runtime state file is unexpectedly large: {path}")
    return value


def _write_state_text(path: Path, value: str) -> None:
    if _is_link_like(path):
        raise RuntimeError(f"refusing linked runtime state file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def pid_file() -> Path:
    return _state_path("uvicorn.pid")


def port_file() -> Path:
    return _state_path("uvicorn.port")


def bind_file() -> Path:
    return _state_path("uvicorn.bind.json")


def log_file() -> Path:
    return _state_path("uvicorn.log")


def write_runtime_bind(host: str, port: int, *, pid: int | None = None) -> None:
    """Persist the current uvicorn bind in both legacy and structured files."""
    normalized = RuntimeBind(host=host.strip() or "127.0.0.1", port=int(port), pid=pid)
    _write_state_text(bind_file(), json.dumps(asdict(normalized), indent=2))
    _write_state_text(port_file(), f"{normalized.port}\n")
    if pid is not None:
        _write_state_text(pid_file(), f"{pid}\n")


def read_runtime_bind() -> RuntimeBind | None:
    """Read the last known bind, falling back to the legacy port file."""
    try:
        raw = json.loads(_read_state_text(bind_file()))
        host = str(raw.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(raw.get("port") or 0)
        pid_raw = raw.get("pid")
        pid = int(pid_raw) if pid_raw is not None else None
        if port > 0:
            return RuntimeBind(host=host, port=port, pid=pid)
    except (OSError, RuntimeError, json.JSONDecodeError, TypeError, ValueError):
        pass

    try:
        port = int(_read_state_text(port_file()).strip())
    except (OSError, RuntimeError, ValueError):
        return None
    if port <= 0:
        return None
    legacy_pid: int | None = None
    try:
        legacy_pid = int(_read_state_text(pid_file()).strip())
    except (OSError, RuntimeError, ValueError):
        pass
    return RuntimeBind(host="127.0.0.1", port=port, pid=legacy_pid)


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
            if sys.platform == "win32":
                subprocess.Popen(  # noqa: S603
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=fh,
                    cwd=str(cwd or Path.cwd()),
                    close_fds=True,
                    creationflags=0x00000008 | 0x08000000,
                )
            else:
                subprocess.Popen(  # noqa: S603
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=fh,
                    cwd=str(cwd or Path.cwd()),
                    close_fds=True,
                    start_new_session=True,
                )
    except OSError:
        return False
    return True


def refresh_current_uvicorn_bind(argv: list[str] | None = None, *, pid: int | None = None) -> RuntimeBind | None:
    """Record or refresh the current uvicorn process bind."""
    args = list(sys.argv if argv is None else argv)
    if not _looks_like_uvicorn_args(args):
        return None
    bind = read_runtime_bind()
    arg_port = _get_option(args, "--port")
    if bind is None:
        try:
            port = int(arg_port) if arg_port is not None else 8000
        except ValueError:
            return None
        host = _get_option(args, "--host") or "127.0.0.1"
        actual_pid = os.getpid() if pid is None else pid
        write_runtime_bind(host, port, pid=actual_pid)
        return RuntimeBind(host=host, port=port, pid=actual_pid)
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
