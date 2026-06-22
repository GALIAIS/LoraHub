"""``lorahub service ...`` — manage the API daemon.

Wraps `uvicorn lorahub.api.app:app` so users can run a single workbench
in the background, query its health, tail logs, and (on Linux/macOS)
register it as a login-time auto-start service. Designed to replace the
ad-hoc shell scripts (``scripts/run.{sh,bat}``) with a discoverable CLI
surface that works the same on every platform.

Layout decisions:

* PID + log + chosen-port live under ``platformdirs.user_state_path``
  (``~/.local/state/lorahub/`` on Linux). State is per-user, not
  per-repo, so a single ``lorahub`` install never clobbers itself when
  the same user has two checkouts.

* No supervisor library. We `nohup` (POSIX) / `START /B` (Windows) the
  uvicorn process and write its PID; ``status`` / ``stop`` use the PID
  to interrogate / signal it. This keeps the dependency surface to
  zero — the CLI was already importing ``psutil`` via the api extras
  and we use it for cross-platform "is this PID a uvicorn process".

* Default port is **random** (free port picked at start time). A user
  who wants a stable port passes ``--port``. The chosen port lands in
  the same state dir, so ``lorahub service status`` can echo it back.

* ``service enable`` writes a systemd **system unit** (the user picked
  this over the user-unit during the design pass). Requires root —
  the command tells the user to re-run via ``sudo`` if it can't write
  the unit file.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from lorahub.api.runtime_bind import (
    clear_runtime_bind,
    log_file,
    pid_file,
    port_file,
    read_runtime_bind,
    record_current_process_bind,
    state_dir,
    write_runtime_bind,
)
from lorahub.cli._i18n import t

console = Console()
err_console = Console(stderr=True)

service_app = typer.Typer(
    name="service",
    help=t("service.help"),
    no_args_is_help=True,
    add_completion=False,
)


def _state_dir() -> Path:
    """Per-user runtime state — pid, log, last-chosen-port."""
    return state_dir()


def _pid_file() -> Path:
    return pid_file()


def _port_file() -> Path:
    return port_file()


def _log_file() -> Path:
    return log_file()


def _read_pid() -> int | None:
    """Return the PID stored in the pid file, or None if missing/dead."""
    p = _pid_file()
    if not p.is_file():
        return None
    try:
        pid = int(p.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _pid_alive(pid):
        # Stale pid file — clean up so callers don't confuse themselves
        # over a terminated daemon.
        clear_runtime_bind(keep_bind=True)
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    """Cross-platform 'is this process running'.

    On POSIX, signal 0 is the canonical "test for existence" probe.
    On Windows there is no equivalent in stock os; psutil's the safe
    way. We import psutil lazily because the CLI module is loaded
    even for ``--help`` and we don't want the dep there.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil  # noqa: PLC0415
        except ImportError:
            return False
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but we can't signal it (different user).
        # Treat as "alive" — better than removing a pid file we don't own.
        return True
    except OSError:
        return False


def _free_port() -> int:
    """Ask the kernel for any free TCP port.

    SO_REUSEADDR + immediate close is the standard race-free way: by
    the time uvicorn binds, the port is overwhelmingly likely to still
    be free, and if it isn't, uvicorn fails fast with a clear error
    message rather than silently picking a different one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _venv_python() -> Path:
    """Locate the active Python — prefer the repo's .venv when running
    against a checkout, fall back to ``sys.executable`` otherwise.

    A ``lorahub`` invocation from the venv's bin/ already runs under
    that interpreter; this helper exists so ``service start`` can
    launch a *new* uvicorn in the background using the same Python
    even when the user invoked us via a system-wide shim.
    """
    return Path(sys.executable)


def _wait_for_health(port: int, *, timeout_s: float = 30.0) -> bool:
    """Poll http://127.0.0.1:<port>/api/health until it answers 200."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


@service_app.command(help=t("service.start.help"))
def start(
    host: Annotated[str, typer.Option(help=t("service.start.host_help"))] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            help=t("service.start.port_help"),
        ),
    ] = 0,
    foreground: Annotated[
        bool,
        typer.Option(
            "--foreground",
            "-f",
            help=t("service.start.foreground_help"),
        ),
    ] = False,
) -> None:
    """Start the API daemon.

    Detaches by default. Pass --foreground to run uvicorn directly
    (useful for development; equivalent to the older ``lorahub serve``).
    """
    existing = _read_pid()
    if existing is not None:
        port_existing = _read_port()
        err_console.print(
            t(
                "service.already_running",
                pid=existing,
                port=(f" port={port_existing}" if port_existing else ""),
            )
        )
        raise typer.Exit(code=2)

    if port == 0:
        port = _free_port()

    if foreground:
        # In-process uvicorn — we're already inside the venv's Python,
        # no need to re-exec (which on Windows mishandles paths
        # containing spaces, e.g. "E:\\Lora Scripts\\..."). Same shape
        # as the legacy ``lorahub serve`` command.
        try:
            import uvicorn  # noqa: PLC0415
        except ImportError as exc:
            err_console.print(t("serve.api_extras_missing"))
            raise typer.Exit(code=1) from exc
        console.print(t("service.foreground_banner", host=host, port=port))
        record_current_process_bind(host, port)
        uvicorn.run(
            "lorahub.api.app:app",
            host=host,
            port=port,
            log_level="info",
        )
        return

    py = _venv_python()
    cmd = [
        str(py),
        "-m",
        "uvicorn",
        "lorahub.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]

    log = _log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log.open("ab")

    if sys.platform == "win32":
        # CREATE_NO_WINDOW + DETACHED_PROCESS lets the child outlive
        # the spawning console (Powershell / cmd) without a console
        # window flashing.
        DETACHED_PROCESS = 0x00000008  # noqa: N806
        CREATE_NO_WINDOW = 0x08000000  # noqa: N806
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
            close_fds=True,
        )
    else:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,  # POSIX: detach from controlling tty
            close_fds=True,
        )

    write_runtime_bind(host, port, pid=proc.pid)

    console.print(t("service.started", pid=proc.pid, port=port))
    console.print(t("service.log_path", path=log))

    if _wait_for_health(port, timeout_s=30.0):
        console.print(t("service.healthy", host=host, port=port))
    else:
        err_console.print(t("service.health_timeout", log=log))
        raise typer.Exit(code=3)


def _read_port() -> int | None:
    bind = read_runtime_bind()
    return bind.port if bind is not None else None


@service_app.command(help=t("service.stop.help"))
def stop(
    timeout: Annotated[
        float,
        typer.Option(help=t("service.stop.timeout_help")),
    ] = 5.0,
) -> None:
    """Stop the API daemon.

    Sends SIGTERM (POSIX) / Ctrl-Break (Windows) first; escalates to
    SIGKILL after ``--timeout`` seconds if the process is still alive.
    """
    pid = _read_pid()
    if pid is None:
        console.print(t("service.stop.not_running"))
        return

    if sys.platform == "win32":
        try:
            import psutil  # noqa: PLC0415

            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                proc.kill()
        except Exception as exc:  # noqa: BLE001
            err_console.print(t("service.stop.failed", pid=pid, err=exc))
            raise typer.Exit(code=1) from exc
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Wait up to `timeout` for graceful exit, then escalate.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.2)
        if _pid_alive(pid):
            with __import__("contextlib").suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)

    clear_runtime_bind(keep_bind=True)
    console.print(t("service.stopped", pid=pid))


@service_app.command(help=t("service.restart.help"))
def restart(
    host: Annotated[str, typer.Option(help=t("service.start.host_help"))] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help=t("service.start.port_help")),
    ] = 0,
) -> None:
    """Stop the daemon (if running) and start a fresh one.

    When ``--port`` is omitted, reuse the last successful service port.
    This keeps ``lorahub service restart`` and the in-app updater from
    moving the UI to a random address after a restart.
    """
    previous = read_runtime_bind()
    if port == 0 and previous is not None:
        port = previous.port
        if host == "127.0.0.1":
            host = previous.host
    if _read_pid() is not None:
        stop()
        time.sleep(0.5)
    start(host=host, port=port, foreground=False)


@service_app.command(help=t("service.status.help"))
def status() -> None:
    """Show whether the daemon is running and on which port."""
    pid = _read_pid()
    if pid is None:
        console.print(t("service.status.stopped"))
        raise typer.Exit(code=3)
    port = _read_port()
    healthy = _wait_for_health(port, timeout_s=2.0) if port else False
    health_label = (
        t("service.status.healthy") if healthy else t("service.status.unhealthy")
    )
    port_label = f"port={port}" if port else "port=?"
    console.print(
        t(
            "service.status.running",
            pid=pid,
            port_label=port_label,
            health_label=health_label,
        )
    )
    console.print(t("service.log_path", path=_log_file()))


@service_app.command(help=t("service.logs.help"))
def logs(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Tail -f instead of full dump."),
    ] = False,
    n: Annotated[int, typer.Option("-n", help="Last N lines.")] = 50,
) -> None:
    """Print the daemon's log file."""
    log = _log_file()
    if not log.is_file():
        console.print(t("service.logs.empty", path=log))
        return
    if follow:
        # Hand off to the platform's tail tool — re-implementing tail-f
        # in pure Python is more code than it's worth. We invoke via
        # subprocess.call instead of os.execvp because Windows' execvp
        # mishandles argv when the spawning process's path or
        # arguments contain spaces (the underlying CreateProcess uses
        # the legacy whitespace-split argv parsing). subprocess uses
        # explicit argv handoff, which is portable.
        if sys.platform == "win32":
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-Content -Path '{log}' -Tail {n} -Wait",
            ]
        else:
            cmd = ["tail", "-n", str(n), "-f", str(log)]
        rc = subprocess.call(cmd)  # noqa: S603
        raise typer.Exit(code=rc)
    # One-shot: read last N lines without loading the whole file.
    with log.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        block = 8192
        data = b""
        pos = size
        while pos > 0 and data.count(b"\n") <= n:
            read = min(block, pos)
            pos -= read
            fh.seek(pos)
            data = fh.read(read) + data
        tail = b"\n".join(data.splitlines()[-n:])
    sys.stdout.buffer.write(tail + b"\n")


@service_app.command(help=t("service.enable.help"))
def enable(
    host: Annotated[str, typer.Option(help="Bind address for the unit.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Fixed port for the unit.")] = 18765,
) -> None:
    """Register LoraHub as a system service (Linux/macOS only).

    Linux: writes ``/etc/systemd/system/lorahub.service`` and runs
    ``systemctl enable --now lorahub``. Requires root — re-run via
    ``sudo lorahub service enable`` if you got a permission error.

    macOS: writes ``/Library/LaunchDaemons/com.lorahub.plist`` and
    loads it with ``launchctl``. Also requires root.

    Windows: not supported by this command — use Task Scheduler.
    Run ``lorahub service install-unit --print`` to get a sample
    schtasks invocation you can adapt.
    """
    if sys.platform == "win32":
        err_console.print(
            t(
                "service.enable.windows",
                exe=sys.executable,
                host=host,
                port=port,
            )
        )
        raise typer.Exit(code=1)

    py = sys.executable
    repo_root = _resolve_repo_root()

    if sys.platform == "darwin":
        plist_path = Path("/Library/LaunchDaemons/com.lorahub.plist")
        plist = _render_launchd_plist(py=py, repo=repo_root, host=host, port=port)
        try:
            plist_path.write_text(plist)
        except PermissionError as exc:
            err_console.print(t("service.enable.perm", path=plist_path))
            raise typer.Exit(code=1) from exc
        subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=False)  # noqa: S603, S607
        console.print(t("service.enable.ok", path=plist_path))
        return

    # Linux: system systemd unit (matches Q3 = system, not user).
    unit_path = Path("/etc/systemd/system/lorahub.service")
    unit = _render_systemd_unit(py=py, repo=repo_root, host=host, port=port)
    try:
        unit_path.write_text(unit)
    except PermissionError as exc:
        err_console.print(t("service.enable.perm", path=unit_path))
        raise typer.Exit(code=1) from exc
    subprocess.run(["systemctl", "daemon-reload"], check=False)  # noqa: S603, S607
    subprocess.run(["systemctl", "enable", "--now", "lorahub"], check=False)  # noqa: S603, S607
    console.print(t("service.enable.ok", path=unit_path))
    console.print(t("service.enable.systemd_hint"))


@service_app.command(help=t("service.disable.help"))
def disable() -> None:
    """Remove the registered system service."""
    if sys.platform == "win32":
        err_console.print(t("service.disable.windows"))
        raise typer.Exit(code=1)
    if sys.platform == "darwin":
        plist = Path("/Library/LaunchDaemons/com.lorahub.plist")
        if plist.is_file():
            subprocess.run(["launchctl", "unload", "-w", str(plist)], check=False)  # noqa: S603, S607
            try:
                plist.unlink()
            except PermissionError as exc:
                err_console.print(t("service.disable.sudo", path=plist))
                raise typer.Exit(code=1) from exc
        console.print(t("service.disable.ok"))
        return
    # Linux
    subprocess.run(["systemctl", "disable", "--now", "lorahub"], check=False)  # noqa: S603, S607
    unit = Path("/etc/systemd/system/lorahub.service")
    if unit.is_file():
        try:
            unit.unlink()
        except PermissionError as exc:
            err_console.print(t("service.disable.sudo", path=unit))
            raise typer.Exit(code=1) from exc
    subprocess.run(["systemctl", "daemon-reload"], check=False)  # noqa: S603, S607
    console.print(t("service.disable.ok"))


@service_app.command("install-unit", help=t("service.install_unit.help"))
def install_unit(
    print_only: Annotated[
        bool,
        typer.Option("--print", help="Print the unit file to stdout instead of writing."),
    ] = False,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 18765,
) -> None:
    """Print the systemd unit / launchd plist / Task Scheduler stub.

    Useful for users who want to inspect or hand-edit the unit before
    enabling, or who want to register the service from a Dockerfile /
    Ansible role / etc.
    """
    py = sys.executable
    repo_root = _resolve_repo_root()
    if sys.platform == "darwin":
        text = _render_launchd_plist(py=py, repo=repo_root, host=host, port=port)
    elif sys.platform == "win32":
        text = (
            "rem schtasks invocation — register at logon\n"
            f'schtasks /Create /SC ONLOGON /TN LoraHub /F /RL HIGHEST '
            f'/TR "{py} -m uvicorn lorahub.api.app:app --host {host} --port {port}"\n'
        )
    else:
        text = _render_systemd_unit(py=py, repo=repo_root, host=host, port=port)
    if print_only:
        sys.stdout.write(text)
    else:
        console.print(text)


def _resolve_repo_root() -> Path:
    """Best-effort repo root for unit files; falls back to cwd."""
    try:
        from lorahub.core.paths import project_root  # noqa: PLC0415

        return project_root()
    except Exception:  # noqa: BLE001
        return Path.cwd().resolve()


def _render_systemd_unit(*, py: str, repo: Path, host: str, port: int) -> str:
    return (
        "[Unit]\n"
        "Description=LoraHub Workbench API\n"
        "After=network.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={repo}\n"
        f"ExecStart={py} -m uvicorn lorahub.api.app:app --host {host} --port {port}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _render_launchd_plist(*, py: str, repo: Path, host: str, port: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        "  <key>Label</key><string>com.lorahub</string>\n"
        "  <key>ProgramArguments</key><array>\n"
        f"    <string>{py}</string>\n"
        "    <string>-m</string><string>uvicorn</string>\n"
        "    <string>lorahub.api.app:app</string>\n"
        f"    <string>--host</string><string>{host}</string>\n"
        f"    <string>--port</string><string>{port}</string>\n"
        "  </array>\n"
        f"  <key>WorkingDirectory</key><string>{repo}</string>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "  <key>KeepAlive</key><true/>\n"
        "</dict></plist>\n"
    )


__all__ = ["service_app"]
