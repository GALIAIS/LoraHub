"""``lorahub manage ...`` — manage the lorahub install itself.

Commands:
    lorahub manage install       Add the `lorahub` command to the user PATH
    lorahub manage uninstall     Remove the shim from user PATH
    lorahub manage path          Print where the active `lorahub` lives
    lorahub manage update        git pull + reinstall deps + rebuild web
    lorahub manage upgrade       Switch to the newest release tag
    lorahub manage build         Rebuild only the web frontend

The user-visible strings are looked up via ``cli._i18n`` so the
group respects ``lorahub --lang``. Internal API / library identifiers
stay in English regardless.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from lorahub.cli._i18n import t

console = Console()
err_console = Console(stderr=True)

manage_app = typer.Typer(
    name="manage",
    # Help is set in main.py before adding the typer so the localised
    # string is materialised after the global --lang flag has been
    # resolved by the typer callback.
    help=t("manage.help"),
    no_args_is_help=True,
    add_completion=False,
)


# Phase → coloured prefix lookup. The dict is rebuilt on every call
# so a late ``--lang`` change still gets honoured (typer evaluates
# subcommand bodies after the root callback has fired).
def _phase_prefix(phase: str) -> str:
    table = {
        "git": t("manage.phase.git"),
        "deps": t("manage.phase.deps"),
        "build": t("manage.phase.build"),
        "done": t("manage.phase.done"),
    }
    return table.get(phase, f"[dim]{phase}[/]")


def _emit(phase: str, level: str, message: str) -> None:
    prefix = _phase_prefix(phase)
    if level == "warn":
        err_console.print(f"{prefix} [yellow]{message}[/]")
    elif level == "error":
        err_console.print(f"{prefix} [red]{message}[/]")
    else:
        console.print(f"{prefix} {message}")


def _shim_dir() -> Path:
    """Where we drop the user-PATH shim per platform."""
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return local / "lorahub" / "bin"
    return Path.home() / ".local" / "bin"


def _shim_path() -> Path:
    name = "lorahub.cmd" if sys.platform == "win32" else "lorahub"
    return _shim_dir() / name


def _venv_lorahub() -> Path | None:
    """Locate the lorahub entry point inside the active venv (if any)."""
    bin_dir = Path(sys.executable).parent
    cand = bin_dir / ("lorahub.exe" if sys.platform == "win32" else "lorahub")
    return cand if cand.is_file() else None


def _venv_python() -> Path | None:
    python = Path(sys.executable)
    return python if python.is_file() else None


def _project_root_for_shim() -> Path:
    from lorahub.core.paths import project_root  # noqa: PLC0415

    return project_root()


def _windows_shim_body(python: Path, root: Path) -> str:
    return (
        "@echo off\r\n"
        "setlocal\r\n"
        f'set "PYTHONPATH={root};%PYTHONPATH%"\r\n'
        f'call "{python}" -m lorahub %*\r\n'
    )


def _posix_shim_body(python: Path, root: Path) -> str:
    return (
        "#!/usr/bin/env sh\n"
        f"PYTHONPATH={shlex.quote(str(root))}:${{PYTHONPATH:-}}\n"
        "export PYTHONPATH\n"
        f"exec {shlex.quote(str(python))} -m lorahub \"$@\"\n"
    )


@manage_app.command(help=t("manage.path.help"))
def path() -> None:
    """Print the location of the currently-running ``lorahub`` command."""
    found = shutil.which("lorahub")
    venv_entry = _venv_lorahub()
    shim = _shim_path()
    not_on_path = f"[dim]{t('manage.path.not_on_path')}[/]"
    none_str = f"[dim]{t('manage.path.none')}[/]"
    exists = t("manage.path.exists") if shim.is_file() else t("manage.path.absent")
    console.print(f"{t('manage.path.shutil_which')}{found or not_on_path}")
    console.print(f"{t('manage.path.venv_entry')}{venv_entry or none_str}")
    console.print(f"{t('manage.path.shim')}{shim}  ({exists})")


@manage_app.command(help=t("manage.install.help"))
def install() -> None:
    """Add a ``lorahub`` shim to the user PATH.

    Linux/macOS: write ``~/.local/bin/lorahub`` launcher.
    Windows: write ``%LOCALAPPDATA%\\lorahub\\bin\\lorahub.cmd`` and
    prepend that directory to the user PATH via ``setx``.
    """
    venv_python = _venv_python()
    if venv_python is None:
        err_console.print(t("manage.install.no_venv_entry"))
        raise typer.Exit(code=1)
    project_root = _project_root_for_shim()

    shim_dir = _shim_dir()
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = _shim_path()

    if sys.platform == "win32":
        # cmd.exe parses .cmd files using the active ANSI code page (mbcs),
        # not ASCII or UTF-8. Using ascii here breaks any path containing
        # non-ASCII characters (e.g. CJK folder names, accented usernames).
        shim_body = _windows_shim_body(venv_python, project_root)
        try:
            shim.write_bytes(shim_body.encode("mbcs"))
        except UnicodeEncodeError as exc:
            err_console.print(
                t(
                    "manage.install.path_unencodable",
                    venv_entry=venv_python,
                    err=exc,
                )
            )
            raise typer.Exit(code=3) from exc
        existing_user_path = _read_user_path_windows()
        if str(shim_dir) not in [p.strip() for p in existing_user_path.split(";") if p.strip()]:
            new_user_path = (
                str(shim_dir) + (";" + existing_user_path if existing_user_path else "")
            )
            # setx silently truncates user PATH at 1024 characters and
            # returns 0, which used to leave users with a wrecked PATH.
            # Refuse rather than corrupt and tell the caller to clean up
            # their PATH first.
            if len(new_user_path) > 1024:
                err_console.print(
                    t(
                        "manage.install.path_too_long",
                        length=len(new_user_path),
                        shim_dir=shim_dir,
                    )
                )
                raise typer.Exit(code=4)
            try:
                subprocess.run(  # noqa: S603, S607
                    ["setx", "PATH", new_user_path],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                err_console.print(
                    t(
                        "manage.install.setx_failed",
                        shim=shim,
                        err=exc,
                        shim_dir=shim_dir,
                    )
                )
                raise typer.Exit(code=2) from exc
        console.print(t("manage.install.windows_done", shim=shim, shim_dir=shim_dir))
        return

    # POSIX: write a tiny launcher instead of symlinking the console
    # script. It keeps working when the editable entry point is stale
    # but the source checkout is present.
    if shim.exists() or shim.is_symlink():
        shim.unlink()
    shim.write_text(_posix_shim_body(venv_python, project_root), encoding="utf-8")
    shim.chmod(0o755)
    console.print(t("manage.install.posix_done", shim=shim, target=venv_python))
    if str(shim_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        console.print(t("manage.install.path_hint", shim_dir=shim_dir))


@manage_app.command(help=t("manage.uninstall.help"))
def uninstall() -> None:
    """Remove the ``lorahub`` shim from the user PATH.

    Doesn't touch the venv or any installed packages — only the shim
    that this user-PATH installer wrote.
    """
    shim = _shim_path()
    if not shim.exists() and not shim.is_symlink():
        console.print(t("manage.uninstall.no_shim", shim=shim))
        return
    shim.unlink()
    console.print(t("manage.uninstall.removed", shim=shim))
    if sys.platform == "win32":
        console.print(t("manage.uninstall.dir_hint"))


def _read_user_path_windows() -> str:
    """Read the current user PATH from the registry (avoids the system PATH)."""
    try:
        import winreg  # noqa: PLC0415
    except ImportError:
        return os.environ.get("PATH", "")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "PATH")
            return value
    except FileNotFoundError:
        return ""
    except OSError:
        return os.environ.get("PATH", "")


@manage_app.command(help=t("manage.update.help"))
def update(
    skip_build: Annotated[
        bool,
        typer.Option("--skip-build", help=t("manage.update.skip_build_help")),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help=t("manage.update.force_help")),
    ] = False,
) -> None:
    """Pull origin/dev, reinstall deps, rebuild SPA.

    Goes through ``lorahub.api.system_update.apply(channel='dev')``
    so the CLI behaves identically to Settings → Update.
    """
    from lorahub.api import system_update  # noqa: PLC0415

    try:
        system_update.apply(
            channel="dev",
            build=not skip_build,
            progress=_emit,
            force=force,
        )
    except RuntimeError as exc:
        err_console.print(t("manage.update.failed", err=exc))
        if not force:
            err_console.print(t("manage.update.force_hint"))
        raise typer.Exit(code=1) from None

    console.print(t("manage.update.complete"))
    console.print(t("manage.restart_hint"))


@manage_app.command(help=t("manage.upgrade.help"))
def upgrade(
    force: Annotated[
        bool,
        typer.Option("--force", help=t("manage.upgrade.force_help")),
    ] = False,
    skip_build: Annotated[
        bool,
        typer.Option("--skip-build", help=t("manage.upgrade.skip_build_help")),
    ] = False,
) -> None:
    """Switch the working tree to the latest published release tag."""
    from lorahub.api import system_update  # noqa: PLC0415

    try:
        system_update.apply(
            channel="tag",
            build=not skip_build,
            progress=_emit,
            force=force,
        )
    except RuntimeError as exc:
        err_console.print(t("manage.upgrade.failed", err=exc))
        if not force:
            err_console.print(t("manage.upgrade.force_hint"))
        raise typer.Exit(code=1) from None

    console.print(t("manage.upgrade.complete"))
    console.print(t("manage.restart_hint"))


@manage_app.command(help=t("manage.build.help"))
def build() -> None:
    """Rebuild the web frontend (vite build)."""
    from lorahub.core.paths import project_root  # noqa: PLC0415

    root = project_root()
    npm = _find_npm(root)
    if npm is None:
        err_console.print(t("manage.build.no_npm"))
        raise typer.Exit(code=1)
    web = root / "web"
    rc = subprocess.run(  # noqa: S603
        [str(npm), "run", "build"],
        cwd=web,
        check=False,
        env=_npm_env(npm),
    ).returncode
    if rc != 0:
        err_console.print(t("manage.build.npm_failed"))
        raise typer.Exit(code=rc)
    console.print(t("manage.build.complete"))


def _find_npm(root: Path) -> Path | None:
    """Locate the npm shim — prefer the portable Node, fall back to PATH."""
    env_node_dir = os.environ.get("NODE_DIR")
    if env_node_dir:
        cand = Path(env_node_dir) / ("npm.cmd" if sys.platform == "win32" else "bin/npm")
        if cand.is_file():
            return cand
    if sys.platform == "win32":
        cand = root / ".node" / "npm.cmd"
        if cand.is_file():
            return cand
    else:
        for cand in (
            root / ".node" / "bin" / "npm",
            Path("/root/autodl-tmp/opt/node20/bin/npm"),
        ):
            if cand.is_file():
                return cand
    found = shutil.which("npm")
    return Path(found) if found else None


def _npm_env(npm: Path) -> dict[str, str]:
    """Ensure npm lifecycle scripts can find the matching node binary."""
    env = os.environ.copy()
    bin_dir = npm.parent
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


__all__ = ["manage_app"]
