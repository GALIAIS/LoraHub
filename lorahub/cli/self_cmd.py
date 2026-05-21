"""``lorahub self ...`` — manage the lorahub install itself.

Commands:
    lorahub self install       Add the `lorahub` command to the user PATH
    lorahub self uninstall     Remove the shim from user PATH
    lorahub self path          Print where the active `lorahub` lives
    lorahub self update        git pull + reinstall deps + rebuild web
    lorahub self upgrade       Switch to the newest release tag
    lorahub self build         Rebuild only the web frontend

Design choices follow the answers to Q1-Q5:

* Q1 = setx PATH on Windows (user-level). The shim itself lives in
  ``%LOCALAPPDATA%\\lorahub\\bin`` so a single ``setx`` covers all
  future commands we add to that directory.
* Q4 = update has both a default (build) and a ``--skip-build`` flag.
* Q5 = scripts/run.{sh,bat} forward to ``lorahub service start`` once
  this module is in place; this file doesn't touch them — that's done
  in the wrapper scripts themselves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()
err_console = Console(stderr=True)

self_app = typer.Typer(
    name="self",
    help="Manage the lorahub install itself (install / update / upgrade).",
    no_args_is_help=True,
    add_completion=False,
)


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


@self_app.command()
def path() -> None:
    """Print the location of the currently-running ``lorahub`` command."""
    # `which`/`where` semantics: we check the shim, the venv entry, and
    # report whichever exists. shutil.which honours the live PATH so it
    # also catches an installation we don't know about.
    found = shutil.which("lorahub")
    venv_entry = _venv_lorahub()
    shim = _shim_path()
    console.print(f"shutil.which:   {found or '[dim]not on PATH[/]'}")
    console.print(f"venv entry:     {venv_entry or '[dim]none[/]'}")
    console.print(f"user-PATH shim: {shim}  ({'exists' if shim.is_file() else 'absent'})")


@self_app.command()
def install() -> None:
    """Add a ``lorahub`` shim to the user PATH.

    Linux/macOS:
        Symlinks ``~/.local/bin/lorahub`` -> ``<venv>/bin/lorahub``.
        ``~/.local/bin`` is part of the systemd-default user PATH on
        every recent distro; we only print a hint if it isn't already.

    Windows:
        Writes ``%LOCALAPPDATA%\\lorahub\\bin\\lorahub.cmd`` and
        prepends that directory to the user PATH via ``setx``. The
        change takes effect in new shells; the current shell still
        uses its old PATH.
    """
    venv_entry = _venv_lorahub()
    if venv_entry is None:
        err_console.print(
            "[red]no lorahub entry in the active venv[/]\n"
            "Run scripts/install.{sh,bat} first so .venv/bin/lorahub exists."
        )
        raise typer.Exit(code=1)

    shim_dir = _shim_dir()
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = _shim_path()

    if sys.platform == "win32":
        # A trivial cmd shim: forward all args to the venv entry. We
        # use `call` so errorlevel propagates without spawning a
        # nested cmd window.
        shim.write_text(
            "@echo off\r\n"
            f"call \"{venv_entry}\" %*\r\n",
            encoding="ascii",
        )
        # Add shim_dir to the user PATH if it isn't already.
        existing_user_path = _read_user_path_windows()
        if str(shim_dir) not in [p.strip() for p in existing_user_path.split(";") if p.strip()]:
            new_user_path = (
                str(shim_dir) + (";" + existing_user_path if existing_user_path else "")
            )
            try:
                subprocess.run(  # noqa: S603, S607
                    ["setx", "PATH", new_user_path],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                err_console.print(
                    f"[yellow]wrote shim {shim}, but setx PATH failed:[/] {exc}\n"
                    f"Add this to your user PATH manually: {shim_dir}"
                )
                raise typer.Exit(code=2) from exc
        console.print(f"[green]installed[/] {shim}")
        console.print(f"[dim]added {shim_dir} to user PATH (open a new shell to use it)[/]")
        return

    # POSIX: replace any existing shim with a fresh symlink so a venv
    # rename doesn't leave a dangling link behind.
    if shim.exists() or shim.is_symlink():
        shim.unlink()
    shim.symlink_to(venv_entry)
    console.print(f"[green]installed[/] {shim} -> {venv_entry}")
    if str(shim_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        console.print(
            f"[yellow]note:[/] {shim_dir} is not on your PATH.\n"
            "Add it to your shell rc, e.g.:\n"
            f"  echo 'export PATH=\"{shim_dir}:$PATH\"' >> ~/.bashrc"
        )


@self_app.command()
def uninstall() -> None:
    """Remove the ``lorahub`` shim from the user PATH.

    Doesn't touch the venv or any installed packages — only the shim
    that this user-PATH installer wrote. Run ``rm -rf .venv`` separately
    if you actually want to wipe the install.
    """
    shim = _shim_path()
    if not shim.exists() and not shim.is_symlink():
        console.print(f"[dim]no shim at {shim}[/]")
        return
    shim.unlink()
    console.print(f"[green]removed[/] {shim}")
    if sys.platform == "win32":
        console.print(
            "[dim]note: the shim directory is still on your user PATH.[/]\n"
            "Remove it via Settings → Environment Variables if you want a clean slate."
        )


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


@self_app.command()
def update(
    skip_build: Annotated[
        bool,
        typer.Option(
            "--skip-build",
            help="Skip the frontend rebuild (faster; backend-only update).",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Discard local changes (git reset --hard + clean -fd) before "
                 "pulling. Same as the Settings UI's 'ignore conflicts' toggle.",
        ),
    ] = False,
) -> None:
    """Pull the latest commits from origin/main, reinstall deps, rebuild SPA.

    Goes through ``lorahub.api.system_update.apply(channel='main')`` so
    the CLI behaves identically to the Settings → Update button:

      * ``configs/``, ``runs/``, ``output/``, ``models/``, ``datasets/``,
        ``.env*`` and ``external/anima_lora/uv.lock`` are user-owned and
        never block the upgrade.
      * Other uncommitted edits get auto-stashed before checkout and
        popped after. Conflicts on pop surface in ``git stash list`` so
        the user can resolve them.

    With ``--force``: skip the stash dance, ``git reset --hard`` + ``git
    clean -fd`` first, then pull. Use when local edits aren't worth
    preserving (the typical CI / VPS path after pushing from the dev box).
    """
    from lorahub.api import system_update  # noqa: PLC0415

    def _emit(phase: str, level: str, message: str) -> None:
        prefix = {
            "git": "[blue]git[/]",
            "deps": "[cyan]deps[/]",
            "build": "[magenta]build[/]",
            "done": "[green]done[/]",
        }.get(phase, f"[dim]{phase}[/]")
        if level == "warn":
            err_console.print(f"{prefix} [yellow]{message}[/]")
        elif level == "error":
            err_console.print(f"{prefix} [red]{message}[/]")
        else:
            console.print(f"{prefix} {message}")

    try:
        system_update.apply(
            channel="main",
            build=not skip_build,
            progress=_emit,
            force=force,
        )
    except RuntimeError as exc:
        err_console.print(f"[red]update failed:[/] {exc}")
        if not force:
            err_console.print(
                "[yellow]Hint:[/] retry with [bold]lorahub self update --force[/] "
                "to discard local changes that block the merge."
            )
        raise typer.Exit(code=1) from None

    console.print("[green]update complete[/]")
    console.print("Restart the daemon: [bold]lorahub service restart[/]")


@self_app.command()
def upgrade(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Discard local changes (git reset --hard + clean -fd) "
                 "before checking out the latest tag.",
        ),
    ] = False,
    skip_build: Annotated[
        bool,
        typer.Option(
            "--skip-build",
            help="Skip the SPA rebuild after checkout.",
        ),
    ] = False,
) -> None:
    """Switch the working tree to the latest published release tag.

    Same flow as ``lorahub self update`` but checks out the highest
    semver ``v*`` tag instead of ``origin/main``. Reuses the same
    user-owned path filter + auto stash/pop machinery as the UI.
    """
    from lorahub.api import system_update  # noqa: PLC0415

    def _emit(phase: str, level: str, message: str) -> None:
        prefix = {
            "git": "[blue]git[/]",
            "deps": "[cyan]deps[/]",
            "build": "[magenta]build[/]",
            "done": "[green]done[/]",
        }.get(phase, f"[dim]{phase}[/]")
        if level == "warn":
            err_console.print(f"{prefix} [yellow]{message}[/]")
        elif level == "error":
            err_console.print(f"{prefix} [red]{message}[/]")
        else:
            console.print(f"{prefix} {message}")

    try:
        system_update.apply(
            channel="tag",
            build=not skip_build,
            progress=_emit,
            force=force,
        )
    except RuntimeError as exc:
        err_console.print(f"[red]upgrade failed:[/] {exc}")
        if not force:
            err_console.print(
                "[yellow]Hint:[/] retry with [bold]lorahub self upgrade --force[/] "
                "to discard local changes."
            )
        raise typer.Exit(code=1) from None

    console.print("[green]upgrade complete[/]")
    console.print("Restart the daemon: [bold]lorahub service restart[/]")


@self_app.command()
def build() -> None:
    """Rebuild the web frontend (vite build) — equivalent to update step 3/3."""
    from lorahub.core.paths import project_root  # noqa: PLC0415

    root = project_root()
    npm = _find_npm(root)
    if npm is None:
        err_console.print(
            "[red]npm not found.[/] Run scripts/install.{sh,bat} first to install "
            "the portable Node toolchain."
        )
        raise typer.Exit(code=1)
    web = root / "web"
    rc = subprocess.run([str(npm), "run", "build"], cwd=web, check=False).returncode  # noqa: S603
    if rc != 0:
        err_console.print("[red]npm run build failed.[/]")
        raise typer.Exit(code=rc)
    console.print("[green]build complete[/]")


def _find_uv() -> Path | None:
    """Locate uv preferring the project-local copy."""
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    found = find_uv()
    return Path(found) if found else None


def _find_npm(root: Path) -> Path | None:
    """Locate the npm shim — prefer the portable Node, fall back to PATH."""
    if sys.platform == "win32":
        cand = root / ".node" / "npm.cmd"
        if cand.is_file():
            return cand
    else:
        cand = root / ".node" / "bin" / "npm"
        if cand.is_file():
            return cand
    found = shutil.which("npm")
    return Path(found) if found else None


__all__ = ["self_app"]
