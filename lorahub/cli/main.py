"""LoraHub CLI entry point.

Commands:
    lorahub validate <recipe>   Check a recipe without launching training.
    lorahub info <recipe>       Show compiled argv and VRAM estimate (dry run).
    lorahub train <recipe>      Run training to completion.
    lorahub init <name>         Scaffold a starter recipe in the current dir.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lorahub import __version__
from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.backends.kohya.compiler import compile_recipe
from lorahub.core.config.loader import load_recipe
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

app = typer.Typer(
    name="lorahub",
    help="Open-source LoRA training workbench for diffusion models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def version() -> None:
    """Print the installed lorahub version."""
    console.print(f"lorahub {__version__}")


@app.command()
def validate(
    recipe: Annotated[Path, typer.Argument(help="Path to a recipe YAML file.")],
) -> None:
    """Validate a recipe without running training."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()
    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)
    console.print("[green]✓ recipe valid[/green]")


@app.command()
def info(
    recipe: Annotated[Path, typer.Argument(help="Path to a recipe YAML file.")],
) -> None:
    """Show what a recipe would compile to, plus VRAM estimate (no training)."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()

    script, argv = compile_recipe(cfg, workspace=Path.cwd() / "_dryrun")
    est = backend.estimate_vram(cfg)

    table = Table(title="Recipe summary", show_header=False, expand=False)
    table.add_row("recipe", str(recipe))
    table.add_row("arch", cfg.base_model.arch)
    table.add_row("network", f"{cfg.network.type} rank={cfg.network.rank} alpha={cfg.network.alpha}")
    table.add_row("schedule", f"{cfg.schedule.epochs} epochs x bs={cfg.schedule.batch_size}")
    table.add_row("precision", cfg.precision)
    table.add_row("entry script", script)
    table.add_row("estimated VRAM", f"{est.total_gib:.1f} GiB")
    console.print(table)

    console.print("\n[bold]Compiled argv:[/bold]")
    for a in argv:
        console.print(f"  {a}")


@app.command()
def train(
    recipe: Annotated[Path, typer.Argument(help="Path to a recipe YAML file.")],
    workspace: Annotated[
        Path | None,
        typer.Option(help="Where to write logs/checkpoints/samples."),
    ] = None,
) -> None:
    """Run training to completion. Press Ctrl+C to stop gracefully."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()

    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)

    ws = workspace or (Path.cwd() / "runs" / cfg.output.name)
    ws.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]workspace:[/dim] {ws}")

    events_log = ws / "events.jsonl"
    with JsonlEventSink(events_log) as sink:

        def on_event(ev: TrainingEvent) -> None:
            sink(ev)
            _render_event(ev)

        handle = backend.launch(cfg, workspace=ws, on_event=on_event)
        console.print(f"[dim]pid:[/dim] {handle.pid}  [dim]job:[/dim] {handle.job_id}")
        try:
            rc = handle.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — stopping training gracefully…[/yellow]")
            handle.stop(graceful=True)
            rc = handle.wait()

    if rc != 0:
        err_console.print(f"[red]training failed (rc={rc})[/red]")
        raise typer.Exit(code=rc)
    console.print("[green]✓ training complete[/green]")


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Name for the new recipe (no extension).")],
    template: Annotated[
        str, typer.Option(help="Built-in template to copy.")
    ] = "sdxl_character_8gb",
) -> None:
    """Scaffold a starter recipe in the current directory."""
    src = _builtin_recipe(template)
    if not src.exists():
        err_console.print(f"[red]unknown template: {template}[/red]")
        raise typer.Exit(code=1)
    dst = Path.cwd() / f"{name}.yaml"
    if dst.exists():
        err_console.print(f"[red]{dst} already exists[/red]")
        raise typer.Exit(code=1)
    shutil.copy2(src, dst)
    console.print(f"[green]created[/green] {dst}")


def _builtin_recipe(name: str) -> Path:
    package_root = Path(__file__).resolve().parent.parent.parent
    return package_root / "recipes" / f"{name}.yaml"


def _render_issues(issues: list[ValidationIssue]) -> None:
    if not issues:
        return
    for i in issues:
        color = {"error": "red", "warning": "yellow", "info": "cyan"}[i.severity.value]
        console.print(f"[{color}]{i.severity.value}[/]: {i.field}: {i.message}")


def _render_event(ev: TrainingEvent) -> None:
    if ev.type is EventType.step:
        step = ev.payload.get("step")
        total = ev.payload.get("total_steps")
        loss = ev.payload.get("loss")
        msg = f"step {step}/{total}"
        if loss is not None:
            msg += f"  loss={loss:.4f}"
        console.print(f"[dim]{msg}[/dim]")
    elif ev.type is EventType.epoch_end:
        console.print(
            f"[cyan]epoch {ev.payload['epoch']}/{ev.payload['total_epochs']} done[/cyan]"
        )
    elif ev.type is EventType.checkpoint_saved:
        console.print(f"[green]checkpoint:[/green] {ev.payload['path']}")
    elif ev.type is EventType.sample_ready:
        console.print(f"[magenta]sample:[/magenta] {ev.payload['path']}")
    elif ev.type is EventType.error:
        err_console.print(f"[red]{ev.payload}[/red]")
    elif ev.type is EventType.log:
        level = ev.payload.get("level", "info")
        msg = ev.payload.get("message", "")
        if level == "error":
            err_console.print(f"[red]{msg}[/red]")
    elif ev.type is EventType.done:
        rc = ev.payload.get("returncode")
        dur = ev.payload.get("duration_s", 0.0)
        console.print(f"[bold]done[/bold]  rc={rc}  duration={dur:.1f}s")


def main() -> None:  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
