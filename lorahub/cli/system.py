"""System CLI — local hardware probes.

These commands collect a system snapshot in-process (no API server
required) and print it in a couple of operator-friendly shapes. Same
data the dashboard sees over SSE; this is just the script angle.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lorahub.api.system_stats import collect_snapshot

console = Console()
system_app = typer.Typer(
    help="Inspect local CPU / GPU / memory state.",
    no_args_is_help=True,
)


@system_app.command("gpu")
def system_gpu(
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Dump the JSON snapshot instead of the table."),
    ] = False,
) -> None:
    """Print one-shot GPU info: name, memory, utilisation, temp, processes."""
    snap = collect_snapshot()
    if raw:
        console.print_json(data={"gpus": [g.to_dict() for g in snap.gpus]})
        return

    if not snap.gpus:
        console.print("[yellow]no GPUs detected[/yellow]")
        if not snap.has_nvidia_smi:
            console.print(
                "[dim]nvidia-smi not on PATH; only CPU stats are available.[/dim]"
            )
        return

    table = Table(show_lines=False, padding=(0, 1))
    table.add_column("idx", style="dim")
    table.add_column("name")
    table.add_column("mem")
    table.add_column("util")
    table.add_column("temp")
    table.add_column("driver", style="dim")
    for g in snap.gpus:
        mem = (
            f"{g.memory_used_bytes / 1024**3:.1f} / "
            f"{g.memory_total_bytes / 1024**3:.1f} GiB"
            if g.memory_used_bytes is not None and g.memory_total_bytes is not None
            else "—"
        )
        util = (
            f"{g.utilization_percent:.0f}%"
            if g.utilization_percent is not None
            else "—"
        )
        temp = (
            f"{g.temperature_c:.0f} ℃" if g.temperature_c is not None else "—"
        )
        table.add_row(str(g.index), g.name or "(unknown)", mem, util, temp, g.driver or "—")
    console.print(table)


@system_app.command("info")
def system_info() -> None:
    """Print the full host snapshot (CPU + memory + disks + network)."""
    snap = collect_snapshot()
    console.print(f"[bold]host:[/bold] {snap.host.hostname}  ({snap.host.system} {snap.host.release})")
    console.print(f"  python: {snap.host.python}")
    console.print(
        f"  CPU: {snap.cpu.cores_logical} logical / {snap.cpu.cores_physical} physical "
        f"@ {snap.cpu.usage_percent:.1f}% load"
    )
    used = snap.memory.used_bytes / 1024**3
    total = snap.memory.total_bytes / 1024**3
    console.print(
        f"  RAM: {used:.1f} / {total:.1f} GiB ({snap.memory.percent:.1f}%)"
    )
    if snap.gpus:
        console.print(f"  GPUs: {len(snap.gpus)} ({', '.join(g.name or '?' for g in snap.gpus)})")
    if snap.disks:
        console.print(f"  Disks: {len(snap.disks)} mount points")


__all__ = ["system_app"]
