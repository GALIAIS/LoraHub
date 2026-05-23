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
from lorahub.cli._i18n import t

console = Console()
system_app = typer.Typer(
    help=t("system.help"),
    no_args_is_help=True,
)


@system_app.command("gpu", help=t("system.gpu.help"))
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
        console.print(t("system.gpu.no_gpus"))
        if not snap.has_nvidia_smi:
            console.print(t("system.gpu.no_smi"))
        return

    table = Table(show_lines=False, padding=(0, 1))
    table.add_column(t("system.col.idx"), style="dim")
    table.add_column(t("system.col.name"))
    table.add_column(t("system.col.mem"))
    table.add_column(t("system.col.util"))
    table.add_column(t("system.col.temp"))
    table.add_column(t("system.col.driver"), style="dim")
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


@system_app.command("info", help=t("system.info.help"))
def system_info() -> None:
    """Print the full host snapshot (CPU + memory + disks + network)."""
    snap = collect_snapshot()
    console.print(
        t(
            "system.info.host",
            hostname=snap.host.hostname,
            system=snap.host.system,
            release=snap.host.release,
        )
    )
    console.print(t("system.info.python", version=snap.host.python))
    console.print(
        t(
            "system.info.cpu",
            logical=snap.cpu.cores_logical,
            physical=snap.cpu.cores_physical,
            usage=snap.cpu.usage_percent,
        )
    )
    used = snap.memory.used_bytes / 1024**3
    total = snap.memory.total_bytes / 1024**3
    console.print(
        t(
            "system.info.ram",
            used=used,
            total=total,
            percent=snap.memory.percent,
        )
    )
    if snap.gpus:
        console.print(
            t(
                "system.info.gpus",
                n=len(snap.gpus),
                names=", ".join(g.name or "?" for g in snap.gpus),
            )
        )
    if snap.disks:
        console.print(t("system.info.disks", n=len(snap.disks)))


__all__ = ["system_app"]
