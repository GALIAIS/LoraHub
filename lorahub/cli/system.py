"""System CLI — local hardware probes + error report registry.

These commands collect a system snapshot in-process (no API server
required) and print it in a couple of operator-friendly shapes. Same
data the dashboard sees over SSE; this is just the script angle.

``lorahub system errors`` exposes the local error-report registry
(:class:`ErrorReportStore`) for users who can't open the Settings UI:
list recent failures, dump one record's context, or stream the whole
registry as ndjson.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lorahub.api.error_reports import (
    ErrorReportStore,
    default_error_report_store_path,
)
from lorahub.api.system_stats import collect_snapshot
from lorahub.cli._i18n import t

console = Console()
err_console = Console(stderr=True)
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
        console.print_json(data={"gpus": [asdict(g) for g in snap.gpus]})
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


# --------------------------------------------------------------------------- #
# Error reports
# --------------------------------------------------------------------------- #


def _open_error_store() -> ErrorReportStore:
    """Open the registry directly. CLI doesn't need the FastAPI lifespan."""
    return ErrorReportStore(default_error_report_store_path())


@system_app.command("errors", help=t("system.errors.help"))
def errors_ls(
    tail: Annotated[
        int,
        typer.Option("--tail", "-n", help=t("system.errors.tail_help")),
    ] = 20,
    severity: Annotated[
        str | None,
        typer.Option("--severity", help=t("system.errors.severity_help")),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help=t("system.errors.source_help")),
    ] = None,
) -> None:
    """List recent error reports newest-first."""
    store = _open_error_store()
    items = store.list(
        limit=max(1, tail),
        severity=severity,  # type: ignore[arg-type]
        source=source,
    )
    if not items:
        console.print(t("system.errors.empty"))
        return
    table = Table(show_lines=False, padding=(0, 1))
    table.add_column(t("system.errors.col_time"), style="dim")
    table.add_column(t("system.errors.col_severity"))
    table.add_column(t("system.errors.col_source"), style="dim")
    table.add_column(t("system.errors.col_title"))
    table.add_column(t("system.errors.col_id"), style="dim", no_wrap=True)
    tone = {
        "fatal": "red",
        "error": "red",
        "warn": "yellow",
        "info": "cyan",
    }
    for r in items:
        colour = tone.get(r.severity, "white")
        table.add_row(
            r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"[{colour}]{r.severity}[/]",
            r.source,
            r.title,
            r.id[-12:],
        )
    console.print(table)


@system_app.command("errors-show", help=t("system.errors_show.help"))
def errors_show(
    report_id: Annotated[str, typer.Argument(help=t("system.errors_show.id_help"))],
) -> None:
    """Print one error report including stack + context."""
    store = _open_error_store()
    rec = store.get(report_id)
    if rec is None:
        # Suffix match so the user can paste the trailing 12 chars
        # printed by ``system errors``.
        candidates = [r for r in store.list(limit=1000) if r.id.endswith(report_id)]
        if not candidates:
            err_console.print(t("system.errors_show.no_match", id=report_id))
            raise typer.Exit(code=1)
        if len(candidates) > 1:
            err_console.print(t("system.errors_show.ambiguous", id=report_id, n=len(candidates)))
            for c in candidates:
                err_console.print(f"  {c.id}")
            raise typer.Exit(code=1)
        rec = candidates[0]
    console.print_json(data=rec.to_dict())


@system_app.command("errors-export", help=t("system.errors_export.help"))
def errors_export(
    output: Annotated[
        Path, typer.Argument(help=t("system.errors_export.output_help")),
    ],
    limit: Annotated[
        int,
        typer.Option("--limit", help=t("system.errors_export.limit_help")),
    ] = 1000,
) -> None:
    """Dump the registry to a newline-delimited JSON file."""
    store = _open_error_store()
    rows = store.list(limit=max(1, limit))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str))
            fh.write("\n")
    console.print(t("system.errors_export.ok", n=len(rows), path=output))


@system_app.command("errors-clear", help=t("system.errors_clear.help"))
def errors_clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help=t("system.errors_clear.yes_help")),
    ] = False,
) -> None:
    """Drop every row in the local error registry."""
    if not yes:
        err_console.print(t("system.errors_clear.confirm_required"))
        raise typer.Exit(code=2)
    store = _open_error_store()
    deleted = store.clear()
    console.print(t("system.errors_clear.ok", n=deleted))


__all__ = ["system_app"]
