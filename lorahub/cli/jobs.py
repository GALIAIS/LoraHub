"""Jobs CLI — list / inspect / cancel / kill / resume / rerun jobs.

These commands read (and for cancel/kill/rerun also write) the on-disk
JobStore directly. That means they work without a running ``lorahub
serve`` — useful for one-off ops on a server that's been killed or
when you want to script around the API.

Cancel / kill semantics:
  * ``cancel <id>`` only flips a queued job to ``canceled``. A running
    job needs ``kill`` instead — direct DB updates can't reach the
    in-process scheduler thread.
  * ``kill <id>`` SIGTERMs the worker process by PID, then upserts the
    record state to ``canceled``. Falls back to SIGKILL if the worker
    doesn't exit within 5 s.

``resume`` and ``rerun`` are wrappers around the existing API endpoints
— they need a running server because the scheduler thread is what
actually re-enqueues. The CLI hits ``http://127.0.0.1:6006`` by default
and respects ``LORAHUB_API_URL`` for non-default ports.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from lorahub.api.state import JobState
from lorahub.api.store import JobStore, default_store_path

console = Console()
jobs_app = typer.Typer(
    help="Inspect and manage training jobs without opening the web UI.",
    no_args_is_help=True,
)


def _store_path() -> Path:
    """Resolve the JobStore SQLite path. Honours the historical default."""
    return default_store_path()


def _api_url() -> str:
    """API base URL for commands that need the running scheduler."""
    return os.environ.get("LORAHUB_API_URL", "http://127.0.0.1:6006")


def _post(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST to the running API. Raises typer.Exit on connection failure."""
    url = f"{_api_url().rstrip('/')}{path}"
    payload = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 (lorahub-controlled URL)
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode("utf-8", errors="replace")
        console.print(f"[red]HTTP {exc.code}[/red] {url}: {body_txt}")
        raise typer.Exit(code=1) from exc
    except urllib.error.URLError as exc:
        console.print(
            f"[red]could not reach[/red] {url}: {exc.reason}\n"
            "[yellow]is `lorahub serve` running on this host?[/yellow]"
        )
        raise typer.Exit(code=1) from exc


def _state_color(state: JobState) -> str:
    return {
        JobState.queued: "yellow",
        JobState.running: "cyan",
        JobState.succeeded: "green",
        JobState.failed: "red",
        JobState.canceled: "dim",
        JobState.canceling: "yellow",
        JobState.interrupted: "magenta",
    }.get(state, "white")


@jobs_app.command("ls")
def jobs_ls(
    state: Annotated[
        str | None,
        typer.Option(
            "--state",
            "-s",
            help="Filter by state (queued/running/succeeded/failed/canceled/interrupted).",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max rows to show.")] = 50,
) -> None:
    """List jobs sorted by creation time, newest first."""
    store = JobStore(_store_path())
    records = store.list()
    if state:
        try:
            target = JobState(state)
        except ValueError as exc:
            console.print(f"[red]unknown state {state!r}[/red]")
            raise typer.Exit(code=2) from exc
        records = [r for r in records if r.state is target]
    records.sort(key=lambda r: r.created_at, reverse=True)
    records = records[:limit]

    table = Table(show_lines=False, padding=(0, 1))
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("state")
    table.add_column("name")
    table.add_column("workspace", style="dim")
    table.add_column("created", style="dim")
    for r in records:
        snap = r.config_snapshot or {}
        out = snap.get("output") if isinstance(snap, dict) else None
        name = (
            out.get("name") if isinstance(out, dict) and isinstance(out.get("name"), str)
            else "—"
        )
        table.add_row(
            r.id[-12:],
            f"[{_state_color(r.state)}]{r.state.value}[/{_state_color(r.state)}]",
            name,
            str(r.workspace),
            r.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)
    if not records:
        console.print("[dim]no jobs[/dim]")


@jobs_app.command("show")
def jobs_show(
    job_id: Annotated[str, typer.Argument(help="Full or trailing job id.")],
) -> None:
    """Print one job's full record (state, metrics, metadata, error)."""
    store = JobStore(_store_path())
    rec = _resolve_record(store, job_id)
    payload = {
        "id": rec.id,
        "state": rec.state.value,
        "workspace": str(rec.workspace),
        "created_at": rec.created_at.isoformat(),
        "started_at": rec.started_at.isoformat() if rec.started_at else None,
        "finished_at": rec.finished_at.isoformat() if rec.finished_at else None,
        "pid": rec.pid,
        "returncode": rec.returncode,
        "error": rec.error,
        "metadata": rec.metadata,
        "metrics": rec.metrics,
        "final_metrics": rec.final_metrics,
    }
    console.print_json(data=payload)


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: Annotated[str, typer.Argument(help="Full or trailing job id.")],
) -> None:
    """Cancel a queued job. Use ``kill`` for running jobs."""
    store = JobStore(_store_path())
    rec = _resolve_record(store, job_id)
    if rec.state is not JobState.queued:
        console.print(
            f"[yellow]job {rec.id[-12:]} is {rec.state.value}, not queued — "
            f"use `lorahub jobs kill` to terminate a running job.[/yellow]"
        )
        raise typer.Exit(code=1)
    rec.state = JobState.canceled
    store.upsert(rec)
    console.print(f"[green]canceled[/green] {rec.id[-12:]}")


@jobs_app.command("kill")
def jobs_kill(
    job_id: Annotated[str, typer.Argument(help="Full or trailing job id.")],
    force: Annotated[bool, typer.Option("--force", "-9", help="SIGKILL immediately.")] = False,
) -> None:
    """Stop a running job by PID, then mark it canceled in the store."""
    store = JobStore(_store_path())
    rec = _resolve_record(store, job_id)
    if rec.pid is None:
        console.print(
            f"[yellow]job {rec.id[-12:]} has no recorded pid (state={rec.state.value})[/yellow]"
        )
        rec.state = JobState.canceled
        store.upsert(rec)
        console.print("[green]marked canceled[/green]")
        return
    pid = rec.pid
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
        console.print(f"[dim]sent {sig.name} to pid {pid}[/dim]")
    except ProcessLookupError:
        console.print(f"[yellow]pid {pid} no longer alive[/yellow]")
    except PermissionError as exc:
        console.print(f"[red]cannot signal pid {pid}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not force:
        # Give graceful shutdown 5s before escalating.
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            console.print("[yellow]worker didn't exit; escalating to SIGKILL[/yellow]")
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)

    rec.state = JobState.canceled
    store.upsert(rec)
    console.print(f"[green]killed[/green] {rec.id[-12:]}")


@jobs_app.command("resume")
def jobs_resume(
    job_id: Annotated[str, typer.Argument(help="Full or trailing job id.")],
) -> None:
    """Resume an interrupted job from its last checkpoint.

    Hits ``POST /api/jobs/{id}/resume`` on the running server. If the
    server isn't up, this command can't help — resuming requires the
    scheduler thread to re-enqueue.
    """
    store = JobStore(_store_path())
    rec = _resolve_record(store, job_id)
    body = _post(f"/api/jobs/{rec.id}/resume")
    new_id = body.get("id") or body.get("job_id")
    console.print(f"[green]resumed[/green] → new job {new_id}")


@jobs_app.command("rerun")
def jobs_rerun(
    job_id: Annotated[str, typer.Argument(help="Full or trailing job id.")],
) -> None:
    """Re-launch a finished job with the same recipe.

    Same caveat as ``resume`` — needs ``lorahub serve`` running.
    """
    store = JobStore(_store_path())
    rec = _resolve_record(store, job_id)
    body = _post(f"/api/jobs/{rec.id}/rerun")
    new_id = body.get("id") or body.get("job_id")
    console.print(f"[green]rerun[/green] → new job {new_id}")


def _resolve_record(store: JobStore, job_id: str) -> Any:
    """Match by exact id, or by trailing suffix when the user typed
    just the last 12 chars (the format `lorahub jobs ls` prints)."""
    rec = store.get(job_id)
    if rec is not None:
        return rec
    candidates = [r for r in store.list() if r.id.endswith(job_id)]
    if not candidates:
        console.print(f"[red]no job matches {job_id!r}[/red]")
        raise typer.Exit(code=1)
    if len(candidates) > 1:
        console.print(
            f"[red]ambiguous suffix {job_id!r} matches {len(candidates)} jobs:[/red]"
        )
        for c in candidates:
            console.print(f"  {c.id} ({c.state.value})")
        raise typer.Exit(code=1)
    return candidates[0]


__all__ = ["jobs_app"]
