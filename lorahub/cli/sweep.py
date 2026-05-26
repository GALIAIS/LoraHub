"""Sweep CLI — submit a sweep config to the running scheduler.

Wraps ``POST /api/sweeps`` so a user can drive grid / random / TPE
search without opening the web UI. Reads a YAML config that maps to
``CreateSweepRequest`` (axes + name_template + mode + n_trials etc.).

Why HTTP instead of in-process: the sweep scheduler shares the same
single-slot worker as `lorahub train`. Going through the API ensures
trials enqueue against the live scheduler instead of running detached.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console

from lorahub.cli._i18n import t

console = Console()
sweep_app = typer.Typer(
    help=t("sweepapp.help"),
    no_args_is_help=True,
)


def _api_url() -> str:
    return os.environ.get("LORAHUB_API_URL", "http://127.0.0.1:6006")


@sweep_app.command("submit", help=t("sweepapp.submit.help"))
def sweep_submit(
    config: Annotated[
        Path, typer.Argument(help="YAML file shaped like CreateSweepRequest.")
    ],
    workspace_root: Annotated[
        Path | None,
        typer.Option(help="Override workspace_root for the sweep."),
    ] = None,
) -> None:
    """Submit a sweep YAML to ``POST /api/sweeps`` and print the response.

    The YAML body must contain ``base_config`` + ``axes`` (list with at
    least one entry); ``mode`` / ``n_trials`` / ``seed`` /
    ``name_template`` are optional. Anything not in the YAML falls back
    to upstream defaults defined on ``CreateSweepRequest``.
    """
    if not config.is_file():
        console.print(t("sweepapp.not_a_file", path=config))
        raise typer.Exit(code=2)
    try:
        body = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        console.print(t("sweepapp.yaml_error", err=exc))
        raise typer.Exit(code=2) from exc
    if not isinstance(body, dict):
        console.print(t("sweepapp.yaml_not_mapping"))
        raise typer.Exit(code=2)
    if workspace_root is not None:
        body["workspace_root"] = str(workspace_root.resolve())

    url = f"{_api_url().rstrip('/')}/api/sweeps"
    req = urllib.request.Request(  # noqa: S310 (lorahub-controlled URL)
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode("utf-8", errors="replace")
        console.print(t("jobs.http_error", code=exc.code, url=url, body=body_txt))
        raise typer.Exit(code=1) from exc
    except urllib.error.URLError as exc:
        console.print(t("sweepapp.unreachable", url=url, reason=exc.reason))
        raise typer.Exit(code=1) from exc

    console.print_json(data=payload)


@sweep_app.command("ls", help=t("sweepapp.ls.help"))
def sweep_ls() -> None:
    """List every sweep on the running server."""
    url = f"{_api_url().rstrip('/')}/api/sweeps"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        console.print(t("sweepapp.unreachable", url=url, reason=exc.reason))
        raise typer.Exit(code=1) from exc

    sweeps = payload.get("sweeps") or []
    if not sweeps:
        console.print(t("sweepapp.empty"))
        return
    for s in sweeps:
        prefix = s.get("name_prefix") or s["sweep_id"][-8:]
        mode = s.get("mode") or "grid"
        console.print(
            f"[cyan]{s['sweep_id'][-12:]}[/cyan] {prefix:<24} "
            f"[dim]{mode:<8}[/dim] "
            f"total={s['total']:>3}  succ={s.get('succeeded', 0):>3}  "
            f"fail={s.get('failed', 0):>3}  run={s.get('running', 0):>3}"
        )


__all__ = ["sweep_app"]
