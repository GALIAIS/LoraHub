"""LoraHub CLI entry point.

Commands:
    lorahub validate <config>   Check a config without launching training.
    lorahub info <config>       Show compiled argv and VRAM estimate (dry run).
    lorahub train <config>      Run training to completion.
    lorahub sweep <config>      Expand a grid sweep into per-variant configs.
    lorahub init <name>         Scaffold a starter config in the current dir.

Language: every user-facing string in this CLI lives in
``lorahub/cli/_i18n.py`` and is selected by the ``--lang zh|en``
global flag (default ``zh``) or the ``LORAHUB_LANG`` env var. The
flag has to be parsed *before* the typer subcommand modules are
imported because their decorators evaluate ``t(...)`` at import time
to populate help strings — see ``_pre_parse_lang`` below.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from lorahub import __version__
from lorahub.cli._i18n import set_lang, t
from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.backends.kohya.compiler import compile_config
from lorahub.core.config.loader import load_config
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.dataset.sources import bangumi_base
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

load_dotenv()  # picks up .env from cwd; existing env vars take precedence


def _pre_parse_lang() -> None:
    """Resolve ``--lang`` before subcommand modules import.

    Every subcommand module evaluates ``t(...)`` at import time to
    populate help strings — by the time typer's callback fires those
    strings are already frozen. Sniff sys.argv for ``--lang`` (and
    ``LORAHUB_LANG`` env via the helper) and pin the language now,
    so subsequent imports pick up the correct locale.
    """
    argv = sys.argv[1:]
    requested: str | None = None
    for i, tok in enumerate(argv):
        if tok in {"--lang", "-L"} and i + 1 < len(argv):
            requested = argv[i + 1]
            break
        if tok.startswith("--lang="):
            requested = tok.split("=", 1)[1]
            break
    set_lang(requested)


_pre_parse_lang()

app = typer.Typer(
    name="lorahub",
    help=t("app.help"),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# Sub-command groups (B9). Each registers as a Typer of its own so the
# CLI surface mirrors the API: `lorahub jobs ls`, `lorahub sweep submit`,
# `lorahub system gpu`. Imported lazily to avoid the import chain pulling
# in the API store at module load when the user just runs `--help`.
from lorahub.cli.jobs import jobs_app  # noqa: E402
from lorahub.cli.manage_cmd import manage_app  # noqa: E402
from lorahub.cli.service import service_app  # noqa: E402
from lorahub.cli.sweep import sweep_app  # noqa: E402
from lorahub.cli.system import system_app  # noqa: E402

app.add_typer(jobs_app, name="jobs")
# Plural: the existing `lorahub sweep <args>` top-level command stays as
# the in-process one-shot grid sweep tool. The new sub-app drives the
# server's /api/sweeps endpoint (mode/n_trials/seed-aware) so power
# users can submit + list adaptive sweeps without the web UI.
app.add_typer(sweep_app, name="sweeps")
app.add_typer(system_app, name="system")
app.add_typer(service_app, name="service")
# `manage` replaces the old `self` group. The rename is hard — old
# `lorahub self update` no longer routes; users must learn the new
# verb. Per-command help / hint strings are localised via _i18n so
# `lorahub --lang en manage --help` still produces a sensible page.
app.add_typer(manage_app, name="manage")
err_console = Console(stderr=True)


@app.callback()
def _root(
    lang: Annotated[
        str | None,
        typer.Option(
            "--lang",
            "-L",
            help=t("app.lang.option"),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Root callback: re-affirms the language for subcommand bodies.

    The actual import-time language pin happens in ``_pre_parse_lang``
    above. Re-running ``set_lang`` here costs nothing and lets typer
    surface the option in ``--help``.
    """
    if lang is not None:
        set_lang(lang)


@app.command(help=t("app.version.help"))
def version() -> None:
    """Print the installed lorahub version."""
    console.print(f"lorahub {__version__}")


@app.command(help=t("app.doctor.help"))
def doctor() -> None:
    """Inspect the local install — venv, Python, Node, web/dist, backends.

    Prints a short table listing each component, where it should live,
    and whether it's actually there. Use this as the first step when
    something feels off ("`lorahub serve` says module not found",
    "the UI shows port :18765 but my browser can't reach it", etc.).
    """
    import sys as _sys  # noqa: PLC0415

    from lorahub.core.paths import lorahub_dir, project_root  # noqa: PLC0415

    root = project_root()
    venv = root / ".venv"
    venv_py = venv / ("Scripts" if _sys.platform == "win32" else "bin") / (
        "python.exe" if _sys.platform == "win32" else "python"
    )
    node_dir = root / ".node"
    node_bin = node_dir / ("node.exe" if _sys.platform == "win32" else "bin/node")
    web_dist = root / "web" / "dist" / "index.html"
    uv_dir = lorahub_dir() / "uv"
    uv_bin = uv_dir / ("uv.exe" if _sys.platform == "win32" else "uv")
    py_dir = lorahub_dir() / "python"

    table = Table(title=t("doctor.title", root=root), show_lines=False)
    table.add_column(t("doctor.col.component"))
    table.add_column(t("doctor.col.location"))
    table.add_column(t("doctor.col.status"))

    install_hint = t("doctor.hint.install")
    build_hint = t("doctor.hint.build")

    def row(label: str, path: Path, present: bool, hint: str = "") -> None:
        status = t("doctor.status.ok") if present else t("doctor.status.missing")
        if hint and not present:
            status += f"  [dim]{hint}[/]"
        table.add_row(label, str(path), status)

    row("interpreter", Path(_sys.executable), Path(_sys.executable).is_file())
    row(".venv", venv_py, venv_py.is_file(), install_hint)
    row(".lorahub/uv", uv_bin, uv_bin.is_file(), install_hint)
    row(".lorahub/python", py_dir, py_dir.is_dir() and any(py_dir.iterdir()) if py_dir.is_dir() else False, install_hint)
    row(".node", node_bin, node_bin.is_file(), install_hint)
    row("web/dist", web_dist, web_dist.is_file(), build_hint)

    console.print(table)

    # --- environment / encoding sanity ---------------------------------
    # Things that don't fit the component table but which still cause
    # opaque mid-training failures when wrong (path encoding, low disk,
    # GPU not visible).
    env_table = Table(title=t("doctor.env.title"), show_lines=False)
    env_table.add_column(t("doctor.env.col.check"))
    env_table.add_column(t("doctor.env.col.detail"))
    env_table.add_column(t("doctor.col.status"))

    def env_row(label: str, detail: str, ok: bool, hint: str = "") -> None:
        status = t("doctor.status.ok") if ok else t("doctor.status.warn")
        if hint and not ok:
            status += f"  [dim]{hint}[/]"
        env_table.add_row(label, detail, status)

    # mbcs path encoding (Windows only). Anything containing characters
    # the active ANSI code page can't represent will eventually corrupt
    # subprocess argv or sd-scripts log paths.
    if _sys.platform == "win32":
        path_str = str(root)
        try:
            path_str.encode("mbcs")
            env_row(t("doctor.env.path_encoding"), path_str, True)
        except UnicodeEncodeError as exc:
            env_row(
                t("doctor.env.path_encoding"),
                t(
                    "doctor.env.path_offending",
                    path=path_str,
                    start=exc.start,
                    end=exc.end,
                ),
                False,
                t("doctor.env.path_hint"),
            )

    # Disk space on the workspace volume.
    import shutil as _shutil  # noqa: PLC0415
    try:
        runs_root = root / "runs"
        probe = runs_root if runs_root.exists() else root
        usage = _shutil.disk_usage(probe)
        free_gib = usage.free / 1024**3
        ok = free_gib >= 5.0
        env_row(
            t("doctor.env.disk"),
            t("doctor.env.disk_detail", free_gib=free_gib, probe=probe),
            ok,
            t("doctor.env.disk_hint"),
        )
    except OSError:
        pass

    # GPU visibility via `nvidia-smi -L`. Best-effort; absence is a
    # warn, not an error (CPU-only installs are valid).
    import subprocess as _sp  # noqa: PLC0415
    try:
        out = _sp.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        env_row(
            t("doctor.env.gpu"),
            t("doctor.env.gpu_count", n=len(gpus)) if gpus else t("doctor.env.gpu_none"),
            bool(gpus),
            t("doctor.env.gpu_hint_cpu"),
        )
    except (FileNotFoundError, OSError, _sp.TimeoutExpired):
        env_row(
            t("doctor.env.gpu"),
            t("doctor.env.gpu_missing"),
            False,
            t("doctor.env.gpu_missing_hint"),
        )

    console.print(env_table)

    # Backend status — best-effort, only if the api extras imported.
    try:
        from lorahub.api import app as _app_mod  # noqa: PLC0415
        from lorahub.api.settings import probe_all_backends  # noqa: PLC0415

        store = getattr(_app_mod, "_settings_store", None)
        if store is not None:
            settings = store.load()
            backends = probe_all_backends(settings)
            be_table = Table(title=t("doctor.backends.title"), show_lines=False)
            be_table.add_column(t("doctor.backends.col.id"))
            be_table.add_column(t("doctor.backends.col.ready"))
            be_table.add_column(t("doctor.backends.col.python"))
            be_table.add_column(t("doctor.backends.col.notes"))
            for bid, info in backends.items():
                ready = (
                    t("doctor.backends.ready_yes")
                    if info.get("ready")
                    else t("doctor.backends.ready_no")
                )
                py = info.get("python") or "-"
                notes: list[str] = []
                if info.get("missing_scripts"):
                    notes.append(
                        t("doctor.backends.note_missing_scripts", n=len(info["missing_scripts"]))
                    )
                if info.get("missing_models"):
                    notes.append(
                        t("doctor.backends.note_missing_models", n=len(info["missing_models"]))
                    )
                if not info.get("venv_detected"):
                    notes.append(t("doctor.backends.note_no_venv"))
                be_table.add_row(bid, ready, str(py), ", ".join(notes) or "—")
            console.print(be_table)
    except Exception:  # noqa: BLE001
        # api extras not installed — that's a valid user choice for the
        # CLI-only flow. doctor's table above is the answer they need.
        pass


@app.command(help=t("validate.help"))
def validate(
    config: Annotated[Path, typer.Argument(help=t("cli.config_arg_help"))],
) -> None:
    """Validate a config without running training."""
    cfg = load_config(config)
    backend = KohyaBackend()
    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)
    console.print(t("validate.ok"))


@app.command(help=t("info.help"))
def info(
    config: Annotated[Path, typer.Argument(help=t("cli.config_arg_help"))],
) -> None:
    """Show what a config would compile to, plus VRAM estimate (no training)."""
    cfg = load_config(config)
    backend = KohyaBackend()

    script, argv, _files, _env = compile_config(cfg, workspace=Path.cwd() / "_dryrun")
    est = backend.estimate_vram(cfg)

    table = Table(title=t("info.title"), show_header=False, expand=False)
    table.add_row("config", str(config))
    table.add_row("arch", cfg.base_model.arch)
    table.add_row("network", f"{cfg.network.type} rank={cfg.network.rank} alpha={cfg.network.alpha}")
    table.add_row("schedule", f"{cfg.schedule.epochs} epochs x bs={cfg.schedule.batch_size}")
    table.add_row("precision", cfg.precision)
    table.add_row("entry script", script)
    table.add_row("estimated VRAM", f"{est.total_gib:.1f} GiB")
    console.print(table)

    console.print("\n" + t("info.compiled_argv"))
    for a in argv:
        console.print(f"  {a}")


@app.command(help=t("train.help"))
def train(
    config: Annotated[Path, typer.Argument(help=t("cli.config_arg_help"))],
    workspace: Annotated[
        Path | None,
        typer.Option(help=t("cli.workspace_help")),
    ] = None,
) -> None:
    """Run training to completion. Press Ctrl+C to stop gracefully."""
    cfg = load_config(config)
    backend = KohyaBackend()

    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)

    ws = (workspace or (Path.cwd() / "runs" / cfg.output.name)).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    console.print(t("train.workspace_label", ws=ws))

    events_log = ws / "events.jsonl"
    with JsonlEventSink(events_log) as sink:

        def on_event(ev: TrainingEvent) -> None:
            sink(ev)
            _render_event(ev)

        handle = backend.launch(cfg, workspace=ws, on_event=on_event)
        console.print(t("train.process_label", pid=handle.pid, job=handle.job_id))
        try:
            rc = handle.wait()
        except KeyboardInterrupt:
            console.print(t("train.interrupt"))
            handle.stop(graceful=True)
            rc = handle.wait()

    if rc != 0:
        err_console.print(t("train.failed", rc=rc))
        raise typer.Exit(code=rc)
    console.print(t("train.ok"))


@app.command(help=t("sweep.help"))
def sweep(
    config: Annotated[Path, typer.Argument(help="Path to the base config YAML file.")],
    axis: Annotated[
        list[str],
        typer.Option(
            "--axis",
            help=(
                "Sweep axis spec, repeatable: 'dotted.path=v1,v2,v3'. "
                "Values are parsed as JSON when possible (so 1e-4, 32, true work) "
                "and fall back to string."
            ),
        ),
    ],
    name_template: Annotated[
        str,
        typer.Option(
            "--name-template",
            help="Per-variant name template ({base} = base output.name, {i} = 1-based index).",
        ),
    ] = "{base}-{i:03d}",
    workspace_root: Annotated[
        Path,
        typer.Option(
            "--workspace-root",
            help="Where each materialised variant config should record its workspace.",
        ),
    ] = Path("./runs"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help=(
                "Where to write generated `variant_NNN.yaml` files plus the "
                "`sweep.json` mapping. Defaults to ./configs/sweep-<name>."
            ),
        ),
    ] = Path("./configs"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print each variant name and config diff; do not write any files.",
        ),
    ] = False,
) -> None:
    """Expand a grid sweep into per-variant config files.

    The CLI never spawns training subprocesses for sweeps — running N kohya
    or diffusion-pipe processes in parallel would deadlock most workstations.
    Instead this command writes ``variant_NNN.yaml`` files plus a ``sweep.json``
    manifest so the operator can pick the dispatcher (kohya in tmux, the API
    server's serial scheduler, slurm, etc.).
    """
    import json  # noqa: PLC0415

    from lorahub.core.config.loader import dump_config  # noqa: PLC0415
    from lorahub.core.sweep import (  # noqa: PLC0415
        SweepAxis,
        SweepError,
        SweepPlan,
    )

    if not axis:
        err_console.print(t("sweep.need_axis"))
        raise typer.Exit(code=1)

    # Validate the base config up front so the user sees schema errors before
    # we materialise N copies of a broken config.
    cfg = load_config(config)
    base_dict = cfg.model_dump(mode="json", exclude_none=True)

    axes: list[SweepAxis] = []
    for spec in axis:
        if "=" not in spec:
            err_console.print(t("sweep.bad_axis", spec=spec))
            raise typer.Exit(code=1)
        path, _, raw_values = spec.partition("=")
        path = path.strip()
        if not path:
            err_console.print(t("sweep.empty_axis_path", spec=spec))
            raise typer.Exit(code=1)
        values = [_coerce_axis_value(tok) for tok in raw_values.split(",") if tok.strip()]
        if not values:
            err_console.print(t("sweep.no_values", path=path))
            raise typer.Exit(code=1)
        axes.append(SweepAxis(path=path, values=values))

    plan = SweepPlan(base_config=base_dict, axes=axes, name_template=name_template)
    try:
        variants = plan.expand()
    except SweepError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    base_name = base_dict.get("output", {}).get("name", "sweep")
    sweep_dir_name = f"sweep-{base_name}"

    if dry_run:
        console.print(t("sweep.dry_run_header", n=len(variants)))
        for i, (variant_name, _variant_config) in enumerate(variants, start=1):
            diff = plan.axis_values_for(i)
            console.print(f"[cyan]{variant_name}[/cyan]  {diff}")
        return

    # Re-validate every variant before writing — catches axis values that
    # violate pydantic constraints (e.g. negative LR, rank > 512).
    for variant_name, variant_config in variants:
        try:
            TrainingConfig.model_validate(variant_config)
        except Exception as exc:  # noqa: BLE001
            err_console.print(t("sweep.variant_invalid", name=variant_name, err=exc))
            raise typer.Exit(code=1) from exc

    target_dir = (output_dir / sweep_dir_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "base_config": str(config.resolve()),
        "name_template": name_template,
        "workspace_root": str(workspace_root.resolve()),
        "axes": [{"path": a.path, "values": list(a.values)} for a in axes],
        "variants": [],
    }
    for i, (variant_name, variant_config) in enumerate(variants, start=1):
        # `dump_config` requires a TrainingConfig — round-trip through validation
        # both confirms the variant is valid and normalises field ordering.
        cfg_v = TrainingConfig.model_validate(variant_config)
        variant_path = target_dir / f"variant_{i:03d}.yaml"
        dump_config(cfg_v, variant_path)
        manifest["variants"].append(  # type: ignore[union-attr]
            {
                "name": variant_name,
                "path": str(variant_path),
                "axis_values": plan.axis_values_for(i),
            }
        )

    manifest_path = target_dir / "sweep.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(t("sweep.ok", n=len(variants), dir=target_dir))
    console.print(t("sweep.manifest", path=manifest_path))


def _coerce_axis_value(token: str) -> Any:
    """Best-effort cast of a CLI axis token to an int / float / bool / null / string.

    JSON parses the obvious literals (`32`, `1e-4`, `true`, `null`) so users
    get the typed values pydantic expects. Anything else (`adamw8bit`,
    `cosine_with_restarts`) falls back to the trimmed string.
    """
    import json as _json  # noqa: PLC0415

    text = token.strip()
    try:
        return _json.loads(text)
    except (ValueError, TypeError):
        return text


@app.command(help=t("serve.help"))
def serve(
    host: Annotated[str, typer.Option(help=t("serve.host_help"))] = "127.0.0.1",
    port: Annotated[int, typer.Option(help=t("serve.port_help"))] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help=t("serve.reload_help")),
    ] = False,
) -> None:
    """Run the LoraHub HTTP API server (REST + WebSocket)."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        err_console.print(t("serve.api_extras_missing"))
        raise typer.Exit(code=1) from exc

    console.print(t("serve.banner", host=host, port=port))
    uvicorn.run(
        "lorahub.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command(help=t("init.help"))
def init(
    name: Annotated[str, typer.Argument(help=t("init.name_help"))],
    template: Annotated[
        str, typer.Option(help=t("init.template_help"))
    ] = "anima_lora_default",
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help=t("init.auto_help"),
        ),
    ] = False,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help=t("init.checkpoint_help")),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help=t("init.dataset_help")),
    ] = None,
    vram_mib: Annotated[
        int | None,
        typer.Option(
            "--vram-mib",
            help=t("init.vram_help"),
        ),
    ] = None,
) -> None:
    """Scaffold a starter config in the current directory."""
    dst = Path.cwd() / f"{name}.yaml"
    if dst.exists():
        err_console.print(t("init.exists", path=dst))
        raise typer.Exit(code=1)

    if auto:
        if checkpoint is None or dataset is None:
            err_console.print(t("init.auto_requires"))
            raise typer.Exit(code=1)
        from lorahub.core.config import scaffold
        from lorahub.core.config.loader import dump_config

        cfg = scaffold.auto_scaffold(
            name=name,
            checkpoint=checkpoint.resolve(),
            dataset=dataset.resolve(),
            vram_mib=vram_mib,
        )
        dump_config(cfg, dst)
        images = scaffold.count_images(dataset.resolve())
        console.print(
            t(
                "init.created",
                dst=dst,
                arch=cfg.base_model.arch,
                rank=cfg.network.rank,
                batch=cfg.schedule.batch_size,
                accum=cfg.schedule.grad_accum,
                images=images,
                repeats=cfg.dataset.num_repeats,
            )
        )
        return

    src = _builtin_config(template)
    if not src.exists():
        err_console.print(t("init.unknown_template", name=template))
        raise typer.Exit(code=1)
    shutil.copy2(src, dst)
    console.print(t("init.copied", dst=dst))


@app.command("bootstrap-kohya", help=t("bootstrap.help"))
def bootstrap_kohya(
    target: Annotated[
        Path,
        typer.Option(help=t("bootstrap.target_help")),
    ] = Path("./sd-scripts"),
    cuda: Annotated[
        str, typer.Option("--cuda", help=t("bootstrap.cuda_help"))
    ] = "cu124",
    torch_version: Annotated[
        str, typer.Option("--torch", help=t("bootstrap.torch_help"))
    ] = "2.6.0",
    torchvision_version: Annotated[
        str, typer.Option("--torchvision", help=t("bootstrap.torchvision_help"))
    ] = "0.21.0",
    no_xformers: Annotated[
        bool,
        typer.Option("--no-xformers", help=t("bootstrap.no_xformers_help")),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help=t("bootstrap.force_help")),
    ] = False,
) -> None:
    """One-shot install of kohya-ss/sd-scripts (clone + venv + PyTorch + deps + xformers)."""
    from lorahub.core.backends.errors import BootstrapError
    from lorahub.core.backends.kohya import installer

    plan = installer.BootstrapPlan(
        target=target.resolve(),
        cuda_version=cuda,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        install_xformers=not no_xformers,
    )

    if plan.target.exists() and any(plan.target.iterdir()):
        if not force:
            err_console.print(t("bootstrap.target_busy", target=plan.target))
            raise typer.Exit(code=1)
        installer.cleanup_partial(plan)

    console.print(
        t(
            "bootstrap.banner",
            target=plan.target,
            cuda=plan.cuda_version,
            torch=plan.torch_version,
            xformers=plan.install_xformers,
        )
    )

    try:
        installer.bootstrap(
            plan,
            progress=lambda step: console.print(t("bootstrap.step", step=step)),
        )
    except BootstrapError as e:
        err_console.print(t("bootstrap.failed", step=e.step, rc=e.returncode))
        raise typer.Exit(code=1) from e

    console.print(t("bootstrap.ok", target=plan.target))
    console.print(t("bootstrap.env_hint", target=plan.target))


@app.command("fetch-bangumi", help=t("bangumi.help"))
def fetch_bangumi(
    repo: Annotated[
        str,
        typer.Argument(help=t("bangumi.repo_help")),
    ],
    character: Annotated[
        str | None,
        typer.Argument(help=t("bangumi.character_help")),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help=t("bangumi.output_help")),
    ] = Path("./datasets/bangumi"),
    limit: Annotated[
        int | None,
        typer.Option(help=t("bangumi.limit_help")),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(help=t("bangumi.preview_help")),
    ] = False,
    seed_captions: Annotated[
        bool,
        typer.Option(
            "--seed-captions/--no-seed-captions",
            help=t("bangumi.seed_help"),
        ),
    ] = True,
) -> None:
    """Download a single character's images from a BangumiBase HF dataset."""
    if character is None:
        chars = bangumi_base.list_characters(repo)
        console.print(
            t("bangumi.list", n=len(chars), repo=repo, names=", ".join(chars))
        )
        return

    if preview:
        for i in range(1, 9):
            path = bangumi_base.download_preview(repo, character, output / character, index=i)
            console.print(t("bangumi.preview_line", i=i, path=path))
        return

    result = bangumi_base.fetch_character(
        repo,
        character,
        output,
        limit=limit,
        seed_captions=seed_captions,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    console.print(t("bangumi.fetched", n=result.image_count, dir=result.output_dir))
    if result.license:
        console.print(t("bangumi.license", license=result.license))
    if result.image_count and seed_captions:
        console.print(t("bangumi.seed_warn"))


@app.command(help=t("tag.help"))
def tag(
    directory: Annotated[
        Path, typer.Argument(help=t("tag.dir_help"))
    ],
    tagger: Annotated[
        str,
        typer.Option(
            "--tagger",
            help=t("tag.tagger_help"),
        ),
    ] = "wd14",
    model: Annotated[
        str, typer.Option(help=t("tag.model_help"))
    ] = "SmilingWolf/wd-eva02-large-tagger-v3",
    general_threshold: Annotated[
        float, typer.Option("--general", help=t("tag.general_help"))
    ] = 0.35,
    character_threshold: Annotated[
        float, typer.Option("--character", help=t("tag.character_help"))
    ] = 0.85,
    joytag_threshold: Annotated[
        float,
        typer.Option(
            "--joytag-threshold",
            help=t("tag.joytag_help"),
        ),
    ] = 0.4,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help=t("tag.recursive_help")),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help=t("tag.overwrite_help"))
    ] = False,
    underscores: Annotated[
        bool, typer.Option("--underscores", help=t("tag.underscores_help"))
    ] = False,
    include_character: Annotated[
        bool,
        typer.Option(
            "--include-character/--no-include-character",
            help=t("tag.include_character_help"),
        ),
    ] = True,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=t("tag.device_help"),
        ),
    ] = "auto",
) -> None:
    """Auto-tag images and write kohya-style .txt captions.

    Supports WD14/WD-v3 (ONNX) and JoyTag (PyTorch). Default is WD14.
    """
    from lorahub.core.tagging.base import BaseTagger  # noqa: PLC0415

    if not directory.is_dir():
        err_console.print(t("tag.not_a_dir", path=directory))
        raise typer.Exit(code=1)

    kind = tagger.lower()
    if kind not in {"wd14", "joytag"}:
        err_console.print(t("tag.unknown_tagger", name=tagger))
        raise typer.Exit(code=1)

    instance: BaseTagger
    if kind == "joytag":
        from lorahub.core.tagging.joytag import JoyTagger, JoyTagModelError  # noqa: PLC0415

        instance = JoyTagger(predict_threshold=joytag_threshold, device=device)
        console.print(t("tag.loading_joytag"))
        try:
            instance.load()
        except JoyTagModelError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    else:
        from lorahub.core.tagging.wd14 import CudaUnavailableError, WD14Tagger  # noqa: PLC0415

        instance = WD14Tagger(
            model_id=model,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            device=device,
        )
        console.print(t("tag.loading_wd", model=model))
        try:
            instance.load()
        except CudaUnavailableError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e

    console.print(t("tag.running_on", provider=instance.active_provider))

    def _on_progress(path: Path, _result: object) -> None:
        console.print(t("tag.tagged_one", name=path.name))

    results = instance.tag_directory(
        directory,
        recursive=recursive,
        write_caption=True,
        skip_existing=not overwrite,
        underscores=underscores,
        include_character=include_character,
        on_progress=_on_progress,
    )

    console.print(t("tag.ok", n=len(results)))




@app.command("anima-caption", help=t("anima.help"))
def anima_caption(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory of *.txt caption files to rewrite in Anima layout."),
    ],
    dataset_tag: Annotated[
        str | None,
        typer.Option(
            "--dataset-tag",
            help="Optional non-anime subset header (e.g. 'ye-pop', 'deviantart').",
        ),
    ] = None,
    quality: Annotated[
        str | None,
        typer.Option(
            "--quality",
            help="Comma-separated quality tags to inject (e.g. 'masterpiece,best quality').",
        ),
    ] = None,
    score: Annotated[
        str | None,
        typer.Option(
            "--score",
            help="Comma-separated PonyV7 score tags to inject (e.g. 'score_7').",
        ),
    ] = None,
    safety: Annotated[
        str | None,
        typer.Option(
            "--safety",
            help="Default safety tag if absent: safe / sensitive / nsfw / explicit. "
            "Pass empty string to disable.",
        ),
    ] = "safe",
    year: Annotated[
        str | None,
        typer.Option(
            "--year",
            help="Comma-separated year tags to inject (e.g. 'year 2025,newest').",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Rewrite existing captions in place."),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Recurse into subdirectories."),
    ] = False,
) -> None:
    """Rewrite *.txt captions to Anima's recommended layout (no tagger inference)."""
    from lorahub.core.dataset.anima import AnimaDatasetTransformer  # noqa: PLC0415

    if not directory.is_dir():
        err_console.print(t("tag.not_a_dir", path=directory))
        raise typer.Exit(code=1)

    def _split(value: str | None) -> list[str] | None:
        if value is None:
            return None
        items = [tok.strip() for tok in value.split(",") if tok.strip()]
        return items or None

    transformer = AnimaDatasetTransformer(
        default_quality=_split(quality),
        default_score=_split(score),
        default_year=_split(year),
        default_safety=safety if safety else None,
        dataset_tag=dataset_tag,
    )

    def _on_progress(path: Path) -> None:
        console.print(t("anima.rewrote_one", name=path.name))

    written = transformer.transform_directory(
        directory,
        recursive=recursive,
        overwrite=overwrite,
        progress=_on_progress,
    )
    if not overwrite:
        console.print(t("anima.dry_run"))
    console.print(t("anima.ok", n=written))


@app.command(help=t("caption.help"))
def caption(
    action: Annotated[
        str,
        typer.Argument(
            help="Caption sub-action. Currently only 'normalize' is supported.",
        ),
    ],
    directory: Annotated[
        Path, typer.Argument(help="Directory of .txt caption files to process.")
    ],
    blacklist: Annotated[
        str,
        typer.Option(
            "--blacklist",
            help="Comma-separated tags to drop (case-insensitive).",
        ),
    ] = "",
    remap: Annotated[
        str,
        typer.Option(
            "--remap",
            help='Comma-separated rewrite rules, "old:new,old2:new2". Empty new deletes.',
        ),
    ] = "",
    known_artists: Annotated[
        str,
        typer.Option(
            "--known-artists",
            help="Comma-separated artist tags to prefix with @ (Animagine convention).",
        ),
    ] = "",
    quality: Annotated[
        str,
        typer.Option(
            "--quality",
            help='Comma-separated quality tags to prepend, e.g. "masterpiece,best quality".',
        ),
    ] = "",
    score: Annotated[
        str,
        typer.Option(
            "--score",
            help='Comma-separated score chain to prepend, e.g. "score_9,score_8_up".',
        ),
    ] = "",
    safety: Annotated[
        str,
        typer.Option("--safety", help="Safety marker to prepend, e.g. 'safe'."),
    ] = "",
    shuffle: Annotated[
        bool, typer.Option("--shuffle", help="Random-shuffle non-anchored tags.")
    ] = False,
    keep_n: Annotated[
        int,
        typer.Option(
            "--keep-n", help="Anchor the first N tags during shuffle (kohya keep_tokens)."
        ),
    ] = 0,
    drop_rate: Annotated[
        float,
        typer.Option(
            "--drop-rate",
            help="Probability of dropping each non-anchored tag (0..1).",
        ),
    ] = 0.0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="PRNG seed for reproducible shuffle/dropout."),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Recurse into subdirectories."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Reserved for future skip-if-already-cleaned logic; currently a no-op.",
        ),
    ] = False,
    booru_alias: Annotated[
        bool,
        typer.Option(
            "--booru-alias",
            help=(
                "Apply the curated Danbooru->Gelbooru alias table after "
                "--remap (off by default; user --remap rules still win)."
            ),
        ),
    ] = False,
) -> None:
    """Clean booru-style captions in place (Illustrious / Pony / Animagine / NoobAI).

    Generic preprocessing toolkit: lowercase, swap underscores for spaces,
    dedupe, drop blacklisted tags, remap one tag to another, prepend
    quality/score/safety markers, optionally shuffle and dropout-regularise.
    Pony score_N tags and quality/safety markers are anchored against
    dropout so the prompt's stylistic spine survives.
    """
    from lorahub.core.dataset.captions import CaptionPipeline  # noqa: PLC0415

    if action != "normalize":
        err_console.print(t("caption.unknown_action", action=action))
        raise typer.Exit(code=1)
    if not directory.is_dir():
        err_console.print(t("tag.not_a_dir", path=directory))
        raise typer.Exit(code=1)

    pipeline = CaptionPipeline(
        blacklist=_parse_csv_set(blacklist),
        remap=_parse_remap(remap),
        known_artists=_parse_csv_set(known_artists),
        quality=_parse_csv_list(quality) or None,
        score=_parse_csv_list(score) or None,
        safety=safety.strip() or None,
        shuffle=shuffle,
        keep_n=keep_n,
        drop_rate=drop_rate,
        seed=seed,
        apply_booru_alias=booru_alias,
    )

    def _on_progress(p: Path, done: int, total: int) -> None:
        console.print(t("caption.progress", done=done, total=total, name=p.name))

    written = pipeline.transform_directory(
        directory,
        recursive=recursive,
        overwrite=overwrite,
        progress=_on_progress,
    )
    console.print(t("caption.ok", n=written))


def _parse_csv_list(raw: str) -> list[str]:
    """Split a comma-separated CLI option into trimmed, non-empty tokens."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_csv_set(raw: str) -> set[str]:
    return set(_parse_csv_list(raw))


def _parse_remap(raw: str) -> dict[str, str]:
    """Parse ``"old:new,old2:new2"`` into ``{old: new, old2: new2}``."""
    rules: dict[str, str] = {}
    for token in _parse_csv_list(raw):
        if ":" not in token:
            continue
        key, _, value = token.partition(":")
        key = key.strip()
        if key:
            rules[key] = value.strip()
    return rules


def _builtin_config(name: str) -> Path:
    package_root = Path(__file__).resolve().parent.parent.parent
    return package_root / "configs" / f"{name}.yaml"


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
