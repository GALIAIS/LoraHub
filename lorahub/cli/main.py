"""LoraHub CLI entry point.

Commands:
    lorahub validate <config>   Check a config without launching training.
    lorahub info <config>       Show compiled argv and VRAM estimate (dry run).
    lorahub train <config>      Run training to completion.
    lorahub sweep <config>      Expand a grid sweep into per-variant configs.
    lorahub init <name>         Scaffold a starter config in the current dir.
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
from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.backends.kohya.compiler import compile_recipe
from lorahub.core.config.loader import load_recipe
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.dataset.sources import bangumi_base
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

load_dotenv()  # picks up .env from cwd; existing env vars take precedence

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
    recipe: Annotated[Path, typer.Argument(help="Path to a config YAML file.")],
) -> None:
    """Validate a config without running training."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()
    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)
    console.print("[green]OK[/] config valid")


@app.command()
def info(
    recipe: Annotated[Path, typer.Argument(help="Path to a config YAML file.")],
) -> None:
    """Show what a config would compile to, plus VRAM estimate (no training)."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()

    script, argv, _files = compile_recipe(cfg, workspace=Path.cwd() / "_dryrun")
    est = backend.estimate_vram(cfg)

    table = Table(title="Config summary", show_header=False, expand=False)
    table.add_row("config", str(recipe))
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
    recipe: Annotated[Path, typer.Argument(help="Path to a config YAML file.")],
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

    ws = (workspace or (Path.cwd() / "runs" / cfg.output.name)).resolve()
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
            console.print("\n[yellow]Ctrl+C - stopping training gracefully...[/yellow]")
            handle.stop(graceful=True)
            rc = handle.wait()

    if rc != 0:
        err_console.print(f"[red]training failed (rc={rc})[/red]")
        raise typer.Exit(code=rc)
    console.print("[green]OK[/] training complete")


@app.command()
def sweep(
    recipe: Annotated[Path, typer.Argument(help="Path to the base config YAML file.")],
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
                "`sweep.json` mapping. Defaults to ./recipes/sweep-<name>."
            ),
        ),
    ] = Path("./recipes"),
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

    from lorahub.core.config.loader import dump_recipe  # noqa: PLC0415
    from lorahub.core.sweep import (  # noqa: PLC0415
        SweepAxis,
        SweepError,
        SweepPlan,
    )

    if not axis:
        err_console.print("[red]at least one --axis is required[/red]")
        raise typer.Exit(code=1)

    # Validate the base recipe up front so the user sees schema errors before
    # we materialise N copies of a broken recipe.
    cfg = load_recipe(recipe)
    base_dict = cfg.model_dump(mode="json", exclude_none=True)

    axes: list[SweepAxis] = []
    for spec in axis:
        if "=" not in spec:
            err_console.print(
                f"[red]bad --axis spec {spec!r}; expected 'dotted.path=v1,v2,...'[/red]"
            )
            raise typer.Exit(code=1)
        path, _, raw_values = spec.partition("=")
        path = path.strip()
        if not path:
            err_console.print(f"[red]empty axis path in {spec!r}[/red]")
            raise typer.Exit(code=1)
        values = [_coerce_axis_value(tok) for tok in raw_values.split(",") if tok.strip()]
        if not values:
            err_console.print(f"[red]axis {path!r} has no values[/red]")
            raise typer.Exit(code=1)
        axes.append(SweepAxis(path=path, values=values))

    plan = SweepPlan(base_recipe=base_dict, axes=axes, name_template=name_template)
    try:
        variants = plan.expand()
    except SweepError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    base_name = base_dict.get("output", {}).get("name", "sweep")
    sweep_dir_name = f"sweep-{base_name}"

    if dry_run:
        console.print(f"[bold]sweep[/bold] {len(variants)} variant(s) [dim](dry run)[/dim]")
        for i, (variant_name, _variant_recipe) in enumerate(variants, start=1):
            diff = plan.axis_values_for(i)
            console.print(f"[cyan]{variant_name}[/cyan]  {diff}")
        return

    # Re-validate every variant before writing — catches axis values that
    # violate pydantic constraints (e.g. negative LR, rank > 512).
    for variant_name, variant_recipe in variants:
        try:
            RecipeConfig.model_validate(variant_recipe)
        except Exception as exc:  # noqa: BLE001
            err_console.print(
                f"[red]variant {variant_name!r} fails schema validation: {exc}[/red]"
            )
            raise typer.Exit(code=1) from exc

    target_dir = (output_dir / sweep_dir_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "base_recipe": str(recipe.resolve()),
        "name_template": name_template,
        "workspace_root": str(workspace_root.resolve()),
        "axes": [{"path": a.path, "values": list(a.values)} for a in axes],
        "variants": [],
    }
    for i, (variant_name, variant_recipe) in enumerate(variants, start=1):
        # `dump_recipe` requires a RecipeConfig — round-trip through validation
        # both confirms the variant is valid and normalises field ordering.
        cfg_v = RecipeConfig.model_validate(variant_recipe)
        variant_path = target_dir / f"variant_{i:03d}.yaml"
        dump_recipe(cfg_v, variant_path)
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
    console.print(
        f"[green]OK[/] wrote {len(variants)} variant(s) to {target_dir}"
    )
    console.print(f"[dim]manifest:[/dim] {manifest_path}")


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


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Auto-reload on code change (dev only)."),
    ] = False,
) -> None:
    """Run the LoraHub HTTP API server (REST + WebSocket)."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        err_console.print(
            "[red]API extras not installed.[/red] Run: pip install lorahub[api]"
        )
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]LoraHub API[/bold] http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(
        "lorahub.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Name for the new config (no extension).")],
    template: Annotated[
        str, typer.Option(help="Built-in template to copy. Ignored when --auto is used.")
    ] = "sdxl_character_8gb",
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help="Probe the GPU + dataset and write a config tuned to this machine.",
        ),
    ] = False,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Base model .safetensors (required for --auto)."),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Dataset directory (required for --auto)."),
    ] = None,
    vram_mib: Annotated[
        int | None,
        typer.Option(
            "--vram-mib",
            help="Override detected VRAM in MiB (e.g. 8192). Skips nvidia-smi.",
        ),
    ] = None,
) -> None:
    """Scaffold a starter config in the current directory."""
    dst = Path.cwd() / f"{name}.yaml"
    if dst.exists():
        err_console.print(f"[red]{dst} already exists[/red]")
        raise typer.Exit(code=1)

    if auto:
        if checkpoint is None or dataset is None:
            err_console.print(
                "[red]--auto requires --checkpoint and --dataset[/red]"
            )
            raise typer.Exit(code=1)
        from lorahub.core.config import scaffold
        from lorahub.core.config.loader import dump_recipe

        cfg = scaffold.auto_scaffold(
            name=name,
            checkpoint=checkpoint.resolve(),
            dataset=dataset.resolve(),
            vram_mib=vram_mib,
        )
        dump_recipe(cfg, dst)
        images = scaffold.count_images(dataset.resolve())
        console.print(
            f"[green]created[/green] {dst}\n"
            f"[dim]arch[/dim] {cfg.base_model.arch}  "
            f"[dim]rank[/dim] {cfg.network.rank}  "
            f"[dim]batch[/dim] {cfg.schedule.batch_size}x{cfg.schedule.grad_accum}  "
            f"[dim]images[/dim] {images}  "
            f"[dim]repeats[/dim] {cfg.dataset.num_repeats}"
        )
        return

    src = _builtin_recipe(template)
    if not src.exists():
        err_console.print(f"[red]unknown template: {template}[/red]")
        raise typer.Exit(code=1)
    shutil.copy2(src, dst)
    console.print(f"[green]created[/green] {dst}")


@app.command("bootstrap-kohya")
def bootstrap_kohya(
    target: Annotated[
        Path,
        typer.Option(help="Where to clone sd-scripts and create its venv."),
    ] = Path("./sd-scripts"),
    cuda: Annotated[
        str, typer.Option("--cuda", help="CUDA wheel suffix (cu118 / cu121 / cu124 / cu128).")
    ] = "cu124",
    torch_version: Annotated[
        str, typer.Option("--torch", help="PyTorch version to install.")
    ] = "2.6.0",
    torchvision_version: Annotated[
        str, typer.Option("--torchvision", help="torchvision version to install.")
    ] = "0.21.0",
    no_xformers: Annotated[
        bool,
        typer.Option("--no-xformers", help="Skip the optional xformers install."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Wipe target if it already exists."),
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
            err_console.print(
                f"[red]target {plan.target} is not empty.[/red] "
                "Pass --force to wipe it first, or pick another path with --target."
            )
            raise typer.Exit(code=1)
        installer.cleanup_partial(plan)

    console.print(
        f"[bold]Installing kohya into[/bold] {plan.target}\n"
        f"[dim]CUDA[/dim] {plan.cuda_version}  "
        f"[dim]torch[/dim] {plan.torch_version}  "
        f"[dim]xformers[/dim] {plan.install_xformers}"
    )

    try:
        installer.bootstrap(
            plan,
            progress=lambda step: console.print(f"[cyan]>[/cyan] {step}"),
        )
    except BootstrapError as e:
        err_console.print(
            f"[red]bootstrap failed at step:[/red] {e.step} "
            f"[dim](exit {e.returncode})[/dim]\n"
            f"Run [bold]lorahub bootstrap-kohya --force[/bold] to retry from scratch."
        )
        raise typer.Exit(code=1) from e

    console.print(f"[green]OK[/] kohya installed at {plan.target}")
    console.print(
        f"[dim]Set LORAHUB_KOHYA_SD_SCRIPTS={plan.target} (or copy .env.example to .env).[/dim]"
    )


@app.command("fetch-bangumi")
def fetch_bangumi(
    repo: Annotated[
        str,
        typer.Argument(help="BangumiBase repo, e.g. 'azurlaneanime' or 'BangumiBase/azurlaneanime'."),
    ],
    character: Annotated[
        str | None,
        typer.Argument(help="Numeric character id (e.g. '3'). Omit to list characters."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Where to unpack images and caption files."),
    ] = Path("./datasets/bangumi"),
    limit: Annotated[
        int | None,
        typer.Option(help="Cap on number of images. Useful for smoke testing."),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(help="Download preview thumbnails 1-8 instead of dataset.zip."),
    ] = False,
    seed_captions: Annotated[
        bool,
        typer.Option(
            "--seed-captions/--no-seed-captions",
            help="Seed empty .txt caption files next to each image. Default on.",
        ),
    ] = True,
) -> None:
    """Download a single character's images from a BangumiBase HF dataset."""
    if character is None:
        chars = bangumi_base.list_characters(repo)
        console.print(f"[bold]{len(chars)} characters[/] in {repo}: {', '.join(chars)}")
        return

    if preview:
        for i in range(1, 9):
            path = bangumi_base.download_preview(repo, character, output / character, index=i)
            console.print(f"[dim]preview {i}[/dim]  {path}")
        return

    result = bangumi_base.fetch_character(
        repo,
        character,
        output,
        limit=limit,
        seed_captions=seed_captions,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    console.print(
        f"[green]OK[/] {result.image_count} images -> {result.output_dir}"
    )
    if result.license:
        console.print(f"[dim]license: {result.license}[/dim]")
    if result.image_count and seed_captions:
        console.print(
            "[yellow]Seeded empty .txt captions - fill them in before training.[/yellow]"
        )


@app.command()
def tag(
    directory: Annotated[
        Path, typer.Argument(help="Directory of images to tag in place.")
    ],
    tagger: Annotated[
        str,
        typer.Option(
            "--tagger",
            help="Which auto-tagger to use: 'wd14' (default) or 'joytag'.",
        ),
    ] = "wd14",
    model: Annotated[
        str, typer.Option(help="Hugging Face model id of the WD tagger (ignored for joytag).")
    ] = "SmilingWolf/wd-v1-4-vit-tagger-v2",
    general_threshold: Annotated[
        float, typer.Option("--general", help="WD14 general-tag score threshold.")
    ] = 0.35,
    character_threshold: Annotated[
        float, typer.Option("--character", help="WD14 character-tag score threshold.")
    ] = 0.85,
    joytag_threshold: Annotated[
        float,
        typer.Option(
            "--joytag-threshold",
            help="JoyTag predict threshold (single value across all tags).",
        ),
    ] = 0.4,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Recurse into subdirectories."),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-tag images that already have a non-empty caption.")
    ] = False,
    underscores: Annotated[
        bool, typer.Option("--underscores", help="Keep underscores in tag names instead of spaces.")
    ] = False,
    include_character: Annotated[
        bool,
        typer.Option(
            "--include-character/--no-include-character",
            help="Include character tags in the caption (WD14 only). Default on.",
        ),
    ] = True,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Runtime: 'auto' (CUDA if available), 'cuda' (force GPU), or 'cpu'.",
        ),
    ] = "auto",
) -> None:
    """Auto-tag images and write kohya-style .txt captions.

    Supports WD14/WD-v3 (ONNX) and JoyTag (PyTorch). Default is WD14.
    """
    from lorahub.core.tagging.base import BaseTagger  # noqa: PLC0415

    if not directory.is_dir():
        err_console.print(f"[red]not a directory: {directory}[/red]")
        raise typer.Exit(code=1)

    kind = tagger.lower()
    if kind not in {"wd14", "joytag"}:
        err_console.print(f"[red]unknown tagger {tagger!r}; expected wd14 or joytag[/red]")
        raise typer.Exit(code=1)

    instance: BaseTagger
    if kind == "joytag":
        from lorahub.core.tagging.joytag import JoyTagger, JoyTagModelError  # noqa: PLC0415

        instance = JoyTagger(predict_threshold=joytag_threshold, device=device)
        console.print("[dim]loading fancyfeast/joytag (first run downloads ~1.2GB)...[/dim]")
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
        console.print(f"[dim]loading {model} (first run downloads ~400MB)...[/dim]")
        try:
            instance.load()
        except CudaUnavailableError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e

    console.print(f"[dim]running on {instance.active_provider}[/dim]")

    def _on_progress(path: Path, _result: object) -> None:
        console.print(f"[dim]tagged[/dim] {path.name}")

    results = instance.tag_directory(
        directory,
        recursive=recursive,
        write_caption=True,
        skip_existing=not overwrite,
        underscores=underscores,
        include_character=include_character,
        on_progress=_on_progress,
    )

    console.print(f"[green]OK[/] tagged {len(results)} images")




@app.command("anima-caption")
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
        err_console.print(f"[red]not a directory: {directory}[/red]")
        raise typer.Exit(code=1)

    def _split(value: str | None) -> list[str] | None:
        if value is None:
            return None
        items = [t.strip() for t in value.split(",") if t.strip()]
        return items or None

    transformer = AnimaDatasetTransformer(
        default_quality=_split(quality),
        default_score=_split(score),
        default_year=_split(year),
        default_safety=safety if safety else None,
        dataset_tag=dataset_tag,
    )

    def _on_progress(path: Path) -> None:
        console.print(f"[dim]rewrote[/dim] {path.name}")

    written = transformer.transform_directory(
        directory,
        recursive=recursive,
        overwrite=overwrite,
        progress=_on_progress,
    )
    if not overwrite:
        console.print(
            "[yellow]dry run[/yellow] (pass --overwrite to actually rewrite captions)"
        )
    console.print(f"[green]OK[/] rewrote {written} caption(s)")


@app.command()
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
        err_console.print(f"[red]unknown caption action: {action}[/red]")
        raise typer.Exit(code=1)
    if not directory.is_dir():
        err_console.print(f"[red]not a directory: {directory}[/red]")
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
        console.print(f"[dim]{done}/{total}[/dim] {p.name}")

    written = pipeline.transform_directory(
        directory,
        recursive=recursive,
        overwrite=overwrite,
        progress=_on_progress,
    )
    console.print(f"[green]OK[/] rewrote {written} caption(s)")


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
