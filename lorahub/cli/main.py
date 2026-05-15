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
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from lorahub import __version__
from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.backends.kohya.compiler import compile_recipe
from lorahub.core.config.loader import load_recipe
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
    recipe: Annotated[Path, typer.Argument(help="Path to a recipe YAML file.")],
) -> None:
    """Validate a recipe without running training."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()
    issues = backend.validate(cfg)
    _render_issues(issues)
    if any(i.severity is Severity.error for i in issues):
        raise typer.Exit(code=1)
    console.print("[green]OK[/] recipe valid")


@app.command()
def info(
    recipe: Annotated[Path, typer.Argument(help="Path to a recipe YAML file.")],
) -> None:
    """Show what a recipe would compile to, plus VRAM estimate (no training)."""
    cfg = load_recipe(recipe)
    backend = KohyaBackend()

    script, argv, _files = compile_recipe(cfg, workspace=Path.cwd() / "_dryrun")
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
    name: Annotated[str, typer.Argument(help="Name for the new recipe (no extension).")],
    template: Annotated[
        str, typer.Option(help="Built-in template to copy. Ignored when --auto is used.")
    ] = "sdxl_character_8gb",
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help="Probe the GPU + dataset and write a recipe tuned to this machine.",
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
    """Scaffold a starter recipe in the current directory."""
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
    except installer.BootstrapError as e:
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
    model: Annotated[
        str, typer.Option(help="Hugging Face model id of the WD tagger.")
    ] = "SmilingWolf/wd-v1-4-vit-tagger-v2",
    general_threshold: Annotated[
        float, typer.Option("--general", help="Score threshold for general tags.")
    ] = 0.35,
    character_threshold: Annotated[
        float, typer.Option("--character", help="Score threshold for character tags.")
    ] = 0.85,
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
            help="Include character tags in the caption. Default on.",
        ),
    ] = True,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="ONNX runtime: 'auto' (CUDA if available), 'cuda' (force GPU), or 'cpu'.",
        ),
    ] = "auto",
) -> None:
    """Auto-tag images with WD14 / WD-v3 and write kohya-style .txt captions."""
    from lorahub.core.tagging.wd14 import CudaUnavailableError, WD14Tagger

    if not directory.is_dir():
        err_console.print(f"[red]not a directory: {directory}[/red]")
        raise typer.Exit(code=1)

    tagger = WD14Tagger(
        model_id=model,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        device=device,
    )

    console.print(f"[dim]loading {model} (first run downloads ~400MB)...[/dim]")
    try:
        tagger.load()
    except CudaUnavailableError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    console.print(f"[dim]running on {tagger.active_provider}[/dim]")

    def _on_progress(path: Path, _result: object) -> None:
        console.print(f"[dim]tagged[/dim] {path.name}")

    results = tagger.tag_directory(
        directory,
        recursive=recursive,
        write_caption=True,
        skip_existing=not overwrite,
        underscores=underscores,
        include_character=include_character,
        on_progress=_on_progress,
    )

    console.print(f"[green]OK[/] tagged {len(results)} images")


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
