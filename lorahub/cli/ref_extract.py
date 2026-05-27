"""``lorahub ref-extract`` — auto-generate reference images for conditioning training.

Wraps controlnet_aux (and optionally HuggingFace transformers for
DepthAnything v2) so users don't have to write a one-off Python
script every time they prepare paired data for the anima_lora 差异训
练 path. Output files share the same stem as the source so the
generated dir can be plugged straight into a LoraHub dataset subset's
``conditioning_data_dir``.

Five built-in processors:

  - dwpose         (whole-body skeleton via DWPose)
  - openpose       (whole-body via legacy OpenPose annotators)
  - canny          (cv2 Canny edge map; no model download)
  - lineart-anime  (anime line art)
  - depth          (MiDaS or DepthAnything-V2 disparity map)

Heavy deps (controlnet_aux / transformers / torch / cv2) are imported
lazily inside the per-processor builder so ``lorahub --help`` and the
top-level Typer scan stay snappy on a fresh checkout where these
extras aren't installed yet.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from lorahub.cli._i18n import t


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Processor(str, Enum):
    dwpose = "dwpose"
    openpose = "openpose"
    canny = "canny"
    lineart_anime = "lineart-anime"
    depth = "depth"


class DepthModel(str, Enum):
    midas = "midas"
    depth_anything_v2 = "depth-anything-v2"


def _missing_dep(pkg: str) -> typer.Exit:
    """Print a clean install hint and exit instead of raising ImportError."""
    Console().print(t("ref_extract.dep_missing", pkg=pkg))
    return typer.Exit(code=2)


def _enumerate_images(src: Path, recursive: bool) -> list[Path]:
    """Walk ``src`` and return image-like files in deterministic order."""
    iterator = src.rglob("*") if recursive else src.iterdir()
    out = [p for p in iterator if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    out.sort()
    return out


def _output_path(img: Path, src: Path, dst: Path, recursive: bool) -> Path:
    """Mirror ``img`` under ``dst`` preserving its relative subdir layout."""
    if recursive:
        rel = img.relative_to(src).parent
        out_dir = dst / rel
    else:
        out_dir = dst
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always write PNG: lossless and what every aux processor emits as
    # the natural default. The differing input ext doesn't matter to
    # the LoraHub pair resolver — it tries png/jpg/jpeg/webp/bmp.
    return out_dir / f"{img.stem}.png"


# --------------------------------------------------------------------------- #
# Per-processor builders. Each returns ``callable(PIL.Image) -> PIL.Image``.
# Heavy imports stay inside the builder so the CLI loads instantly.
# --------------------------------------------------------------------------- #


def _build_dwpose():
    try:
        from controlnet_aux import DWposeDetector
    except ImportError as exc:
        raise _missing_dep("controlnet_aux") from exc
    detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
    return lambda img: detector(img)


def _build_openpose():
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError as exc:
        raise _missing_dep("controlnet_aux") from exc
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return lambda img: detector(img, include_face=True, include_hand=True)


def _build_canny(low: int, high: int):
    try:
        from controlnet_aux import CannyDetector
    except ImportError as exc:
        raise _missing_dep("controlnet_aux") from exc
    detector = CannyDetector()
    return lambda img: detector(img, low_threshold=low, high_threshold=high)


def _build_lineart_anime():
    try:
        from controlnet_aux import LineartAnimeDetector
    except ImportError as exc:
        raise _missing_dep("controlnet_aux") from exc
    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    return lambda img: detector(img)


def _build_depth(model: DepthModel):
    if model is DepthModel.midas:
        try:
            from controlnet_aux import MidasDetector
        except ImportError as exc:
            raise _missing_dep("controlnet_aux") from exc
        detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        return lambda img: detector(img)
    # depth-anything-v2 — uses transformers' depth-estimation pipeline
    # so we don't need the standalone DepthAnything checkout.
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise _missing_dep("transformers") from exc
    pipe = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    return lambda img: pipe(img)["depth"]


_PROCESSOR_BUILDERS = {
    Processor.dwpose: _build_dwpose,
    Processor.openpose: _build_openpose,
    Processor.lineart_anime: _build_lineart_anime,
}


def _build(processor: Processor, *, canny_low: int, canny_high: int, depth_model: DepthModel):
    if processor in _PROCESSOR_BUILDERS:
        return _PROCESSOR_BUILDERS[processor]()
    if processor is Processor.canny:
        return _build_canny(canny_low, canny_high)
    if processor is Processor.depth:
        return _build_depth(depth_model)
    msg = f"unhandled processor: {processor}"
    raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# Typer entry point
# --------------------------------------------------------------------------- #


def ref_extract(
    src: Path = typer.Argument(..., help=t("ref_extract.src.help")),
    dst: Path = typer.Argument(..., help=t("ref_extract.dst.help")),
    processor: Processor = typer.Option(
        Processor.dwpose, "--processor", "-p", help=t("ref_extract.processor.help"),
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help=t("ref_extract.overwrite.help")),
    recursive: bool = typer.Option(False, "--recursive", "-r", help=t("ref_extract.recursive.help")),
    canny_low: int = typer.Option(100, help=t("ref_extract.canny_low.help")),
    canny_high: int = typer.Option(200, help=t("ref_extract.canny_high.help")),
    depth_model: DepthModel = typer.Option(
        DepthModel.midas, "--depth-model", help=t("ref_extract.depth_model.help"),
    ),
) -> None:
    """Generate paired reference images for anima_lora conditioning training."""
    console = Console()
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_dir():
        console.print(f"[red]src is not a directory:[/red] {src}")
        raise typer.Exit(code=2)

    try:
        from PIL import Image
    except ImportError as exc:
        raise _missing_dep("Pillow") from exc

    console.print(t("ref_extract.start", processor=processor.value, src=src, dst=dst))

    images = _enumerate_images(src, recursive)
    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for img in images:
        out = _output_path(img, src, dst, recursive)
        if out.exists() and not overwrite:
            skipped += 1
            continue
        pairs.append((img, out))

    console.print(t("ref_extract.scanned", n=len(pairs), skipped=skipped))
    if not pairs:
        console.print(t("ref_extract.done", ok=0, fail=0, skipped=skipped))
        return

    runner = _build(
        processor,
        canny_low=canny_low,
        canny_high=canny_high,
        depth_model=depth_model,
    )

    ok = 0
    fail = 0
    with Progress(
        SpinnerColumn(),
        TextColumn(t("ref_extract.processing")),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("", total=len(pairs))
        for img_path, out_path in pairs:
            try:
                img = Image.open(img_path).convert("RGB")
                result = runner(img)
                # All controlnet_aux detectors + transformers depth
                # pipeline return PIL.Image directly; defensive coerce
                # for anything that wandered off-spec.
                if not isinstance(result, Image.Image):
                    result = Image.fromarray(result)
                result.save(out_path)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                console.print(t("ref_extract.failed", path=img_path.name, err=str(exc)))
                fail += 1
            progress.advance(task_id)

    console.print(t("ref_extract.done", ok=ok, fail=fail, skipped=skipped))
