"""``lorahub ref-extract`` — minimal Canny edge generator for conditioning training.

差异训练 (anima_lora conditioning) 需要每张目标图配一张参考图。
本命令只覆盖最轻量的一类:cv2 Canny 边缘检测。零外部模型下载、零
重型依赖,跑得快。

更复杂的参考图 (DWPose 骨架 / OpenPose / 线稿 / 深度图) 不在本程
序范围内 — 那些预处理器对 mmpose / mediapipe / matplotlib /
onnxruntime + 模型权重等一长串依赖有强约束,集成成本远高于产出
价值。**生成那类 ref 图请直接走 ComfyUI 生态** (controlnet_aux
节点 / DWPose 节点),完成后把目录路径填到 LoraHub 数据集子集的
"参考图目录"即可。

Output files share the same stem as the source so the generated
dir can be plugged straight into a LoraHub dataset subset's
``conditioning_data_dir``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lorahub.cli._i18n import t


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Processor(str, Enum):
    canny = "canny"


def _missing_dep(exc: ImportError, hint_pkg: str) -> typer.Exit:
    """Print a clean install hint and exit instead of raising ImportError."""
    real = getattr(exc, "name", None)
    if real and real != hint_pkg:
        Console().print(
            t("ref_extract.dep_missing_real", missing=real, hint=hint_pkg, err=str(exc))
        )
    else:
        Console().print(t("ref_extract.dep_missing", pkg=hint_pkg))
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
    # the natural default. Differing input ext doesn't matter to the
    # LoraHub pair resolver — it tries png/jpg/jpeg/webp/bmp.
    return out_dir / f"{img.stem}.png"


def _build_canny(low: int, high: int):
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415, F401
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise _missing_dep(exc, "opencv-python") from exc

    def _run(img):
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rgb = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        # Stack to 3-channel so downstream pipelines that expect RGB
        # see a consistent shape regardless of processor choice.
        canvas = np.stack([edges, edges, edges], axis=-1)
        return Image.fromarray(canvas)

    return _run


# --------------------------------------------------------------------------- #
# Typer entry point
# --------------------------------------------------------------------------- #


def ref_extract(
    src: Path = typer.Argument(..., help=t("ref_extract.src.help")),
    dst: Path = typer.Argument(..., help=t("ref_extract.dst.help")),
    processor: Processor = typer.Option(
        Processor.canny, "--processor", "-p", help=t("ref_extract.processor.help"),
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help=t("ref_extract.overwrite.help")),
    recursive: bool = typer.Option(False, "--recursive", "-r", help=t("ref_extract.recursive.help")),
    canny_low: int = typer.Option(100, help=t("ref_extract.canny_low.help")),
    canny_high: int = typer.Option(200, help=t("ref_extract.canny_high.help")),
) -> None:
    """Generate paired reference images for anima_lora conditioning training."""
    console = Console()
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_dir():
        console.print(f"[red]src is not a directory:[/red] {src}")
        raise typer.Exit(code=2)

    try:
        from PIL import Image  # noqa: F401, PLC0415
    except ImportError as exc:
        raise _missing_dep(exc, "Pillow") from exc

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

    # Only one processor (canny) survives the simplification — heavier
    # processors (dwpose / openpose / lineart / depth) live in ComfyUI.
    if processor is Processor.canny:
        runner = _build_canny(canny_low, canny_high)
    else:
        msg = f"unhandled processor: {processor}"
        raise RuntimeError(msg)

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
                from PIL import Image  # noqa: PLC0415

                img = Image.open(img_path).convert("RGB")
                result = runner(img)
                if not isinstance(result, Image.Image):
                    result = Image.fromarray(result)
                result.save(out_path)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                console.print(t("ref_extract.failed", path=img_path.name, err=str(exc)))
                fail += 1
            progress.advance(task_id)

    console.print(t("ref_extract.done", ok=ok, fail=fail, skipped=skipped))
