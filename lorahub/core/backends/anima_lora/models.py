"""anima_lora model presence check + one-click download.

Anima needs three safetensors checkpoints to run inference / training:
  - diffusion_models/anima-base-v1.0.safetensors  (DiT, ~12 GB)
  - text_encoders/qwen_3_06b_base.safetensors      (Qwen3 TE)
  - vae/qwen_image_vae.safetensors                 (Qwen Image VAE)

All three live in the ModelScope / HuggingFace repo ``circlestone-labs/Anima``
under ``split_files/{kind}/{name}.safetensors``. The trainer reads them via
relative paths from its own cwd (``external/anima_lora/``), so we
download into ``<lorahub_root>/models/{kind}/`` (the unified models
directory) and link ``external/anima_lora/models`` -> the project
``models/`` so the relative paths still resolve.

The link is created on first download. Existing files are left in
place — re-running the download is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lorahub.core.backends._common.bootstrap import ensure_models_link
from lorahub.core.backends.anima_lora.bootstrap import default_repo_path
from lorahub.core.paths import project_root
from lorahub.core.models.downloader import (
    DownloadProgress,
    DownloadRequest,
    download,
)

ProgressCallback = Callable[["DownloadEvent"], None]
Source = Literal["modelscope", "huggingface"]

ANIMA_REPO_ID = "circlestone-labs/Anima"

# (subdir under models/, filename, repo path under split_files/)
_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "diffusion_models",
        "anima-base-v1.0.safetensors",
        "split_files/diffusion_models/anima-base-v1.0.safetensors",
    ),
    (
        "text_encoders",
        "qwen_3_06b_base.safetensors",
        "split_files/text_encoders/qwen_3_06b_base.safetensors",
    ),
    (
        "vae",
        "qwen_image_vae.safetensors",
        "split_files/vae/qwen_image_vae.safetensors",
    ),
)


@dataclass(frozen=True, slots=True)
class DownloadEvent:
    """One progress update emitted while downloading."""

    message: str
    percent: float
    files_done: int
    files_total: int


def models_root() -> Path:
    """The unified models directory at the LoRaHub project root."""
    return project_root() / "models"


def expected_files() -> list[Path]:
    """Absolute paths every anima model file should land at."""
    root = models_root()
    return [root / sub / name for sub, name, _ in _TARGETS]


def missing_files() -> list[str]:
    """Names of model files that aren't on disk yet (relative to models/)."""
    out: list[str] = []
    for sub, name, _ in _TARGETS:
        p = models_root() / sub / name
        if not p.is_file() or p.stat().st_size == 0:
            out.append(f"{sub}/{name}")
    return out


def models_ok() -> bool:
    return not missing_files()


def _link_anima_models_dir() -> None:
    """Make ``external/anima_lora/models`` point at ``<root>/models``.

    anima's trainer / inference scripts hardcode relative paths like
    ``models/diffusion_models/...`` and run with cwd=``external/anima_lora``,
    so the link makes those paths resolve to the unified ``<root>/models/``
    without having to patch upstream.

    On Windows we use a directory junction (``mklink /J``) which doesn't
    require admin rights — symlink would. On Linux/macOS a regular
    symlink is fine.
    """
    ensure_models_link(default_repo_path())


def download_models(
    *,
    source: Source = "modelscope",
    huggingface_endpoint: str | None = None,
    huggingface_token: str | None = None,
    modelscope_token: str | None = None,
    proxy: str | None = None,
    threads: int = 3,
    progress: ProgressCallback | None = None,
) -> None:
    """Download every missing anima model file into ``<root>/models/``.

    The download is parallel over the (up to) 3 files. We re-emit the
    same per-file event shape as the existing ``models/download``
    endpoint so the front end can render a uniform progress UI.
    """
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    for sub, _, _ in _TARGETS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Filter to files that aren't already present.
    pending: list[tuple[str, str, str]] = []
    for sub, name, repo_path in _TARGETS:
        dest = root / sub / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        pending.append((sub, name, repo_path))

    total = len(pending)
    if total == 0:
        if progress:
            progress(DownloadEvent("all anima models already present", 100, 0, 0))
        _link_anima_models_dir()
        return

    if progress:
        progress(
            DownloadEvent(
                f"{source}: {total} anima model file(s) to download from {ANIMA_REPO_ID}",
                2,
                0,
                total,
            )
        )

    paths = tuple(repo_path for _, _, repo_path in pending)

    def forward(event: DownloadProgress) -> None:
        if progress:
            progress(
                DownloadEvent(
                    message=event.message,
                    percent=event.percent if event.percent is not None else 0,
                    files_done=event.files_done,
                    files_total=event.files_total,
                )
            )

    download(
        DownloadRequest(
            source=source,
            repo_id=ANIMA_REPO_ID,
            revision="master" if source == "modelscope" else "main",
            target_dir=root,
            huggingface_endpoint=huggingface_endpoint,
            huggingface_token=huggingface_token,
            modelscope_token=modelscope_token,
            threads=threads,
            proxy=proxy,
            paths=paths,
        ),
        forward,
    )

    for sub, name, repo_path in pending:
        cached_path = root / repo_path
        dest = root / sub / name
        if cached_path.is_file() and cached_path.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            cached_path.replace(dest)

    leftover = root / "split_files"
    if leftover.is_dir():
        import shutil  # noqa: PLC0415

        shutil.rmtree(leftover, ignore_errors=True)

    _link_anima_models_dir()
    if progress:
        progress(
            DownloadEvent(
                message=f"done — {total}/{total} files",
                percent=100,
                files_done=total,
                files_total=total,
            )
        )


__all__ = [
    "ANIMA_REPO_ID",
    "DownloadEvent",
    "download_models",
    "expected_files",
    "missing_files",
    "models_ok",
    "models_root",
]
