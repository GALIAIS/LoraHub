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

import os
import shutil
import tempfile
import threading
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

_DOWNLOAD_LOCK = threading.Lock()


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _prepare_models_root(root: Path) -> None:
    """Create download destinations without traversing links or junctions."""
    if _is_link_like(root):
        raise OSError(f"models root cannot be a link: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise OSError(f"models root is not a directory: {root}")
    for sub, _, _ in _TARGETS:
        directory = root / sub
        if _is_link_like(directory):
            raise OSError(f"model destination cannot be a link: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise OSError(f"model destination is not a directory: {directory}")


def _publish_downloaded_file(cached_path: Path, dest: Path) -> None:
    """Publish one downloaded checkpoint without exposing a partial file."""
    if _is_link_like(dest):
        raise OSError(f"model destination cannot be a link: {dest}")
    if dest.exists():
        if not dest.is_file():
            raise OSError(f"model destination is not a regular file: {dest}")
        if dest.stat().st_size > 0:
            return
    if not cached_path.is_file() or cached_path.stat().st_size <= 0:
        raise OSError(f"download completed without a valid checkpoint: {cached_path}")

    # Hugging Face may materialize local_dir entries as cache links. Never move
    # such a link into the canonical models tree; copy its bytes to a temporary
    # regular file and atomically publish that file instead.
    if _is_link_like(cached_path):
        fd, raw_temp = tempfile.mkstemp(
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temp_path = Path(raw_temp)
        try:
            shutil.copyfile(cached_path, temp_path)
            if temp_path.stat().st_size <= 0:
                raise OSError(f"downloaded checkpoint is empty: {cached_path}")
            temp_path.replace(dest)
        finally:
            temp_path.unlink(missing_ok=True)
        cached_path.unlink(missing_ok=True)
        return

    cached_path.replace(dest)


def _remove_empty_download_dirs(root: Path) -> None:
    """Remove only now-empty directories created by the known repo paths."""
    split_root = root / "split_files"
    for sub, _, _ in _TARGETS:
        directory = split_root / sub
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        split_root.rmdir()
    except OSError:
        pass


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

    Raises ``OSError`` if the link cannot be created. ``ensure_models_link``
    silently returns the *target* directory (``<root>/models``) when link
    creation fails, but without the link anima's hardcoded ``models/...``
    paths don't resolve and training / preview can't find the checkpoints
    that ``download_models`` just wrote. Surfacing the failure here stops
    ``download_models`` from reporting success in that case — e.g. Docker
    ``--user`` runs or read-only checkouts that can't create the junction
    or symlink.
    """
    repo = default_repo_path()
    link = repo / "models"
    result = ensure_models_link(repo)
    # ``ensure_models_link`` returns the *link* (repo/models) on success or
    # when an existing real directory is left in place, and falls back to
    # returning the *target* (<root>/models) ONLY when link creation raised.
    # That fallback is a silent failure for anima — detect and surface it.
    if result != link:
        raise OSError(
            f"Failed to link {link} -> {result}: the anima backend resolves "
            "hardcoded `models/...` paths relative to its checkout, so "
            "without this link training/preview cannot find the downloaded "
            "checkpoints. Ensure the backend checkout is writable (Docker "
            "`--user` runs and read-only repos cannot create the "
            "junction/symlink)."
        )
    unavailable = [
        f"{sub}/{name}"
        for sub, name, _ in _TARGETS
        if not (link / sub / name).is_file()
        or (link / sub / name).stat().st_size <= 0
    ]
    if unavailable:
        raise OSError(
            f"{link} does not expose the downloaded Anima checkpoints: "
            + ", ".join(unavailable)
        )


def download_models(
    *,
    source: Source = "modelscope",
    huggingface_endpoint: str | None = None,
    huggingface_token: str | None = None,
    modelscope_token: str | None = None,
    proxy: str | None = None,
    threads: int = 3,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Download every missing anima model file into ``<root>/models/``.

    The download is parallel over the (up to) 3 files. We re-emit the
    same per-file event shape as the existing ``models/download``
    endpoint so the front end can render a uniform progress UI.
    """
    with _DOWNLOAD_LOCK:
        _download_models_locked(
            source=source,
            huggingface_endpoint=huggingface_endpoint,
            huggingface_token=huggingface_token,
            modelscope_token=modelscope_token,
            proxy=proxy,
            threads=threads,
            progress=progress,
            cancel_event=cancel_event,
        )


def _download_models_locked(
    *,
    source: Source,
    huggingface_endpoint: str | None,
    huggingface_token: str | None,
    modelscope_token: str | None,
    proxy: str | None,
    threads: int,
    progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> None:
    root = models_root()
    _prepare_models_root(root)

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
            cancel_event=cancel_event,
        ),
        forward,
    )

    for sub, name, repo_path in pending:
        cached_path = root / repo_path
        dest = root / sub / name
        if cached_path.absolute() != dest.absolute():
            _publish_downloaded_file(cached_path, dest)

    # Never recursively delete models/split_files: it may contain assets the
    # user placed there. Only remove known generated directories when empty.
    _remove_empty_download_dirs(root)

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
