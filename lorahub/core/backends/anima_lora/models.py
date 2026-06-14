"""anima_lora model presence check + one-click download.

Anima needs three safetensors checkpoints to run inference / training:
  - diffusion_models/anima-base-v1.0.safetensors  (DiT, ~12 GB)
  - text_encoders/qwen_3_06b_base.safetensors      (Qwen3 TE)
  - vae/qwen_image_vae.safetensors                 (Qwen Image VAE)

All three live in the HuggingFace repo ``circlestone-labs/Anima`` under
``split_files/{kind}/{name}.safetensors``. The trainer reads them via
relative paths from its own cwd (``external/anima_lora/``), so we
download into ``<lorahub_root>/models/{kind}/`` (the unified models
directory) and link ``external/anima_lora/models`` -> the project
``models/`` so the relative paths still resolve.

The link is created on first download. Existing files are left in
place — re-running the download is a no-op.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends.anima_lora.bootstrap import default_repo_path
from lorahub.core.net import hf_endpoint, proxy_env

ProgressCallback = Callable[["DownloadEvent"], None]

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
    return Path.cwd() / "models"


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
    target = models_root()
    target.mkdir(parents=True, exist_ok=True)
    link = default_repo_path() / "models"

    if link.exists() or link.is_symlink():
        # Already set up (symlink, junction, or a real dir from a prior
        # install). If it's a real directory with content, leave it alone
        # — the user may have downloaded models there manually.
        return

    if sys.platform == "win32":
        # Junction works for directories on every NTFS volume without
        # SeCreateSymbolicLink privilege. mklink is a cmd-builtin so we
        # have to go through cmd.exe.
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def download_models(
    *,
    huggingface_endpoint: str | None = None,
    huggingface_token: str | None = None,
    proxy: str | None = None,
    threads: int = 3,
    progress: ProgressCallback | None = None,
) -> None:
    """Download every missing anima model file into ``<root>/models/``.

    The download is parallel over the (up to) 3 files. We re-emit the
    same per-file event shape as the existing ``models/download``
    endpoint so the front end can render a uniform progress UI.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    endpoint = hf_endpoint(huggingface_endpoint)
    token = (huggingface_token or "").strip() or None

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
                (
                    f"hf: {total} anima model file(s) to download from "
                    f"{ANIMA_REPO_ID} <- {endpoint or 'huggingface.co'}"
                ),
                2,
                0,
                total,
            )
        )

    def fetch(sub: str, name: str, repo_path: str) -> str:
        kw: dict[str, object] = {
            "repo_id": ANIMA_REPO_ID,
            "filename": repo_path,
            "local_dir": str(root),
        }
        if endpoint:
            kw["endpoint"] = endpoint
        if token:
            kw["token"] = token
        with proxy_env(proxy):
            cached = hf_hub_download(**kw)
        # hf_hub_download writes under <local_dir>/split_files/<kind>/<name>.
        # Move the file to <local_dir>/<kind>/<name> so anima's trainer
        # picks it up at the expected path.
        cached_path = Path(cached)
        dest = root / sub / name
        if cached_path.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            shutil.move(str(cached_path), str(dest))
        return f"{sub}/{name}"

    workers = max(1, min(threads, total))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, sub, name, rp) for sub, name, rp in pending]
        for fut in as_completed(futures):
            completed += 1
            try:
                done_name = fut.result()
                msg = f"hf: [{completed}/{total}] {done_name}"
            except Exception as exc:  # noqa: BLE001
                msg = f"hf: [{completed}/{total}] failed: {exc}"
            if progress:
                progress(
                    DownloadEvent(
                        message=msg,
                        percent=2 + (completed / total) * 95,
                        files_done=completed,
                        files_total=total,
                    )
                )

    # Clean up the intermediate split_files/ directory hf_hub_download leaves behind.
    leftover = root / "split_files"
    if leftover.is_dir():
        shutil.rmtree(leftover, ignore_errors=True)

    _link_anima_models_dir()
    if progress:
        progress(
            DownloadEvent(
                message=f"done — {completed}/{total} files",
                percent=100,
                files_done=completed,
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
