"""Model downloader supporting HuggingFace and ModelScope.

HuggingFace: uses `huggingface_hub.snapshot_download`. The HF_ENDPOINT
environment variable is honored when the user has configured a mirror in
Settings — `lorahub.api.settings.env_overrides` injects it before the
download starts.

ModelScope: downloads files directly via the public HTTP API
(https://modelscope.cn/api/v1/models/{owner}/{name}/repo?Revision=...&FilePath=...)
so we don't need the heavyweight `modelscope` package as a runtime dependency.

Both paths are designed to run in a worker thread and feed progress through
a callback so the UI can stream events live.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

ProgressCallback = Callable[[str], None]
Source = Literal["huggingface", "modelscope"]


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    source: Source
    repo_id: str  # "owner/name"
    revision: str = "master"  # ModelScope default; HF will be remapped to "main"
    target_dir: Path | None = None  # default: <cwd>/models/<repo_id without slash>
    huggingface_endpoint: str | None = None
    modelscope_token: str | None = None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    target: Path
    files: int
    total_bytes: int


# --------------------------------------------------------------------------- #
# HuggingFace                                                                 #
# --------------------------------------------------------------------------- #


def _hf_download(req: DownloadRequest, progress: ProgressCallback | None) -> DownloadResult:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    if req.huggingface_endpoint:
        os.environ["HF_ENDPOINT"] = req.huggingface_endpoint.rstrip("/")
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = req.huggingface_endpoint.rstrip("/")
    revision = "main" if req.revision == "master" else req.revision
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
        progress(f"hf: snapshot_download {req.repo_id} ({revision}) <- {endpoint}")
    snapshot_download(
        repo_id=req.repo_id,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    files = list(target.rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    if progress:
        progress(f"hf: done — {len([f for f in files if f.is_file()])} files, {total} bytes")
    return DownloadResult(target=target, files=sum(1 for f in files if f.is_file()), total_bytes=total)


# --------------------------------------------------------------------------- #
# ModelScope                                                                  #
# --------------------------------------------------------------------------- #


_MS_BASE = "https://www.modelscope.cn/api/v1/models"


def _ms_list_files(repo_id: str, revision: str, token: str | None) -> list[dict]:
    """List repo files via ModelScope's tree API.

    Returns a flat list of {Path, Size, Type} entries; directories are
    skipped (we only download files).
    """
    import json  # noqa: PLC0415

    url = f"{_MS_BASE}/{repo_id}/repo/files?Revision={quote(revision)}&Recursive=True"
    headers: dict[str, str] = {"User-Agent": "lorahub/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
    items = body.get("Data", {}).get("Files") or body.get("Data", {}).get("Files", [])
    if isinstance(items, dict):
        items = items.get("Files", []) or []
    return [it for it in items if it.get("Type") != "tree"]


def _ms_download_file(
    repo_id: str,
    revision: str,
    file_path: str,
    target: Path,
    token: str | None,
) -> int:
    url = (
        f"{_MS_BASE}/{repo_id}/repo?Revision={quote(revision)}"
        f"&FilePath={quote(file_path)}"
    )
    headers: dict[str, str] = {"User-Agent": "lorahub/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    target.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with urlopen(req, timeout=120) as resp, target.open("wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            bytes_written += len(chunk)
    return bytes_written


def _ms_download(req: DownloadRequest, progress: ProgressCallback | None) -> DownloadResult:
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"ms: list files for {req.repo_id} (rev={req.revision})")
    files = _ms_list_files(req.repo_id, req.revision, req.modelscope_token)
    if progress:
        progress(f"ms: {len(files)} files to download")
    total = 0
    for i, it in enumerate(files, start=1):
        path = str(it.get("Path") or it.get("FilePath") or "")
        if not path:
            continue
        out = target / path
        if progress:
            progress(f"ms: [{i}/{len(files)}] {path}")
        try:
            n = _ms_download_file(req.repo_id, req.revision, path, out, req.modelscope_token)
            total += n
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(f"ms: skip {path}: {exc}")
            continue
    if progress:
        progress(f"ms: done — {len(files)} files, {total} bytes")
    return DownloadResult(target=target, files=len(files), total_bytes=total)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def download(req: DownloadRequest, progress: ProgressCallback | None = None) -> DownloadResult:
    if req.source == "huggingface":
        return _hf_download(req, progress)
    if req.source == "modelscope":
        return _ms_download(req, progress)
    msg = f"unknown source: {req.source!r}"
    raise ValueError(msg)


def cleanup_partial(target: Path) -> None:
    """Best-effort cleanup of a half-finished download directory."""
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


__all__ = [
    "DownloadRequest",
    "DownloadResult",
    "Source",
    "cleanup_partial",
    "download",
]
