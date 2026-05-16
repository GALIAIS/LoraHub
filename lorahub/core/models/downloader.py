"""Model downloader supporting HuggingFace and ModelScope."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

ProgressCallback = Callable[["DownloadProgress"], None]
Source = Literal["huggingface", "modelscope"]


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    source: Source
    repo_id: str
    revision: str = "master"
    target_dir: Path | None = None
    huggingface_endpoint: str | None = None
    modelscope_token: str | None = None
    threads: int = 4
    proxy: str | None = None  # socks5h://user:pass@host:port or http://...


@dataclass(frozen=True, slots=True)
class DownloadResult:
    target: Path
    files: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    message: str
    percent: float | None = None
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0


def _emit(progress: ProgressCallback | None, event: DownloadProgress) -> None:
    if progress:
        progress(event)


def _file_size(item: dict[str, Any]) -> int:
    raw = item.get("Size") or item.get("size") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# HuggingFace                                                                 #
# --------------------------------------------------------------------------- #


def _hf_list_files(repo_id: str, revision: str) -> list[tuple[str, int]]:
    """Return [(rfilename, size_bytes)] for every file in the snapshot."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    out: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None) or 0
        out.append((sibling.rfilename, int(size)))
    return out


def _hf_download(req: DownloadRequest, progress: ProgressCallback | None) -> DownloadResult:
    """Pull a model snapshot from HuggingFace with per-file progress events.

    We resolve the file list via `HfApi.model_info` and then call
    `hf_hub_download` once per file under a thread pool. Going one file at a
    time gives the UI a meaningful progress signal — `snapshot_download`
    routes its progress through tqdm, which we can't capture from an HTTP
    callback. The trade-off is one extra HTTP round-trip for the metadata
    listing; for repos with many small files it pays for itself within a
    second.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    import huggingface_hub.constants as _hf_constants  # noqa: PLC0415

    if req.huggingface_endpoint:
        endpoint_url = req.huggingface_endpoint.rstrip("/")
        os.environ["HF_ENDPOINT"] = endpoint_url
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint_url
        _hf_constants.ENDPOINT = endpoint_url
    if req.proxy:
        os.environ["HTTPS_PROXY"] = req.proxy
        os.environ["HTTP_PROXY"] = req.proxy
        os.environ["ALL_PROXY"] = req.proxy
    revision = "main" if req.revision == "master" else req.revision
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"

    _emit(
        progress,
        DownloadProgress(
            message=f"hf: list files for {req.repo_id} (rev={revision}) <- {endpoint}",
            percent=2,
        ),
    )
    files = _hf_list_files(req.repo_id, revision)
    bytes_total = sum(size for _, size in files)
    _emit(
        progress,
        DownloadProgress(
            message=f"hf: {len(files)} files to download",
            percent=5 if files else 100,
            files_total=len(files),
            bytes_total=bytes_total,
        ),
    )

    def fetch(name: str, size: int) -> tuple[str, int]:
        hf_hub_download(
            repo_id=req.repo_id,
            filename=name,
            revision=revision,
            local_dir=str(target),
        )
        # If size metadata is missing (rare), fall back to the on-disk size.
        if size <= 0:
            size = (target / name).stat().st_size
        return name, size

    workers = max(1, min(req.threads, len(files) or 1))
    completed = 0
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, name, size) for name, size in files]
        for future in as_completed(futures):
            completed += 1
            try:
                name, size = future.result()
                bytes_done += size
                message = f"hf: [{completed}/{len(files)}] {name}"
            except Exception as exc:  # noqa: BLE001
                message = f"hf: [{completed}/{len(files)}] skip: {exc}"
            percent = 5 + (completed / len(files) * 95) if files else 100
            _emit(
                progress,
                DownloadProgress(
                    message=message,
                    percent=percent,
                    files_done=completed,
                    files_total=len(files),
                    bytes_done=bytes_done,
                    bytes_total=bytes_total,
                ),
            )

    file_count = len(files)
    _emit(
        progress,
        DownloadProgress(
            message=f"hf: done - {file_count} files, {bytes_done} bytes",
            percent=100,
            files_done=file_count,
            files_total=file_count,
            bytes_done=bytes_done,
            bytes_total=bytes_total or bytes_done,
        ),
    )
    return DownloadResult(target=target, files=file_count, total_bytes=bytes_done)


# --------------------------------------------------------------------------- #
# ModelScope                                                                  #
# --------------------------------------------------------------------------- #


_MS_BASE = "https://www.modelscope.cn/api/v1/models"


def _ms_list_files(repo_id: str, revision: str, token: str | None) -> list[dict[str, Any]]:
    """List repo files via ModelScope's tree API."""
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
    if req.proxy:
        os.environ["HTTPS_PROXY"] = req.proxy
        os.environ["HTTP_PROXY"] = req.proxy
        os.environ["ALL_PROXY"] = req.proxy
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    _emit(
        progress,
        DownloadProgress(
            message=f"ms: list files for {req.repo_id} (rev={req.revision})",
            percent=2,
        ),
    )
    files = _ms_list_files(req.repo_id, req.revision, req.modelscope_token)
    bytes_total = sum(_file_size(it) for it in files)
    _emit(
        progress,
        DownloadProgress(
            message=f"ms: {len(files)} files to download",
            percent=5 if files else 100,
            files_total=len(files),
            bytes_total=bytes_total,
        ),
    )
    total = 0
    completed = 0
    workers = max(1, min(req.threads, len(files) or 1))

    def submit_file(it: dict[str, Any]) -> tuple[str, int]:
        path = str(it.get("Path") or it.get("FilePath") or "")
        if not path:
            return "", 0
        out = target / path
        return path, _ms_download_file(req.repo_id, req.revision, path, out, req.modelscope_token)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(submit_file, it) for it in files]
        for future in as_completed(futures):
            completed += 1
            try:
                path, n = future.result()
                total += n
                message = f"ms: [{completed}/{len(files)}] {path}"
            except Exception as exc:  # noqa: BLE001
                message = f"ms: [{completed}/{len(files)}] skip: {exc}"
            percent = 5 + (completed / len(files) * 95) if files else 100
            _emit(
                progress,
                DownloadProgress(
                    message=message,
                    percent=percent,
                    files_done=completed,
                    files_total=len(files),
                    bytes_done=total,
                    bytes_total=bytes_total,
                ),
            )
    _emit(
        progress,
        DownloadProgress(
            message=f"ms: done - {len(files)} files, {total} bytes",
            percent=100,
            files_done=len(files),
            files_total=len(files),
            bytes_done=total,
            bytes_total=bytes_total or total,
        ),
    )
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
    "DownloadProgress",
    "DownloadRequest",
    "DownloadResult",
    "Source",
    "cleanup_partial",
    "download",
]
