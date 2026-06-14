"""Model downloader supporting HuggingFace and ModelScope."""

from __future__ import annotations

import shutil
from fnmatch import fnmatch
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

from lorahub.core.net import hf_endpoint, proxy_env

ProgressCallback = Callable[["DownloadProgress"], None]
Source = Literal["huggingface", "modelscope"]

DEFAULT_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.gguf",
    "*.onnx",
    "*.json",
    "*.txt",
    "*.model",
    "*.vocab",
    "*.merges",
)

DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".gitattributes",
    "README*",
    "LICENSE*",
    "*.md",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.webp",
    "*.gif",
    "*.mp4",
    "*.zip",
    "*.tar",
    "*.tar.gz",
)


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    source: Source
    repo_id: str
    revision: str = "master"
    target_dir: Path | None = None
    huggingface_endpoint: str | None = None
    huggingface_token: str | None = None
    modelscope_token: str | None = None
    threads: int = 4
    proxy: str | None = None  # socks5h://user:pass@host:port or http://...
    paths: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS


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


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size: int
    selected: bool
    reason: str


def _emit(progress: ProgressCallback | None, event: DownloadProgress) -> None:
    if progress:
        progress(event)


def _file_size(item: dict[str, Any]) -> int:
    raw = item.get("Size") or item.get("size") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _normalise_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("/")


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch(path, pat) or fnmatch(name, pat) for pat in patterns)


def _selection_reason(
    path: str,
    selected_paths: set[str],
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> tuple[bool, str]:
    if selected_paths:
        return (path in selected_paths, "selected" if path in selected_paths else "not selected")
    if _matches_any(path, ignore_patterns):
        return False, "ignored by default"
    if _matches_any(path, allow_patterns):
        return True, "model asset"
    return False, "not a model asset"


def select_files(
    files: list[tuple[str, int]],
    *,
    paths: tuple[str, ...] = (),
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS,
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS,
) -> list[RemoteFile]:
    selected_paths = {_normalise_path(p) for p in paths if _normalise_path(p)}
    out: list[RemoteFile] = []
    seen: set[str] = set()
    for raw_path, size in files:
        path = _normalise_path(raw_path)
        if not path or path in seen:
            continue
        seen.add(path)
        selected, reason = _selection_reason(
            path,
            selected_paths,
            allow_patterns,
            ignore_patterns,
        )
        out.append(RemoteFile(path=path, size=size, selected=selected, reason=reason))
    out.sort(key=lambda f: f.path.lower())
    return out


# --------------------------------------------------------------------------- #
# HuggingFace                                                                 #
# --------------------------------------------------------------------------- #


def _hf_list_files(
    repo_id: str,
    revision: str,
    endpoint: str | None,
    token: str | None = None,
) -> list[tuple[str, int]]:
    """Return [(rfilename, size_bytes)] for every file in the snapshot."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(endpoint=endpoint, token=token) if (endpoint or token) else HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True, token=token)
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

    endpoint = hf_endpoint(req.huggingface_endpoint)
    token = (req.huggingface_token or "").strip() or None
    revision = "main" if req.revision == "master" else req.revision
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)

    _emit(
        progress,
        DownloadProgress(
            message=f"hf: list files for {req.repo_id} (rev={revision}) <- {endpoint or 'huggingface.co'}",
            percent=2,
        ),
    )
    with proxy_env(req.proxy):
        remote_files = select_files(
            _hf_list_files(req.repo_id, revision, endpoint, token=token),
            paths=req.paths,
            allow_patterns=req.allow_patterns,
            ignore_patterns=req.ignore_patterns,
        )
    files = [(f.path, f.size) for f in remote_files if f.selected]
    bytes_total = sum(size for _, size in files)
    if remote_files and not files:
        msg = (
            "no files selected for download; list the remote files and select "
            "the required weights/config/tokenizer files explicitly"
        )
        raise ValueError(msg)
    _emit(
        progress,
        DownloadProgress(
            message=(
                f"hf: {len(files)}/{len(remote_files)} files selected for download"
            ),
            percent=5 if files else 100,
            files_total=len(files),
            bytes_total=bytes_total,
        ),
    )

    def fetch(name: str, size: int) -> tuple[str, int]:
        kw: dict[str, Any] = {
            "repo_id": req.repo_id,
            "filename": name,
            "revision": revision,
            "local_dir": str(target),
        }
        if endpoint:
            kw["endpoint"] = endpoint
        if token:
            kw["token"] = token
        with proxy_env(req.proxy):
            hf_hub_download(**kw)
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
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    _emit(
        progress,
        DownloadProgress(
            message=f"ms: list files for {req.repo_id} (rev={req.revision})",
            percent=2,
        ),
    )
    with proxy_env(req.proxy):
        listed = _ms_list_files(req.repo_id, req.revision, req.modelscope_token)
    by_path: dict[str, dict[str, Any]] = {}
    for it in listed:
        path = _normalise_path(str(it.get("Path") or it.get("FilePath") or ""))
        if path:
            by_path[path] = it
    remote_files = select_files(
        [(path, _file_size(it)) for path, it in by_path.items()],
        paths=req.paths,
        allow_patterns=req.allow_patterns,
        ignore_patterns=req.ignore_patterns,
    )
    files = [by_path[f.path] for f in remote_files if f.selected]
    bytes_total = sum(_file_size(it) for it in files)
    if remote_files and not files:
        msg = (
            "no files selected for download; list the remote files and select "
            "the required weights/config/tokenizer files explicitly"
        )
        raise ValueError(msg)
    _emit(
        progress,
        DownloadProgress(
            message=(
                f"ms: {len(files)}/{len(remote_files)} files selected for download"
            ),
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
        with proxy_env(req.proxy):
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


def list_remote_files(req: DownloadRequest) -> list[RemoteFile]:
    """List remote repo files and mark the default download selection."""
    if req.source == "huggingface":
        endpoint = hf_endpoint(req.huggingface_endpoint)
        token = (req.huggingface_token or "").strip() or None
        revision = "main" if req.revision == "master" else req.revision
        with proxy_env(req.proxy):
            files = _hf_list_files(req.repo_id, revision, endpoint, token=token)
    elif req.source == "modelscope":
        with proxy_env(req.proxy):
            items = _ms_list_files(req.repo_id, req.revision, req.modelscope_token)
        files = [
            (
                str(it.get("Path") or it.get("FilePath") or ""),
                _file_size(it),
            )
            for it in items
        ]
    else:
        msg = f"unknown source: {req.source!r}"
        raise ValueError(msg)
    return select_files(
        files,
        paths=req.paths,
        allow_patterns=req.allow_patterns,
        ignore_patterns=req.ignore_patterns,
    )


def cleanup_partial(target: Path) -> None:
    """Best-effort cleanup of a half-finished download directory."""
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


__all__ = [
    "DownloadProgress",
    "DownloadRequest",
    "DownloadResult",
    "RemoteFile",
    "Source",
    "cleanup_partial",
    "download",
    "list_remote_files",
]
