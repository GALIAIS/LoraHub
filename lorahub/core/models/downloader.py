"""Model downloader supporting HuggingFace and ModelScope."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

from lorahub.core.net import hf_api, hf_download, hf_endpoint, proxy_env
from lorahub.core.redaction import redact_command_text

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

_MAX_REMOTE_MANIFEST_BYTES = 64 * 1024 * 1024
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(?:\d+|\*)$", re.IGNORECASE)


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
    cancel_event: threading.Event | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        validate_repo_id(self.repo_id)


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


class DownloadCanceledError(InterruptedError):
    """Raised when a caller cancels an in-flight model download."""


@dataclass(slots=True)
class _PathLockState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, _PathLockState] = {}


@contextmanager
def _download_path_lock(path: Path) -> Iterator[None]:
    """Serialize writes to one checkpoint across concurrent API sessions."""
    key = path.expanduser().absolute()
    with _PATH_LOCKS_GUARD:
        state = _PATH_LOCKS.setdefault(key, _PathLockState())
        state.users += 1
    state.lock.acquire()
    try:
        yield
    finally:
        state.lock.release()
        with _PATH_LOCKS_GUARD:
            state.users -= 1
            if state.users == 0 and _PATH_LOCKS.get(key) is state:
                _PATH_LOCKS.pop(key, None)


_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def validate_repo_id(repo_id: str) -> str:
    """Validate the two-segment repository id used in URLs and local paths."""
    parts = repo_id.strip().split("/")
    if (
        len(parts) != 2
        or any(part in {".", ".."} for part in parts)
        or any(_REPO_SEGMENT_RE.fullmatch(part) is None for part in parts)
    ):
        raise ValueError("repo_id must be a safe 'owner/name' identifier")
    return "/".join(parts)


def _raise_if_canceled(req: DownloadRequest) -> None:
    if req.cancel_event is not None and req.cancel_event.is_set():
        raise DownloadCanceledError("download canceled by user")


def _safe_error(exc: BaseException) -> str:
    """Return an actionable error without persisting credentials from URLs."""
    return redact_command_text(str(exc))


def _cancel_tqdm_class(cancel_event: threading.Event) -> type[Any]:
    """Make Hugging Face's streaming loop observe the session cancel event."""
    from tqdm.auto import tqdm  # noqa: PLC0415

    class _CancelAwareTqdm(tqdm):  # type: ignore[misc]
        def update(self, n: int | float = 1) -> bool | None:
            if cancel_event.is_set():
                raise DownloadCanceledError("download canceled by user")
            return super().update(n)

    return _CancelAwareTqdm


def _emit(progress: ProgressCallback | None, event: DownloadProgress) -> None:
    if progress:
        progress(event)


def _file_size(item: dict[str, Any]) -> int:
    raw = item.get("Size") or item.get("size") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _normalise_path(path: str) -> str | None:
    normalised = path.strip().replace("\\", "/")
    if not normalised or normalised.startswith("/") or ":" in normalised:
        return None
    parts = [part for part in normalised.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    for part in parts:
        stem = part.split(".", 1)[0].upper()
        if (
            part[-1] in {" ", "."}
            or any(char in '<>"|?*\0' for char in part)
            or stem in _WINDOWS_RESERVED_NAMES
            or len(part.encode("utf-8")) > 240
        ):
            return None
    return "/".join(parts)


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _safe_download_target(root: Path, relative_path: str) -> Path:
    """Resolve one selected file without traversing pre-existing links."""
    if _is_link(root):
        raise ValueError("download target root must not be a symlink or junction")
    candidate = root.joinpath(*relative_path.split("/"))
    current = root
    for part in relative_path.split("/")[:-1]:
        current = current / part
        if current.exists() and _is_link(current):
            raise ValueError(f"download path traverses a linked directory: {relative_path}")
    if candidate.exists() and _is_link(candidate):
        raise ValueError(f"download target is a linked file: {relative_path}")
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError(f"download path escapes target root: {relative_path}") from exc
    return candidate


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
    selected_paths: set[str] = set()
    for raw in paths:
        path = _normalise_path(raw)
        if path is None:
            raise ValueError(f"invalid selected path: {raw!r}")
        selected_paths.add(path)
    out: list[RemoteFile] = []
    seen: set[str] = set()
    for raw_path, size in files:
        path = _normalise_path(raw_path)
        if path is None or path in seen:
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
    api = hf_api(endpoint=endpoint, token=token) if (endpoint or token) else hf_api()
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
    _raise_if_canceled(req)
    endpoint = hf_endpoint(req.huggingface_endpoint)
    token = (req.huggingface_token or "").strip() or None
    revision = "main" if req.revision == "master" else req.revision
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    if _is_link(target):
        raise ValueError("download target root must not be a symlink or junction")
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
    _raise_if_canceled(req)
    files = [(f.path, f.size) for f in remote_files if f.selected]
    bytes_total = sum(size for _, size in files)
    if not remote_files:
        raise RuntimeError(
            "remote repository returned no downloadable files; verify the repository, "
            "revision, access token, and mirror"
        )
    if not files:
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
        _raise_if_canceled(req)
        _safe_download_target(target, name)
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
        if req.cancel_event is not None:
            kw["tqdm_class"] = _cancel_tqdm_class(req.cancel_event)
        hf_download(**kw)
        _raise_if_canceled(req)
        # If size metadata is missing (rare), fall back to the on-disk size.
        if size <= 0:
            size = (target / name).stat().st_size
        return name, size

    workers = max(1, min(req.threads, len(files) or 1, 16))
    completed = 0
    bytes_done = 0
    failures: list[tuple[str, str]] = []
    with proxy_env(req.proxy):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch, name, size): name
                for name, size in files
            }
            for future in as_completed(futures):
                completed += 1
                try:
                    name, size = future.result()
                    bytes_done += size
                    message = f"hf: [{completed}/{len(files)}] {name}"
                except DownloadCanceledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    name = futures[future]
                    error = _safe_error(exc)
                    failures.append((name, error))
                    message = f"hf: [{completed}/{len(files)}] failed {name}: {error}"
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
    _raise_if_canceled(req)
    if failures:
        detail = "; ".join(f"{name}: {error}" for name, error in failures[:5])
        if len(failures) > 5:
            detail = f"{detail}; +{len(failures) - 5} more"
        raise RuntimeError(f"hf download failed for {len(failures)} file(s): {detail}")

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
        raw = resp.read(_MAX_REMOTE_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_REMOTE_MANIFEST_BYTES:
        raise RuntimeError("ModelScope file manifest exceeds the safety limit")
    body = json.loads(raw.decode("utf-8"))
    items = body.get("Data", {}).get("Files") or body.get("Data", {}).get("Files", [])
    if isinstance(items, dict):
        items = items.get("Files", []) or []
    return [it for it in items if it.get("Type") != "tree"]


def _ms_download_file_unlocked(
    repo_id: str,
    revision: str,
    file_path: str,
    target: Path,
    token: str | None,
    cancel_event: threading.Event | None = None,
    *,
    expected_size: int = 0,
) -> int:
    url = (
        f"{_MS_BASE}/{repo_id}/repo?Revision={quote(revision)}"
        f"&FilePath={quote(file_path)}"
    )
    headers: dict[str, str] = {"User-Agent": "lorahub/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_link(target):
        raise ValueError(f"download target is a linked file: {target.name}")
    if expected_size > 0 and target.is_file() and target.stat().st_size == expected_size:
        return expected_size

    partial = target.with_name(f".{target.name}.lorahub.part")
    if _is_link(partial):
        raise ValueError(f"download partial file cannot be a link: {partial.name}")
    if partial.exists() and not partial.is_file():
        raise ValueError(f"download partial path is not a regular file: {partial.name}")
    offset = partial.stat().st_size if partial.is_file() and not _is_link(partial) else 0
    if expected_size > 0 and offset > expected_size:
        partial.unlink(missing_ok=True)
        offset = 0
    if expected_size > 0 and offset == expected_size and offset > 0:
        partial.replace(target)
        return expected_size
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:  # noqa: S310
        status = int(getattr(resp, "status", 0) or 0)
        if not status:
            getcode = getattr(resp, "getcode", None)
            status = int(getcode() or 200) if getcode is not None else 200
        append = offset > 0 and status == 206
        if append:
            content_range = str(resp.headers.get("Content-Range") or "").strip()
            match = _CONTENT_RANGE_RE.fullmatch(content_range)
            if match is None or int(match.group(1)) != offset:
                raise RuntimeError(
                    f"invalid ranged response for {file_path}; "
                    f"expected byte {offset}, got {content_range or 'no Content-Range'}"
                )
        mode = "ab" if append else "wb"
        bytes_written = offset if append else 0
        with partial.open(mode) as fh:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCanceledError("download canceled by user")
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bytes_written += len(chunk)
            fh.flush()
            os.fsync(fh.fileno())

    if expected_size > 0 and bytes_written != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {file_path}: "
            f"expected {expected_size}, got {bytes_written}"
        )
    partial.replace(target)
    return bytes_written


def _ms_download_file(
    repo_id: str,
    revision: str,
    file_path: str,
    target: Path,
    token: str | None,
    cancel_event: threading.Event | None = None,
    *,
    expected_size: int = 0,
) -> int:
    with _download_path_lock(target):
        return _ms_download_file_unlocked(
            repo_id,
            revision,
            file_path,
            target,
            token,
            cancel_event,
            expected_size=expected_size,
        )


def _ms_download(req: DownloadRequest, progress: ProgressCallback | None) -> DownloadResult:
    _raise_if_canceled(req)
    target = req.target_dir or (Path.cwd() / "models" / req.repo_id.replace("/", "__"))
    if _is_link(target):
        raise ValueError("download target root must not be a symlink or junction")
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
    _raise_if_canceled(req)
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
    files = [(f.path, by_path[f.path]) for f in remote_files if f.selected]
    bytes_total = sum(_file_size(item) for _path, item in files)
    if not remote_files:
        raise RuntimeError(
            "remote repository returned no downloadable files; verify the repository, "
            "revision, access token, and mirror"
        )
    if not files:
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
    workers = max(1, min(req.threads, len(files) or 1, 16))
    failures: list[tuple[str, str]] = []

    def submit_file(path: str) -> tuple[str, int]:
        _raise_if_canceled(req)
        out = _safe_download_target(target, path)
        expected_size = _file_size(by_path[path])
        if req.cancel_event is None:
            return path, _ms_download_file(
                req.repo_id,
                req.revision,
                path,
                out,
                req.modelscope_token,
                expected_size=expected_size,
            )
        return path, _ms_download_file(
            req.repo_id,
            req.revision,
            path,
            out,
            req.modelscope_token,
            req.cancel_event,
            expected_size=expected_size,
        )

    with proxy_env(req.proxy):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(submit_file, path): path
                for path, _item in files
            }
            for future in as_completed(futures):
                completed += 1
                try:
                    path, n = future.result()
                    total += n
                    message = f"ms: [{completed}/{len(files)}] {path}"
                except DownloadCanceledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    path = futures[future]
                    error = _safe_error(exc)
                    failures.append((path, error))
                    message = f"ms: [{completed}/{len(files)}] failed {path}: {error}"
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
    _raise_if_canceled(req)
    if failures:
        detail = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        if len(failures) > 5:
            detail = f"{detail}; +{len(failures) - 5} more"
        raise RuntimeError(f"ms download failed for {len(failures)} file(s): {detail}")
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
    """Remove download-owned partial files without deleting completed assets."""
    if _is_link(target) or not target.is_dir():
        return
    stack = [target]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            if _is_link(path):
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".part"):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue


__all__ = [
    "DownloadProgress",
    "DownloadRequest",
    "DownloadResult",
    "DownloadCanceledError",
    "RemoteFile",
    "Source",
    "cleanup_partial",
    "download",
    "list_remote_files",
    "validate_repo_id",
]
