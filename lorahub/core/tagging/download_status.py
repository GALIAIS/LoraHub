"""Cross-thread download progress for the WD14 / JoyTag tagger checkpoints.

WD14Tagger's first ``tag_image()`` blocks on a ~700MB ONNX download
when no copy lives in the HuggingFace cache yet. The web UI used to
just freeze on the spinner the whole time, with no signal to
distinguish "model is downloading" from "the request hung".

This module is the in-process bridge:

  * :func:`mark_start` / :func:`mark_chunk` / :func:`mark_done` are
    called from the download thread (via the custom tqdm class in
    :func:`tqdm_class_for`).
  * :func:`snapshot` is called from the FastAPI request handler that
    serves ``GET /api/tagging/wd14/download-status``. The web UI
    polls that endpoint every second and shows / hides a floating
    progress card based on what it sees.

The state is intentionally per-process and not persisted: a download
that survived a restart starts over at the network layer too (HF
cache resume keeps the bytes; we just lose the visible progress).
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class _Job:
    """One observable file download.

    Attributes:
        repo_id: The HF repo id, e.g. ``SmilingWolf/wd-eva02-large-tagger-v3``.
        filename: The file inside the repo.
        total: Total bytes (``None`` until the first tqdm update).
        downloaded: Bytes downloaded so far.
        started_at: ``time.time()`` when the download started.
        finished_at: ``time.time()`` when the download finished or errored.
        status: ``"running"`` / ``"done"`` / ``"error"``.
        error: Truncated repr of the exception when status == ``"error"``.
    """

    repo_id: str
    filename: str
    total: int | None = None
    downloaded: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    error: str | None = None

    def percent(self) -> float | None:
        if not self.total:
            return None
        return min(100.0, (self.downloaded / self.total) * 100.0)


_lock = threading.Lock()
_jobs: dict[str, _Job] = {}
# Last-finished job key, kept around briefly so the web UI can flash
# a "下载完成" confirmation before the toast disappears. Pruned by
# :func:`snapshot` after the linger window passes.
_FINISHED_LINGER_S = 4.0


def _key(repo_id: str, filename: str) -> str:
    return f"{repo_id}::{filename}"


def mark_start(repo_id: str, filename: str, total: int | None = None) -> None:
    """Register a new download. Idempotent: re-registering resets bytes to 0."""
    with _lock:
        _jobs[_key(repo_id, filename)] = _Job(
            repo_id=repo_id,
            filename=filename,
            total=total,
        )


def mark_total(repo_id: str, filename: str, total: int) -> None:
    """Late-bind the total when the tqdm header arrives after the start hook."""
    with _lock:
        job = _jobs.get(_key(repo_id, filename))
        if job is not None and job.status == "running":
            job.total = total


def mark_chunk(repo_id: str, filename: str, n: int) -> None:
    """Bump the downloaded counter by ``n`` bytes."""
    with _lock:
        job = _jobs.get(_key(repo_id, filename))
        if job is not None and job.status == "running":
            job.downloaded += max(0, int(n))


def mark_done(repo_id: str, filename: str) -> None:
    with _lock:
        job = _jobs.get(_key(repo_id, filename))
        if job is None:
            return
        job.status = "done"
        job.finished_at = time.time()
        if job.total is not None:
            job.downloaded = job.total


def mark_error(repo_id: str, filename: str, err: BaseException) -> None:
    with _lock:
        job = _jobs.get(_key(repo_id, filename))
        if job is None:
            return
        job.status = "error"
        job.finished_at = time.time()
        job.error = repr(err)[:200]


def snapshot() -> dict[str, Any]:
    """Return the current state, pruning long-finished jobs along the way.

    Active jobs are kept; finished / errored jobs disappear from the
    response after :data:`_FINISHED_LINGER_S` so the front-end stops
    showing the toast on its own.
    """
    now = time.time()
    out: list[dict[str, Any]] = []
    with _lock:
        for key, job in list(_jobs.items()):
            if job.status != "running":
                if job.finished_at is None or now - job.finished_at > _FINISHED_LINGER_S:
                    _jobs.pop(key, None)
                    continue
            row = asdict(job)
            row["percent"] = job.percent()
            out.append(row)
    out.sort(key=lambda r: r.get("started_at") or 0.0)
    return {"jobs": out}


def reset() -> None:
    """Test seam — drop everything."""
    with _lock:
        _jobs.clear()


def tqdm_class_for(repo_id: str, filename: str) -> type:
    """Return a tqdm-shaped class that reports into this module's state.

    huggingface_hub accepts ``tqdm_class`` on ``hf_hub_download``. The
    returned class only implements the four tqdm methods that
    huggingface_hub actually calls (``__init__``, ``update``, ``close``,
    ``__enter__`` / ``__exit__``). Anything else is a no-op so we don't
    have to vendor the full tqdm surface.
    """
    captured_repo = repo_id
    captured_filename = filename

    class _ReportingTqdm:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._total = kwargs.get("total")
            self._n = 0
            mark_start(captured_repo, captured_filename, total=self._total)

        # tqdm uses positional args for the iterable; we don't iterate.
        def __enter__(self) -> "_ReportingTqdm":
            return self

        def __exit__(self, *_exc: Any) -> None:
            mark_done(captured_repo, captured_filename)

        def update(self, n: int | float = 1) -> None:
            n_int = int(n) if n is not None else 0
            self._n += n_int
            mark_chunk(captured_repo, captured_filename, n_int)

        def close(self) -> None:
            mark_done(captured_repo, captured_filename)

        # tqdm's ``set_description`` / ``write`` get called occasionally;
        # accept them so a hf_hub_download's progress prints don't crash.
        def set_description(self, *_a: Any, **_k: Any) -> None:
            pass

        def write(self, *_a: Any, **_k: Any) -> None:
            pass

        def reset(self, total: int | None = None) -> None:
            if total is not None:
                self._total = total
                mark_total(captured_repo, captured_filename, total)

        def refresh(self) -> None:
            pass

        def __iter__(self):  # pragma: no cover — tqdm's iter wrapping
            return iter(())

    return _ReportingTqdm


__all__ = [
    "mark_chunk",
    "mark_done",
    "mark_error",
    "mark_start",
    "mark_total",
    "reset",
    "snapshot",
    "tqdm_class_for",
]
