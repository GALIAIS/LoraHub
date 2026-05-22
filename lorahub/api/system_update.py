"""Self-update orchestration: GitHub Releases polling + git-based upgrade.

Mirrors the *shape* of ShiroManager's app-updater (mirror pool, cached
release metadata, periodic background check) but the *artifact* is
different: ShiroManager downloads a NSIS installer; LoraHub upgrades by
running ``git pull`` (or ``git checkout v…``) inside the working tree.

Public surface:

* ``check(channel="main")`` — return ``UpdateInfo`` for the resolved
  channel. Uses a 5-minute on-disk cache so opening the Settings page
  doesn't re-hit the GitHub API every refresh.
* ``apply(channel, *, restart, build, progress)`` — perform the upgrade,
  emitting structured progress events through ``progress``.
* ``last_check()`` — return the cached payload if any.

Channels:
  ``main`` — checkout origin/main (rolling release).
  ``tag``  — checkout the highest semver ``v*`` tag.

Mirrors:
  Read from ``Settings.github_proxy``; if empty, the request goes to
  ``api.github.com`` directly. The proxy prefix is **not** applied to
  the GitHub API itself (gh-proxy variants only forward release
  binaries / repo tarballs, not the JSON API). Only the ``git pull``
  step honours the proxy via the existing ``apply_github_proxy()``.

Internals:
  ``apply()`` is a thin orchestrator over five stage functions —
  ``_pre_check``, ``_snapshot_configs``, ``_fetch``, ``_apply_ref``,
  ``_install_deps``. The shared mutable state (stash flag, snapshot
  archive path) lives in ``_UpdateContext``, whose ``__exit__``
  guarantees stash-pop / snapshot-restore even when a stage raises.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from platformdirs import user_state_path

ChannelName = Literal["main", "tag"]
ProgressCallback = Callable[[str, str, str], None]

GITHUB_OWNER = "GALIAIS"
GITHUB_REPO = "LoraHub"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
TAGS_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/tags"
COMMITS_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/main"
WEB_RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
WEB_COMMITS_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commits/main"

CACHE_TTL_SECONDS = 5 * 60
HTTP_TIMEOUT_S = 12.0


def _state_dir() -> Path:
    p = user_state_path("lorahub", "lorahub")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_file() -> Path:
    return _state_dir() / "update-cache.json"


@dataclass
class UpdateInfo:
    """Snapshot of remote-vs-local state for one channel."""

    channel: ChannelName
    current: str
    latest: str | None
    update_available: bool
    release_url: str
    release_notes: str = ""
    checked_at: str = ""
    is_dirty: bool = False
    error: str | None = None
    # Optional metadata: tag-only (None for "main" channel).
    tag_name: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _CacheBlob:
    data: dict[str, dict[str, Any]] = field(default_factory=dict)  # channel -> UpdateInfo dict
    updated_at: float = 0.0


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _current_version() -> str:
    """Return the running lorahub version (hatch-vcs string).

    Falls back to ``0.0.0`` if the version module isn't materialised
    yet — this happens for source checkouts that haven't been
    ``pip install -e .``'d.
    """
    try:
        from lorahub import __version__  # noqa: PLC0415

        return str(__version__) or "0.0.0"
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _normalize_version(raw: str) -> str:
    """Drop a leading 'v' and any pre-release/local suffix junk we don't compare on."""
    s = raw.strip()
    if s.lower().startswith("v"):
        s = s[1:]
    return s


def _compare_versions(left: str, right: str) -> int:
    """``-1 / 0 / 1``. Falls back to lexicographic when packaging can't parse."""
    try:
        return (Version(_normalize_version(left)) > Version(_normalize_version(right))) - (
            Version(_normalize_version(left)) < Version(_normalize_version(right))
        )
    except InvalidVersion:
        return (left > right) - (left < right)


def _read_cache() -> _CacheBlob:
    f = _cache_file()
    if not f.is_file():
        return _CacheBlob()
    try:
        raw = json.loads(f.read_text("utf-8"))
        if not isinstance(raw, dict):
            return _CacheBlob()
        return _CacheBlob(
            data=raw.get("data", {}) or {},
            updated_at=float(raw.get("updated_at") or 0.0),
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return _CacheBlob()


def _write_cache(blob: _CacheBlob) -> None:
    try:
        _cache_file().write_text(
            json.dumps({"data": blob.data, "updated_at": blob.updated_at}, indent=2),
            "utf-8",
        )
    except OSError:
        # Cache is best-effort; failure to write is not a hard error.
        pass


def last_check() -> dict[str, dict[str, Any]] | None:
    """Return the most recent cached payload, ignoring TTL.

    The lifespan startup hook uses this to seed the API response so
    the very first request after boot doesn't have to wait for the
    background fetch to land.
    """
    blob = _read_cache()
    return blob.data if blob.data else None


def _fetch_json(url: str) -> dict[str, Any]:
    """Tiny urllib wrapper (we don't pull in requests for one call).

    Bypasses any system-level HTTP_PROXY env vars: we only want the
    settings-configured proxy for *git*; the GitHub API itself runs
    over plain HTTPS and is fast enough from anywhere.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"lorahub/{_current_version()}",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
        if resp.status >= 400:
            msg = f"HTTP {resp.status}: {resp.reason}"
            raise OSError(msg)
        return json.loads(resp.read().decode("utf-8"))


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """``git ...`` with stdout+stderr captured. Never raises on non-zero."""
    return subprocess.run(  # noqa: S603, S607
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _detect_dirty(cwd: Path) -> bool:
    """``True`` iff the working tree has uncommitted changes the user
    *cares about preserving across upgrade*.

    Naive ``git status --porcelain`` returns every modified file,
    which causes a routine "I edited my training config" to block
    upgrade. We filter out paths the user is expected to mutate
    locally:

      * ``configs/`` — user's own training recipes
      * ``runs/``, ``output/``, ``models/`` — runtime artefacts
      * ``.env``, ``.env.local`` — local secrets / overrides
      * ``external/anima_lora/uv.lock`` — anima locks itself

    Anything still flagged after that filter is genuine "you edited
    LoraHub source code" territory; the upgrade flow's stash-pop
    cycle handles those non-conflicting cases automatically (see
    ``apply()``), and the dirty flag exists mainly as a UI hint.
    """
    out = _git(["status", "--porcelain"], cwd=cwd)
    if out.returncode != 0:
        return False
    for raw in out.stdout.splitlines():
        # Porcelain v1 format: ``XY <path>`` (status codes + space + path).
        # Untracked rows look like ``?? <path>``.
        line = raw.rstrip()
        if len(line) < 4:
            continue
        path = line[3:].lstrip()
        # Renames show as ``orig -> new``; pick the destination so the
        # ignore filter applies to where the user expects to see it.
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        path = path.strip('"')
        if _is_user_owned_path(path):
            continue
        return True
    return False


_USER_OWNED_PREFIXES = (
    "configs/",
    "runs/",
    "output/",
    "models/",
    "datasets/",
    ".env",
    "external/anima_lora/uv.lock",
    "external/anima_lora/output/",
    "external/anima_lora/post_image_dataset/",
)


def _is_user_owned_path(path: str) -> bool:
    """Match against the curated ignore list for the dirty check."""
    norm = path.replace("\\", "/")
    for prefix in _USER_OWNED_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return True
    return False


def _git_root() -> Path | None:
    """Locate the LoraHub working tree root (None for non-git installs)."""
    try:
        from lorahub.core.paths import project_root  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    root = project_root()
    if (root / ".git").exists():
        return root
    return None


def _detect_detached_head(cwd: Path) -> str | None:
    """Return the current commit SHA if HEAD is detached, else None.

    A detached HEAD points at a commit (often a tag like ``v0.3.0``)
    without an attached branch ref. ``git checkout origin/main`` or
    ``git checkout v…`` from a detached state silently abandons the
    current commit if the user had committed work on top of it; that's
    exactly the failure mode self-update is supposed to prevent.

    We use ``git symbolic-ref -q HEAD`` rather than parsing
    ``HEAD`` ourselves so submodule / worktree / packed-refs setups
    also resolve correctly.
    """
    sym = _git(["symbolic-ref", "-q", "HEAD"], cwd=cwd)
    if sym.returncode == 0:
        # ``symbolic-ref`` succeeded → HEAD points at a branch.
        return None
    # Non-zero exit means detached. Read the resolved SHA so the error
    # message can show the user where they are.
    rev = _git(["rev-parse", "--short", "HEAD"], cwd=cwd)
    if rev.returncode == 0:
        return rev.stdout.strip() or "(unknown)"
    return "(unknown)"


def _refresh_main(cwd: Path) -> dict[str, Any]:
    """Probe origin/main via the GitHub commits API (no auth, 60/hr)."""
    info = _fetch_json(COMMITS_API)
    sha = str(info.get("sha") or "")
    short_sha = sha[:7] if sha else ""
    msg = (info.get("commit") or {}).get("message") or ""
    return {
        "tag_name": None,
        "version_str": short_sha or "main",
        "release_notes": msg.split("\n", 1)[0][:300],
        "published_at": (info.get("commit") or {}).get("committer", {}).get("date") or None,
    }


def _refresh_tag() -> dict[str, Any]:
    """Probe the latest GitHub release.

    Falls back to the ``/tags`` API when ``/releases/latest`` 404s.
    The repo currently uses lightweight git tags without published
    GitHub Releases, so the canonical endpoint always returns 404 —
    we still want the user to see the latest tag in the UI rather
    than a network error.
    """
    try:
        info = _fetch_json(RELEASES_API)
    except OSError as exc:
        if not _is_not_found(exc):
            raise
        return _refresh_tag_via_tags_api()

    tag = str(info.get("tag_name") or "")
    return {
        "tag_name": tag or None,
        "version_str": _normalize_version(tag) if tag else "",
        "release_notes": str(info.get("body") or "")[:8000],
        "published_at": str(info.get("published_at") or "") or None,
    }


def _is_not_found(exc: BaseException) -> bool:
    """Match the HTTPError-shaped 404 we get from urllib.

    Prefers the structured ``.code`` attribute on
    ``urllib.error.HTTPError`` (which subclasses ``OSError``); falls
    back to substring matching only when the exception wasn't an
    HTTPError (e.g. our own manual ``raise OSError("HTTP 404 …")``
    above).
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code == 404
    msg = str(exc)
    return "404" in msg or "Not Found" in msg


def _refresh_tag_via_tags_api() -> dict[str, Any]:
    """Pick the highest semver ``v*`` tag from the lightweight tags API.

    GitHub returns tags in commit-creation order, not version order, so
    we sort them ourselves with ``packaging.version`` and skip non-semver
    entries (e.g. ``docs-v1`` annotated tags).
    """
    tags_raw = _fetch_json(TAGS_API)
    if not isinstance(tags_raw, list):
        return _empty_tag_payload()

    candidates: list[tuple[Version, str, str]] = []
    for entry in tags_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name.startswith("v"):
            continue
        normalized = _normalize_version(name)
        try:
            ver = Version(normalized)
        except InvalidVersion:
            continue
        sha = str(entry.get("commit", {}).get("sha") or "")
        candidates.append((ver, name, sha))

    if not candidates:
        return _empty_tag_payload()

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_ver, best_name, _sha = candidates[0]
    return {
        "tag_name": best_name,
        "version_str": str(best_ver),
        # Lightweight tags don't carry release notes; the UI just gets
        # an empty string and the "open in GitHub" link still works.
        "release_notes": "",
        "published_at": None,
    }


def _empty_tag_payload() -> dict[str, Any]:
    return {
        "tag_name": None,
        "version_str": "",
        "release_notes": "",
        "published_at": None,
    }


def check(channel: ChannelName = "tag", *, force: bool = False) -> UpdateInfo:
    """Resolve current-vs-remote for the given channel.

    Returns even on network errors — the ``error`` field carries the
    failure message so the UI can render the cached state plus an
    "offline" hint.
    """
    cwd = _git_root()
    is_dirty = _detect_dirty(cwd) if cwd else False
    current = _current_version()

    blob = _read_cache()
    cached = blob.data.get(channel)
    fresh_enough = (
        cached
        and not force
        and (time.time() - blob.updated_at) < CACHE_TTL_SECONDS
    )

    if fresh_enough:
        info = UpdateInfo(**{**cached, "current": current, "is_dirty": is_dirty})
        return info

    try:
        if channel == "main":
            remote = _refresh_main(cwd or Path.cwd())
        else:
            remote = _refresh_tag()
    except (OSError, ValueError) as exc:
        # Network failure — degrade gracefully to the cached payload.
        if cached:
            info = UpdateInfo(**{**cached, "current": current, "is_dirty": is_dirty})
            info.error = f"refresh failed: {exc}"
            return info
        return UpdateInfo(
            channel=channel,
            current=current,
            latest=None,
            update_available=False,
            release_url=WEB_RELEASES_URL if channel == "tag" else WEB_COMMITS_URL,
            checked_at=_now_iso(),
            is_dirty=is_dirty,
            error=f"refresh failed: {exc}",
        )

    latest = remote["version_str"] or None
    update_available = False
    if channel == "tag" and latest:
        update_available = _compare_versions(latest, current) > 0
    elif channel == "main" and latest and cwd:
        # For main, "update available" means HEAD is not on the
        # remote sha. We compare short SHA prefixes via git
        # rev-parse so a forced reset still counts as up-to-date.
        head = _git(["rev-parse", "HEAD"], cwd=cwd).stdout.strip()
        update_available = bool(head) and not head.startswith(latest)

    info = UpdateInfo(
        channel=channel,
        current=current,
        latest=latest,
        update_available=update_available,
        release_url=WEB_RELEASES_URL if channel == "tag" else WEB_COMMITS_URL,
        release_notes=remote["release_notes"],
        checked_at=_now_iso(),
        is_dirty=is_dirty,
        tag_name=remote["tag_name"],
        published_at=remote["published_at"],
    )

    blob.data[channel] = info.to_dict()
    blob.updated_at = time.time()
    _write_cache(blob)
    return info


# --------------------------------------------------------------------- #
# apply() — five-stage pipeline
#
#   1. _pre_check        — git root check, detached HEAD probe, dirty fence
#   2. _snapshot_configs — tarfile-backed configs/ snapshot to a temp file
#   3. _fetch            — git fetch --tags origin
#   4. _apply_ref        — checkout the resolved ref (origin/main or v…)
#   5. _install_deps     — pip install + optional npm run build
#
# Shared mutable state (stash flag, snapshot archive path) lives in
# `_UpdateContext`. Its `__exit__` is the single rollback point: if
# any stage raises, the archive is restored to disk and any active
# stash is popped. Successful exits clean up the temp archive.
# --------------------------------------------------------------------- #


def apply(
    channel: ChannelName = "tag",
    *,
    build: bool = True,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> None:
    """Execute the upgrade in the current working tree.

    Steps:
      1. ``git fetch --tags origin``
      2. ``git checkout origin/main`` (channel=main) or
         ``git checkout v<latest>`` (channel=tag)
      3. ``uv pip install -e .[api,dev]`` if the project has uv on PATH,
         else ``pip install -e .[api,dev]``
      4. ``npm run build`` if ``build`` is True

    When ``force`` is True, local working-tree changes that would
    conflict with the upgrade are wiped via ``git reset --hard`` +
    ``git clean -fd`` (untracked files included). ``force`` also
    suppresses the detached-HEAD safety gate. Use only when the user
    explicitly opted in via a confirmation dialog — this is destructive
    and unrecoverable.

    Raises ``RuntimeError`` on any non-zero step. ``progress`` is invoked
    with ``(phase, level, message)`` for each line of subprocess output
    so the API can stream the update to the UI like the bootstrap flow.
    """
    cwd = _git_root()
    if cwd is None:
        msg = "this install is not a git checkout — `lorahub manage update` is required."
        raise RuntimeError(msg)

    emit = progress if progress is not None else _NULL_EMIT

    _pre_check(cwd, force=force, emit=emit)

    with _UpdateContext(cwd, emit) as ctx:
        ctx.snapshot_path = _snapshot_configs(cwd, emit)
        # configs/ may have local tracked-file modifications; reset to
        # HEAD so the upcoming checkout sees a clean tree there. The
        # snapshot we just took will overwrite it again at the end.
        if ctx.snapshot_path is not None:
            _stream_subprocess(
                ["git", "checkout", "HEAD", "--", "configs"],
                cwd=cwd, phase="git", emit=emit,
            )

        if force:
            emit(
                "git", "warn",
                "force=True: discarding local changes (git reset --hard + clean -fd); "
                "configs/ is preserved by the snapshot/restore step",
            )
            _stream_subprocess(
                ["git", "reset", "--hard", "HEAD"], cwd=cwd, phase="git", emit=emit,
            )
            _stream_subprocess(
                ["git", "clean", "-fd", "-e", "configs"], cwd=cwd, phase="git", emit=emit,
            )
        elif _has_any_local_changes(cwd):
            # Stash + pop to preserve local edits across checkout.
            # Cheap no-op when the tree is already clean. Conflicts on
            # pop fall through to a clear warning so the user can
            # resolve them rather than silently lose the change.
            emit("git", "info", "git stash --include-untracked (preserving local edits)")
            rc = _stream_subprocess(
                ["git", "stash", "push", "--include-untracked",
                 "-m", "lorahub-self-update"],
                cwd=cwd, phase="git", emit=emit,
            )
            if rc == 0:
                ctx.stash_active = True

        _fetch(cwd, emit)
        _apply_ref(cwd, channel=channel, force=force, emit=emit)
        _restore_configs(cwd, ctx.snapshot_path, emit)
        # The snapshot has been re-laid into the working tree; the
        # context's __exit__ no longer needs to restore it on success.
        ctx.snapshot_consumed = True

        _install_deps(cwd, build=build, emit=emit)

        if ctx.stash_active:
            emit("git", "info", "git stash pop (restoring local edits)")
            rc = _stream_subprocess(
                ["git", "stash", "pop"], cwd=cwd, phase="git", emit=emit,
            )
            if rc != 0:
                emit(
                    "git", "warn",
                    "stash pop reported conflicts — your local edits are still "
                    "in `git stash list`; resolve manually after this update.",
                )
            ctx.stash_active = False

    emit("done", "info", "update applied")


def _NULL_EMIT(_phase: str, _level: str, _message: str) -> None:
    pass


@dataclass
class _UpdateContext:
    """Tracks the rollback state across the upgrade stages.

    On a clean exit the snapshot tar is unlinked and any leftover
    stash is left to the caller. On an exception:

      * ``snapshot_path`` (if set and not yet consumed) is unpacked
        back over ``configs/`` so the user's recipes survive even
        when the upgrade aborted mid-flight.
      * ``stash_active`` triggers a final ``git stash pop`` so the
        user's other local edits aren't trapped in the stash list.

    The context is intentionally narrow — failure handling for the
    individual stages (reset --hard rollback, npm exit codes, …) lives
    inside each stage function so the recovery logic stays close to
    the code that knows what could fail.
    """

    cwd: Path
    emit: ProgressCallback
    snapshot_path: Path | None = None
    snapshot_consumed: bool = False
    stash_active: bool = False

    def __enter__(self) -> _UpdateContext:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            # Best-effort rollback. We don't re-raise from here; the
            # original exception propagates out of the ``with`` block.
            if self.snapshot_path is not None and not self.snapshot_consumed:
                with contextlib.suppress(Exception):
                    self.emit(
                        "git", "warn",
                        "upgrade failed; restoring configs/ from pre-flight snapshot",
                    )
                    _restore_configs(self.cwd, self.snapshot_path, self.emit)
            if self.stash_active:
                with contextlib.suppress(Exception):
                    self.emit(
                        "git", "warn",
                        "upgrade failed; popping stash to restore local edits",
                    )
                    _stream_subprocess(
                        ["git", "stash", "pop"], cwd=self.cwd, phase="git", emit=self.emit,
                    )
        # Always remove the temp archive — it's only useful as a
        # rollback bridge and would otherwise accumulate in TMPDIR.
        if self.snapshot_path is not None:
            with contextlib.suppress(Exception):
                self.snapshot_path.unlink(missing_ok=True)


# --------------------------------------------------------------------- #
# Stage 1 — pre-flight
# --------------------------------------------------------------------- #


def _pre_check(cwd: Path, *, force: bool, emit: ProgressCallback) -> None:
    """Refuse upgrade in states the rest of the pipeline can't recover from.

    Currently:
      * detached HEAD (``force=False`` only) — checking out
        ``origin/main`` from a detached state silently abandons any
        commits the user made on top of the detached SHA.

    ``force=True`` callers have already opted in to destructive
    behaviour via the UI confirm dialog; we still emit a warning so
    the SSE log shows the override happened.
    """
    head_sha = _detect_detached_head(cwd)
    if head_sha is None:
        return
    if not force:
        msg = (
            f"HEAD is detached at {head_sha}. Self-update from a detached "
            "state would silently abandon any commits made on top of it. "
            "Either run `git checkout main` first, or pass --force to "
            "discard the detached commits."
        )
        raise RuntimeError(msg)
    emit(
        "git", "warn",
        f"force=True: HEAD detached at {head_sha}; commits on top of it "
        "will be abandoned by the upcoming checkout",
    )


# --------------------------------------------------------------------- #
# Stage 2 — configs/ snapshot
# --------------------------------------------------------------------- #


def _snapshot_configs(cwd: Path, emit: ProgressCallback) -> Path | None:
    """Capture every regular file under ``configs/`` into a tarball.

    The archive lives in ``tempfile.gettempdir()`` so a multi-megabyte
    yaml collection doesn't have to be held in memory while the
    upgrade runs. Failure to add a single file logs a warning and
    skips that file rather than aborting the upgrade — partial
    coverage is better than refusing to update.

    Returns ``None`` if ``configs/`` doesn't exist or is empty (no
    snapshot needed).
    """
    root = cwd / "configs"
    if not root.is_dir():
        return None
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return None

    fd, raw = tempfile.mkstemp(prefix="lorahub-configs-", suffix=".tar")
    archive = Path(raw)
    # mkstemp leaves the FD open; close it so tarfile can re-open by path.
    import os as _os  # noqa: PLC0415
    _os.close(fd)

    written = 0
    try:
        with tarfile.open(archive, "w") as tar:
            for full in files:
                try:
                    arcname = full.relative_to(cwd).as_posix()
                    tar.add(full, arcname=arcname, recursive=False)
                    written += 1
                except OSError as exc:
                    emit("git", "warn", f"could not snapshot {full}: {exc}")
    except OSError as exc:
        # Tar creation itself failed (disk full, perms). Discard the
        # half-written archive and bail out without a snapshot — the
        # caller's ``_UpdateContext`` will see ``snapshot_path is
        # None`` and skip the restore branch.
        emit("git", "warn", f"snapshot tar failed: {exc}; configs/ will not be protected")
        archive.unlink(missing_ok=True)
        return None

    if written == 0:
        archive.unlink(missing_ok=True)
        return None
    emit("git", "info", f"snapshotted {written} file(s) under configs/ to {archive.name}")
    return archive


def _restore_configs(
    cwd: Path, snapshot: Path | None, emit: ProgressCallback,
) -> None:
    """Unpack the configs/ tar back over the working tree.

    Idempotent and safe when ``snapshot`` is None. Members extracted
    by name within ``cwd`` only — ``tarfile.extract`` resolves the
    path so this is the same risk surface as ``tar xf`` (we trust
    our own snapshot output).
    """
    if snapshot is None or not snapshot.is_file():
        return
    extracted = 0
    try:
        with tarfile.open(snapshot, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Defence-in-depth: refuse absolute paths or "..".
                # Our own snapshot writer never emits these, but a
                # tampered file shouldn't escape cwd either.
                arcname = member.name.replace("\\", "/")
                if arcname.startswith("/") or ".." in arcname.split("/"):
                    emit("git", "warn", f"skipping suspicious archive entry {arcname}")
                    continue
                target = cwd / arcname
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                try:
                    target.write_bytes(source.read())
                    extracted += 1
                except OSError as exc:
                    emit("git", "warn", f"could not restore {arcname}: {exc}")
    except (OSError, tarfile.TarError) as exc:
        emit("git", "warn", f"restore from snapshot failed: {exc}")
        return
    if extracted:
        emit("git", "info", f"restored {extracted} file(s) under configs/")


# --------------------------------------------------------------------- #
# Stage 3 — fetch
# --------------------------------------------------------------------- #


def _fetch(cwd: Path, emit: ProgressCallback) -> None:
    emit("git", "info", "git fetch --tags origin")
    rc = _stream_subprocess(
        ["git", "fetch", "--tags", "--prune", "origin"], cwd=cwd, phase="git", emit=emit,
    )
    if rc != 0:
        msg = f"git fetch failed (exit {rc})"
        raise RuntimeError(msg)


# --------------------------------------------------------------------- #
# Stage 4 — checkout
# --------------------------------------------------------------------- #


def _apply_ref(
    cwd: Path,
    *,
    channel: ChannelName,
    force: bool,
    emit: ProgressCallback,
) -> None:
    if channel == "tag":
        target_ref = _resolve_latest_tag(cwd)
        if not target_ref:
            msg = "no v* tag reachable from origin; switch to channel=main."
            raise RuntimeError(msg)
    else:
        target_ref = "origin/main"
    emit("git", "info", f"git checkout {target_ref}")
    checkout_cmd = ["git", "checkout"]
    if force:
        checkout_cmd.append("--force")
    checkout_cmd.append(target_ref)
    rc = _stream_subprocess(checkout_cmd, cwd=cwd, phase="git", emit=emit)
    if rc != 0:
        msg = f"git checkout {target_ref} failed (exit {rc})"
        raise RuntimeError(msg)


# --------------------------------------------------------------------- #
# Stage 5 — install + build
# --------------------------------------------------------------------- #


def _install_deps(cwd: Path, *, build: bool, emit: ProgressCallback) -> None:
    emit("deps", "info", "reinstalling Python dependencies")
    py_cmd = _build_pip_command(cwd)
    rc = _stream_subprocess(py_cmd, cwd=cwd, phase="deps", emit=emit)
    if rc != 0:
        msg = f"pip install failed (exit {rc})"
        raise RuntimeError(msg)
    if not build:
        return
    emit("build", "info", "npm run build (web/)")
    npm = _find_npm(cwd)
    if npm is None:
        emit("build", "warn", "npm not found; skipping SPA rebuild.")
        return
    rc = _stream_subprocess([npm, "run", "build"], cwd=cwd / "web", phase="build", emit=emit)
    if rc != 0:
        msg = f"npm run build failed (exit {rc})"
        raise RuntimeError(msg)


def _has_any_local_changes(cwd: Path) -> bool:
    """Raw ``git status --porcelain`` check — no filter. Used by the
    stash gate where we want to preserve *every* modified path."""
    out = _git(["status", "--porcelain"], cwd=cwd)
    return out.returncode == 0 and bool(out.stdout.strip())


def _resolve_latest_tag(cwd: Path) -> str | None:
    """Return the highest reachable v* tag (e.g. ``v1.0.5``) or None."""
    out = _git(["tag", "-l", "v*", "--sort=-v:refname"], cwd=cwd).stdout
    pattern = re.compile(r"^v\d+\.\d+\.\d+$")
    for line in out.splitlines():
        line = line.strip()
        if pattern.match(line):
            return line
    return None


def _build_pip_command(cwd: Path) -> list[str]:
    """Pick uv when available; otherwise fall back to plain pip."""
    import sys as _sys  # noqa: PLC0415

    py = _sys.executable
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415

        uv = find_uv()
        if uv:
            return [uv, "pip", "install", "-e", ".[api,dev]", "--python", py]
    except Exception:  # noqa: BLE001
        pass
    return [py, "-m", "pip", "install", "-e", ".[api,dev]"]


def _find_npm(repo: Path) -> str | None:
    """Prefer the portable Node, fall back to PATH."""
    import shutil  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    if _sys.platform == "win32":
        cand = repo / ".node" / "npm.cmd"
        if cand.is_file():
            return str(cand)
    else:
        cand = repo / ".node" / "bin" / "npm"
        if cand.is_file():
            return str(cand)
    found = shutil.which("npm")
    return found if found else None


def _stream_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    phase: str,
    emit: ProgressCallback,
) -> int:
    """Run ``cmd`` and forward stdout+stderr lines through ``emit``.

    Buffered line-by-line so a long-running ``npm run build`` shows up
    in the UI as it progresses, not as a single 30-second freeze.
    """
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None  # noqa: S101
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            emit(phase, "info", line)
    return proc.wait()


# --------------------------------------------------------------------- #
# Test seams
# --------------------------------------------------------------------- #
#
# Public for tests: small surface so unit tests can call the stages
# without a live network or git remote. Not exported via __all__.
def _iter_user_owned_prefixes() -> Iterator[str]:
    yield from _USER_OWNED_PREFIXES


__all__ = [
    "ChannelName",
    "UpdateInfo",
    "apply",
    "check",
    "last_check",
]
