"""Self-update orchestration: GitHub Releases polling + git-based upgrade.

Mirrors the *shape* of ShiroManager's app-updater (mirror pool, cached
release metadata, periodic background check) but the *artifact* is
different: ShiroManager downloads a NSIS installer; LoraHub upgrades by
running ``git pull`` (or ``git checkout v…``) inside the working tree.

Public surface:

* ``check(channel="dev")`` — return ``UpdateInfo`` for the resolved
  channel. Uses a 5-minute on-disk cache so opening the Settings page
  doesn't re-hit the GitHub API every refresh.
* ``apply(channel, *, restart, build, progress)`` — perform the upgrade,
  emitting structured progress events through ``progress``.
* ``last_check()`` — return the cached payload if any.

Channels:
  ``dev`` — checkout origin/dev (rolling pre-release; was named
            "main" through v1.0.3 — see ``_LEGACY_CHANNEL_ALIASES``
            for the back-compat shim).
  ``tag`` — checkout the highest semver ``v*`` tag.

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
import os
import queue
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from platformdirs import user_state_path

from lorahub.api.system_update_types import (
    CacheBlob,
    ChannelName,
    ProgressCallback,
    UpdateInfo,
)

# Pre-v1.0.4 the rolling channel was called "main". Old on-disk
# update-cache.json files and any client that hard-coded the name
# get translated transparently in :func:`check`.
_LEGACY_CHANNEL_ALIASES: dict[str, ChannelName] = {"main": "dev"}

GITHUB_OWNER = "GALIAIS"
GITHUB_REPO = "LoraHub"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
TAGS_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/tags"
COMMITS_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/dev"
WEB_RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
WEB_COMMITS_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commits/dev"

CACHE_TTL_SECONDS = 5 * 60
HTTP_TIMEOUT_S = 12.0


def _state_dir() -> Path:
    p = user_state_path("lorahub", "lorahub")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_file() -> Path:
    return _state_dir() / "update-cache.json"


_CacheBlob = CacheBlob


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _current_version() -> str:
    """Resolve the running lorahub version through a chain of fallbacks.

    The historical path was ``hatch-vcs → _version.py → __version__``,
    which only works when the user installed from a real git checkout.
    Users who download a ZIP from GitHub and run scripts/install.{sh,bat}
    against it have no ``.git/`` to read tags from, so hatch-vcs writes
    ``0.0.0`` (or skips writing entirely) and the UI reports a useless
    ``Current: 0.0.0``.

    Resolution chain:

      1. ``lorahub._version.__version__`` (hatch-vcs result) — only
         taken if it isn't the placeholder ``0.0.0`` / ``0.0.0+unknown``.
      2. Installed distribution metadata via ``importlib.metadata`` —
         covers wheel installs (``pip install lorahub``) and any zip-
         install where ``pyproject.toml`` wrote a static version into
         the dist info.
      3. The latest released entry in ``CHANGELOG.md`` — last-resort
         heuristic for hand-extracted ZIP trees that never went through
         pip at all. Only the version *number* line is parsed
         (e.g. ``## [0.3.0] - 2026-05-18``).
      4. ``0.0.0+unknown`` — the original fallback, only when every
         source above silently failed.

    The returned string is the literal version; callers that need to
    differentiate "real" vs "guessed" use ``_resolve_version()`` below.
    """
    return _resolve_version()[0]


# Source label so the UI can mark a version as "guessed" rather than
# implying parity with hatch-vcs precision.
_VERSION_SOURCES = ("git-describe", "hatch-vcs", "dist-metadata", "changelog", "fallback")


def _subprocess_no_window() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return 0


def _git_describe_runtime() -> str | None:
    """Resolve the version via runtime ``git describe`` against the repo.

    Why this lives in front of ``hatch-vcs`` in the resolution chain:
    ``_version.py`` is *generated at install time* — running
    ``pip install -e .`` writes the current describe output, but git
    advancing afterwards (the user committed, switched branch, or did
    a force-push) leaves the file frozen on the older sha. The user
    then sees ``Backend 1.0.3.dev85+g7ddfe78`` in the UI even though
    the source tree is on a different commit, and the version chip's
    sha-comparison flags it as a mismatch against the freshly-built
    frontend bundle whose ``__APP_VERSION__`` came from running git
    describe at vite build time.

    This shells out to ``git describe --tags --dirty --always`` from
    the project root, mirroring exactly what ``vite.config.ts`` does
    on the frontend side. Same tool + same flags + same cwd → the two
    halves agree without anyone having to re-run ``pip install``.

    Returns ``None`` (so callers fall through) when:

      * git isn't on PATH (CI containers without git, ZIP installs)
      * ``project_root()`` isn't a git checkout
      * the call times out (5s ceiling so a hung ``index.lock`` can't
        wedge ``/api/health``)

    The frontend's leading ``v`` from a tag is stripped here too, so
    the two halves produce byte-identical strings on the same commit.
    """
    try:
        from lorahub.core.paths import project_root  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    try:
        root = project_root()
    except Exception:  # noqa: BLE001
        return None

    # Quick bail-out: if there's no .git directory we'd just spawn
    # subprocess for nothing. ``Path.is_dir`` covers both the regular
    # case and worktrees where ``.git`` is a file pointing elsewhere.
    git_dir = root / ".git"
    if not git_dir.exists():
        return None

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=_subprocess_no_window(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    return raw[1:] if raw.startswith("v") else raw


def _resolve_version() -> tuple[str, str]:
    """Return ``(version_string, source_label)``.

    ``source_label`` is one of the constants in ``_VERSION_SOURCES``.
    The web UI surfaces it as a tooltip when the source is anything
    but ``hatch-vcs`` so users understand why the number might lag
    by a commit or two.
    """
    placeholders = {"", "0.0.0", "0.0.0+unknown"}

    # 1. Live ``git describe`` — preferred over the static hatch-vcs
    # snapshot because ``_version.py`` only refreshes when pip runs.
    # Same command + flags as the frontend's vite.config.ts so a
    # commit that touches both halves yields identical strings on
    # both sides.
    git_v = _git_describe_runtime()
    if git_v and git_v not in placeholders:
        return git_v, "git-describe"

    # 2. hatch-vcs (the canonical path).
    try:
        from lorahub import __version__  # noqa: PLC0415

        v = str(__version__).strip()
        if v and v not in placeholders:
            return v, "hatch-vcs"
    except Exception:  # noqa: BLE001
        pass

    # 3. importlib.metadata — wheel / pip install / static metadata.
    try:
        from importlib.metadata import PackageNotFoundError, version as dist_version  # noqa: PLC0415

        try:
            v = dist_version("lorahub").strip()
            if v and v not in placeholders:
                return v, "dist-metadata"
        except PackageNotFoundError:
            pass
    except Exception:  # noqa: BLE001
        pass

    # 4. CHANGELOG.md — read the latest non-Unreleased ``## [X.Y.Z]`` line.
    changelog_v = _read_changelog_version()
    if changelog_v:
        return changelog_v, "changelog"

    # 5. Last-resort marker.
    return "0.0.0+unknown", "fallback"


def _read_changelog_version() -> str | None:
    """Pluck the most recent released version from CHANGELOG.md.

    Matches lines shaped like ``## [0.3.0] - 2026-05-18`` and skips
    ``## [Unreleased]``. Stops at the first match — entries are written
    newest-first per the project's CHANGELOG convention.
    """
    try:
        from lorahub.core.paths import project_root  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    candidates = [
        project_root() / "CHANGELOG.md",
        Path(__file__).resolve().parents[2] / "CHANGELOG.md",
    ]
    pattern = re.compile(r"^##\s*\[\s*(\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?)\s*\]")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                m = pattern.match(line.strip())
                if m:
                    return m.group(1)
        except OSError:
            continue
    return None


def _normalize_version(raw: str) -> str:
    """Drop a leading 'v' plus git-describe dirtiness markers."""
    s = raw.strip()
    if s.lower().startswith("v"):
        s = s[1:]
    if s.endswith("-dirty"):
        s = s[:-6]
    return s


def _compare_versions(left: str, right: str) -> int:
    """``-1 / 0 / 1``. Falls back to lexicographic when packaging can't parse."""
    try:
        return (Version(_normalize_version(left)) > Version(_normalize_version(right))) - (
            Version(_normalize_version(left)) < Version(_normalize_version(right))
        )
    except InvalidVersion:
        return (left > right) - (left < right)


def _tag_update_available(
    *,
    latest: str | None,
    current: str,
    latest_commit: str | None,
    current_commit: str | None,
) -> bool:
    if not latest:
        return False
    version_cmp = _compare_versions(latest, current)
    return version_cmp > 0 or (
        version_cmp == 0
        and bool(latest_commit)
        and bool(current_commit)
        and latest_commit != current_commit
    )


def _release_notes_from_git(
    cwd: Path | None,
    tag_name: str | None,
    current_commit: str | None,
) -> str:
    if cwd is None or not tag_name or not cwd.is_dir():
        return ""
    start = current_commit or "HEAD"
    try:
        out = _git(
            ["log", "--no-merges", "--pretty=format:%h %s", f"{start}..{tag_name}", "--"],
            cwd=cwd,
        )
    except OSError:
        return ""
    if out.returncode != 0 or not out.stdout.strip():
        return f"{tag_name} 正式版"
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return "\n".join(lines[:80])[:8000]


def _release_notes_from_commit(commit_sha: str | None) -> str:
    if not commit_sha:
        return ""
    try:
        info = _fetch_json(
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{commit_sha}"
        )
    except (OSError, ValueError):
        return ""
    msg = (info.get("commit") or {}).get("message") or ""
    return msg.split("\n", 1)[0][:300]


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
    # Same UTF-8 reasoning as _stream_subprocess: git on Windows emits
    # UTF-8 by default for paths and commit messages, so we must avoid
    # the locale-dependent fallback that gbk-defaults to on zh-CN hosts.
    return subprocess.run(  # noqa: S603, S607
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_subprocess_no_window(),
    )


def _detect_dirty(cwd: Path) -> bool:
    """``True`` iff the working tree has uncommitted changes the user
    *cares about preserving across upgrade*.

    Naive ``git status --porcelain`` returns every modified file,
    which causes a routine "I edited my training config" to block
    upgrade. We filter out paths the user is expected to mutate
    locally:

      * ``configs/`` — user's own training configs
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
    without an attached branch ref. ``git checkout origin/dev`` or
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


def _refresh_dev(cwd: Path) -> dict[str, Any]:
    """Probe origin/dev via the GitHub commits API (no auth, 60/hr)."""
    info = _fetch_json(COMMITS_API)
    sha = str(info.get("sha") or "")
    short_sha = sha[:7] if sha else ""
    msg = (info.get("commit") or {}).get("message") or ""
    return {
        "tag_name": None,
        "version_str": short_sha or "dev",
        "commit": sha or None,
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
    commit = str((info.get("target_commitish") or "")).strip() or None
    return {
        "tag_name": tag or None,
        "version_str": _normalize_version(tag) if tag else "",
        "commit": commit,
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
    best_ver, best_name, sha = candidates[0]
    return {
        "tag_name": best_name,
        "version_str": str(best_ver),
        "commit": sha or None,
        # Lightweight tags don't carry release notes; the UI just gets
        # an empty string and the "open in GitHub" link still works.
        "release_notes": "",
        "published_at": None,
    }


def _empty_tag_payload() -> dict[str, Any]:
    return {
        "tag_name": None,
        "version_str": "",
        "commit": None,
        "release_notes": "",
        "published_at": None,
    }


def _current_commit(cwd: Path | None) -> str | None:
    if cwd is None:
        return None
    out = _git(["rev-parse", "HEAD"], cwd=cwd)
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _remote_tag_commit(cwd: Path | None, tag_name: str | None) -> str | None:
    if cwd is None or not tag_name:
        return None
    # Annotated tags need the peeled ^{} ref; lightweight tags only
    # have the direct ref. Query both without relying on local stale tags.
    for ref in (f"refs/tags/{tag_name}^{{}}", f"refs/tags/{tag_name}"):
        out = _git(["ls-remote", "--tags", "origin", ref], cwd=cwd)
        if out.returncode != 0:
            continue
        line = out.stdout.strip().splitlines()
        if not line:
            continue
        sha = line[0].split()[0].strip()
        if sha:
            return sha
    return None


def check(channel: ChannelName = "tag", *, force: bool = False) -> UpdateInfo:
    """Resolve current-vs-remote for the given channel.

    Returns even on network errors — the ``error`` field carries the
    failure message so the UI can render the cached state plus an
    "offline" hint.
    """
    # Old clients / cached payloads still carry the pre-v1.0.4
    # ``main`` channel name. Translate so they keep working without
    # forcing the user to clear the on-disk cache.
    channel = _LEGACY_CHANNEL_ALIASES.get(channel, channel)  # type: ignore[arg-type]
    cwd = _git_root()
    is_dirty = _detect_dirty(cwd) if cwd else False
    current, version_source = _resolve_version()
    is_git_checkout = cwd is not None
    current_commit = _current_commit(cwd)

    blob = _read_cache()
    cached = blob.data.get(channel)
    fresh_enough = (
        cached
        and not force
        and (time.time() - blob.updated_at) < CACHE_TTL_SECONDS
    )
    if fresh_enough and channel == "tag" and cached.get("tag_name") and not cached.get("latest_commit"):
        fresh_enough = False

    if fresh_enough:
        cached_latest = cached.get("latest")
        cached_latest_commit = (
            _remote_tag_commit(cwd, cached.get("tag_name"))
            if channel == "tag"
            else cached.get("latest_commit")
        )
        if cached_latest_commit is None:
            cached_latest_commit = cached.get("latest_commit")
        info = UpdateInfo(
            **{
                **cached,
                "current": current,
                "is_dirty": is_dirty,
                "version_source": version_source,
                "git_checkout": is_git_checkout,
                "current_commit": current_commit,
                "latest_commit": cached_latest_commit,
                "update_available": (
                    _tag_update_available(
                        latest=cached_latest,
                        current=current,
                        latest_commit=cached_latest_commit,
                        current_commit=current_commit,
                    )
                    if channel == "tag"
                    else cached.get("update_available", False)
                ),
            }
        )
        return info

    try:
        if channel == "dev":
            remote = _refresh_dev(cwd or Path.cwd())
        else:
            remote = _refresh_tag()
    except (OSError, ValueError) as exc:
        # Network failure — degrade gracefully to the cached payload.
        if cached:
            info = UpdateInfo(
                **{
                    **cached,
                    "current": current,
                    "is_dirty": is_dirty,
                    "version_source": version_source,
                    "git_checkout": is_git_checkout,
                    "current_commit": current_commit,
                }
            )
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
            version_source=version_source,
            git_checkout=is_git_checkout,
            current_commit=current_commit,
        )

    latest = remote["version_str"] or None
    latest_commit = (
        _remote_tag_commit(cwd, remote.get("tag_name"))
        if channel == "tag"
        else remote.get("commit")
    )
    if latest_commit is None:
        latest_commit = remote.get("commit")
    release_notes = remote["release_notes"]
    if channel == "tag":
        release_notes = (
            _release_notes_from_commit(latest_commit)
            or release_notes
            or _release_notes_from_git(
            cwd,
            remote.get("tag_name"),
            current_commit,
            )
        )
    if channel == "tag" and latest:
        update_available = _tag_update_available(
            latest=latest,
            current=current,
            latest_commit=latest_commit,
            current_commit=current_commit,
        )
    elif channel == "dev" and latest and cwd:
        # For dev, "update available" means HEAD is not on the
        # remote sha. We compare short SHA prefixes via git
        # rev-parse so a forced reset still counts as up-to-date.
        head = current_commit or ""
        update_available = bool(head) and not head.startswith(latest)
    else:
        update_available = False

    info = UpdateInfo(
        channel=channel,
        current=current,
        latest=latest,
        update_available=update_available,
        release_url=WEB_RELEASES_URL if channel == "tag" else WEB_COMMITS_URL,
        release_notes=release_notes,
        checked_at=_now_iso(),
        is_dirty=is_dirty,
        tag_name=remote["tag_name"],
        published_at=remote["published_at"],
        current_commit=current_commit,
        latest_commit=latest_commit,
        version_source=version_source,
        git_checkout=is_git_checkout,
    )

    blob.data[channel] = info.to_dict()
    blob.updated_at = time.time()
    _write_cache(blob)
    return info


# --------------------------------------------------------------------- #
# apply() — five-stage pipeline
#
#   1. _pre_check        — git root check, detached HEAD probe, dirty fence
#   2. _snapshot_configs — tarball the user's untracked configs/* (presets
#                          remain owned by git so upstream changes apply)
#   3. _fetch            — git fetch --tags origin
#   4. _apply_ref        — checkout the resolved ref (origin/dev or v…)
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
      2. ``git checkout origin/dev`` (channel=dev) or
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
    # Translate legacy ``main`` channel name (pre-v1.0.4 callers) so
    # tests / old scripts / cached UI state keep working without a
    # forced rename pass.
    channel = _LEGACY_CHANNEL_ALIASES.get(channel, channel)  # type: ignore[arg-type]
    cwd = _git_root()
    if cwd is None:
        # This is the path ZIP-extracted users hit. The check() endpoint
        # already exposes ``git_checkout=False`` so the UI can grey out
        # the "Apply" button before they click it; this string is the
        # narrative version of that, surfaced when an outdated UI (or
        # the CLI) reaches this point anyway.
        msg = (
            "无法在线更新:当前安装目录不是 git 检出(常见于直接下载 ZIP 解压)。\n"
            "请按以下任一方式重装为可更新形态:\n"
            "  1) 推荐:在新目录 `git clone https://github.com/GALIAIS/LoraHub` "
            "再跑 scripts/install.{sh,bat};把旧目录的 datasets/、configs/、models/ "
            "搬过去即可继承数据。\n"
            "  2) 在当前目录 `git init && git remote add origin "
            "https://github.com/GALIAIS/LoraHub.git && git fetch && git reset "
            "--hard origin/dev`,然后重跑 `pip install -e .` 让 hatch-vcs "
            "重写 _version.py。\n"
            "完成任一步骤后,Web 设置页 → 软件更新 即可正常使用。"
        )
        raise RuntimeError(msg)

    emit = progress if progress is not None else _NULL_EMIT

    _pre_check(cwd, channel=channel, force=force, emit=emit)

    with _UpdateContext(cwd, emit) as ctx:
        ctx.snapshot_path = _snapshot_configs(cwd, emit)
        # `_snapshot_configs` captures user-created configs and locally edited
        # tracked configs. Untouched repo presets still ride along with the
        # checkout so upstream default fixes can land.

        if force:
            emit(
                "git", "warn",
                "force=True: discarding local changes (git reset --hard + clean -fd); "
                "user-owned paths (configs/, runs/, models/, output/, datasets/, "
                ".env*, external/anima_lora/{output,post_image_dataset}) are "
                "preserved via -e excludes",
            )
            _stream_subprocess(
                ["git", "reset", "--hard", "HEAD"], cwd=cwd, phase="git", emit=emit,
            )
            # ``git clean -fd`` only deletes *untracked* files, so a tracked
            # path that lives behind a user-owned prefix (e.g. configs/*.yaml)
            # is already safe — we still pass it as an exclude so a future
            # ``.gitignore`` change that un-tracks the directory doesn't
            # silently turn the upgrade into ``rm -rf``. The big risk paths
            # are the ones a user creates fresh: datasets/, models/, output/,
            # runs/ — those *aren't* in every .gitignore at every commit, so
            # we need explicit excludes here to keep ``force`` from wiping
            # local artefacts the user has produced since the last sync.
            clean_cmd: list[str] = ["git", "clean", "-fd"]
            for prefix in _USER_OWNED_PREFIXES:
                # Pathspec form: drop the trailing slash, use forward slashes.
                spec = prefix.rstrip("/").replace("\\", "/")
                if spec:
                    clean_cmd.extend(["-e", spec])
            _stream_subprocess(
                clean_cmd, cwd=cwd, phase="git", emit=emit,
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

        _restore_configs(cwd, ctx.snapshot_path, emit)
        # The snapshot has been re-laid into the working tree; the
        # context's __exit__ no longer needs to restore it on success.
        ctx.snapshot_consumed = True

    emit("done", "info", "update applied")


def _NULL_EMIT(_phase: str, _level: str, _message: str) -> None:
    pass


@dataclass
class _UpdateContext:
    """Tracks the rollback state across the upgrade stages.

    On a clean exit the snapshot tar is unlinked and any leftover
    stash is left to the caller. On an exception:

      * ``snapshot_path`` (if set and not yet consumed) is unpacked
        back over ``configs/`` so the user's configs survive even
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
            if self.stash_active:
                with contextlib.suppress(Exception):
                    self.emit(
                        "git", "warn",
                        "upgrade failed; popping stash to restore local edits",
                    )
                    _stream_subprocess(
                        ["git", "stash", "pop"], cwd=self.cwd, phase="git", emit=self.emit,
                    )
            if self.snapshot_path is not None and not self.snapshot_consumed:
                with contextlib.suppress(Exception):
                    self.emit(
                        "git", "warn",
                        "upgrade failed; restoring configs/ from pre-flight snapshot",
                    )
                    _restore_configs(self.cwd, self.snapshot_path, self.emit)
        # Always remove the temp archive — it's only useful as a
        # rollback bridge and would otherwise accumulate in TMPDIR.
        if self.snapshot_path is not None:
            with contextlib.suppress(Exception):
                self.snapshot_path.unlink(missing_ok=True)


# --------------------------------------------------------------------- #
# Stage 1 — pre-flight
# --------------------------------------------------------------------- #


def _pre_check(
    cwd: Path,
    *,
    channel: ChannelName,
    force: bool,
    emit: ProgressCallback,
) -> None:
    """Refuse upgrade in states the rest of the pipeline can't recover from.

    Currently:
      * detached HEAD when the SHA isn't reachable from
        ``origin/<channel>`` (``force=False`` only) — checking out
        ``origin/dev`` from a detached state with local commits ahead
        of the channel would silently abandon them.

    When the detached SHA IS reachable from the channel head — the
    common "I checked out v1.0.4 by tag and now want to track ``dev``
    again" case — we auto-attach by ``git checkout <channel>`` instead
    of refusing. No commits are lost because everything we have is
    already in the remote channel's history. ``force=True`` short-
    circuits the safety gate entirely.
    """
    head_sha = _detect_detached_head(cwd)
    if head_sha is None:
        return
    if force:
        emit(
            "git", "warn",
            f"force=True: ignoring detached HEAD at {head_sha}; the upcoming "
            "checkout would discard any commits made on top of the detached SHA",
        )
        return
    # Try the friendly path first: fetch the remote channel and see if
    # our SHA is in its history. If so, the detached state is just
    # "I checked out a tag" rather than "I committed local work" —
    # safe to auto-attach. We do an explicit fetch here even though
    # ``apply()`` will fetch again later: ``_pre_check`` runs before
    # the snapshot stage, so without this the reachable check works
    # off whatever stale ``origin/<channel>`` ref the local repo last
    # synced. The duplicate fetch is cheap (a few hundred KB of refs)
    # and idempotent.
    target_branch = "main" if channel == "tag" else channel
    target_remote = f"refs/remotes/origin/{target_branch}"
    fetch = _git(
        [
            "fetch",
            "--quiet",
            "origin",
            f"{target_branch}:refs/remotes/origin/{target_branch}",
        ],
        cwd=cwd,
    )
    if fetch.returncode != 0:
        # Network failure — fall through to the abandon-commits error
        # rather than silently mis-classifying the detached state.
        msg = (
            f"HEAD is detached at {head_sha} and `git fetch origin "
            f"{target_branch}:refs/remotes/origin/{target_branch}` failed "
            f"({fetch.stderr.strip() or 'unknown'}). "
            f"Run `git checkout {target_branch}` first, or pass --force to "
            "discard the detached commits."
        )
        raise RuntimeError(msg)
    reachable = _git(
        ["merge-base", "--is-ancestor", "HEAD", target_remote],
        cwd=cwd,
    )
    if reachable.returncode == 0:
        # Detached SHA is on the remote channel's first-parent line —
        # safe to attach without losing anything.
        emit(
            "git", "info",
            f"detached at {head_sha} (reachable from {target_remote}); "
            f"auto-attaching to local branch `{target_branch}`",
        )
        # Create or fast-forward the local branch to whatever HEAD is
        # pointing at. ``git checkout -B`` resets the branch ref if it
        # exists and creates it otherwise; using ``HEAD`` as the
        # explicit start-point avoids accidentally fast-forwarding
        # past our current SHA when the remote moved on between the
        # fetch above and this checkout.
        switch = _git(
            ["checkout", "-B", target_branch, "HEAD"],
            cwd=cwd,
        )
        if switch.returncode != 0:
            msg = (
                f"detected reachable detached HEAD at {head_sha} but "
                f"`git checkout -B {target_branch}` failed: "
                f"{switch.stderr.strip() or 'unknown'}"
            )
            raise RuntimeError(msg)
        return
    # Not reachable — preserve the original safety gate.
    msg = (
        f"HEAD is detached at {head_sha} with commits not reachable from "
        f"{target_remote}. Self-update from this state would silently "
        f"abandon them. Run `git checkout {target_branch}` and merge / "
        f"cherry-pick first, or pass --force to discard the detached commits."
    )
    raise RuntimeError(msg)


# --------------------------------------------------------------------- #
# Stage 2 — configs/ snapshot
# --------------------------------------------------------------------- #


def _snapshot_configs(cwd: Path, emit: ProgressCallback) -> Path | None:
    """Capture user-authored and locally edited files under ``configs/``.

    The snapshot includes untracked config files plus tracked config files
    with staged or unstaged local edits. Untouched tracked presets are not
    captured, so upstream default fixes still reach users who did not edit
    those files. In force mode this snapshot is the only protection tracked
    config edits have before ``git reset --hard``.

    The archive lives in ``tempfile.gettempdir()`` so a multi-megabyte
    yaml collection doesn't have to be held in memory while the upgrade
    runs. Failure to add a single file logs a warning and skips that
    file rather than aborting the upgrade — partial coverage beats
    refusing to update.

    Returns ``None`` when ``configs/`` is missing, empty, or contains no
    user-created or locally edited files to protect.
    """
    root = cwd / "configs"
    if not root.is_dir():
        return None

    rels: set[str] = set()
    git_failed: list[str] = []
    commands = (
        ["ls-files", "--others", "--exclude-standard", "--", "configs"],
        ["diff", "--name-only", "--", "configs"],
        ["diff", "--name-only", "--cached", "--", "configs"],
    )
    for command in commands:
        result = _git(command, cwd=cwd)
        if result.returncode == 0:
            rels.update(line.strip() for line in result.stdout.splitlines() if line.strip())
        else:
            git_failed.append(result.stderr.strip() or "unknown")

    if not git_failed:
        files = [cwd / rel for rel in sorted(rels) if (cwd / rel).is_file()]
    else:
        # Defensive fallback: corrupted index or non-git layout. Warn so
        # the "save everything" behaviour is observable and preserve the
        # old semantics — we'd rather risk overwriting a preset than
        # silently lose user data on a degraded git state.
        emit(
            "git", "warn",
            f"git config snapshot scan failed ({'; '.join(git_failed)}); "
            "falling back to full configs/ snapshot — tracked-preset edits "
            "will override upstream changes this run.",
        )
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
    emit("git", "info", "git fetch --tags --force origin")
    rc = _stream_subprocess(
        ["git", "fetch", "--tags", "--force", "--prune", "origin"],
        cwd=cwd,
        phase="git",
        emit=emit,
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
            msg = "no v* tag reachable from origin; switch to channel=dev."
            raise RuntimeError(msg)
    else:
        target_ref = "origin/dev"
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
    emit("deps", "info", "running " + _format_cmd(py_cmd))
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
    web = cwd / "web"
    _ensure_frontend_deps(web, npm=Path(npm), emit=emit)
    rc = _stream_subprocess(
        [npm, "run", "build"],
        cwd=web,
        phase="build",
        emit=emit,
        env=_npm_env(Path(npm)),
    )
    if rc != 0:
        msg = f"npm run build failed (exit {rc})"
        raise RuntimeError(msg)


def _ensure_frontend_deps(web: Path, *, npm: Path, emit: ProgressCallback) -> None:
    env = _npm_env(npm)
    rc = _stream_subprocess(
        [str(npm), "ls", "--depth=0"],
        cwd=web,
        phase="build",
        emit=emit,
        env=env,
    )
    if rc == 0:
        emit("build", "info", "web dependencies already installed")
        return
    emit("build", "warn", "web dependencies incomplete; running npm ci")
    rc = _stream_subprocess(
        [
            str(npm),
            "ci",
            "--verbose",
            "--no-audit",
            "--no-fund",
            "--fetch-timeout=60000",
            "--fetch-retries=2",
            "--fetch-retry-mintimeout=5000",
            "--fetch-retry-maxtimeout=20000",
        ],
        cwd=web,
        phase="build",
        emit=emit,
        env=env,
    )
    if rc != 0:
        msg = f"npm ci failed (exit {rc})"
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
    pypi_index = _configured_pypi_index()
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415

        uv = find_uv()
        if uv:
            cmd = [uv, "pip", "install"]
            if os.environ.get("LORAHUB_INSTALL_VERBOSE") == "1":
                cmd.append("-v")
            if pypi_index:
                cmd += ["--index-url", pypi_index]
            return [*cmd, "-e", ".[api,dev]", "--python", py, "--link-mode=copy"]
    except Exception:  # noqa: BLE001
        pass
    cmd = [py, "-m", "pip", "install"]
    if pypi_index:
        cmd += ["--index-url", pypi_index]
    return [*cmd, "-e", ".[api,dev]"]


def _configured_pypi_index() -> str | None:
    env_index = (os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL") or "").strip()
    if env_index:
        return env_index
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        return (_app._settings_store.load().pypi_index_url or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _format_cmd(cmd: list[str]) -> str:
    import shlex  # noqa: PLC0415

    return " ".join(shlex.quote(part) for part in cmd)


def _find_npm(repo: Path) -> str | None:
    """Prefer the portable Node, fall back to PATH."""
    import shutil  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    env_node_dir = os.environ.get("NODE_DIR")
    if env_node_dir:
        cand = Path(env_node_dir) / ("npm.cmd" if _sys.platform == "win32" else "bin/npm")
        if cand.is_file():
            return str(cand)
    if _sys.platform == "win32":
        cand = repo / ".node" / "npm.cmd"
        if cand.is_file():
            return str(cand)
    else:
        for cand in (
            repo / ".node" / "bin" / "npm",
            Path("/root/autodl-tmp/opt/node20/bin/npm"),
        ):
            if cand.is_file():
                return str(cand)
    found = shutil.which("npm")
    return found if found else None


def _npm_env(npm: Path) -> dict[str, str]:
    """Ensure npm lifecycle scripts can find the matching node binary."""
    env = os.environ.copy()
    env["PATH"] = f"{npm.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def _stream_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    phase: str,
    emit: ProgressCallback,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``cmd`` and forward stdout+stderr lines through ``emit``.

    Buffered line-by-line so a long-running ``npm run build`` shows up
    in the UI as it progresses, not as a single 30-second freeze.
    """
    # Force UTF-8 + replace on undecodable bytes. Without this, Python
    # falls back to locale.getpreferredencoding() — which on a zh-CN
    # Windows host is gbk/cp936. Vite, npm, and pip all emit UTF-8
    # status glyphs ("✓", "▲", boxed CJK) and gbk chokes on the very
    # first banner with UnicodeDecodeError, killing the whole upgrade.
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None  # noqa: S101
    lines: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw.rstrip())
        finally:
            lines.put(None)

    threading.Thread(target=_reader, name="lorahub-update-stream", daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=30)
        except queue.Empty:
            if proc.poll() is None:
                emit(phase, "info", "still running ...")
                continue
            break
        if line is None:
            break
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
