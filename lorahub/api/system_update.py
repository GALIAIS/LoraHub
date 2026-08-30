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
  ``tag`` — display the highest semver ``v*`` tag as the formal
            version, but update to ``origin/main`` so hotfix commits
            pushed after the tag are visible and installable.

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
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
MAIN_COMMITS_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/main"
WEB_RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
WEB_COMMITS_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commits/dev"
WEB_MAIN_COMMITS_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commits/main"

CACHE_TTL_SECONDS = 5 * 60
HTTP_TIMEOUT_S = 12.0
_MAX_GITHUB_API_BYTES = 8 * 1024 * 1024
_RELEASE_TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
_CACHE_WRITE_LOCK = threading.Lock()


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
_VERSION_SOURCES = ("env", "git-describe", "hatch-vcs", "dist-metadata", "changelog", "fallback")


def _subprocess_no_window() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _git_describe_runtime() -> str | None:
    """Resolve a readable runtime version from the git checkout.

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

    The user-facing shape is ``<tag>-<branch>-g<sha8>``:
    ``1.1.0-main-g2e1c81ef`` for release branch builds and
    ``1.1.0-dev-g2e1c81ef`` for dev builds. Dirty working-tree state
    is reported separately by the update endpoint; it should not be
    folded into the version identity.

    Returns ``None`` (so callers fall through) when:

      * git isn't on PATH (CI containers without git, ZIP installs)
      * ``project_root()`` isn't a git checkout
      * the call times out (5s ceiling so a hung ``index.lock`` can't
        wedge ``/api/health``)

    The frontend strips a leading ``v`` too, so the two halves produce
    byte-identical strings on the same commit.
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
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
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
    base = result.stdout.strip()
    if not base:
        return None
    sha = _git(["rev-parse", "--short=8", "HEAD"], cwd=root).stdout.strip()
    branch = _git(["branch", "--show-current"], cwd=root).stdout.strip()
    if not branch:
        branch = _branch_from_remote_contains(root) or "detached"
    branch = re.sub(r"[^0-9A-Za-z._-]+", "-", branch).strip("-") or "detached"
    version = base[1:] if base.startswith("v") else base
    return f"{version}-{branch}-g{sha}" if sha else version


def _branch_from_remote_contains(root: Path) -> str | None:
    out = _git(["branch", "-r", "--contains", "HEAD"], cwd=root)
    if out.returncode != 0:
        return None
    refs = {line.strip().removeprefix("origin/") for line in out.stdout.splitlines()}
    if "main" in refs:
        return "main"
    if "dev" in refs:
        return "dev"
    return next((ref for ref in sorted(refs) if ref and " -> " not in ref), None)


def _resolve_version() -> tuple[str, str]:
    """Return ``(version_string, source_label)``.

    ``source_label`` is one of the constants in ``_VERSION_SOURCES``.
    The web UI surfaces it as a tooltip when the source is anything
    but ``hatch-vcs`` so users understand why the number might lag
    by a commit or two.
    """
    placeholders = {"", "0.0.0", "0.0.0+unknown"}

    env_v = os.environ.get("LORAHUB_APP_VERSION", "").strip().removeprefix("v")
    if env_v and env_v not in placeholders:
        return env_v, "env"

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
        from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
        from importlib.metadata import version as dist_version

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
    s = re.sub(r"-[0-9A-Za-z._-]+-g[0-9a-f]{7,40}$", "", s, flags=re.IGNORECASE)
    return s


def _compare_versions(left: str, right: str) -> int:
    """``-1 / 0 / 1``. Falls back to lexicographic when packaging can't parse."""
    try:
        return (Version(_normalize_version(left)) > Version(_normalize_version(right))) - (
            Version(_normalize_version(left)) < Version(_normalize_version(right))
        )
    except InvalidVersion:
        return (left > right) - (left < right)


def _commit_from_version(raw: str) -> str | None:
    """Extract a build commit from LoraHub or hatch-vcs version strings."""
    match = re.search(r"(?:^|[-+])g([0-9a-f]{7,40})(?:$|[.+-])", raw, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _commits_match(left: str | None, right: str | None) -> bool:
    """Compare full and abbreviated git object ids without false mismatches."""
    if not left or not right:
        return False
    first = left.strip().lower()
    second = right.strip().lower()
    if first == second:
        return True
    if not re.fullmatch(r"[0-9a-f]{7,40}", first) or not re.fullmatch(
        r"[0-9a-f]{7,40}", second
    ):
        return False
    return first.startswith(second) or second.startswith(first)


def _tag_update_available(
    *,
    latest: str | None,
    current: str,
    latest_commit: str | None,
    current_commit: str | None,
    cwd: Path | None = None,
) -> bool:
    if not latest:
        return False
    # A locally newer semantic version must never be presented as older just
    # because its commit diverges from the release branch. Commit ancestry is
    # only meaningful once both builds identify as the same release.
    version_cmp = _compare_versions(latest, current)
    if version_cmp > 0:
        return True
    if version_cmp < 0:
        return False
    if latest_commit and current_commit:
        if _commits_match(latest_commit, current_commit):
            return False
        relation = _commit_relation(cwd, current_commit, latest_commit)
        if relation == "remote_ahead":
            return True
        if relation == "local_ahead":
            return False
        if relation == "diverged":
            return True

    if (
        not latest_commit
        or not current_commit
        or _commits_match(latest_commit, current_commit)
    ):
        return False
    # If git cannot prove ancestry, keep the older same-version retag
    # behavior so users can still recover from a moved release ref.
    return True


def _branch_update_available(
    *,
    latest_commit: str | None,
    current_commit: str | None,
    cwd: Path | None = None,
) -> bool:
    """Return true only when the remote branch can advance the current commit."""
    if (
        not latest_commit
        or not current_commit
        or _commits_match(latest_commit, current_commit)
    ):
        return False
    relation = _commit_relation(cwd, current_commit, latest_commit)
    if relation == "local_ahead":
        return False
    return relation in {"remote_ahead", "diverged", "unknown"}


def _commit_relation(
    cwd: Path | None,
    current_commit: str,
    latest_commit: str,
) -> str:
    """Return local/remote ancestry for same-version update decisions."""
    if cwd is None or not cwd.is_dir():
        return "unknown"
    try:
        current_is_ancestor = _git(
            ["merge-base", "--is-ancestor", current_commit, latest_commit],
            cwd=cwd,
        )
        if current_is_ancestor.returncode == 0:
            return "remote_ahead"
        latest_is_ancestor = _git(
            ["merge-base", "--is-ancestor", latest_commit, current_commit],
            cwd=cwd,
        )
        if latest_is_ancestor.returncode == 0:
            return "local_ahead"
        merge_base = _git(["merge-base", current_commit, latest_commit], cwd=cwd)
        if merge_base.returncode == 0 and merge_base.stdout.strip():
            return "diverged"
    except OSError:
        return "unknown"
    return "unknown"


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
        info = _api_object(
            _fetch_json(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{commit_sha}"
            ),
            "commit",
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
        raw_data = raw.get("data", {})
        data = (
            {
                str(channel): payload
                for channel, payload in raw_data.items()
                if isinstance(payload, dict)
            }
            if isinstance(raw_data, dict)
            else {}
        )
        return _CacheBlob(data=data, updated_at=float(raw.get("updated_at") or 0.0))
    except (OSError, json.JSONDecodeError, ValueError):
        return _CacheBlob()


def _write_cache(blob: _CacheBlob) -> None:
    target = _cache_file()
    if target.is_symlink():
        return
    temp_path: Path | None = None
    try:
        fd, raw_temp = tempfile.mkstemp(
            dir=target.parent,
            prefix=".update-cache-",
            suffix=".tmp",
        )
        os.close(fd)
        temp_path = Path(raw_temp)
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"data": blob.data, "updated_at": blob.updated_at},
                handle,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target)
    except OSError:
        # Cache is best-effort; failure to write is not a hard error.
        pass
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_channel_cache(channel: ChannelName, payload: dict[str, Any]) -> None:
    """Merge one channel under a process lock instead of replacing its peer."""
    with _CACHE_WRITE_LOCK:
        latest = _read_cache()
        latest.data[channel] = payload
        latest.updated_at = time.time()
        _write_cache(latest)


def _cache_entry_is_fresh(
    cached: dict[str, Any] | None,
    legacy_updated_at: float,
    *,
    now: float | None = None,
) -> bool:
    """Evaluate cache age per channel, with compatibility for old blobs.

    ``CacheBlob.updated_at`` predates the two-channel update UI and is shared
    by the whole file. Using it directly lets a refresh of one channel make a
    stale entry for the other channel appear fresh. New entries already carry
    their own ISO ``checked_at`` value, so prefer that and only fall back to
    the blob timestamp for caches written by older releases.
    """
    if not cached:
        return False
    checked_ts = legacy_updated_at
    raw_checked_at = cached.get("checked_at")
    if isinstance(raw_checked_at, str) and raw_checked_at.strip():
        try:
            checked = datetime.fromisoformat(raw_checked_at.strip())
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=UTC)
            checked_ts = checked.timestamp()
        except ValueError:
            pass
    age = (time.time() if now is None else now) - checked_ts
    return 0 <= age < CACHE_TTL_SECONDS


_UPDATE_INFO_FIELD_NAMES = frozenset(item.name for item in fields(UpdateInfo))


def _update_info_from_cache(
    cached: dict[str, Any],
    *,
    channel: ChannelName,
    **overrides: Any,
) -> UpdateInfo:
    """Read old/new cache payloads without trusting their exact schema."""
    payload = {
        key: value
        for key, value in cached.items()
        if key in _UPDATE_INFO_FIELD_NAMES
    }
    payload.update(overrides)
    payload["channel"] = channel
    payload.setdefault("current", "0.0.0+unknown")
    payload.setdefault("latest", None)
    payload.setdefault("update_available", False)
    payload.setdefault(
        "release_url",
        WEB_RELEASES_URL if channel == "tag" else WEB_COMMITS_URL,
    )
    return UpdateInfo(**payload)


def last_check() -> dict[str, dict[str, Any]] | None:
    """Return the most recent cached payload, ignoring TTL.

    The lifespan startup hook uses this to seed the API response so
    the very first request after boot doesn't have to wait for the
    background fetch to land.
    """
    blob = _read_cache()
    return blob.data if blob.data else None


def _fetch_json(url: str) -> Any:
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
        raw = resp.read(_MAX_GITHUB_API_BYTES + 1)
    if len(raw) > _MAX_GITHUB_API_BYTES:
        raise OSError("GitHub API response exceeds the safety limit")
    return json.loads(raw.decode("utf-8"))


def _api_object(payload: Any, resource: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub {resource} response is not an object")
    return payload


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


def _is_container_install() -> bool:
    """Return True when LoraHub is running from the Docker image."""
    marker = os.environ.get("LORAHUB_DOCKER", "").strip().lower()
    if marker:
        return marker not in {"0", "false", "no", "off"}
    if not Path("/.dockerenv").exists():
        return False
    return (
        os.environ.get("LORAHUB_HOME") == "/data"
        or os.environ.get("LORAHUB_WEB_DIST") == "/app/web/dist"
    )


def _install_kind(cwd: Path | None) -> str:
    if _is_container_install():
        return "docker"
    return "git" if cwd is not None else "archive"


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
    info = _api_object(_fetch_json(COMMITS_API), "dev commit")
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


def _refresh_main() -> dict[str, Any]:
    """Probe origin/main via the GitHub commits API."""
    info = _api_object(_fetch_json(MAIN_COMMITS_API), "main commit")
    sha = str(info.get("sha") or "")
    msg = (info.get("commit") or {}).get("message") or ""
    return {
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
        info = _api_object(_fetch_json(RELEASES_API), "release")
    except OSError as exc:
        if not _is_not_found(exc):
            raise
        return _refresh_tag_via_tags_api()

    tag = str(info.get("tag_name") or "")
    target_commitish = str(info.get("target_commitish") or "").strip()
    # GitHub permits ``target_commitish`` to be a branch name (commonly
    # ``main``), not the immutable commit behind the release tag. Treating
    # that label as a SHA makes Docker builds compare unequal forever.
    commit = (
        target_commitish
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", target_commitish)
        else None
    )
    if tag and commit is None:
        with contextlib.suppress(OSError, ValueError):
            commit = _commit_for_tag(_fetch_json(TAGS_API), tag)
    main: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        main = _refresh_main()
    return {
        "tag_name": tag or None,
        "version_str": _normalize_version(tag) if tag else "",
        "commit": commit,
        "branch_commit": main.get("commit"),
        "release_notes": str(info.get("body") or "")[:8000],
        "published_at": str(info.get("published_at") or "") or main.get("published_at"),
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
    candidates = _stable_tag_candidates(tags_raw)
    if not candidates:
        return _empty_tag_payload()

    best_ver, best_name, sha = candidates[0]
    main: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        main = _refresh_main()
    return {
        "tag_name": best_name,
        "version_str": str(best_ver),
        "commit": sha or None,
        "branch_commit": main.get("commit"),
        # Lightweight tags don't carry release notes; the UI just gets
        # an empty string and the "open in GitHub" link still works.
        "release_notes": main.get("release_notes") or "",
        "published_at": main.get("published_at"),
    }


def _stable_tag_candidates(raw: Any) -> list[tuple[Version, str, str]]:
    if not isinstance(raw, list):
        return []
    candidates: list[tuple[Version, str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not _RELEASE_TAG_PATTERN.fullmatch(name):
            continue
        try:
            version = Version(name.removeprefix("v"))
        except InvalidVersion:
            continue
        sha = str(entry.get("commit", {}).get("sha") or "")
        candidates.append((version, name, sha))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _commit_for_tag(raw: Any, tag_name: str) -> str | None:
    """Resolve one release tag from GitHub's tags payload."""
    if not isinstance(raw, list):
        return None
    for entry in raw:
        if not isinstance(entry, dict) or str(entry.get("name") or "") != tag_name:
            continue
        sha = str(entry.get("commit", {}).get("sha") or "").strip()
        return sha if re.fullmatch(r"[0-9a-fA-F]{7,40}", sha) else None
    return None


def list_release_history(limit: int = 6) -> list[dict[str, str | None]]:
    """Return recent stable release tags for the rollback picker."""
    bounded_limit = max(1, min(limit, 20))
    return [
        {
            "tag_name": name,
            "commit": sha or None,
        }
        for _version, name, sha in _stable_tag_candidates(_fetch_json(TAGS_API))[
            :bounded_limit
        ]
    ]


def is_release_tag(value: str) -> bool:
    return bool(_RELEASE_TAG_PATTERN.fullmatch(value))


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


def _remote_branch_commit(cwd: Path | None, branch: str) -> str | None:
    if cwd is None:
        return None
    try:
        out = _git(["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], cwd=cwd)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    line = out.stdout.strip().splitlines()
    if not line:
        return None
    sha = line[0].split()[0].strip()
    return sha or None


def _release_channel_commit(
    cwd: Path | None,
    tag_name: str | None,
    cached_commit: str | None = None,
) -> str | None:
    """Resolve the formal channel commit.

    The displayed formal version still comes from the newest semver tag,
    but the update target is the current ``main`` head. This prevents the
    Settings page from treating a local build after the tag as "behind"
    merely because the latest tag points at an older SHA.
    """
    return (
        _remote_branch_commit(cwd, "main")
        or cached_commit
        or _remote_tag_commit(cwd, tag_name)
    )


def _formal_target_commit(
    cwd: Path | None,
    install_kind: str,
    tag_name: str | None,
    *,
    cached_commit: str | None = None,
    advertised_tag_commit: str | None = None,
) -> str | None:
    """Resolve the artifact that the current installation can actually use.

    Git checkouts update from ``main`` and therefore track its head after the
    latest release tag. Docker/archive builds are immutable release artifacts;
    they only exist for a tag and must compare against that tag's commit.
    """
    if install_kind == "git":
        return _release_channel_commit(cwd, tag_name, cached_commit)
    return (
        advertised_tag_commit
        or _remote_tag_commit(cwd, tag_name)
        or cached_commit
    )


def check(channel: ChannelName = "tag", *, force: bool = False) -> UpdateInfo:
    """Resolve current-vs-remote for the given channel.

    Returns even on network errors — the ``error`` field carries the
    failure message so the UI can render the cached state plus an
    "offline" hint.
    """
    # Old clients / cached payloads still carry the pre-v1.0.4
    # ``main`` channel name. Translate so they keep working without
    # forcing the user to clear the on-disk cache.
    channel = _LEGACY_CHANNEL_ALIASES.get(channel, channel)
    cwd = _git_root()
    is_dirty = _detect_dirty(cwd) if cwd else False
    current, version_source = _resolve_version()
    install_kind = _install_kind(cwd)
    is_git_checkout = cwd is not None and install_kind == "git"
    current_commit = _current_commit(cwd) or _commit_from_version(current)

    blob = _read_cache()
    cached = blob.data.get(channel)
    fresh_enough = not force and _cache_entry_is_fresh(cached, blob.updated_at)
    if (
        fresh_enough
        and cached is not None
        and channel == "tag"
        and cached.get("tag_name")
        and not cached.get("latest_commit")
    ):
        fresh_enough = False

    if fresh_enough and cached is not None:
        cached_latest = cached.get("latest")
        cached_latest_commit = cached.get("latest_commit")
        if channel == "tag":
            cached_latest_commit = _formal_target_commit(
                cwd,
                install_kind,
                cached.get("tag_name"),
                cached_commit=cached_latest_commit,
            )
        info = _update_info_from_cache(
            cached,
            channel=channel,
            current=current,
            is_dirty=is_dirty,
            version_source=version_source,
            git_checkout=is_git_checkout,
            install_kind=install_kind,
            current_commit=current_commit,
            latest_commit=cached_latest_commit,
            update_available=(
                _tag_update_available(
                    latest=cached_latest,
                    current=current,
                    latest_commit=cached_latest_commit,
                    current_commit=current_commit,
                    cwd=cwd,
                )
                if channel == "tag"
                else _branch_update_available(
                    latest_commit=cached_latest_commit,
                    current_commit=current_commit,
                    cwd=cwd,
                )
            ),
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
            cached_latest = cached.get("latest")
            cached_latest_commit = cached.get("latest_commit")
            if channel == "tag":
                cached_latest_commit = _formal_target_commit(
                    cwd,
                    install_kind,
                    cached.get("tag_name"),
                    cached_commit=cached_latest_commit,
                )
            cached_available = (
                _tag_update_available(
                    latest=cached_latest,
                    current=current,
                    latest_commit=cached_latest_commit,
                    current_commit=current_commit,
                    cwd=cwd,
                )
                if channel == "tag"
                else _branch_update_available(
                    latest_commit=cached_latest_commit,
                    current_commit=current_commit,
                    cwd=cwd,
                )
            )
            info = _update_info_from_cache(
                cached,
                channel=channel,
                current=current,
                is_dirty=is_dirty,
                version_source=version_source,
                git_checkout=is_git_checkout,
                install_kind=install_kind,
                current_commit=current_commit,
                latest_commit=cached_latest_commit,
                update_available=cached_available,
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
            install_kind=install_kind,
            current_commit=current_commit,
        )

    latest = remote["version_str"] or None
    if channel == "tag":
        latest_commit = _formal_target_commit(
            cwd,
            install_kind,
            remote.get("tag_name"),
            cached_commit=remote.get("branch_commit"),
            advertised_tag_commit=remote.get("commit"),
        )
    else:
        latest_commit = remote.get("commit")
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
            cwd=cwd,
        )
    elif channel == "dev":
        update_available = _branch_update_available(
            latest_commit=latest_commit,
            current_commit=current_commit,
            cwd=cwd,
        )
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
        install_kind=install_kind,
    )

    _write_channel_cache(channel, info.to_dict())
    return info


# --------------------------------------------------------------------- #
# apply() — five-stage pipeline
#
#   1. _pre_check        — git root check, detached HEAD probe, dirty fence
#   2. _snapshot_configs — tarball the user's untracked configs/* (presets
#                          remain owned by git so upstream changes apply)
#   3. _fetch            — git fetch --tags origin
#   4. _apply_ref        — checkout origin/dev, origin/main, or a verified tag
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
    target_tag: str | None = None,
) -> None:
    """Execute the upgrade in the current working tree.

    Steps:
      1. ``git fetch --tags origin``
      2. attach and fast-forward the local ``dev`` / ``main`` branch, or
         checkout a verified stable release tag when ``target_tag`` is
         provided. Only ``force=True`` resets a local branch.
      3. reinstall ``.[api,dev,cpu|gpu]`` into the running venv. Windows uses
         a regular local install to avoid PEP 660 editable launcher
         edge cases; POSIX keeps editable installs for developer trees.
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
    channel = _LEGACY_CHANNEL_ALIASES.get(channel, channel)
    if target_tag and (channel != "tag" or not is_release_tag(target_tag)):
        raise ValueError("target_tag must be a stable vX.Y.Z tag on the formal channel")
    cwd = _git_root()
    if _is_container_install():
        msg = (
            "Docker 容器内不支持维护页直接改写应用源码。容器文件系统是镜像产物,"
            "在容器内 git checkout / pip install / npm build 会造成运行态与镜像层漂移,"
            "容器重建后也会丢失。\n"
            "请在宿主机项目目录执行:\n"
            "  git pull\n"
            "  docker compose -f docker/docker-compose.yml --profile gpu up -d --build\n"
            "如使用 CPU profile,将 gpu 改为 cpu。命名卷中的 runs/、models/、datasets/、configs/ "
            "会保留。"
        )
        raise RuntimeError(msg)
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
            "--hard origin/dev`,然后重跑 `pip install .` 重装当前工作树。\n"
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
            reset_rc = _stream_subprocess(
                ["git", "reset", "--hard", "HEAD"], cwd=cwd, phase="git", emit=emit,
            )
            if reset_rc != 0:
                raise RuntimeError(f"git reset --hard failed (exit {reset_rc})")
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
                    clean_cmd.extend(["-e", ".env*" if spec == ".env" else spec])
            clean_rc = _stream_subprocess(
                clean_cmd, cwd=cwd, phase="git", emit=emit,
            )
            if clean_rc != 0:
                raise RuntimeError(f"git clean failed (exit {clean_rc})")
        else:
            ctx.stash_active = _stash_update_changes(cwd, emit)

        _fetch(cwd, channel=channel, emit=emit)
        _apply_ref(
            cwd,
            channel=channel,
            force=force,
            emit=emit,
            target_tag=target_tag,
        )

        # From this point a failed pip/npm step may have partially changed the
        # installed package or SPA. The rollback context will restore the old
        # source ref and best-effort reinstall that source before returning the
        # original update error.
        ctx.runtime_may_be_changed = True
        ctx.repair_build = build
        ctx.onnx_extra = _preferred_onnx_extra()
        _install_deps(
            cwd,
            build=build,
            emit=emit,
            onnx_extra=ctx.onnx_extra,
        )

        if ctx.stash_active:
            emit("git", "info", "git stash pop (restoring local edits)")
            rc = _stream_subprocess(
                ["git", "stash", "pop"], cwd=cwd, phase="git", emit=emit,
            )
            if rc != 0:
                raise RuntimeError(
                    "stash restore conflicted with the updated source; "
                    "rolling back so the service is not restarted from a conflicted tree"
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
    runtime_may_be_changed: bool = False
    repair_build: bool = False
    onnx_extra: str = "cpu"
    original_commit: str | None = field(default=None, init=False)
    original_branch: str | None = field(default=None, init=False)

    def __enter__(self) -> _UpdateContext:
        head = _git(["rev-parse", "HEAD"], cwd=self.cwd)
        if head.returncode == 0:
            self.original_commit = head.stdout.strip() or None
        branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=self.cwd)
        if branch.returncode == 0:
            self.original_branch = branch.stdout.strip() or None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        preserve_snapshot = False
        source_restored = True
        if exc_type is not None:
            # Best-effort rollback. We don't re-raise from here; the
            # original exception propagates out of the ``with`` block.
            try:
                source_restored = self._restore_source_ref()
            except Exception as rollback_exc:  # noqa: BLE001
                source_restored = False
                self.emit("git", "error", f"automatic source rollback failed: {rollback_exc}")
            if self.stash_active:
                try:
                    self.emit(
                        "git", "warn",
                        "upgrade failed; popping stash to restore local edits",
                    )
                    rc = _stream_subprocess(
                        ["git", "stash", "pop"], cwd=self.cwd, phase="git", emit=self.emit,
                    )
                    if rc != 0:
                        self.emit(
                            "git",
                            "error",
                            "could not restore all stashed edits; the stash entry was kept",
                        )
                except Exception as stash_exc:  # noqa: BLE001
                    self.emit("git", "error", f"could not restore stash: {stash_exc}")
                self.stash_active = False
            if self.snapshot_path is not None and not self.snapshot_consumed:
                try:
                    self.emit(
                        "git", "warn",
                        "upgrade failed; restoring configs/ from pre-flight snapshot",
                    )
                    _restore_configs(self.cwd, self.snapshot_path, self.emit)
                except Exception as restore_exc:  # noqa: BLE001
                    preserve_snapshot = True
                    self.emit(
                        "git",
                        "error",
                        f"automatic config restore failed: {restore_exc}; "
                        f"snapshot preserved at {self.snapshot_path}",
                    )
            if source_restored and self.runtime_may_be_changed:
                self._repair_runtime()
        # Always remove the temp archive — it's only useful as a
        # rollback bridge and would otherwise accumulate in TMPDIR. Keep it
        # when automatic restoration failed so recovery remains possible.
        if self.snapshot_path is not None and not preserve_snapshot:
            with contextlib.suppress(Exception):
                self.snapshot_path.unlink(missing_ok=True)

    def _restore_source_ref(self) -> bool:
        """Return the working tree to the branch/commit captured on entry."""
        if self.original_commit is None:
            return False
        head = _git(["rev-parse", "HEAD"], cwd=self.cwd)
        branch = _git(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=self.cwd,
        )
        current_commit = head.stdout.strip() if head.returncode == 0 else None
        current_branch = branch.stdout.strip() if branch.returncode == 0 else None
        if _commits_match(current_commit, self.original_commit) and (
            current_branch == self.original_branch
        ):
            return True

        target = self.original_branch or self.original_commit
        self.emit(
            "git",
            "warn",
            f"upgrade failed; restoring source to {target} at {self.original_commit[:8]}",
        )
        if self.original_branch:
            checkout = ["git", "checkout", "--force", self.original_branch]
        else:
            checkout = ["git", "checkout", "--force", "--detach", self.original_commit]
        if _stream_subprocess(
            checkout,
            cwd=self.cwd,
            phase="git",
            emit=self.emit,
        ) != 0:
            self.emit("git", "error", f"source rollback checkout failed: {target}")
            return False
        if _stream_subprocess(
            ["git", "reset", "--hard", self.original_commit],
            cwd=self.cwd,
            phase="git",
            emit=self.emit,
        ) != 0:
            self.emit(
                "git",
                "error",
                f"source rollback reset failed: {self.original_commit[:8]}",
            )
            return False
        return True

    def _repair_runtime(self) -> None:
        self.emit(
            "deps",
            "warn",
            "repairing dependencies and frontend from the restored source",
        )
        try:
            _install_deps(
                self.cwd,
                build=self.repair_build,
                emit=self.emit,
                onnx_extra=self.onnx_extra,
            )
        except Exception as repair_exc:  # noqa: BLE001
            self.emit(
                "deps",
                "error",
                f"automatic runtime repair failed: {repair_exc}; run `lorahub manage install`",
            )


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
        local_exists = _git(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{target_branch}"],
            cwd=cwd,
        )
        head = _git(["rev-parse", "HEAD"], cwd=cwd)
        if local_exists.returncode == 0 and head.returncode == 0:
            local_ref = _git(["rev-parse", f"refs/heads/{target_branch}"], cwd=cwd)
            if local_ref.returncode != 0:
                raise RuntimeError(
                    f"could not inspect local branch {target_branch}: "
                    f"{local_ref.stderr.strip() or local_ref.returncode}"
                )
            if local_ref.stdout.strip() != head.stdout.strip():
                emit(
                    "git",
                    "info",
                    f"detached at {head_sha} (reachable from {target_remote}); "
                    f"local `{target_branch}` points elsewhere and will be "
                    "checked out after local edits are preserved",
                )
                return
            switch_cmd = ["checkout", target_branch]
        elif local_exists.returncode == 1:
            switch_cmd = ["checkout", "-b", target_branch, "HEAD"]
        else:
            raise RuntimeError(
                f"could not inspect local branch {target_branch}: "
                f"{local_exists.stderr.strip() or local_exists.returncode}"
            )
        emit(
            "git", "info",
            f"detached at {head_sha} (reachable from {target_remote}); "
            f"auto-attaching to local branch `{target_branch}`",
        )
        switch = _git(switch_cmd, cwd=cwd)
        if switch.returncode != 0:
            msg = (
                f"detected reachable detached HEAD at {head_sha} but "
                f"`git {' '.join(switch_cmd)}` failed: "
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


def _is_link_path(path: Path) -> bool:
    """Return True for symbolic links and Windows directory junctions."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _remove_link_path(path: Path) -> None:
    """Remove a link itself without touching the directory it targets."""
    if path.is_symlink():
        path.unlink()
    else:
        os.rmdir(path)


def _path_uses_link(path: Path, root: Path) -> bool:
    current = path
    while True:
        if _is_link_path(current):
            return True
        if current == root:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _snapshot_configs(cwd: Path, emit: ProgressCallback) -> Path | None:
    """Capture user-authored and locally edited files under ``configs/``.

    The snapshot includes untracked config files plus tracked config files
    with staged or unstaged local edits. Untouched tracked presets are not
    captured, so upstream default fixes still reach users who did not edit
    those files. In force mode this snapshot is the only protection tracked
    config edits have before ``git reset --hard``.

    The archive lives in ``tempfile.gettempdir()`` so a multi-megabyte
    yaml collection doesn't have to be held in memory while the upgrade
    runs. Snapshot failures abort the update: proceeding with partial
    coverage could destroy the one file that failed to archive.

    Returns ``None`` when ``configs/`` is missing, empty, or contains no
    user-created or locally edited files to protect.
    """
    root = cwd / "configs"
    if not root.is_dir():
        return None
    if _is_link_path(root):
        raise RuntimeError("configs/ is a link and cannot be snapshotted safely")

    rels: set[str] = set()
    git_failed: list[str] = []
    commands = (
        ["ls-files", "-z", "--others", "--exclude-standard", "--", "configs"],
        ["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", "configs"],
        ["diff", "--name-only", "-z", "--", "configs"],
        ["diff", "--name-only", "--cached", "-z", "--", "configs"],
    )
    for command in commands:
        result = _git(command, cwd=cwd)
        if result.returncode == 0:
            # NUL framing preserves non-ASCII, whitespace and newline-bearing
            # filenames exactly; Git's default quoted output is not a path.
            rels.update(path for path in result.stdout.split("\0") if path)
        else:
            git_failed.append(result.stderr.strip() or "unknown")

    if not git_failed:
        files = [
            cwd / rel
            for rel in sorted(rels)
            if (cwd / rel).is_file() or (cwd / rel).is_symlink()
        ]
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
        files = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
    if not files:
        return None

    fd, raw = tempfile.mkstemp(prefix="lorahub-configs-", suffix=".tar")
    archive = Path(raw)
    # mkstemp leaves the FD open; close it so tarfile can re-open by path.
    import os as _os  # noqa: PLC0415
    _os.close(fd)

    written = 0
    failures: list[str] = []
    try:
        with tarfile.open(archive, "w") as tar:
            for full in files:
                try:
                    if _path_uses_link(full, root):
                        raise OSError("linked configs cannot be restored safely")
                    full.resolve(strict=True).relative_to(root.resolve())
                    arcname = full.relative_to(cwd).as_posix()
                    tar.add(full, arcname=arcname, recursive=False)
                    written += 1
                except (OSError, ValueError, tarfile.TarError) as exc:
                    failures.append(f"{full}: {exc}")
    except (OSError, tarfile.TarError) as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"could not snapshot configs/: {exc}") from exc

    if failures:
        archive.unlink(missing_ok=True)
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"could not safely snapshot configs/: {detail}")

    if written == 0:
        archive.unlink(missing_ok=True)
        return None
    emit("git", "info", f"snapshotted {written} file(s) under configs/ to {archive.name}")
    return archive


def _restore_configs(
    cwd: Path, snapshot: Path | None, emit: ProgressCallback,
) -> None:
    """Unpack the configs/ tar back over the working tree.

    Idempotent and safe when ``snapshot`` is None. Every destination is
    checked again immediately before writing so a replaced directory or
    symbolic link cannot redirect restoration outside ``configs/``.
    """
    if snapshot is None or not snapshot.is_file():
        return
    extracted = 0
    failures: list[str] = []
    expected_root = cwd.resolve() / "configs"
    config_dir = cwd / "configs"
    if _is_link_path(config_dir):
        _remove_link_path(config_dir)
    elif config_dir.exists() and not config_dir.is_dir():
        raise RuntimeError("cannot restore configs/: path is not a directory")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_root = config_dir.resolve()
    if config_root != expected_root:
        raise RuntimeError("cannot restore configs/: directory escapes project root")
    try:
        with tarfile.open(snapshot, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Defence-in-depth: refuse absolute paths or "..".
                # Our own snapshot writer never emits these, but a
                # tampered file shouldn't escape cwd either.
                arcname = member.name.replace("\\", "/")
                parts = tuple(arcname.split("/"))
                if (
                    arcname.startswith("/")
                    or len(parts) < 2
                    or parts[0] != "configs"
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    emit("git", "warn", f"skipping suspicious archive entry {arcname}")
                    continue
                relative = Path(*parts[1:])
                parent = config_root
                unsafe_parent = False
                for component in relative.parent.parts:
                    parent /= component
                    if _is_link_path(parent):
                        _remove_link_path(parent)
                    elif parent.exists() and not parent.is_dir():
                        failures.append(f"{arcname}: parent is not a directory")
                        emit(
                            "git",
                            "warn",
                            f"skipping config blocked by non-directory: {arcname}",
                        )
                        unsafe_parent = True
                        break
                    parent.mkdir(exist_ok=True)
                if unsafe_parent:
                    continue
                target = config_root / relative
                if _is_link_path(target):
                    _remove_link_path(target)
                elif target.exists() and not target.is_file():
                    failures.append(f"{arcname}: target is not a file")
                    emit("git", "warn", f"skipping non-file config target: {arcname}")
                    continue
                tmp: Path | None = None
                fd: int | None = None
                try:
                    source = tar.extractfile(member)
                    if source is None:
                        raise OSError("archive member has no readable content")
                    fd, tmp_raw = tempfile.mkstemp(
                        dir=target.parent,
                        prefix=f".{target.name}.",
                        suffix=".restore",
                    )
                    tmp = Path(tmp_raw)
                    with source, os.fdopen(fd, "wb") as destination:
                        fd = None
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                    tmp.replace(target)
                    extracted += 1
                except (OSError, tarfile.TarError) as exc:
                    failures.append(f"{arcname}: {exc}")
                    emit("git", "warn", f"could not restore {arcname}: {exc}")
                finally:
                    if fd is not None:
                        with contextlib.suppress(OSError):
                            os.close(fd)
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"restore from config snapshot failed: {exc}") from exc
    if failures:
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"could not restore all config files: {detail}")
    if extracted:
        emit("git", "info", f"restored {extracted} file(s) under configs/")


# --------------------------------------------------------------------- #
# Stage 3 — fetch
# --------------------------------------------------------------------- #


def _fetch(
    cwd: Path,
    *,
    channel: ChannelName,
    emit: ProgressCallback,
) -> None:
    target_branch = "main" if channel == "tag" else "dev"
    branch_refspec = (
        f"+refs/heads/{target_branch}:refs/remotes/origin/{target_branch}"
    )
    emit("git", "info", f"git fetch --tags --force origin {target_branch}")
    rc = _stream_subprocess(
        [
            "git",
            "fetch",
            "--tags",
            "--force",
            "--prune",
            "origin",
            branch_refspec,
        ],
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
    target_tag: str | None = None,
) -> None:
    if target_tag:
        remote_sha = _remote_tag_commit(cwd, target_tag)
        local = _git(
            ["rev-parse", "--verify", f"refs/tags/{target_tag}^{{commit}}"],
            cwd=cwd,
        )
        if not remote_sha or local.returncode != 0 or local.stdout.strip() != remote_sha:
            raise RuntimeError(f"release tag is not available from origin: {target_tag}")
        target_ref = f"refs/tags/{target_tag}"
        emit("git", "info", f"git checkout {target_ref}")
        checkout_cmd = ["git", "checkout"]
        if force:
            checkout_cmd.append("--force")
        checkout_cmd.append(target_ref)
        rc = _stream_subprocess(checkout_cmd, cwd=cwd, phase="git", emit=emit)
        if rc != 0:
            raise RuntimeError(f"git checkout {target_ref} failed (exit {rc})")
        return

    branch = "main" if channel == "tag" else "dev"
    target_ref = f"origin/{branch}"
    if force:
        emit("git", "info", f"git checkout --force -B {branch} {target_ref}")
        checkout_cmd = ["git", "checkout", "--force", "-B", branch, target_ref]
        rc = _stream_subprocess(checkout_cmd, cwd=cwd, phase="git", emit=emit)
        if rc != 0:
            raise RuntimeError(f"git checkout {target_ref} failed (exit {rc})")
    else:
        local_ref = f"refs/heads/{branch}"
        local = _git(["show-ref", "--verify", "--quiet", local_ref], cwd=cwd)
        if local.returncode == 0:
            emit("git", "info", f"git checkout {branch}")
            rc = _stream_subprocess(
                ["git", "checkout", branch], cwd=cwd, phase="git", emit=emit,
            )
            if rc != 0:
                raise RuntimeError(f"git checkout {branch} failed (exit {rc})")
            emit("git", "info", f"git merge --ff-only {target_ref}")
            rc = _stream_subprocess(
                ["git", "merge", "--ff-only", target_ref],
                cwd=cwd,
                phase="git",
                emit=emit,
            )
            if rc != 0:
                raise RuntimeError(
                    f"local {branch} has diverged from {target_ref}; "
                    "merge or rebase it manually, or retry with --force"
                )
        elif local.returncode == 1:
            emit("git", "info", f"git checkout -b {branch} {target_ref}")
            rc = _stream_subprocess(
                ["git", "checkout", "-b", branch, target_ref],
                cwd=cwd,
                phase="git",
                emit=emit,
            )
            if rc != 0:
                raise RuntimeError(f"git checkout {target_ref} failed (exit {rc})")
        else:
            raise RuntimeError(
                f"could not inspect local branch {branch}: "
                f"{local.stderr.strip() or local.returncode}"
            )

    upstream = _stream_subprocess(
        ["git", "branch", f"--set-upstream-to={target_ref}", branch],
        cwd=cwd,
        phase="git",
        emit=emit,
    )
    if upstream != 0:
        emit("git", "warn", f"could not set {branch} upstream to {target_ref}")


# --------------------------------------------------------------------- #
# Stage 5 — install + build
# --------------------------------------------------------------------- #


def _install_deps(
    cwd: Path,
    *,
    build: bool,
    emit: ProgressCallback,
    onnx_extra: str | None = None,
) -> None:
    emit("deps", "info", "reinstalling Python dependencies")
    selected_onnx = onnx_extra or _preferred_onnx_extra()
    _remove_legacy_onnx_conflict(cwd, emit)
    py_cmd = _build_pip_command(cwd, onnx_extra=selected_onnx)
    emit("deps", "info", "running " + _format_cmd(py_cmd))
    rc = _stream_subprocess(py_cmd, cwd=cwd, phase="deps", emit=emit)
    if rc != 0:
        msg = f"pip install failed (exit {rc})"
        raise RuntimeError(msg)
    if not build:
        return
    emit("build", "info", f"npm run build (web/, v{_current_version()})")
    npm = _find_npm(cwd)
    if npm is None:
        raise RuntimeError(
            "npm not found; cannot rebuild the frontend. Run the environment "
            "installer or retry with frontend build disabled."
        )
    web = cwd / "web"
    _ensure_frontend_deps(web, npm=Path(npm), emit=emit)
    rc = _stream_subprocess(
        [npm, "--silent", "run", "build"],
        cwd=web,
        phase="build",
        emit=emit,
        env=_npm_env(Path(npm), app_version=_current_version()),
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


def _stash_pathspecs() -> list[str]:
    """Pathspecs for source/config edits without runtime or user data."""
    specs = ["."]
    for prefix in _USER_OWNED_PREFIXES:
        if prefix == "configs/":
            continue
        spec = prefix.rstrip("/").replace("\\", "/")
        if spec == ".env":
            specs.append(":(exclude).env*")
        elif prefix.endswith("/"):
            specs.append(f":(exclude){spec}/**")
        else:
            specs.append(f":(exclude){spec}")
    return specs


def _stash_update_changes(cwd: Path, emit: ProgressCallback) -> bool:
    """Stash code and configs while leaving user-owned runtime data in place."""
    pathspecs = _stash_pathspecs()
    status = _git(
        ["status", "--porcelain", "--untracked-files=all", "--", *pathspecs],
        cwd=cwd,
    )
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr.strip() or status.returncode}")
    if not status.stdout.strip():
        return False

    emit("git", "info", "stashing source and config edits; user data stays in place")
    rc = _stream_subprocess(
        [
            "git",
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "lorahub-self-update",
            "--",
            *pathspecs,
        ],
        cwd=cwd,
        phase="git",
        emit=emit,
    )
    if rc != 0:
        raise RuntimeError(f"git stash failed (exit {rc})")
    return True


def _resolve_latest_tag(cwd: Path) -> str | None:
    """Return the highest reachable v* tag (e.g. ``v1.0.5``) or None."""
    out = _git(["tag", "-l", "v*", "--sort=-v:refname"], cwd=cwd).stdout
    pattern = re.compile(r"^v\d+\.\d+\.\d+$")
    for line in out.splitlines():
        line = line.strip()
        if pattern.match(line):
            return line
    return None


def _installed_onnx_distributions() -> set[str]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    installed: set[str] = set()
    for distribution in ("onnxruntime", "onnxruntime-gpu"):
        try:
            version(distribution)
        except PackageNotFoundError:
            continue
        installed.add(distribution)
    return installed


def _preferred_onnx_extra() -> str:
    installed = _installed_onnx_distributions()
    return "gpu" if "onnxruntime-gpu" in installed else "cpu"


def _remove_legacy_onnx_conflict(cwd: Path, emit: ProgressCallback) -> None:
    """Remove legacy dual ORT installs before reinstalling one selected extra."""
    if _installed_onnx_distributions() != {"onnxruntime", "onnxruntime-gpu"}:
        return
    emit(
        "deps",
        "warn",
        "removing conflicting onnxruntime and onnxruntime-gpu installations",
    )
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415

        uv = find_uv()
    except Exception:  # noqa: BLE001
        uv = None
    if uv:
        cmd = [
            uv,
            "pip",
            "uninstall",
            "onnxruntime",
            "onnxruntime-gpu",
            "--python",
            sys.executable,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "onnxruntime",
            "onnxruntime-gpu",
        ]
    rc = _stream_subprocess(cmd, cwd=cwd, phase="deps", emit=emit)
    if rc != 0:
        raise RuntimeError(f"could not remove conflicting ONNX Runtime packages (exit {rc})")


def _build_pip_command(cwd: Path, *, onnx_extra: str | None = None) -> list[str]:
    """Pick uv when available; otherwise fall back to plain pip."""
    import sys as _sys  # noqa: PLC0415

    py = _sys.executable
    pypi_index = _configured_pypi_index()
    selected_onnx = onnx_extra or _preferred_onnx_extra()
    if selected_onnx not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported ONNX Runtime extra: {selected_onnx}")
    package = f".[api,dev,{selected_onnx}]"
    project_spec = [package] if _sys.platform == "win32" else ["-e", package]
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415

        uv = find_uv()
        if uv:
            cmd = [uv, "pip", "install"]
            if os.environ.get("LORAHUB_INSTALL_VERBOSE") == "1":
                cmd.append("-v")
            if pypi_index:
                cmd += ["--index-url", pypi_index]
            return [*cmd, *project_spec, "--python", py, "--link-mode=copy"]
    except Exception:  # noqa: BLE001
        pass
    cmd = [py, "-m", "pip", "install"]
    if pypi_index:
        cmd += ["--index-url", pypi_index]
    return [*cmd, *project_spec]


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


def _npm_env(npm: Path, *, app_version: str | None = None) -> dict[str, str]:
    """Ensure npm lifecycle scripts can find the matching node binary."""
    env = os.environ.copy()
    env["PATH"] = f"{npm.parent}{os.pathsep}{env.get('PATH', '')}"
    if app_version:
        env["LORAHUB_APP_VERSION"] = app_version
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
        creationflags=_subprocess_no_window(),
    )
    stdout = proc.stdout
    assert stdout is not None  # noqa: S101
    lines: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            for raw in stdout:
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
    "is_release_tag",
    "last_check",
    "list_release_history",
]
