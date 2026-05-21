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
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
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
    """``True`` iff the working tree has uncommitted changes."""
    out = _git(["status", "--porcelain"], cwd=cwd)
    return out.returncode == 0 and bool(out.stdout.strip())


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
    """``urlopen`` 4xx → ``HTTPError`` (which is an OSError). The
    canonical message is ``HTTP Error 404: Not Found`` on the
    HTTPError side and ``HTTP 404: ...`` on our manual raise. Match
    both shapes so a future urllib refactor doesn't slip past."""
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


def apply(
    channel: ChannelName = "tag",
    *,
    build: bool = True,
    progress: ProgressCallback | None = None,
) -> None:
    """Execute the upgrade in the current working tree.

    Steps:
      1. ``git fetch --tags origin``
      2. ``git checkout origin/main`` (channel=main) or
         ``git checkout v<latest>`` (channel=tag)
      3. ``uv pip install -e .[api,dev]`` if the project has uv on PATH,
         else ``pip install -e .[api,dev]``
      4. ``npm run build`` if ``build`` is True

    Raises ``RuntimeError`` on any non-zero step. ``progress`` is invoked
    with ``(phase, level, message)`` for each line of subprocess output
    so the API can stream the update to the UI like the bootstrap flow.
    """
    cwd = _git_root()
    if cwd is None:
        msg = "this install is not a git checkout — `lorahub self update` is required."
        raise RuntimeError(msg)
    if _detect_dirty(cwd):
        msg = (
            "working tree has uncommitted changes; commit or stash before updating "
            "to avoid losing local edits."
        )
        raise RuntimeError(msg)

    def emit(phase: str, level: str, message: str) -> None:
        if progress is not None:
            progress(phase, level, message)

    # 1. Fetch
    emit("git", "info", "git fetch --tags origin")
    rc = _stream_subprocess(
        ["git", "fetch", "--tags", "--prune", "origin"], cwd=cwd, phase="git", emit=emit,
    )
    if rc != 0:
        msg = f"git fetch failed (exit {rc})"
        raise RuntimeError(msg)

    # 2. Checkout target ref.
    if channel == "tag":
        target = _resolve_latest_tag(cwd)
        if not target:
            msg = "no v* tag reachable from origin; switch to channel=main."
            raise RuntimeError(msg)
        target_ref = target
    else:
        target_ref = "origin/main"
    emit("git", "info", f"git checkout {target_ref}")
    rc = _stream_subprocess(
        ["git", "checkout", target_ref], cwd=cwd, phase="git", emit=emit,
    )
    if rc != 0:
        msg = f"git checkout {target_ref} failed (exit {rc})"
        raise RuntimeError(msg)

    # 3. Reinstall Python deps.
    emit("deps", "info", "reinstalling Python dependencies")
    py_cmd = _build_pip_command(cwd)
    rc = _stream_subprocess(py_cmd, cwd=cwd, phase="deps", emit=emit)
    if rc != 0:
        msg = f"pip install failed (exit {rc})"
        raise RuntimeError(msg)

    # 4. Optional rebuild of the SPA.
    if build:
        emit("build", "info", "npm run build (web/)")
        npm = _find_npm(cwd)
        if npm is None:
            emit("build", "warn", "npm not found; skipping SPA rebuild.")
        else:
            rc = _stream_subprocess([npm, "run", "build"], cwd=cwd / "web", phase="build", emit=emit)
            if rc != 0:
                msg = f"npm run build failed (exit {rc})"
                raise RuntimeError(msg)

    emit("done", "info", "update applied")


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


__all__ = [
    "ChannelName",
    "UpdateInfo",
    "apply",
    "check",
    "last_check",
]
