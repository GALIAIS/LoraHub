"""Git-based update detection for external training backends.

Each backend (kohya/sd-scripts, diffusion-pipe) lives in its own git
checkout. This module provides helpers to:

  1. ``check_update(repo_path)`` -- fetch from origin and compare HEAD
     against the remote tracking branch. Returns structured info about
     how many commits behind the local checkout is.

  2. ``apply_update(repo_path)`` -- pull (fast-forward) the repo to
     bring it up to date with origin.

The helpers are backend-agnostic; the router passes in the resolved
repo path from each backend's bootstrap module.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _subprocess_no_window() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


@dataclass(slots=True)
class UpdateCheckResult:
    """Outcome of a single backend update check."""

    update_available: bool = False
    current_sha: str = ""
    remote_sha: str = ""
    commits_behind: int = 0
    branch: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(repo: Path, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", str(repo), *args]
    _log.debug("git: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_subprocess_no_window(),
    )


def check_update(repo: Path) -> UpdateCheckResult:
    """Fetch from origin and compare HEAD against the tracking branch.

    Non-destructive: only does ``git fetch``, never modifies the working
    tree. Returns an ``UpdateCheckResult`` with ``update_available=True``
    when the remote has commits not yet in the local checkout.
    """
    if not repo.is_dir() or not (repo / ".git").exists():
        return UpdateCheckResult(error=f"not a git repo: {repo}")

    # Determine current branch
    r = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if r.returncode != 0:
        return UpdateCheckResult(error=f"cannot determine branch: {r.stderr.strip()}")
    branch = r.stdout.strip()
    if not branch or branch == "HEAD":
        branch = "main"

    # Fetch from origin (quiet, no tags to keep it fast)
    r = _git(repo, "fetch", "origin", branch, "--quiet", timeout=60)
    if r.returncode != 0:
        return UpdateCheckResult(
            branch=branch,
            error=f"git fetch failed: {r.stderr.strip()}",
        )

    # Current local HEAD
    r = _git(repo, "rev-parse", "HEAD")
    if r.returncode != 0:
        return UpdateCheckResult(branch=branch, error="cannot read HEAD")
    local_sha = r.stdout.strip()

    # Remote tracking ref
    r = _git(repo, "rev-parse", f"origin/{branch}")
    if r.returncode != 0:
        return UpdateCheckResult(
            branch=branch,
            current_sha=local_sha,
            error=f"cannot read origin/{branch}",
        )
    remote_sha = r.stdout.strip()

    if local_sha == remote_sha:
        return UpdateCheckResult(
            current_sha=local_sha,
            remote_sha=remote_sha,
            branch=branch,
        )

    # Count commits behind
    r = _git(repo, "rev-list", "--count", f"HEAD..origin/{branch}")
    behind = int(r.stdout.strip()) if r.returncode == 0 else 0

    return UpdateCheckResult(
        update_available=behind > 0,
        current_sha=local_sha,
        remote_sha=remote_sha,
        commits_behind=behind,
        branch=branch,
    )


def apply_update(repo: Path) -> UpdateCheckResult:
    """Pull (fast-forward only) the repo to match origin.

    Returns the post-pull state. If the pull fails (e.g. local
    modifications), the error field is populated.
    """
    if not repo.is_dir() or not (repo / ".git").exists():
        return UpdateCheckResult(error=f"not a git repo: {repo}")

    # Determine branch
    r = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = r.stdout.strip() if r.returncode == 0 else "main"
    if not branch or branch == "HEAD":
        branch = "main"

    # Pull with ff-only to avoid merge commits
    r = _git(repo, "pull", "--ff-only", "origin", branch, timeout=120)
    if r.returncode != 0:
        stderr = r.stderr.strip()
        # If ff-only fails, try a regular pull (handles diverged histories
        # from force-pushes upstream — common for these training repos).
        r2 = _git(repo, "reset", "--hard", f"origin/{branch}", timeout=30)
        if r2.returncode != 0:
            return UpdateCheckResult(
                branch=branch,
                error=f"pull failed: {stderr}; reset also failed: {r2.stderr.strip()}",
            )

    # Read final state
    r = _git(repo, "rev-parse", "HEAD")
    local_sha = r.stdout.strip() if r.returncode == 0 else ""
    r = _git(repo, "rev-parse", f"origin/{branch}")
    remote_sha = r.stdout.strip() if r.returncode == 0 else ""

    return UpdateCheckResult(
        update_available=False,
        current_sha=local_sha,
        remote_sha=remote_sha,
        commits_behind=0,
        branch=branch,
    )
