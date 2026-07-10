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

from lorahub.core.redaction import redact_command_text

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


def _git_error(value: str) -> str:
    return redact_command_text(value.strip())


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
        return UpdateCheckResult(error=f"cannot determine branch: {_git_error(r.stderr)}")
    branch = r.stdout.strip()
    if not branch or branch == "HEAD":
        return UpdateCheckResult(
            branch="HEAD",
            error=(
                "backend update check refused: repository is on a detached HEAD; "
                "check out the intended branch first"
            ),
        )

    # Fetch from origin (quiet, no tags to keep it fast)
    r = _git(repo, "fetch", "origin", branch, "--quiet", timeout=60)
    if r.returncode != 0:
        return UpdateCheckResult(
            branch=branch,
            error=f"git fetch failed: {_git_error(r.stderr)}",
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
    count_error = r.returncode != 0
    try:
        behind = int(r.stdout.strip()) if r.returncode == 0 else 0
    except ValueError:
        behind = 0
        count_error = True

    return UpdateCheckResult(
        update_available=behind > 0,
        current_sha=local_sha,
        remote_sha=remote_sha,
        commits_behind=behind,
        branch=branch,
        error=(
            "cannot determine how many remote commits are pending"
            if count_error
            else None
        ),
    )


def apply_update(repo: Path) -> UpdateCheckResult:
    """Fast-forward a clean backend checkout without discarding local work.

    Local modifications, detached HEADs, and diverged histories are reported
    to the caller. They are never resolved with an implicit hard reset.
    """
    if not repo.is_dir() or not (repo / ".git").exists():
        return UpdateCheckResult(error=f"not a git repo: {repo}")

    # Determine branch
    r = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = r.stdout.strip() if r.returncode == 0 else "main"
    if not branch or branch == "HEAD":
        return UpdateCheckResult(
            branch="HEAD",
            error=(
                "backend update refused: repository is on a detached HEAD; "
                "check out the intended branch first"
            ),
        )

    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    if status.returncode != 0:
        return UpdateCheckResult(
            branch=branch,
            error=f"cannot inspect backend working tree: {_git_error(status.stderr)}",
        )
    if status.stdout.strip():
        return UpdateCheckResult(
            branch=branch,
            error=(
                "backend update refused: working tree contains local changes; "
                "commit or stash them before updating"
            ),
        )

    # Pull with ff-only to avoid merge commits
    r = _git(repo, "pull", "--ff-only", "origin", branch, timeout=120)
    if r.returncode != 0:
        detail = _git_error(r.stderr or r.stdout) or "git pull --ff-only failed"
        return UpdateCheckResult(
            branch=branch,
            error=(
                f"backend update could not fast-forward: {detail}. "
                "Resolve or back up the local branch manually; no files were reset"
            ),
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
