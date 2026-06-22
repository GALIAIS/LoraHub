"""Typed payloads shared by update check and apply flows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChannelName = Literal["dev", "tag"]
ProgressCallback = Callable[[str, str, str], None]


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
    # Optional metadata: tag-only (None for "dev" channel).
    tag_name: str | None = None
    published_at: str | None = None
    # Where the ``current`` string was sourced from. ``hatch-vcs``
    # is the canonical path (real git checkout, real install). The
    # other values mark a degraded discovery; values match
    # ``system_update._VERSION_SOURCES``.
    version_source: str = "hatch-vcs"
    # ``True`` iff this install is a real ``git`` checkout.
    git_checkout: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CacheBlob:
    data: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_at: float = 0.0
