"""Shared interface for auto-taggers.

Every concrete tagger (WD14, JoyTag, ...) implements `BaseTagger` so the API
router and CLI can stay backend-agnostic. The protocol is intentionally a
superset of `WD14Tagger`'s pre-existing signatures: `tag_directory`'s kwargs
match the original WD14 implementation so we don't break old call sites.

`predict_tags` is a small string-only adapter — concrete taggers may keep
returning richer dataclasses from their native `tag_image` methods, but the
caption-writing pipeline only needs the flat tag-name list this exposes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

TaggerKind = Literal["wd14", "joytag"]

# (image_path, native_per_image_result) — the second arg is intentionally
# typed as Any because each tagger has its own result dataclass.
ProgressCallback = Callable[[Path, Any], None]
StopCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class TaggingProgress:
    """Snapshot suitable for surfacing to a UI mid-run.

    Concrete taggers don't have to emit this themselves — the API session
    layer is the primary consumer. It's exported here so other call sites
    (CLI progress bars, tests) can share the shape.
    """

    image: Path
    written: int
    total: int | None = None


@runtime_checkable
class BaseTagger(Protocol):
    """Common interface implemented by `WD14Tagger` and `JoyTagger`."""

    @property
    def active_provider(self) -> str:
        """Identifier for the runtime that's actually running the model.

        Examples: ``"CPUExecutionProvider"`` (ONNX), ``"cuda"`` / ``"cpu"``
        (PyTorch). Empty until `load()` has run.
        """
        ...

    def load(self, *, should_stop: StopCallback | None = None) -> None:
        """Eagerly download weights and warm up the runtime."""
        ...

    def predict_tags(self, image_path: Path) -> list[str]:
        """Return a flat list of tag names predicted for `image_path`.

        Order matches what would appear in the caption file (highest-priority
        first). Tag names use underscores; replace them at format time.
        """
        ...

    def tag_directory(
        self,
        directory: Path,
        *,
        recursive: bool = False,
        write_caption: bool = True,
        skip_existing: bool = True,
        underscores: bool = False,
        include_character: bool = True,
        on_progress: ProgressCallback | None = None,
        should_stop: StopCallback | None = None,
    ) -> Sequence[Any]:
        """Tag every image under `directory` and (optionally) write `.txt` captions.

        Returns the per-image native results. Implementations should honour
        `skip_existing` to avoid re-tagging files that already have a non-empty
        sidecar caption — the API route translates ``overwrite`` into
        ``skip_existing=not overwrite``.
        """
        ...


__all__ = [
    "BaseTagger",
    "ProgressCallback",
    "StopCallback",
    "TaggerKind",
    "TaggingProgress",
]
