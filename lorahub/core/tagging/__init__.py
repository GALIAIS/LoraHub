"""Auto-taggers: turn images into descriptive tag strings for kohya-style captions."""

from lorahub.core.tagging.base import (
    BaseTagger,
    ProgressCallback,
    TaggerKind,
    TaggingProgress,
)
from lorahub.core.tagging import download_status

__all__ = [
    "BaseTagger",
    "ProgressCallback",
    "TaggerKind",
    "TaggingProgress",
    "download_status",
]
