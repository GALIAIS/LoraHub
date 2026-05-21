"""Shared dataset prep helpers used by every backend's ``launch()``.

The three concrete backends (kohya / diffusion-pipe / anima_lora) all
need to: (1) optionally rewrite the dataset's caption files when the
recipe asks to drop certain tokens, and (2) thread the resulting
sanitised path back into the cfg so the compiler sees the cleaned
data instead of the user's master copy.

The block was duplicated in three places verbatim. Centralised here
so a future addition (e.g. caption-fragment normalisation) only
edits one site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_caption_dropouts(cfg: Any, workspace: Path) -> None:
    """Run :func:`lorahub.core.config.caption_filter.sanitise_dataset`
    on ``cfg.dataset.source`` and rebind ``cfg.dataset.source`` to the
    sanitised path when one was actually written.

    No-op when the recipe has no ``drop_tokens`` configured (the
    sanitiser short-circuits in that case). The mutation is in-place
    so callers don't have to re-thread a return value.
    """
    # Lazy import — caption_filter pulls PIL etc; keep _common cheap
    # at import time.
    from lorahub.core.config.caption_filter import sanitise_dataset  # noqa: PLC0415

    sanitised_source = sanitise_dataset(
        source=cfg.dataset.source,
        drop_tokens=list(cfg.dataset.caption.drop_tokens),
        workspace=workspace,
    )
    if sanitised_source != cfg.dataset.source:
        cfg.dataset.source = sanitised_source


__all__ = ["apply_caption_dropouts"]
