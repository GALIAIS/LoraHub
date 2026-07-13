"""Shared dataset prep helpers used by every backend's ``launch()``.

The three concrete backends (kohya / diffusion-pipe / anima_lora) all
need to: (1) optionally rewrite every active training directory's caption
files when the recipe asks to drop certain tokens, and (2) thread the
resulting sanitised paths back into the cfg so the compiler sees the cleaned
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
    on the active training dataset paths and rebind them to their sanitised
    mirrors when one was actually written.

    Kohya, diffusion-pipe, and ai-toolkit use ``dataset.subsets`` in
    preference to ``dataset.source``. Each subset therefore gets its own
    generated mirror. Anima's subsets only carry conditioning metadata, so
    its single ``dataset.source`` remains the active directory. The mutation
    is in-place so callers do not have to re-thread a return value.
    """
    # Lazy import — caption_filter pulls PIL etc; keep _common cheap
    # at import time.
    from lorahub.core.config.caption_filter import sanitise_dataset  # noqa: PLC0415

    drop_tokens = list(cfg.dataset.caption.drop_tokens)
    if not any(token and token.strip() for token in drop_tokens):
        return

    if cfg.dataset.subsets and cfg.backend.type in {
        "kohya",
        "diffusion-pipe",
        "ai_toolkit",
    }:
        mirror_root = workspace / "captions_sanitized"
        for index, subset in enumerate(cfg.dataset.subsets):
            if subset.path is None:
                # The backend compiler emits the actionable validation error
                # for this incomplete form state after dataset preparation.
                continue
            subset.path = sanitise_dataset(
                source=subset.path,
                drop_tokens=drop_tokens,
                workspace=workspace,
                target_dir=mirror_root / f"subset-{index + 1}",
            )
        return

    source = cfg.dataset.source
    if source is None:
        return
    sanitised_source = sanitise_dataset(
        source=source,
        drop_tokens=drop_tokens,
        workspace=workspace,
    )
    if sanitised_source != source:
        cfg.dataset.source = sanitised_source


__all__ = ["apply_caption_dropouts"]
