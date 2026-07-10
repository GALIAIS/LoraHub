"""Caption / tag management endpoints.

Goes beyond the existing ``annotations`` router (which is per-image
PUT/DELETE caption) — this is the dataset-level batch edit + introspection
layer the F3 design calls for:

  - GET  /captions/vocab        tag frequency table for the whole dataset
  - POST /captions/find-replace global find-replace, dry-run + apply
  - POST /captions/inject-trigger prepend / append a trigger word
                                  to every caption that doesn't have it
  - POST /captions/blacklist     remove a list of tags (case-insensitive)

Every write goes through ``_backup_file`` from the curate router so a
mistaken find-replace is rolled back via the same /curate/restore-backup
endpoint already exposed.

Tag normalisation here matches what audit.py uses (lowercase + comma split
+ trim) so vocab counts agree across the two views.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import (
    IMAGE_SUFFIXES,
    iter_safe_files,
    resolve_dataset_directory,
    resolve_file_under,
)
from lorahub.api.routers.image_studio.curate import _backup_file

from ._shared import _atomic_write_text, _file_mutation

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _walk_caption_files(root: Path, recursive: bool):
    """Yield (image_path, caption_path) for every captioned image."""
    for image in iter_safe_files(
        root,
        recursive=recursive,
        skip_dirs=frozenset({".workbench"}),
    ):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        caption = resolve_file_under(root, image.with_suffix(".txt"))
        if caption is not None:
            yield image, caption


def _split_tags(caption: str) -> list[str]:
    """Comma-separated, trimmed, lowercase. Empty entries dropped."""
    return [t.strip() for t in caption.split(",") if t.strip()]


def _join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def _ensure_dataset(dataset_path: str) -> Path:
    try:
        return resolve_dataset_directory(dataset_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #


@router.get("/captions/vocab")
def captions_vocab(
    dataset_path: str,
    recursive: bool = True,
    limit: int = 200,
    case_sensitive: bool = False,
) -> dict:
    """Return ``[{tag, count}, ...]`` sorted by frequency.

    Used by the Annotate stage's vocab pane to drive the bar list and
    one-click blacklist actions. ``limit=200`` covers the long tail
    that the audit report's top-50 truncates.
    """
    root = _ensure_dataset(dataset_path)
    counter: Counter[str] = Counter()
    files_seen = 0
    for _img, cap_path in _walk_caption_files(root, recursive):
        files_seen += 1
        try:
            txt = cap_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for tag in _split_tags(txt):
            if not case_sensitive:
                tag = tag.lower()
            counter[tag] += 1

    # Stable secondary sort (alphabetic) when counts tie.
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {
        "files_seen": files_seen,
        "tag_count": len(counter),
        "vocab": [{"tag": t, "count": c} for t, c in rows],
    }


# --------------------------------------------------------------------------- #
# Find-replace
# --------------------------------------------------------------------------- #


class FindReplaceRequest(BaseModel):
    dataset_path: str
    pattern: str = Field(..., min_length=1, description="Substring or regex to find")
    replacement: str = Field(default="", description="Replacement text (empty = delete)")
    is_regex: bool = False
    case_sensitive: bool = False
    # Match against the whole caption text, not per-tag. Default False
    # because most users want tag-aware replace ("1girl" → "1 girl"
    # without touching "1girlish_pose"). When True we do raw string /
    # regex replace on the whole caption.
    whole_caption: bool = False
    dry_run: bool = True
    recursive: bool = True
    # Optional path filter so the user can scope a replace to a
    # selection from the grid.
    paths: list[str] | None = None


@dataclass
class _Diff:
    path: str
    before: str
    after: str
    matches: int


@router.post("/captions/find-replace")
def captions_find_replace(req: FindReplaceRequest) -> dict:
    """Global find / replace across captions.

    Tag-aware (default): the pattern is matched per individual tag,
    so ``"1girl"`` swaps just the ``1girl`` tag and leaves
    ``1girls`` alone. ``whole_caption=True`` falls back to raw
    string / regex replace on the whole caption text.

    ``dry_run=True`` (default) returns the list of diffs without
    writing. The UI calls dry-run, shows the diff, then re-calls
    with ``dry_run=False`` to apply.
    """
    root = _ensure_dataset(req.dataset_path)
    flags = 0 if req.case_sensitive else re.IGNORECASE
    try:
        pattern_re = (
            re.compile(req.pattern, flags)
            if req.is_regex
            else re.compile(re.escape(req.pattern), flags)
        )
    except re.error as exc:
        raise HTTPException(400, f"invalid regular expression: {exc}") from exc
    only: set[str] | None = (
        {str(Path(p).resolve()) for p in req.paths} if req.paths else None
    )

    diffs: list[dict] = []
    written: list[str] = []
    matched_files = 0
    matched_count = 0
    for img, cap_path in _walk_caption_files(root, req.recursive):
        if only is not None and str(img.resolve()) not in only:
            continue
        with _file_mutation(img):
            try:
                text = cap_path.read_text(encoding="utf-8")
            except OSError:
                continue

            if req.whole_caption:
                new_text, n = pattern_re.subn(req.replacement, text)
            else:
                tags = _split_tags(text)
                n = 0
                new_tags: list[str] = []
                for tag in tags:
                    if pattern_re.fullmatch(tag) is not None:
                        n += 1
                        if req.replacement:
                            replaced = pattern_re.sub(req.replacement, tag)
                            if replaced.strip():
                                new_tags.append(replaced.strip())
                        # If replacement is empty, the tag is dropped.
                    elif pattern_re.search(tag) is not None and req.is_regex:
                        # Regex partial-match within a tag still counts.
                        replaced = pattern_re.sub(req.replacement, tag)
                        if replaced != tag:
                            n += 1
                            if replaced.strip():
                                new_tags.append(replaced.strip())
                    else:
                        new_tags.append(tag)
                new_text = _join_tags(new_tags)
                # Preserve trailing newline on disk if the original had one.
                if text.endswith("\n") and not new_text.endswith("\n"):
                    new_text += "\n"

            if n == 0 or new_text == text:
                continue

            matched_files += 1
            matched_count += n
            diffs.append(
                {
                    "path": str(img),
                    "caption_path": str(cap_path),
                    "before": text.strip(),
                    "after": new_text.strip(),
                    "matches": n,
                },
            )
            if not req.dry_run:
                _backup_file(req.dataset_path, img)
                _atomic_write_text(cap_path, new_text)
                written.append(str(cap_path))

    return {
        "dry_run": req.dry_run,
        "matched_files": matched_files,
        "matched_count": matched_count,
        "diffs": diffs[:500],  # UI doesn't need to render 10k diffs
        "diffs_truncated": len(diffs) > 500,
        "written": written,
    }


# --------------------------------------------------------------------------- #
# Trigger word injection
# --------------------------------------------------------------------------- #


class InjectTriggerRequest(BaseModel):
    dataset_path: str
    trigger_word: str = Field(..., min_length=1)
    position: Literal["prepend", "append"] = "prepend"
    skip_existing: bool = True
    recursive: bool = True
    paths: list[str] | None = None


@router.post("/captions/inject-trigger")
def captions_inject_trigger(req: InjectTriggerRequest) -> dict:
    """Add a trigger word to every caption (idempotent by default).

    With ``skip_existing=True``, captions already containing the
    trigger (case-insensitive substring match) are left alone — safe
    to run repeatedly. ``position`` controls whether the trigger lands
    at the start or end of the comma-separated tag list.
    """
    root = _ensure_dataset(req.dataset_path)
    trigger = req.trigger_word.strip()
    only: set[str] | None = (
        {str(Path(p).resolve()) for p in req.paths} if req.paths else None
    )

    injected: list[str] = []
    skipped = 0
    for img, cap_path in _walk_caption_files(root, req.recursive):
        if only is not None and str(img.resolve()) not in only:
            continue
        with _file_mutation(img):
            try:
                text = cap_path.read_text(encoding="utf-8")
            except OSError:
                continue
            stripped = text.strip()
            already = trigger.lower() in stripped.lower()
            if req.skip_existing and already:
                skipped += 1
                continue
            tags = _split_tags(stripped)
            if req.position == "prepend":
                new_tags = [trigger] + [
                    t for t in tags if t.lower() != trigger.lower()
                ]
            else:
                new_tags = [
                    t for t in tags if t.lower() != trigger.lower()
                ] + [trigger]
            new_text = _join_tags(new_tags)
            if text.endswith("\n"):
                new_text += "\n"
            if new_text == text:
                skipped += 1
                continue
            _backup_file(req.dataset_path, img)
            _atomic_write_text(cap_path, new_text)
            injected.append(str(cap_path))

    return {
        "trigger": trigger,
        "position": req.position,
        "injected_count": len(injected),
        "skipped_count": skipped,
    }


# --------------------------------------------------------------------------- #
# Blacklist tags
# --------------------------------------------------------------------------- #


class BlacklistRequest(BaseModel):
    dataset_path: str
    tags: list[str] = Field(..., min_length=1)
    case_sensitive: bool = False
    recursive: bool = True
    paths: list[str] | None = None


@router.post("/captions/blacklist")
def captions_blacklist(req: BlacklistRequest) -> dict:
    """Remove every occurrence of the listed tags from all captions.

    Comparison is per-tag (a blacklisted ``"smile"`` won't strip
    ``"smiley"``). Whitespace and case are normalised so a tag list
    pasted from anywhere matches.
    """
    root = _ensure_dataset(req.dataset_path)
    blacklist = {
        (t if req.case_sensitive else t.lower()).strip()
        for t in req.tags
        if t.strip()
    }
    only: set[str] | None = (
        {str(Path(p).resolve()) for p in req.paths} if req.paths else None
    )

    edited: list[str] = []
    removed_total = 0
    for img, cap_path in _walk_caption_files(root, req.recursive):
        if only is not None and str(img.resolve()) not in only:
            continue
        with _file_mutation(img):
            try:
                text = cap_path.read_text(encoding="utf-8")
            except OSError:
                continue
            tags = _split_tags(text)
            kept: list[str] = []
            removed = 0
            for tag in tags:
                key = tag if req.case_sensitive else tag.lower()
                if key in blacklist:
                    removed += 1
                else:
                    kept.append(tag)
            if removed == 0:
                continue
            new_text = _join_tags(kept)
            if text.endswith("\n"):
                new_text += "\n"
            _backup_file(req.dataset_path, img)
            _atomic_write_text(cap_path, new_text)
            edited.append(str(cap_path))
            removed_total += removed

    return {
        "edited_count": len(edited),
        "removed_count": removed_total,
        "blacklisted_tags": sorted(blacklist),
    }
