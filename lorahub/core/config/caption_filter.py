"""Caption sanitisation helper — drop user-specified strings from training captions.

Lives at compile time so backend trainers don't need any code changes:
we mirror the user's dataset into ``<workspace>/captions_sanitized/``
where every image is symlinked back to the source, and every ``.txt``
caption sidecar is filtered (case-insensitive substring removal of
each entry in ``cfg.dataset.caption.drop_tokens``).

Returns the new dataset root the trainer should read from. When the
drop list is empty, returns the original ``cfg.dataset.source`` so
callers can wire the helper in unconditionally.

Implementation notes:
- Symlinks are used for images so the mirror is ~free on disk
  (Linux only; on Windows the helper falls back to a copy).
- Caption files are always rewritten — symlinking would propagate
  edits back to the user's source which is exactly what we want to
  avoid.
- Subdirectories are mirrored recursively, preserving the relative
  layout the trainer expects (kohya / dp / anima_lora all walk
  recursively into ``[[datasets.subsets]] image_dir``).
- The mirror is regenerated on every compile, so editing
  ``drop_tokens`` and re-launching always reflects the current list.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _iter_source_files(source: Path) -> Iterator[tuple[Path, Path]]:
    """Yield regular files below source without following linked entries."""
    for current, dirs, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if not _is_link_like(current_path / name)]
        for name in files:
            candidate = current_path / name
            if _is_link_like(candidate):
                continue
            try:
                resolved = candidate.resolve()
                resolved.relative_to(source)
            except (OSError, RuntimeError, ValueError):
                continue
            if resolved.is_file():
                yield resolved, candidate.relative_to(source)


def _remove_generated_tree(path: Path) -> None:
    if _is_link_like(path):
        raise RuntimeError(f"refusing to remove linked caption mirror: {path}")
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"caption mirror path is not a directory: {path}")
        shutil.rmtree(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved tree contains the other."""
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def sanitise_dataset(
    *,
    source: Path,
    drop_tokens: list[str],
    workspace: Path,
) -> Path:
    """Mirror ``source`` into ``workspace/captions_sanitized/`` with caption filters.

    No-op when ``drop_tokens`` is empty: returns ``source`` unchanged
    so callers can wire this in front of every backend without
    branching themselves.

    Returns the path the trainer's ``image_dir`` should point to.
    """
    cleaned = [t.strip() for t in drop_tokens or [] if t and t.strip()]
    if not cleaned:
        return source

    source = source.resolve()
    if not source.is_dir():
        logger.warning(
            "caption_filter: dataset.source %s is not a directory — skipping mirror",
            source,
        )
        return source

    workspace = workspace.expanduser().resolve()
    target = workspace / "captions_sanitized"
    if _paths_overlap(source, target):
        raise RuntimeError(
            "caption source and generated mirror must be separate directories: "
            f"source={source}, target={target}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    if _is_link_like(target):
        raise RuntimeError(f"caption mirror cannot be a link: {target}")
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"caption mirror path is not a directory: {target}")

    # Build beside the destination, then swap it into place. A failed read or
    # disk-full error leaves the previous complete mirror available instead of
    # exposing a half-written dataset to a resumed training job.
    stage = Path(tempfile.mkdtemp(prefix=".captions_sanitized-", dir=workspace))
    backup = workspace / f".captions_sanitized-backup-{uuid.uuid4().hex}"

    image_count = 0
    caption_count = 0
    edited_count = 0

    try:
        for src_path, rel in _iter_source_files(source):
            dst_path = stage / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = src_path.suffix.lower()
            if suffix == ".txt":
                try:
                    text = src_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # Caption written in some odd encoding — fall back to
                    # a byte-level read and best-effort decode so the
                    # mirror still works.
                    text = src_path.read_bytes().decode("utf-8", errors="replace")
                filtered = _strip_tokens(text, cleaned)
                dst_path.write_text(filtered, encoding="utf-8")
                caption_count += 1
                if filtered != text:
                    edited_count += 1
            elif suffix in _IMAGE_EXTS:
                _link_or_copy(src_path, dst_path)
                image_count += 1
            else:
                # Mask sidecars, .npz caches, anything else — mirror via
                # symlink/copy so the trainer's bucketing helpers see the
                # exact same layout the source has.
                _link_or_copy(src_path, dst_path)

        if target.exists():
            target.replace(backup)
        stage.replace(target)
        try:
            _remove_generated_tree(backup)
        except OSError as exc:
            logger.warning("caption_filter: could not remove stale backup %s: %s", backup, exc)
    except Exception:
        try:
            _remove_generated_tree(stage)
        except OSError:
            pass
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise

    logger.info(
        "caption_filter: mirrored dataset %s -> %s "
        "(%d images, %d captions, %d edited; drop_tokens=%d)",
        source,
        target,
        image_count,
        caption_count,
        edited_count,
        len(cleaned),
    )
    return target


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``; fall back to copy on platforms that
    can't create symlinks without elevation (Windows non-admin).
    Replaces any existing entry so the mirror stays in sync with edits.
    """
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            os.symlink(src, dst)
            return
        except (OSError, NotImplementedError):
            # Lacking SeCreateSymbolicLinkPrivilege — copy instead.
            # The cost is real (full image copy) but at least the
            # training job works; advise the user via the warning log.
            logger.warning(
                "caption_filter: symlink failed on Windows for %s; "
                "copying instead. Enable Developer Mode or run as "
                "admin to keep the mirror cheap.",
                src,
            )
            shutil.copy2(src, dst)
            return
    try:
        os.symlink(src, dst)
    except OSError as exc:
        logger.warning(
            "caption_filter: symlink %s -> %s failed (%s); falling back to copy",
            src, dst, exc,
        )
        shutil.copy2(src, dst)


def _strip_tokens(text: str, drop_tokens: list[str]) -> str:
    """Remove every drop_tokens entry from ``text`` (case-insensitive
    substring match), then tidy up the resulting comma list so the
    output looks like a hand-written caption rather than ``,, ,``.

    The cleanup is conservative — only collapses whitespace and
    repeated separators that fall out of the removal step. Multi-line
    captions are joined back the way they came.
    """
    out = text
    # Sort longest-first so a phrase ("looking at viewer") is removed
    # before a token that's a substring of it ("looking").
    for token in sorted(drop_tokens, key=len, reverse=True):
        # Case-insensitive verbatim match. We intentionally don't add
        # word-boundaries — natural-language phrases and tags with
        # punctuation both need to match across spaces / punctuation.
        out = re.sub(re.escape(token), "", out, flags=re.IGNORECASE)
    # Cleanup: collapse runs of comma + whitespace into a single
    # ", ", then strip leading/trailing punctuation per line so a
    # caption that started with a stripped token doesn't keep its
    # leading comma.
    lines: list[str] = []
    for line in out.splitlines():
        cleaned = re.sub(r"\s*,(?:\s*,)+\s*", ", ", line)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ,\t")
        lines.append(cleaned)
    return "\n".join(lines)


__all__ = ["sanitise_dataset"]
