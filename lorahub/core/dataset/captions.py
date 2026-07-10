"""Generic caption preprocessing toolkit for SDXL anime-derivative models.

This module provides a small, pure-functional toolbox for cleaning up
booru-style captions shared by Illustrious, Pony, Animagine, NoobAI, and
similar SDXL anime forks. The functions are stateless so they're easy to
chain in any order; `CaptionPipeline` bundles a common, sane order plus
per-directory batch I/O for callers who just want "process this folder".

Anchored prefixes (quality / score / safety) are protected against
``drop_tags`` so dropout regularisation can't accidentally strip the
prompt's stylistic anchor and turn the resulting caption into garbage.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Tag vocabularies
# --------------------------------------------------------------------------- #
#
# These constants are intentionally small and curated. Every entry below maps
# to a *literal* booru-style tag string after `normalise_underscores` runs:
# spaces, lowercase, no underscores (except the score_N family which is
# preserved verbatim — that's a Pony naming choice the upstream model relies
# on).
QUALITY_TAGS: frozenset[str] = frozenset(
    {
        "masterpiece",
        "best quality",
        "good quality",
        "normal quality",
        "low quality",
        "worst quality",
    }
)

# `score_1` ... `score_9` plus the "_up" rollup variants Pony's prompt prefix
# uses (`score_9, score_8_up, score_7_up`). We also keep `score_9_up` even
# though the canonical Pony chain doesn't use it — some forks do, and treating
# it as anchorable costs us nothing.
SCORE_TAGS: frozenset[str] = frozenset(
    {f"score_{i}" for i in range(1, 10)}
    | {"score_9_up", "score_8_up", "score_7_up", "score_6_up"}
)

SAFETY_TAGS: frozenset[str] = frozenset(
    {"safe", "sensitive", "questionable", "nsfw", "explicit"}
)

# Booru meta tags that describe the *file* rather than the depicted scene.
# Useful as a hint set for callers that want to demote them via blacklist.
META_TAGS: frozenset[str] = frozenset(
    {
        "highres",
        "absurdres",
        "anime screenshot",
        "jpeg artifacts",
        "official art",
        "sketch",
        "monochrome",
    }
)

# NoobAI-style era buckets. Year tags like ``year 2024`` come in via regex
# (`_YEAR_RE`) since enumerating every year would be silly.
TIME_TAGS: frozenset[str] = frozenset({"newest", "recent", "mid", "early", "old"})

_SCORE_RE: re.Pattern[str] = re.compile(r"^score_\d+(_up)?$")
_YEAR_RE: re.Pattern[str] = re.compile(r"^year\s+\d{4}$")


def is_score_tag(tag: str) -> bool:
    """Return True when ``tag`` is part of the Pony score_N family.

    Matches both the bare ``score_7`` form and the ``score_7_up`` rollup
    Pony's canonical prefix uses. Comparison is case-insensitive on the
    leading word; the digits and the optional ``_up`` suffix must be exact.
    """
    return bool(_SCORE_RE.match(tag.strip().lower()))


def is_year_tag(tag: str) -> bool:
    """Return True for NoobAI-style ``year 2024`` markers."""
    return bool(_YEAR_RE.match(tag.strip().lower()))


# --------------------------------------------------------------------------- #
# Atomic transformations
# --------------------------------------------------------------------------- #


def normalise_underscores(tag: str) -> str:
    """Replace underscores with spaces, but keep score_N tags intact.

    Booru exports tag names with underscores (``oomuro_sakurako``) while most
    SDXL anime forks expect spaces in prompts. The Pony score family is the
    one exception — the model was trained with literal ``score_7_up``, so
    swapping it to ``score 7 up`` would break the prompt.
    """
    stripped = tag.strip()
    if is_score_tag(stripped):
        return stripped
    return stripped.replace("_", " ")


def split_tags(text: str) -> list[str]:
    """Split a comma-separated caption string into trimmed, non-empty tags."""
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def join_tags(tags: list[str], *, separator: str = ", ") -> str:
    """Inverse of :func:`split_tags`; joins tags with ``separator``."""
    return separator.join(t for t in tags if t)


def normalise_tags(text: str, *, separator: str = ", ") -> str:
    """Lowercase, swap underscores, dedupe (preserving order), and rejoin.

    The de-duplication compares the post-normalisation form so
    ``"Blue_Hair, blue hair"`` collapses to a single ``blue hair``. This is
    the single function most callers will reach for; the others compose
    around it.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in split_tags(text):
        normalised = normalise_underscores(raw).lower()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        out.append(normalised)
    return join_tags(out, separator=separator)


# --------------------------------------------------------------------------- #
# Random-modifying transformations
# --------------------------------------------------------------------------- #


def _make_rng(seed: int | None) -> random.Random:
    """Return a fresh PRNG; ``None`` -> non-deterministic, otherwise seeded."""
    return random.Random(seed) if seed is not None else random.Random()


def _is_anchored(tag: str) -> bool:
    """True for tags that should never be shuffled / dropped.

    Anchored tags are the prompt's stylistic spine — quality keywords, the
    score_N family, and safety markers. Killing them via random dropout
    would produce captions the model wasn't trained on.
    """
    lowered = tag.strip().lower()
    return (
        lowered in QUALITY_TAGS
        or lowered in SAFETY_TAGS
        or is_score_tag(lowered)
    )


def shuffle_tags(
    tags: list[str], *, keep_n: int = 0, seed: int | None = None
) -> list[str]:
    """Shuffle ``tags`` in place-style but anchor the first ``keep_n``.

    This mirrors kohya's ``keep_tokens`` knob: the first ``keep_n`` entries
    stay glued to the prompt's front (typically the trigger word and any
    anchoring style tags), everything after shuffles. ``seed`` is plumbed
    through so tests can pin the order.
    """
    keep_n = max(0, min(keep_n, len(tags)))
    head = list(tags[:keep_n])
    tail = list(tags[keep_n:])
    rng = _make_rng(seed)
    rng.shuffle(tail)
    return head + tail


def drop_tags(
    tags: list[str], *, drop_rate: float, seed: int | None = None
) -> list[str]:
    """Independently drop each non-anchored tag with probability ``drop_rate``.

    ``drop_rate=0`` is the identity. Quality / score / safety tags are
    *always* preserved regardless of the rate, so dropout regularisation
    won't accidentally turn ``masterpiece, 1girl, blue hair`` into
    ``blue hair`` and ruin a Pony/Animagine-style training run.
    """
    if drop_rate <= 0.0:
        return list(tags)
    rate = min(drop_rate, 1.0)
    rng = _make_rng(seed)
    out: list[str] = []
    for tag in tags:
        if _is_anchored(tag) or rng.random() >= rate:
            out.append(tag)
    return out


# --------------------------------------------------------------------------- #
# Vocabulary edits
# --------------------------------------------------------------------------- #


def add_artist_prefix(
    tags: list[str], *, known_artists: set[str]
) -> list[str]:
    """Prefix tags in ``known_artists`` with ``@`` (Animagine convention).

    Idempotent: tags that already start with ``@`` are passed through as-is,
    and tags that aren't in the artist set are untouched. Comparison against
    ``known_artists`` is case-insensitive but the original tag spelling is
    preserved in the output.
    """
    artists_lower = {a.strip().lower() for a in known_artists if a.strip()}
    out: list[str] = []
    for tag in tags:
        stripped = tag.strip()
        if not stripped:
            continue
        if stripped.startswith("@"):
            out.append(stripped)
            continue
        if stripped.lower() in artists_lower:
            out.append(f"@{stripped}")
        else:
            out.append(stripped)
    return out


def inject_quality(
    tags: list[str],
    quality: list[str] | None = None,
    *,
    score: list[str] | None = None,
    safety: str | None = None,
) -> list[str]:
    """Prepend quality / score / safety markers if they're not already present.

    Order on the prefix is ``score -> quality -> safety`` so a Pony config's
    ``score_9, score_8_up, masterpiece, safe`` pattern materialises naturally.
    Each input list is treated as a sequence of literals; nothing is
    re-normalised here, the caller has already chosen the spelling.
    """
    existing_lower = {t.strip().lower() for t in tags}
    prefix: list[str] = []

    if score:
        for tag in score:
            t = tag.strip()
            if t and t.lower() not in existing_lower:
                prefix.append(t)
                existing_lower.add(t.lower())

    if quality:
        for tag in quality:
            t = tag.strip()
            if t and t.lower() not in existing_lower:
                prefix.append(t)
                existing_lower.add(t.lower())

    if safety:
        s = safety.strip()
        if s and s.lower() not in existing_lower:
            prefix.append(s)
            existing_lower.add(s.lower())

    return prefix + list(tags)


def filter_blacklist(tags: list[str], blacklist: set[str]) -> list[str]:
    """Drop any tag in ``blacklist`` (case-insensitive)."""
    if not blacklist:
        return list(tags)
    bl = {b.strip().lower() for b in blacklist if b.strip()}
    return [t for t in tags if t.strip().lower() not in bl]


def remap_tags(tags: list[str], rules: dict[str, str]) -> list[str]:
    """Replace tags per ``rules``; an empty target string deletes the tag.

    ``rules`` keys match case-insensitively; values are inserted verbatim.
    A value containing commas expands into multiple tags so ``"1girl"`` ->
    ``"solo, 1girl"`` is a one-rule transformation. Order is preserved.
    """
    if not rules:
        return list(tags)
    rules_lower = {k.strip().lower(): v for k, v in rules.items()}
    out: list[str] = []
    for tag in tags:
        key = tag.strip().lower()
        if key in rules_lower:
            replacement = rules_lower[key]
            if not replacement.strip():
                continue  # delete
            for part in split_tags(replacement):
                out.append(part)
        else:
            out.append(tag)
    return out


# --------------------------------------------------------------------------- #
# Pipeline + directory batch
# --------------------------------------------------------------------------- #


_CAPTION_GLOB = "*.txt"


@dataclass(frozen=True)
class CaptionPipeline:
    """Composable, declarative caption preprocessor.

    Apply order::

        filter_blacklist -> remap -> normalise_tags -> add_artist_prefix
        -> inject_quality -> drop_tags -> shuffle_tags

    Every step is a no-op when the relevant config evidence is absent — an
    empty blacklist skips filter, ``shuffle=False`` skips shuffle, etc. This
    means a default-constructed pipeline is the identity on already-clean
    captions, which is the behaviour callers expect from "process this
    directory" workflows.
    """

    blacklist: set[str] = field(default_factory=set)
    remap: dict[str, str] = field(default_factory=dict)
    known_artists: set[str] = field(default_factory=set)
    quality: list[str] | None = None
    score: list[str] | None = None
    safety: str | None = None
    shuffle: bool = False
    keep_n: int = 0
    drop_rate: float = 0.0
    seed: int | None = None
    # Apply the curated Danbooru -> Gelbooru alias table after the user
    # ``remap`` step. Off by default so existing configs round-trip
    # bit-identically; opt-in via the CLI's ``--booru-alias`` flag or the
    # API's ``apply_booru_alias`` field.
    apply_booru_alias: bool = False
    # Optional override / extension for the alias table. Keys here win over
    # the default :data:`DANBOORU_TO_GELBOORU`; ignored unless
    # ``apply_booru_alias`` is true.
    booru_alias_extra: dict[str, str] = field(default_factory=dict)

    def transform_text(self, text: str) -> str:
        """Run the pipeline against a raw caption string and return the result."""
        tags = split_tags(text)

        if self.blacklist:
            tags = filter_blacklist(tags, self.blacklist)
        if self.remap:
            tags = remap_tags(tags, self.remap)
        if self.apply_booru_alias:
            # Lazy import keeps the dataset module's surface lean.
            from lorahub.core.dataset.booru_alias import (  # noqa: PLC0415
                load_aliases,
            )

            # User ``remap`` runs first, so any conflicting key the user
            # already supplied has already replaced the tag and the alias
            # step won't see it. That is intentional: user rules trump the
            # curated alias table.
            tags = remap_tags(tags, load_aliases(self.booru_alias_extra or None))

        # `normalise_tags` works on the joined string form because that's
        # the canonical entry point for dedup / lowercasing. Round-trip
        # through split/join keeps the data type consistent for the next
        # steps.
        tags = split_tags(normalise_tags(join_tags(tags)))

        if self.known_artists:
            tags = add_artist_prefix(tags, known_artists=self.known_artists)

        if self.quality or self.score or self.safety:
            tags = inject_quality(
                tags,
                self.quality,
                score=self.score,
                safety=self.safety,
            )

        if self.drop_rate > 0.0:
            tags = drop_tags(tags, drop_rate=self.drop_rate, seed=self.seed)

        if self.shuffle:
            tags = shuffle_tags(tags, keep_n=self.keep_n, seed=self.seed)

        return join_tags(tags)

    def transform_directory(
        self,
        path: Path,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        progress: Callable[[Path, int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Apply ``transform_text`` to every ``.txt`` caption under ``path``.

        Mirrors ``WD14Tagger.tag_directory``'s contract: returns the number
        of files written, fires ``progress(file, done, total)`` after each
        write so an HTTP session can render percent + last-touched filename.
        ``overwrite=False`` is a no-op here (we're not creating files, just
        rewriting existing ones); the flag exists for symmetry with the
        tagger and is reserved for future "skip-if-already-cleaned" logic.

        ``overwrite=True`` rewrites caption files even when the new contents
        match the old ones; ``False`` skips no-op writes to keep mtimes stable.
        """
        del overwrite  # currently unused; see docstring.
        files = list(_iter_caption_files(path, recursive=recursive))
        total = len(files)
        written = 0
        for idx, caption_file in enumerate(files, start=1):
            if should_stop is not None and should_stop():
                raise InterruptedError("normalization canceled by user")
            old = caption_file.read_text(encoding="utf-8")
            new = self.transform_text(old)
            if new != old:
                caption_file.write_text(new, encoding="utf-8")
                written += 1
            if progress is not None:
                progress(caption_file, idx, total)
        return written


def _iter_caption_files(directory: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/" + _CAPTION_GLOB if recursive else _CAPTION_GLOB
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            yield p


__all__ = [
    "CaptionPipeline",
    "META_TAGS",
    "QUALITY_TAGS",
    "SAFETY_TAGS",
    "SCORE_TAGS",
    "TIME_TAGS",
    "add_artist_prefix",
    "drop_tags",
    "filter_blacklist",
    "inject_quality",
    "is_score_tag",
    "is_year_tag",
    "join_tags",
    "normalise_tags",
    "normalise_underscores",
    "remap_tags",
    "shuffle_tags",
    "split_tags",
]
