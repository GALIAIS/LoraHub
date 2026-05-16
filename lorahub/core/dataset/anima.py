"""Anima caption formatter and dataset transformer.

Anima (https://huggingface.co/circlestone-labs/Anima) is tdrussell's 2B
text-to-image model derived from Cosmos-Predict2 (NOT SDXL). Its training
captions follow a strict Danbooru-style layout:

    [quality / meta / year / safety] [1girl/1boy/1other] [character] [series]
    [artist] [general]

Local rules:

* lowercase only
* underscores -> spaces, EXCEPT ``score_N`` (PonyV7 aesthetic scores) which
  keep the underscore as a single token (regex ``^score_\\d+$``)
* artist tags get an ``@`` prefix (``@nnn yryr``); without it Anima barely
  responds to artist conditioning
* when Danbooru and Gelbooru disagree, prefer the Gelbooru spelling — this
  module cannot enforce that, it's a data-prep convention

Non-anime subsets use a multi-line layout:

    line 1: dataset tag (e.g. ``ye-pop`` / ``deviantart``)
    line 2: alt-text / natural-language summary
    line 3+: the regular tag string

This module only rewrites caption text; it never invokes a tagger model
(WD14/JoyTag handle that upstream).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Vocabularies (best-effort classifiers used by parse_caption)
# --------------------------------------------------------------------------- #

# Human-rated quality bucket. Order here is the "good -> bad" order that
# Anima's docs use for prefix/suffix conditioning suggestions.
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

# Anima inherits PonyV7's score_N aesthetic ladder. score_1 .. score_9.
# We keep the underscore form on the wire even though every other tag
# normalises ``_`` to space.
SCORE_TAG_RE = re.compile(r"^score_\d+$")

# Year tags: either ``year YYYY`` or one of the bucket names.
YEAR_BUCKET_TAGS: frozenset[str] = frozenset({"newest", "recent", "mid", "early", "old"})
YEAR_NUMERIC_RE = re.compile(r"^year \d{4}$")

# Production / source meta.
META_TAGS: frozenset[str] = frozenset(
    {
        "highres",
        "absurdres",
        "anime screenshot",
        "jpeg artifacts",
        "official art",
    }
)

# Booru rating buckets.
SAFETY_TAGS: frozenset[str] = frozenset({"safe", "sensitive", "nsfw", "explicit"})

# Subject head-count tags: 1girl, 1boy, 1other, 2girls, multiple_girls, etc.
# Keep the regex strict so we don't sweep stray general tags into "subject".
SUBJECT_RE = re.compile(r"^(?:\d+(?:girls?|boys?|others?)|multiple (?:girls|boys|others)|no humans)$")


# --------------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------------- #


def _norm_tag(tag: str) -> str:
    """Lowercase + ``_``-to-space, with score_N preserved verbatim."""
    t = tag.strip().lower()
    if not t:
        return ""
    if SCORE_TAG_RE.match(t):
        return t
    return t.replace("_", " ")


def _norm_artist(tag: str) -> str:
    """Lowercase + ``_``-to-space and ensure a single leading ``@``."""
    t = tag.strip().lower()
    if not t:
        return ""
    had_prefix = t.startswith("@")
    if had_prefix:
        t = t[1:].lstrip()
    # Artist names never use the score_N exception.
    t = t.replace("_", " ")
    return f"@{t}"


def _dedup(seq: list[str]) -> list[str]:
    """Stable de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


@dataclass(frozen=True, slots=True)
class AnimaCaptionFormatter:
    """Reorder + normalise a tag list to Anima's recommended layout.

    Construct one per caption, then call :meth:`format` to render. Use
    :func:`parse_caption` to go the other way.
    """

    quality: list[str] = field(default_factory=list)
    score: list[str] = field(default_factory=list)
    year: list[str] = field(default_factory=list)
    meta: list[str] = field(default_factory=list)
    safety: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    character: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    # Stored without the ``@`` prefix; format() adds it.
    artist: list[str] = field(default_factory=list)
    general: list[str] = field(default_factory=list)
    dataset_tag: str | None = None
    natural_language: str | None = None

    def _ordered_tags(self) -> list[str]:
        ordered: list[str] = []
        ordered += [_norm_tag(t) for t in self.quality]
        ordered += [_norm_tag(t) for t in self.score]
        ordered += [_norm_tag(t) for t in self.year]
        ordered += [_norm_tag(t) for t in self.meta]
        ordered += [_norm_tag(t) for t in self.safety]
        ordered += [_norm_tag(t) for t in self.subject]
        ordered += [_norm_tag(t) for t in self.character]
        ordered += [_norm_tag(t) for t in self.series]
        ordered += [_norm_artist(t) for t in self.artist]
        ordered += [_norm_tag(t) for t in self.general]
        return _dedup([t for t in ordered if t])

    def format(self) -> str:
        """Render to a single Anima-compliant caption string."""
        body = ", ".join(self._ordered_tags())
        if self.dataset_tag is None:
            return body
        # Multi-line layout for non-anime subsets.
        head = self.dataset_tag.strip().lower()
        lines = [head]
        if self.natural_language:
            lines.append(self.natural_language.strip())
        if body:
            lines.append(body)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _split_caption(text: str) -> tuple[str | None, str | None, list[str]]:
    """Pull off the optional dataset_tag (line 1) and natural_language (line 2).

    Returns ``(dataset_tag, natural_language, raw_tag_tokens)``. The third
    element is the comma-split tag list from the remaining body, lower-cased
    and trimmed (but not yet normalised — that happens in parse_caption).
    """
    # Strip BOM/whitespace.
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return None, None, []

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None, None, []

    # Heuristic: the first line is a dataset tag iff it has no commas AND it
    # does not look like the regular tag stream. Single-token / two-token
    # slugs like ``ye-pop``, ``deviantart`` qualify; ``masterpiece, ...`` does
    # not. We also require at least one extra line so a one-line caption
    # without commas (rare but possible) does not get mistaken for a header.
    dataset_tag: str | None = None
    natural_language: str | None = None
    body_lines = lines

    if len(lines) >= 2 and "," not in lines[0] and " " not in lines[0].rstrip():
        dataset_tag = lines[0]
        # Treat the next line as natural language if it contains sentence-y
        # prose (a period or 4+ words) — otherwise it's just more tags.
        candidate = lines[1]
        if "." in candidate or len(candidate.split()) >= 4:
            natural_language = candidate
            body_lines = lines[2:]
        else:
            body_lines = lines[1:]

    body = ", ".join(body_lines)
    raw_tokens = [t.strip().lower() for t in body.split(",") if t.strip()]
    return dataset_tag, natural_language, raw_tokens


def parse_caption(text: str) -> AnimaCaptionFormatter:
    """Best-effort split of an arbitrary caption into Anima sections.

    The classifier is keyword-driven: only the buckets we have vocab for
    (quality, score, year, meta, safety, subject) plus the ``@``-prefixed
    artist convention are recognised. Everything else lands in ``general``
    so callers can re-classify by hand if they need to.
    """
    dataset_tag, natural_language, tokens = _split_caption(text)

    quality: list[str] = []
    score: list[str] = []
    year: list[str] = []
    meta: list[str] = []
    safety: list[str] = []
    subject: list[str] = []
    artist: list[str] = []
    general: list[str] = []

    for raw in tokens:
        # Artist marker is the strongest signal — handle it first.
        if raw.startswith("@"):
            artist.append(raw[1:].lstrip())
            continue

        # score_N keeps the underscore; classify before normalisation.
        if SCORE_TAG_RE.match(raw):
            score.append(raw)
            continue

        # Normalise underscores to spaces for vocab lookups.
        norm = raw.replace("_", " ")

        if norm in QUALITY_TAGS:
            quality.append(norm)
        elif norm in YEAR_BUCKET_TAGS or YEAR_NUMERIC_RE.match(norm):
            year.append(norm)
        elif norm in META_TAGS:
            meta.append(norm)
        elif norm in SAFETY_TAGS:
            safety.append(norm)
        elif SUBJECT_RE.match(norm):
            subject.append(norm)
        else:
            general.append(norm)

    return AnimaCaptionFormatter(
        quality=_dedup(quality),
        score=_dedup(score),
        year=_dedup(year),
        meta=_dedup(meta),
        safety=_dedup(safety),
        subject=_dedup(subject),
        artist=_dedup(artist),
        general=_dedup(general),
        dataset_tag=dataset_tag,
        natural_language=natural_language,
    )


# --------------------------------------------------------------------------- #
# Bulk transformer
# --------------------------------------------------------------------------- #

ProgressFn = Callable[[Path], None]


def _iter_caption_files(directory: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def _merge_defaults(
    parsed: AnimaCaptionFormatter,
    *,
    default_quality: list[str] | None,
    default_safety: str | None,
    default_score: list[str] | None,
    default_year: list[str] | None,
    dataset_tag: str | None,
) -> AnimaCaptionFormatter:
    quality = list(parsed.quality)
    if default_quality:
        for q in default_quality:
            n = _norm_tag(q)
            if n and n not in quality:
                quality.append(n)

    safety = list(parsed.safety)
    if default_safety:
        n = _norm_tag(default_safety)
        if n and not safety:
            safety.append(n)

    score = list(parsed.score)
    if default_score:
        for s in default_score:
            n = _norm_tag(s)
            if n and n not in score:
                score.append(n)

    year = list(parsed.year)
    if default_year:
        for y in default_year:
            n = _norm_tag(y)
            if n and n not in year:
                year.append(n)

    return replace(
        parsed,
        quality=quality,
        score=score,
        year=year,
        safety=safety,
        dataset_tag=dataset_tag if dataset_tag is not None else parsed.dataset_tag,
    )


@dataclass(frozen=True, slots=True)
class AnimaDatasetTransformer:
    """Bulk-rewrite Danbooru-style caption files into Anima layout.

    The transformer never invents tags — it only re-orders the existing
    caption and (optionally) injects user-provided defaults for quality /
    safety / score / year / dataset_tag.
    """

    default_quality: list[str] | None = None
    default_safety: str | None = "safe"
    default_score: list[str] | None = None
    default_year: list[str] | None = None
    dataset_tag: str | None = None

    def transform_file(self, path: Path) -> str:
        """Read, transform, and return the new caption text (no write)."""
        text = path.read_text(encoding="utf-8")
        parsed = parse_caption(text)
        merged = _merge_defaults(
            parsed,
            default_quality=self.default_quality,
            default_safety=self.default_safety,
            default_score=self.default_score,
            default_year=self.default_year,
            dataset_tag=self.dataset_tag,
        )
        return merged.format()

    def transform_directory(
        self,
        path: Path,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        progress: ProgressFn | None = None,
    ) -> int:
        """Walk ``path`` and rewrite each ``*.txt`` caption in place.

        Returns the number of files written. With ``overwrite=False`` (the
        default) every existing caption is left untouched — call sites must
        opt in explicitly. Files that don't exist are silently skipped.
        """
        if not path.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")

        written = 0
        for caption_path in _iter_caption_files(path, recursive=recursive):
            if not overwrite:
                continue
            new_text = self.transform_file(caption_path)
            caption_path.write_text(new_text, encoding="utf-8")
            written += 1
            if progress is not None:
                progress(caption_path)
        return written


__all__ = [
    "QUALITY_TAGS",
    "SCORE_TAG_RE",
    "YEAR_BUCKET_TAGS",
    "YEAR_NUMERIC_RE",
    "META_TAGS",
    "SAFETY_TAGS",
    "SUBJECT_RE",
    "AnimaCaptionFormatter",
    "AnimaDatasetTransformer",
    "parse_caption",
]
