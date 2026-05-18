"""Caption-source helpers used by the tagger CLI build/eval modes.

Tag-taxonomy / caption-format constants (``SLOT_ORDER``, ``TAG_TYPE_NAMES``,
``RATINGS``, ``PEOPLE_COUNT_LABELS``) are the single source of truth for the
trainer's view of the corpus and live in
``library/captioning/anima_tagger.py`` so the inference wrapper, training CLI,
and any downstream consumer all see the same definitions. The script-local
helpers below (caption-file discovery, count-tag detection, people-count
bucketing) consume those constants.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Count-tag detection. Matches ``1girl``, ``2girls``, ``1boy``, ``3others``,
# ``multiple_girls``, ``multiple_boys``. Also matches ``6+girls`` /
# ``6+boys`` (booru convention for "≥6 of"). Underscores or spaces both fine.
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

# Pull the leading integer off a count tag like "3girls" / "6+girls" → 3 / 6.
_LEADING_INT_RE = re.compile(r"^(\d+)")

# Image extensions we look for next to each .txt caption file. Order is
# preference; first hit wins.
IMAGE_EXTS: Tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png")


def find_image_for_caption(caption_path: Path) -> Optional[Path]:
    """Return the sibling image file matching ``{stem}.<ext>``, or None."""
    for ext in IMAGE_EXTS:
        candidate = caption_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def is_count_tag(tag: str) -> bool:
    return bool(_COUNT_RE.match(tag))


def classify_people(tags: Iterable[str]) -> int:
    """Derive the 8-class :data:`PEOPLE_COUNT_LABELS` index for a parsed-tag list.

    Bucketing rules:

    * ``no_people`` (0) — no count tag at all
    * ``1girl`` (1), ``2girls`` (3), ``1boy`` (6) — exact-girls-no-boy /
      exact-boys-no-girl combos
    * ``1girl_1boy`` (2), ``2girls_1boy`` (4), ``2boys_1girl`` (5) —
      the three explicit mixed combos
    * ``multi`` (7) — anything else with a count tag: ``3+girls``,
      ``3+boys``, ``2girls+2+boys``, ``Nothers``, or a ``multiple_*`` tag
      with no explicit numeric companion. ``others`` count tags ride into
      ``multi`` since the head is girls/boys-shaped.

    Booru auto-fires ``multiple_girls`` / ``multiple_boys`` whenever the
    count is ≥2, not just ≥3 — so it cannot be treated as a ≥3 signal on
    its own. We defer to the explicit numeric count tag when one is
    present; ``multiple_*`` only contributes as a floor of 2 when no
    numeric tag for that gender was seen.

    Tag order in ``tags`` doesn't matter — counts are reduced first.
    """
    girls = boys = 0
    saw_multi_g = saw_multi_b = False
    saw_other = False
    for t in tags:
        if not is_count_tag(t):
            continue
        if t.startswith("multiple"):
            if "girl" in t:
                saw_multi_g = True
            elif "boy" in t:
                saw_multi_b = True
            elif "other" in t:
                saw_other = True
            continue
        m = _LEADING_INT_RE.match(t)
        if m is None:                    # e.g. malformed; defensive
            continue
        n = int(m.group(1))
        if "girl" in t:
            girls = max(girls, n)
        elif "boy" in t:
            boys = max(boys, n)
        # "others" count tags are recorded as a "multi" indicator without
        # changing girls/boys directly — they don't fit the 7 buckets.
        elif "other" in t:
            saw_other = True
    # ``multiple_*`` only kicks in when the explicit numeric tag is missing
    # (rare — booru attaches both). Treat it as ≥2, not ≥3, since that's
    # what the booru auto-tag actually means.
    if saw_multi_g and girls == 0:
        girls = 2
    if saw_multi_b and boys == 0:
        boys = 2
    if saw_other or girls >= 3 or boys >= 3 or (boys >= 2 and girls >= 2):
        return 7                          # multi: 3+girls / 3+boys / 2g+2b+ / lonely multiple_* / Nothers
    if girls == 0 and boys == 0:
        return 0                          # no_people (only when no count tag fired)
    if girls == 1 and boys == 0:
        return 1                          # 1girl
    if girls == 1 and boys == 1:
        return 2                          # 1girl_1boy
    if girls == 2 and boys == 0:
        return 3                          # 2girls
    if girls == 2 and boys == 1:
        return 4                          # 2girls_1boy
    if girls == 1 and boys == 2:
        return 5                          # 2boys_1girl
    if girls == 0 and boys == 1:
        return 6                          # 1boy
    return 7                              # fallback (e.g. 0g/2b without "others")
