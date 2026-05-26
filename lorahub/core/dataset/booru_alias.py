"""Curated Danbooru -> Gelbooru tag aliases.

When the same concept is tagged differently between Danbooru and Gelbooru,
the upstream Anima config (and most of the SDXL anime-fork community) asks
authors to take the Gelbooru spelling, since that is what the public NoobAI
/ Pony / Animagine prompt examples use.

This table is intentionally **conservative**. The list only contains
entries where Gelbooru's tag wiki / aliases / "implies" graph publicly
agrees that the Gelbooru spelling is the canonical one; everything else is
left out. If you need a more aggressive mapping, ``load_aliases`` lets you
extend or fully replace the default in code.

A few rules of thumb that shaped the list:

- Pure underscore -> space conversions are *not* in here. They are already
  handled by ``normalise_underscores`` in :mod:`captions`.
- Tags whose Danbooru / Gelbooru spelling is identical post-normalisation
  are also omitted; there is nothing to remap.
- Numeric / counting tags (``1girl``, ``2girls``) are kept untouched
  because both sites use the same spellings.

Sources (manually verified):
- gelbooru.com wiki "implies" / "aliased to" entries
- danbooru.donmai.us tag aliases
"""

from __future__ import annotations

# Keys are the Danbooru spelling (post ``normalise_tags``: lowercase, spaces
# instead of underscores). Values are the Gelbooru-canonical replacement.
DANBOORU_TO_GELBOORU: dict[str, str] = {
    # hair colour: alternate spellings ----------------------------------
    "blond hair": "blonde hair",
    "platinum blond hair": "platinum blonde hair",
    "gray hair": "grey hair",
    "silver-haired": "silver hair",
    "white-haired": "white hair",
    # eye colour ---------------------------------------------------------
    "gray eyes": "grey eyes",
    # camera framing: concat vs spaced spellings -------------------------
    "fullbody": "full body",
    "upperbody": "upper body",
    "lowerbody": "lower body",
    "halfbody": "half body",
    # composition / scene direction --------------------------------------
    "outside": "outdoors",
    "inside": "indoors",
    "indoor": "indoors",
    "outdoor": "outdoors",
    # facial expression: aliased to base form ---------------------------
    "blushing": "blush",
    "smiling": "smile",
    "smirking": "smirk",
    "laughing": "laugh",
    "crying": "tears",
    # eye state ---------------------------------------------------------
    "closed eye": "closed eyes",
    "eye closed": "closed eyes",
    # outfit alternate spellings ---------------------------------------
    "swim suit": "swimsuit",
    "schoolgirl uniform": "school uniform",
    "sukumizu": "school swimsuit",
    # NSFW vocab spellings ---------------------------------------------
    "uncensor": "uncensored",
}


def load_aliases(
    extra: dict[str, str] | None = None,
    *,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge ``extra`` over ``base`` (default :data:`DANBOORU_TO_GELBOORU`).

    Keys in ``extra`` win over keys in ``base`` — this lets a user inject a
    site- or config-specific override on top of the curated default without
    having to copy the whole table. ``base=None`` falls back to the curated
    default; ``base={}`` yields a remap built only from ``extra`` (handy for
    tests that want to verify the mechanism with a tiny synthetic table).

    The returned dict is a *copy*; mutating it does not change the module
    constants.
    """
    merged: dict[str, str] = dict(
        DANBOORU_TO_GELBOORU if base is None else base
    )
    if extra:
        merged.update(extra)
    return merged


__all__ = ["DANBOORU_TO_GELBOORU", "load_aliases"]
