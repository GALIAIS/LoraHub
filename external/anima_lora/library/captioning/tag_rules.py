"""Anima caption tag-rules — replacements, removals, clothing-base dedup.

Mirrors the rule semantics of ``gelcrawl/postprocess.py::dedup_tags_in_file`` so
the trained tagger can apply the same normalization at inference without a
runtime dependency on the gelcrawl repo. The ``rules.yaml`` snapshot inside
the tagger checkpoint dir is the canonical source at runtime; ``gelcrawl``'s
copy is only consulted at vocab-build time.

Three rule families:

* ``replacements``: whole-string ``str.replace`` applied before tokenization.
  Used for HTML-entity decode (``&#039; → '``), rating collapse
  (``questionable → sensitive``), and artist-alias rewrites.
* ``remove``: tag literals that are unconditionally stripped.
* dedup map: ``{base: {variants}}``. If any variant of ``base`` is present,
  the base tag is removed. Used to drop generic clothing tags like ``bra``
  when ``black bra`` is already in the caption.

The :func:`apply_rules` function operates on a tag list and returns a tag
list — string-level work happens in :func:`load_rules` (replacements get
fused into a single ``str.replace`` chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import yaml


@dataclass(frozen=True)
class TagRules:
    """Compiled rule set ready to apply to caption strings or tag lists."""

    replacements: Tuple[Tuple[str, str], ...]
    remove: frozenset
    dedup: Dict[str, frozenset]
    # Per-tag category override consulted *before* the booru tag cache in
    # ``vocab.categorize()``. Useful when the cache mis-types a tag (e.g.
    # GFL character tags stored as type_id=0 ``general``). Empty by default
    # — overrides only apply when the curator lists the tag explicitly.
    category_overrides: Dict[str, str]
    # Substring patterns suppressed from the build-time "top-20
    # uncategorized" coverage log. Pure logging filter — does not change
    # categorization. Use for noisy general descriptors the booru cache
    # doesn't track (e.g. ``"another's"`` catches "grabbing another's
    # breast", "holding another's hair", etc.). Substring match is
    # case-sensitive; booru tags are already lowercase.
    coverage_ignore: Tuple[str, ...]

    def to_dict(self) -> dict:
        """Round-trippable dict for snapshotting into the checkpoint dir."""
        out: dict = {
            "replacements": dict(self.replacements),
            "remove": sorted(self.remove),
        }
        if self.category_overrides:
            out["category_overrides"] = dict(self.category_overrides)
        if self.coverage_ignore:
            out["coverage_ignore"] = list(self.coverage_ignore)
        out.update(
            {base: sorted(variants) for base, variants in self.dedup.items()}
        )
        return out


# Top-level YAML keys that are NOT dedup base→variants entries. Centralized
# so :func:`load_rules` and :func:`from_dict` agree on what to exclude.
_RESERVED_KEYS = frozenset({
    "replacements",
    "remove",
    "category_overrides",
    "coverage_ignore",
})


def load_rules(path: str | Path) -> TagRules:
    """Load a ``tag_rules.yaml`` (gelcrawl format) into a :class:`TagRules`."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    repl_map = raw.pop("replacements", {}) or {}
    remove = frozenset(raw.pop("remove", []) or [])
    overrides = dict(raw.pop("category_overrides", {}) or {})
    coverage_ignore = tuple(str(s) for s in (raw.pop("coverage_ignore", []) or []))
    # Everything else in the YAML is a dedup base→variants entry.
    dedup = {str(base): frozenset(variants) for base, variants in raw.items()}
    replacements = tuple((str(k), str(v)) for k, v in repl_map.items())
    return TagRules(
        replacements=replacements,
        remove=remove,
        dedup=dedup,
        category_overrides={str(k): str(v) for k, v in overrides.items()},
        coverage_ignore=coverage_ignore,
    )


def from_dict(d: dict) -> TagRules:
    """Inverse of :meth:`TagRules.to_dict` — load a snapshot from JSON."""
    repl_map = d.get("replacements", {}) or {}
    remove = frozenset(d.get("remove", []) or [])
    overrides = dict(d.get("category_overrides", {}) or {})
    coverage_ignore = tuple(str(s) for s in (d.get("coverage_ignore", []) or []))
    dedup = {
        k: frozenset(v)
        for k, v in d.items()
        if k not in _RESERVED_KEYS
    }
    replacements = tuple((str(k), str(v)) for k, v in repl_map.items())
    return TagRules(
        replacements=replacements,
        remove=remove,
        dedup=dedup,
        category_overrides={str(k): str(v) for k, v in overrides.items()},
        coverage_ignore=coverage_ignore,
    )


def apply_replacements(content: str, rules: TagRules) -> str:
    """Apply whole-string replacements (rating collapse, HTML decode, alias)."""
    for find, replace in rules.replacements:
        if find in content:
            content = content.replace(find, replace)
    return content


def parse_caption(content: str, rules: TagRules) -> List[str]:
    """Split a raw caption string into a clean tag list under ``rules``.

    Equivalent to gelcrawl's ``dedup_tags_in_file`` but pure: no file IO,
    no in-place mutation. Returns the *kept* tags in their original order.
    """
    content = apply_replacements(content, rules).strip()
    if not content:
        return []
    tags = [t.strip() for t in content.split(",")]
    tags = [t for t in tags if t]
    return apply_rules(tags, rules)


def apply_rules(tags: Iterable[str], rules: TagRules) -> List[str]:
    """Drop ``remove``-listed tags and dedup base tags whose variants fired."""
    tag_list = list(tags)
    tag_set: Set[str] = set(tag_list)
    to_remove: Set[str] = set(tag_set & rules.remove)
    for base, variants in rules.dedup.items():
        if base in tag_set and tag_set & variants:
            to_remove.add(base)
    if not to_remove:
        return tag_list
    return [t for t in tag_list if t not in to_remove]
