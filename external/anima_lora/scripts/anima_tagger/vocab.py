"""Vocab build — caption discovery, tag categorization, frequency cuts.

Produces three artifacts under ``out_dir/``:

* ``vocab.json``   — kept tag list (with category + median emit position),
                     rating list, slot order, coverage stats, train/val split.
* ``rules.yaml``   — snapshot of the source ``tag_rules.yaml`` so the
                     inference wrapper has zero runtime dep on the corpus.
* ``dataset.json`` — per-stem ``(image_path, multi_hot_indices, rating_idx)``
                     manifest, filtered to captions with a sibling image,
                     a recognized rating, and at least one in-vocab tag.

The build is intentionally self-contained: every other CLI mode reads from
the manifest + vocab snapshot, never from the source corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from library.captioning import tag_groups as tg
from library.captioning import tag_rules as tr
from library.captioning.anima_tagger import (
    PEOPLE_COUNT_LABELS,
    RATINGS,
    SLOT_ORDER,
    TAG_TYPE_NAMES,
)

from .constants import (
    classify_people,
    find_image_for_caption,
    is_count_tag,
)

logger = logging.getLogger(__name__)


# ── Caption source discovery ──────────────────────────────────────────────


def find_caption_files(roots: Sequence[Path]) -> List[Path]:
    """Discover all ``.txt`` caption files under the given roots.

    Skips dotfiles and the ``tag_cache``/``hash_cache`` JSON sidecars.
    Returns a deduplicated list (by absolute path); a stem appearing under
    multiple roots is *not* deduped here — that's the caller's job (see
    :func:`build_caption_index`).
    """
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            logger.warning("caption root %s does not exist — skipping", root)
            continue
        for p in root.rglob("*.txt"):
            if any(part.startswith(".") for part in p.parts):
                continue
            out.append(p)
    return out


def build_caption_index(
    paths: Iterable[Path],
    rules: tr.TagRules,
) -> Dict[str, Tuple[Path, Optional[Path], List[str]]]:
    """Map ``stem → (caption_path, image_path | None, parsed_tags)``.

    When a stem appears in multiple caption sources, the *first* path wins
    (caller controls precedence via root order). Stems whose sibling image
    file can't be found are still indexed (caption-only entries) so the
    coverage scan reflects what's *captioned*, not what's *trainable*; the
    image-required filter happens at manifest-build time.
    """
    index: Dict[str, Tuple[Path, Optional[Path], List[str]]] = {}
    for path in sorted(paths):
        stem = path.stem
        if stem in index:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("non-utf8 caption %s — skipped", path)
            continue
        tags = tr.parse_caption(content, rules)
        if not tags:
            continue
        image_path = find_image_for_caption(path)
        index[stem] = (path, image_path, tags)
    return index


# ── Categorization ────────────────────────────────────────────────────────


def load_tag_cache(path: Path) -> Dict[str, str]:
    """Load the corpus tag-taxonomy cache and map tag → category name."""
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for tag, type_id in raw.items():
        cat = TAG_TYPE_NAMES.get(int(type_id))
        if cat is not None:
            # Cache uses underscored tag names; the canonical caption format
            # writes them with spaces. Normalize to space form so lookups
            # against parsed captions hit.
            out[tag.replace("_", " ")] = cat
    return out


# Categories the user is allowed to assign via ``category_overrides:`` in
# tag_rules.yaml. ``rating`` and ``count`` are intentionally excluded:
# rating is a separate field on the corpus and shouldn't be retrofitted
# onto a tag, and count tags are matched by regex (overriding would just
# create dead aliases).
_OVERRIDABLE_CATEGORIES = frozenset(TAG_TYPE_NAMES.values())


def categorize(
    tag: str,
    cache: Dict[str, str],
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Return ``rating`` / ``count`` / ``character`` / ``copyright`` /
    ``artist`` / ``general`` / ``metadata`` / ``deprecated`` for ``tag``.

    Resolution order:

    1. Rating literals (``general``/``sensitive``/``explicit``) → ``rating``.
       Note ``general`` is *both* a rating value and a category name, so
       rating-tag membership is checked before any category lookup.
    2. ``@<non-space>...`` tags → ``artist``. Anima's caption format
       prefixes artists with ``@`` directly followed by the name (e.g.
       ``@sincos``, ``@sumiyao (amam)``); the underlying tag-cache key
       drops the ``@``, so cache lookups need the bare name. Emoticons
       like ``@ @`` (booru ``@_@`` after ``_``→`` `` normalization) fall
       through to the cache so they get their real category (``general``).
    3. Count-tag regex → ``count`` (overrides ``general`` typing for
       ``1girl`` etc.).
    4. ``category_overrides`` lookup (curator-supplied via tag_rules.yaml).
       Fixes booru cache mistypings — e.g. GFL character tags stored as
       ``general`` (type_id=0) get corrected here.
    5. Cache lookup.
    6. Fallback: ``general``.
    """
    # Note: rating literals collide with the "general" category name. We
    # treat them as their own slot regardless of cache typing — the cache
    # doesn't actually carry rating values anyway (those come from a
    # separate corpus field, not the tag system).
    if tag in RATINGS:
        return "rating"
    if len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace():
        return "artist"
    if is_count_tag(tag):
        return "count"
    bare = tag[1:] if tag.startswith("@") else tag
    if overrides:
        ov = overrides.get(tag) or overrides.get(bare)
        if ov is not None:
            return ov
    cat = cache.get(bare)
    if cat is None:
        return "general"
    return cat


def validate_overrides(overrides: Dict[str, str]) -> List[str]:
    """Return a list of human-readable validation errors for overrides.

    Empty list → all entries are well-formed. Catches typos like
    ``caracter`` and unsupported categories (``rating`` / ``count``) up
    front so :func:`cmd_build_vocab` can fail loudly rather than silently
    typing tags into a slot the trainer doesn't understand.
    """
    errors: List[str] = []
    for tag, cat in overrides.items():
        if cat not in _OVERRIDABLE_CATEGORIES:
            errors.append(
                f"category_overrides[{tag!r}] = {cat!r} — must be one of "
                f"{sorted(_OVERRIDABLE_CATEGORIES)}"
            )
    return errors


# ── Vocab build ───────────────────────────────────────────────────────────


def build_vocab(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    tag_cache: Dict[str, str],
    min_freq: int,
    category_overrides: Optional[Dict[str, str]] = None,
) -> Dict:
    """Compute frequencies, categories, median emit positions; cut by min_freq."""
    freq: Counter = Counter()
    sum_pos: Dict[str, int] = defaultdict(int)
    pos_counts: Dict[str, int] = defaultdict(int)

    rating_freq: Counter = Counter()
    n_with_rating = 0
    people_freq: Counter = Counter()

    for stem, (_, _, tags) in caption_index.items():
        # Pull rating off the front if present; everything else feeds the
        # multi-label vocab. Anima's format puts rating first, but be
        # defensive — scan the first few tags.
        rating_seen: Optional[str] = None
        for t in tags[:2]:
            if t in RATINGS:
                rating_seen = t
                break
        if rating_seen is not None:
            rating_freq[rating_seen] += 1
            n_with_rating += 1

        # People-count bucket — derived from count tags. Distribution is
        # informational; the per-stem label is recomputed at manifest-build
        # time so it stays in sync with the labelling rule (don't read this
        # at training).
        people_freq[PEOPLE_COUNT_LABELS[classify_people(tags)]] += 1

        for i, tag in enumerate(tags):
            if tag in RATINGS:
                continue
            freq[tag] += 1
            sum_pos[tag] += i
            pos_counts[tag] += 1

    kept = sorted(
        (t for t, c in freq.items() if c >= min_freq),
        key=lambda t: (-freq[t], t),
    )
    dropped_lowfreq = sum(1 for c in freq.values() if c < min_freq)

    cat_buckets: Counter = Counter()
    cache_hits = 0
    for tag in kept:
        cat = categorize(tag, tag_cache, category_overrides)
        cat_buckets[cat] += 1
        bare = tag[1:] if tag.startswith("@") else tag
        if bare in tag_cache:
            cache_hits += 1

    tags_payload: List[Dict] = []
    for idx, tag in enumerate(kept):
        cat = categorize(tag, tag_cache, category_overrides)
        median_pos = sum_pos[tag] / max(pos_counts[tag], 1)
        tags_payload.append(
            {
                "name": tag,
                "index": idx,
                "category": cat,
                "freq": freq[tag],
                "median_pos": round(median_pos, 2),
            }
        )

    return {
        "tags": tags_payload,
        "ratings": list(RATINGS),
        "people_count_labels": list(PEOPLE_COUNT_LABELS),
        "slot_order": list(SLOT_ORDER),
        "min_freq": min_freq,
        "n_captions_seen": len(caption_index),
        "n_unique_tags_seen": len(freq),
        "n_tags_kept": len(kept),
        "n_tags_dropped_lowfreq": dropped_lowfreq,
        "category_counts": dict(cat_buckets),
        "cache_hit_rate": round(cache_hits / max(len(kept), 1), 4),
        "rating_distribution": dict(rating_freq),
        "rating_coverage": round(n_with_rating / max(len(caption_index), 1), 4),
        "people_count_distribution": dict(people_freq),
    }


def make_split(
    stems: Sequence[str],
    val_frac: float,
    seed: int,
) -> Dict[str, List[str]]:
    """Deterministic random split keyed by ``seed``."""
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return {
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "seed": seed,
        "val_frac": val_frac,
    }


# ── Training manifest ─────────────────────────────────────────────────────


def build_manifest(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    vocab: Dict,
    split: Dict,
) -> Dict:
    """Compact dataset.json: per-stem image path, multi-hot indices, rating, people-count.

    Stems lacking a sibling image file are dropped from the manifest (the
    coverage scan in :func:`scan_cache_coverage` still counts them in vocab
    statistics — we just can't *train* on captions without pixels). The split
    is filtered to match. Per-stem people-count label is recomputed from
    parsed tags via :func:`classify_people` so the bucketing rule is the
    single source of truth (no plumbing through vocab).
    """
    tag_to_idx: Dict[str, int] = {t["name"]: t["index"] for t in vocab["tags"]}
    rating_to_idx: Dict[str, int] = {r: i for i, r in enumerate(vocab["ratings"])}

    stems: List[str] = []
    image_paths: List[str] = []
    tag_indices: List[List[int]] = []
    rating_indices: List[int] = []
    people_count_indices: List[int] = []
    n_no_image = 0
    n_no_rating = 0
    n_no_tags = 0

    for stem in sorted(caption_index.keys()):
        _, image_path, tags = caption_index[stem]
        if image_path is None:
            n_no_image += 1
            continue
        rating_idx: Optional[int] = None
        for t in tags[:2]:
            if t in rating_to_idx:
                rating_idx = rating_to_idx[t]
                break
        if rating_idx is None:
            n_no_rating += 1
            continue
        idxs = sorted(
            tag_to_idx[t] for t in tags if t in tag_to_idx and t not in rating_to_idx
        )
        if not idxs:
            n_no_tags += 1
            continue
        stems.append(stem)
        image_paths.append(str(image_path.resolve()))
        tag_indices.append(idxs)
        rating_indices.append(rating_idx)
        people_count_indices.append(classify_people(tags))

    kept = set(stems)
    filtered_split = {
        "val": [s for s in split["val"] if s in kept],
        "train": [s for s in split["train"] if s in kept],
        "seed": split["seed"],
        "val_frac": split["val_frac"],
    }

    return {
        "stems": stems,
        "image_paths": image_paths,
        "tag_indices": tag_indices,
        "rating_indices": rating_indices,
        "people_count_indices": people_count_indices,
        "split": filtered_split,
        "n_tags": len(vocab["tags"]),
        "n_ratings": len(vocab["ratings"]),
        "n_people_counts": len(PEOPLE_COUNT_LABELS),
        "dropped_no_image": n_no_image,
        "dropped_no_rating": n_no_rating,
        "dropped_no_invocab_tags": n_no_tags,
    }


# ── Coverage scan ─────────────────────────────────────────────────────────


def scan_cache_coverage(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    tag_cache: Dict[str, str],
    category_overrides: Optional[Dict[str, str]] = None,
    coverage_ignore: Optional[Tuple[str, ...]] = None,
) -> Dict:
    """How many caption tags lack a category in gelcrawl's cache?

    A high miss rate would mean ``categorize()`` is falling back to
    ``general`` for too many tags and we should run the gelbooru API fill-in
    pass before training. <5 % miss → safe to default-to-general.

    Tags listed in ``category_overrides`` are treated as covered (they
    *are* explicitly typed — just by the curator rather than the cache).
    Tags whose name contains any substring in ``coverage_ignore`` are
    silently skipped from both the seen and missing tallies — used to
    drop noisy general descriptors ("grabbing another's …") that the
    booru cache doesn't track but the curator knows are general.
    """
    overrides = category_overrides or {}
    ignore_subs = tuple(coverage_ignore or ())
    seen: Counter = Counter()
    missing: Counter = Counter()
    for _, (_, _, tags) in caption_index.items():
        for tag in tags:
            if tag in RATINGS:
                continue
            if ignore_subs and any(sub in tag for sub in ignore_subs):
                continue
            seen[tag] += 1
            is_artist = len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()
            bare = tag[1:] if is_artist else tag
            if (
                is_artist
                or is_count_tag(tag)
                or bare in tag_cache
                or tag in overrides
                or bare in overrides
            ):
                continue
            missing[tag] += 1
    return {
        "n_unique_tags": len(seen),
        "n_unique_missing": len(missing),
        "n_total_tag_occurrences": sum(seen.values()),
        "n_missing_occurrences": sum(missing.values()),
        "missing_top20": missing.most_common(20),
    }


# ── CLI entry ─────────────────────────────────────────────────────────────


def cmd_build_vocab(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_src = Path(args.rules)
    rules = tr.load_rules(rules_src)
    logger.info(
        "rules: %d replacements, %d remove, %d dedup base tags, "
        "%d category overrides",
        len(rules.replacements),
        len(rules.remove),
        len(rules.dedup),
        len(rules.category_overrides),
    )
    override_errors = validate_overrides(rules.category_overrides)
    if override_errors:
        for e in override_errors:
            logger.error(e)
        raise SystemExit(
            f"category_overrides has {len(override_errors)} invalid entry(ies) "
            f"in {rules_src} — fix and re-run"
        )
    if rules.category_overrides:
        # Distribution of override targets, useful as a sanity printout.
        target_counts: Counter = Counter(rules.category_overrides.values())
        logger.info(
            "category_overrides targets: %s",
            {k: target_counts[k] for k in sorted(target_counts)},
        )

    roots = [Path(r) for r in args.caption_roots]
    cap_paths = find_caption_files(roots)
    logger.info("scanning %d caption files across %d roots", len(cap_paths), len(roots))
    index = build_caption_index(cap_paths, rules)
    logger.info("kept %d unique stems with non-empty captions", len(index))

    tag_cache = load_tag_cache(Path(args.tag_cache))
    logger.info("loaded tag cache with %d entries", len(tag_cache))

    coverage = scan_cache_coverage(
        index,
        tag_cache,
        rules.category_overrides,
        rules.coverage_ignore,
    )
    miss_rate = coverage["n_missing_occurrences"] / max(
        coverage["n_total_tag_occurrences"], 1
    )
    logger.info(
        "cache coverage: %d/%d unique tags categorized "
        "(%.2f%% of occurrences missing)",
        coverage["n_unique_tags"] - coverage["n_unique_missing"],
        coverage["n_unique_tags"],
        100 * miss_rate,
    )
    if coverage["missing_top20"]:
        logger.info("top-20 uncategorized tags (will fall back to 'general'):")
        for tag, n in coverage["missing_top20"]:
            logger.info("  %5d × %s", n, tag)

    vocab = build_vocab(
        index,
        tag_cache,
        min_freq=args.min_freq,
        category_overrides=rules.category_overrides,
    )
    vocab["caption_roots"] = [str(r.resolve()) for r in roots]
    vocab["tag_cache_path"] = str(Path(args.tag_cache).resolve())
    vocab["rules_source_path"] = str(rules_src.resolve())
    vocab["coverage"] = coverage

    # Resolve typed groups (eye_color, hair_color, …) against the kept
    # vocab and embed the index sets into vocab.json. Optional — when
    # --groups isn't passed (or the file is missing) we build a flat-vocab
    # checkpoint and the trainer falls back to pure BCE.
    groups_src = Path(args.groups) if args.groups else None
    if groups_src is not None and groups_src.exists():
        groups = tg.load_groups(groups_src)
        tag_to_idx = {t["name"]: t["index"] for t in vocab["tags"]}
        resolved, dropped = tg.resolve_groups(groups, tag_to_idx)
        vocab["groups"] = tg.resolved_to_dict(resolved)
        vocab["groups_source_path"] = str(groups_src.resolve())
        logger.info(
            "groups: %d typed groups, %d tag/escape names dropped (not_in_vocab)",
            len(resolved),
            len(dropped),
        )
        for g in resolved:
            n_drop = len(g.tag_names) < sum(1 for _ in groups.by_name(g.name).tags)
            logger.info(
                "  %-14s mode=%-18s n_tags=%3d n_escape=%2d%s",
                g.name, g.mode, len(g.tag_indices), len(g.escape_indices),
                "  (some tags dropped)" if n_drop else "",
            )
        if dropped:
            sample = list(dropped)[:10]
            logger.info(
                "first %d dropped: %s%s",
                len(sample), sample, " …" if len(dropped) > len(sample) else "",
            )
    else:
        vocab["groups"] = []
        vocab["groups_source_path"] = None
        if args.groups:
            logger.warning(
                "--groups=%s does not exist — building flat-vocab checkpoint",
                args.groups,
            )
        else:
            logger.info(
                "no --groups given — building flat-vocab checkpoint "
                "(pure BCE on every tag)",
            )

    split = make_split(
        sorted(index.keys()),
        val_frac=args.val_frac,
        seed=args.seed,
    )
    vocab["split"] = split

    # Write the vocab + split.
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", vocab_path)

    # Snapshot the groups YAML alongside rules.yaml so the inference wrapper
    # has zero runtime dependency on the source corpus.
    if groups_src is not None and groups_src.exists():
        groups_snap = out_dir / "groups.yaml"
        with open(groups_snap, "w") as f:
            import yaml as _yaml

            _yaml.safe_dump(groups.to_dict(), f, sort_keys=False)
        logger.info("wrote %s", groups_snap)

    # Snapshot the rules into the checkpoint dir so the inference wrapper
    # has zero runtime dependency on the source corpus.
    snap_path = out_dir / "rules.yaml"
    with open(snap_path, "w") as f:
        import yaml as _yaml

        _yaml.safe_dump(rules.to_dict(), f, sort_keys=False)
    logger.info("wrote %s", snap_path)

    # Build and persist the training manifest (drops captions without a
    # sibling image, without a rating tag, or with no in-vocab tags).
    manifest = build_manifest(index, vocab, split)
    manifest_path = out_dir / "dataset.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "wrote %s — %d trainable samples (dropped %d no_image, %d no_rating, "
        "%d no_invocab_tags)",
        manifest_path,
        len(manifest["stems"]),
        manifest["dropped_no_image"],
        manifest["dropped_no_rating"],
        manifest["dropped_no_invocab_tags"],
    )

    # Compact summary printout.
    print()
    print(f"  caption stems indexed:  {vocab['n_captions_seen']}")
    print(f"  unique tags seen:       {vocab['n_unique_tags_seen']}")
    print(f"  vocab size (≥{args.min_freq}):       {vocab['n_tags_kept']}")
    print(f"  dropped (low-freq):     {vocab['n_tags_dropped_lowfreq']}")
    print(f"  cache hit rate:         {vocab['cache_hit_rate']}")
    print("  category counts:")
    for cat, n in sorted(vocab["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<12} {n}")
    print(f"  rating coverage:        {vocab['rating_coverage']}")
    print(f"  rating distribution:    {vocab['rating_distribution']}")
    print(f"  people distribution:    {vocab['people_count_distribution']}")
    print(f"  split:                  {len(split['train'])} train / {len(split['val'])} val")
    print(f"  cache miss rate:        {miss_rate:.2%}")
    print(f"  trainable samples:      {len(manifest['stems'])}")
    print(
        f"    (dropped {manifest['dropped_no_image']} no-image, "
        f"{manifest['dropped_no_rating']} no-rating, "
        f"{manifest['dropped_no_invocab_tags']} no-invocab-tags)"
    )
