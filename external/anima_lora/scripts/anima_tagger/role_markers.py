"""Role-marker scan — find character-typed tags that behave like copyrights.

Reads ``vocab.json`` + ``dataset.json`` from a tagger checkpoint dir and
ranks every ``category=='character'`` tag by its conditional co-occurrence
with *another* character tag on **solo** training samples, then auto-buckets
each candidate into one of four classes via a string-prefix heuristic over
the partner list.

Buckets
~~~~~~~

* **A_costume** — candidate and a top partner share the same name prefix
  (everything before the first ``(``). The longer-parenthetical name is
  the variant; the shorter one is the base. Curate via ``tag_rules.yaml``
  dedup blocks (``base: [variant1, variant2, ...]``) so the variant wins
  whenever both fire.
* **D_role** — broad partner pool (``≥ min_role_partners`` distinct
  partners across solo co-occurrences). Tag is acting as an affiliation
  marker (``sensei (blue archive)``, ``producer (idolmaster)``,
  ``doctor (arknights)``). Curate via ``tag_rules.yaml`` ``remove:``.
* **C_pair** — narrow partner pool (top-1 partner accounts for ≥60% of
  co-occurrences, no costume-variant prefix match). Genuine couple/
  sibling tag pairs (``kousaka kyousuke ↔ kirino``,
  ``takasu ryuuji ↔ aisaka taiga``). Leave alone — the data is correct,
  the flat character softmax just absorbs the loss on these.
* **B_review** — everything else. Likely aliases (``smithee a. haysaca`` ↔
  ``hayasaka ai``, traveler/protagonist split) or noisy edge cases.
  Eyeball the ~20 rows that land here and decide per-row.

Output is a printed table (with a ``bucket`` column) and an optional YAML
stub split into pasteable sections — A bucket as dedup blocks, D bucket
under ``remove:``, B and C as commented hints. No files in the
checkpoint dir are mutated — this mode is read-only.

The "solo" predicate matches the trainer's :class:`GroupRouter` logic
exactly: at least one of ``solo``/``1girl``/``1boy``/``1other`` fires and
nothing matching the multi-count regex (``2girls``, ``multiple_girls``…)
fires.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


_SINGLE_COUNT_NAMES = {"solo", "1girl", "1boy", "1other"}
_MULTI_COUNT_RE = re.compile(
    r"^(?:\d+(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)


def _solo_index_sets(
    vocab_tags: Sequence[dict],
) -> Tuple[Set[int], Set[int]]:
    """Return ``(single_count_indices, multi_count_indices)`` from vocab."""
    single_idx: Set[int] = set()
    multi_idx: Set[int] = set()
    for t in vocab_tags:
        name = t["name"]
        idx = int(t["index"])
        if name in _SINGLE_COUNT_NAMES:
            single_idx.add(idx)
        elif _MULTI_COUNT_RE.match(name):
            multi_idx.add(idx)
    return single_idx, multi_idx


def _name_prefix(name: str) -> str:
    """Everything before the first ``(``, stripped. Used for costume detection.

    ``"toki (bunny) (blue archive)"`` → ``"toki"``;
    ``"gawr gura"``                   → ``"gawr gura"``.
    """
    return name.split("(", 1)[0].strip()


def _name_prefix_no_first_token(name: str) -> str:
    """Like :func:`_name_prefix` but additionally strips the first space token.

    Catches "adjective + base" costume patterns the bare prefix misses:

    * ``"cool mita (miside)"``       → ``"mita"`` (matches base ``"mita (miside)"``)
    * ``"male rover (wuthering waves)"`` → ``"rover"``
    * ``"the herta (honkai: star rail)"`` → ``"herta"``

    Returns an empty string when the prefix has only one token (no
    adjective to drop) — callers should treat empty as "no match".
    """
    pre = _name_prefix(name)
    parts = pre.split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[1:])


def classify(
    name: str,
    partners_full: List[Tuple[str, int]],
    n_co: int,
    min_role_partners: int,
    pair_dominance: float,
) -> Tuple[str, str]:
    """Bucket a candidate based on its full partner distribution.

    Returns ``(bucket, base_or_top_partner)`` where ``bucket`` is one of
    ``A_costume`` / ``C_pair`` / ``D_role`` / ``B_review``. The second
    element is the partner whose name should be paired with the candidate
    in the output: the prefix-matching base for A, the top partner for C
    and B, and an empty string for D (D is broad-pool by definition, no
    single base).

    Order of checks matters: A wins over D when both could apply (a
    popular character with both a costume variant and many other partners
    is still primarily a costume-variant case from the dedup standpoint).
    """
    if not partners_full or n_co == 0:
        return "B_review", ""

    n_distinct = len(partners_full)
    top_name, top_count = partners_full[0]
    top_share = top_count / n_co

    # D — role marker: broad partner pool wins over A. A character with
    # 5+ distinct co-partners is acting as an affiliation tag regardless
    # of any incidental prefix overlap with one of them ("doodle sensei"
    # has prefix-overlap with "sensei" but is also broad-pool — D wins so
    # it lands under `remove:` instead of getting demoted to a dedup row).
    if n_distinct >= min_role_partners:
        return "D_role", ""

    # A — costume/version variant: prefix-before-paren matches a partner
    # (tier 1, strong), OR drop-first-token matches (tier 2, catches
    # "cool mita ↔ mita", "male rover ↔ rover", "the herta ↔ herta").
    # Already past the D gate, so the pool is narrow enough to trust the
    # prefix relationship. The matched partner becomes the *base* if the
    # candidate is more specific; otherwise the candidate is the base
    # and the row is informational only (A_base, suppressed in YAML out).
    cand_prefix = _name_prefix(name)
    cand_prefix_drop1 = _name_prefix_no_first_token(name)
    matched_partner = None
    if cand_prefix:
        for pname, _ in partners_full:
            if pname == name:
                continue
            if _name_prefix(pname) == cand_prefix:
                matched_partner = pname
                break
    if matched_partner is None and cand_prefix_drop1:
        for pname, _ in partners_full:
            if pname == name:
                continue
            if _name_prefix(pname) == cand_prefix_drop1:
                matched_partner = pname
                break
    if matched_partner is not None:
        if _is_more_specific(name, matched_partner):
            return "A_costume", matched_partner
        return "A_base", matched_partner

    # C — couple/sibling pair: narrow pool dominated by top partner, but
    # only when there are ≥2 distinct partners. The n_distinct==1 case is
    # ambiguous (alias vs. always-paired-couple — both look identical in
    # data) and gets pushed to B_review for an eyeball decision.
    if top_share >= pair_dominance and n_distinct >= 2:
        return "C_pair", top_name

    # Everything else needs human review (likely B aliases / noisy).
    return "B_review", top_name


def _is_more_specific(a: str, b: str) -> bool:
    """True if ``a`` is the more-specific (variant) form of ``b``.

    Heuristic: the name with more parenthetical groups is the variant; on
    a tie, the longer string is the variant. Used to pick the dedup
    direction so we emit ``base: [variant]`` and not the reverse.
    """
    a_paren = a.count("(")
    b_paren = b.count("(")
    if a_paren != b_paren:
        return a_paren > b_paren
    return len(a) > len(b)


def scan(
    vocab: dict,
    manifest: dict,
    min_solo: int,
    min_ratio: float,
    top_partners: int,
    min_role_partners: int = 5,
    pair_dominance: float = 0.6,
) -> List[dict]:
    """Return a list of candidate role markers ranked by co-occurrence ratio.

    Each entry is::

        {
          "name": "sensei (blue archive)",
          "index": 620,
          "freq": 130,
          "n_solo": 142,                  # solo samples where this tag fires
          "n_co": 138,                    # of those, how many also have another char
          "ratio": 0.971,
          "partners": [(name, count), ...],  # top-K partner chars by count
          "n_distinct_partners": 47,       # full pool breadth
          "bucket": "D_role",
          "base": "",                      # the variant base (A) or top pair-mate
        }

    Sorted by descending ratio, then descending ``n_solo``.
    """
    tags = vocab["tags"]
    idx2name: Dict[int, str] = {int(t["index"]): t["name"] for t in tags}
    char_idx: Set[int] = {int(t["index"]) for t in tags if t["category"] == "character"}
    single_idx, multi_idx = _solo_index_sets(tags)

    # Per-character tallies.
    n_solo: Dict[int, int] = {i: 0 for i in char_idx}
    n_co: Dict[int, int] = {i: 0 for i in char_idx}
    partner: Dict[int, Dict[int, int]] = {i: {} for i in char_idx}

    for tags_list in manifest["tag_indices"]:
        s = set(tags_list)
        is_solo = bool(s & single_idx) and not (s & multi_idx)
        if not is_solo:
            continue
        chars_here = s & char_idx
        if not chars_here:
            continue
        is_multi = len(chars_here) > 1
        for c in chars_here:
            n_solo[c] += 1
            if is_multi:
                n_co[c] += 1
                for p in chars_here:
                    if p == c:
                        continue
                    partner[c][p] = partner[c].get(p, 0) + 1

    rows: List[dict] = []
    name_to_freq = {t["name"]: int(t["freq"]) for t in tags}
    for c in char_idx:
        ns = n_solo[c]
        if ns < min_solo:
            continue
        ratio = n_co[c] / ns
        if ratio < min_ratio:
            continue
        partners_full_sorted = sorted(partner[c].items(), key=lambda kv: -kv[1])
        partners_named = [(idx2name[p], n) for p, n in partners_full_sorted]
        bucket, base = classify(
            idx2name[c],
            partners_named,
            n_co[c],
            min_role_partners=min_role_partners,
            pair_dominance=pair_dominance,
        )
        rows.append(
            {
                "name": idx2name[c],
                "index": c,
                "freq": name_to_freq.get(idx2name[c], -1),
                "n_solo": ns,
                "n_co": n_co[c],
                "ratio": ratio,
                "partners": partners_named[:top_partners],
                "n_distinct_partners": len(partners_named),
                "bucket": bucket,
                "base": base,
            }
        )
    rows.sort(key=lambda r: (-r["ratio"], -r["n_solo"]))
    return rows


def _yaml_safe(s: str) -> str:
    """Return ``s`` in a YAML-safe form for use as a sequence item or key.

    Single-quotes the string when it contains characters that would confuse
    a YAML parser at this position: ``": "`` (key/value separator —
    `trailblazer (honkai: star rail)` would otherwise parse as a mapping),
    a trailing ``":"``, a leading reserved indicator (``@``, ``-``, ``?``,
    etc.), or an internal apostrophe (which gets doubled inside the quotes
    per YAML 1.2 § 7.3.2).

    Bare strings are preferred for readability — only quote when needed.
    """
    needs_quote = (
        ": " in s
        or s.endswith(":")
        or "'" in s
        or (s and s[0] in "@#&*!|>'\"%`-?,[]{}")
    )
    if not needs_quote:
        return s
    return "'" + s.replace("'", "''") + "'"


def _format_table(rows: List[dict], limit: int) -> str:
    """Render the candidate table as fixed-width text."""
    if not rows:
        return "(no candidates above threshold)"
    head = (
        f"{'bucket':<10}  {'ratio':>5}  {'n_solo':>6}  {'n_co':>5}  "
        f"{'np':>3}  {'freq':>5}  {'tag':<40}  partners (count)"
    )
    sep = "-" * len(head)
    lines = [head, sep]
    for r in rows[:limit]:
        partners_str = ", ".join(f"{n}×{name}" for name, n in r["partners"])
        lines.append(
            f"{r['bucket']:<10}  {r['ratio']:>5.2f}  {r['n_solo']:>6d}  "
            f"{r['n_co']:>5d}  {r['n_distinct_partners']:>3d}  "
            f"{r['freq']:>5d}  {r['name']:<40}  {partners_str}"
        )
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more)")
    return "\n".join(lines)


def _emit_yaml_stub(rows: List[dict], min_solo: int, min_ratio: float) -> str:
    """Build a single yaml-shaped string with three pasteable sections.

    The output is **not** a single valid YAML document — it's a working
    file the curator copies snippets out of. Sections:

    * ``# A_costume`` — dedup blocks (``base: [variants]``) ready to paste
      under top-level keys in ``tag_rules.yaml``.
    * ``# D_role``   — items ready to paste under ``remove:``.
    * ``# B_review`` / ``# C_pair`` — commented hints; nothing to paste
      verbatim, but useful for triage decisions.
    """
    by_bucket: Dict[str, List[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)

    lines: List[str] = []
    lines.append("# Auto-classified role-marker scan output.")
    lines.append(
        f"# Threshold: n_solo>={min_solo}, ratio>={min_ratio:.2f}. "
        f"{len(rows)} candidate(s)."
    )
    lines.append("# Sections below are ready to paste into tag_rules.yaml — see")
    lines.append("# headers for the target location.")
    lines.append("")

    # ── A_costume — dedup blocks ───────────────────────────────────────
    a_rows = by_bucket.get("A_costume", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ A_costume — paste these as top-level dedup blocks in         ║")
    lines.append("# ║ tag_rules.yaml (when any variant fires, the base is dropped) ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if a_rows:
        # Group variants by base.
        base_to_variants: Dict[str, List[Tuple[str, dict]]] = {}
        for r in a_rows:
            base = r["base"]
            base_to_variants.setdefault(base, []).append((r["name"], r))
        for base in sorted(base_to_variants):
            variants = base_to_variants[base]
            lines.append(f"{_yaml_safe(base)}:")
            for vname, vr in sorted(variants, key=lambda kv: kv[0]):
                lines.append(
                    f"  - {_yaml_safe(vname)}  # "
                    f"ratio={vr['ratio']:.2f} n_solo={vr['n_solo']}"
                )
        lines.append("")
    else:
        lines.append("# (no A_costume candidates)")
        lines.append("")

    # ── D_role — under `remove:` ───────────────────────────────────────
    d_rows = by_bucket.get("D_role", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ D_role — paste these under `remove:` in tag_rules.yaml.      ║")
    lines.append("# ║ These are class/affiliation markers (broad partner pool).    ║")
    lines.append("# ║ Removing strips them from training; alternatively keep the   ║")
    lines.append("# ║ tag and recategorize via a future force_general: override.   ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    lines.append("remove:")
    if d_rows:
        for r in d_rows:
            lines.append(
                f"  - {_yaml_safe(r['name'])}  # "
                f"ratio={r['ratio']:.2f} n_solo={r['n_solo']} "
                f"n_partners={r['n_distinct_partners']}"
            )
    else:
        lines.append("  # (no D_role candidates)")
    lines.append("")

    # ── B_review — needs eyeballing ────────────────────────────────────
    b_rows = by_bucket.get("B_review", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ B_review — eyeball each. Likely aliases (use `replacements:`)║")
    lines.append("# ║ or genuine pair-mates.                                       ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if b_rows:
        for r in b_rows:
            top = r["partners"][0] if r["partners"] else ("", 0)
            lines.append(
                f"#  - {r['name']:<40}  top: {top[1]}×{top[0]} "
                f"(ratio={r['ratio']:.2f}, n_solo={r['n_solo']}, "
                f"n_partners={r['n_distinct_partners']})"
            )
    else:
        lines.append("# (none)")
    lines.append("")

    # ── C_pair — informational only ────────────────────────────────────
    c_rows = by_bucket.get("C_pair", [])
    lines.append("# ╔══════════════════════════════════════════════════════════════╗")
    lines.append("# ║ C_pair — leave these alone. Genuine couple/sibling tag pairs;║")
    lines.append("# ║ the data is correct.                                         ║")
    lines.append("# ╚══════════════════════════════════════════════════════════════╝")
    if c_rows:
        for r in c_rows:
            top = r["partners"][0] if r["partners"] else ("", 0)
            lines.append(
                f"#  - {r['name']:<40}  pair: {top[0]} ({top[1]}×, "
                f"ratio={r['ratio']:.2f})"
            )
    else:
        lines.append("# (none)")
    lines.append("")

    return "\n".join(lines)


def cmd_scan_role_markers(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.json"
    manifest_path = out_dir / "dataset.json"
    if not vocab_path.exists() or not manifest_path.exists():
        raise SystemExit(
            f"need both {vocab_path} and {manifest_path} — run --mode build_vocab first"
        )
    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = scan(
        vocab,
        manifest,
        min_solo=args.min_solo,
        min_ratio=args.min_ratio,
        top_partners=args.top_partners,
        min_role_partners=args.min_role_partners,
        pair_dominance=args.pair_dominance,
    )
    bucket_counts: Dict[str, int] = {}
    for r in rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1
    logger.info(
        "scanned %d trainable samples, %d character tags in vocab — "
        "%d candidates with n_solo≥%d and ratio≥%.2f",
        len(manifest["tag_indices"]),
        sum(1 for t in vocab["tags"] if t["category"] == "character"),
        len(rows),
        args.min_solo,
        args.min_ratio,
    )
    logger.info(
        "  buckets: A_costume=%d, A_base=%d, D_role=%d, C_pair=%d, B_review=%d",
        bucket_counts.get("A_costume", 0),
        bucket_counts.get("A_base", 0),
        bucket_counts.get("D_role", 0),
        bucket_counts.get("C_pair", 0),
        bucket_counts.get("B_review", 0),
    )
    print(_format_table(rows, args.limit))

    if args.out_yaml:
        out_path = Path(args.out_yaml)
        out_path.write_text(
            _emit_yaml_stub(rows, args.min_solo, args.min_ratio), encoding="utf-8"
        )
        logger.info("wrote bucketed stub → %s", out_path)
