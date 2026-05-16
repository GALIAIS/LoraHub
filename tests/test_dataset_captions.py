"""Tests for the generic anime caption preprocessing toolkit."""

from __future__ import annotations

from pathlib import Path

from lorahub.core.dataset.captions import (
    CaptionPipeline,
    META_TAGS,
    QUALITY_TAGS,
    SAFETY_TAGS,
    SCORE_TAGS,
    TIME_TAGS,
    add_artist_prefix,
    drop_tags,
    filter_blacklist,
    inject_quality,
    is_score_tag,
    is_year_tag,
    join_tags,
    normalise_tags,
    normalise_underscores,
    remap_tags,
    shuffle_tags,
    split_tags,
)


# --------------------------------------------------------------------------- #
# Atomic transformations
# --------------------------------------------------------------------------- #


def test_normalise_underscores_preserves_score_n() -> None:
    """score_N family must survive verbatim; other tags lose underscores."""
    assert normalise_underscores("score_7") == "score_7"
    assert normalise_underscores("score_9_up") == "score_9_up"
    assert normalise_underscores("oomuro_sakurako") == "oomuro sakurako"
    # Stripping is part of the contract — caption files arrive with messy
    # whitespace from upstream taggers.
    assert normalise_underscores("  blue_hair  ") == "blue hair"


def test_normalise_tags_full_pipeline() -> None:
    """Lowercase + underscore swap + dedupe + order preserving."""
    out = normalise_tags("Blue_Hair, BLUE HAIR, 1girl, score_7")
    assert out == "blue hair, 1girl, score_7"


def test_split_join_round_trip() -> None:
    """split -> join with default separator returns a trimmed canonical form."""
    raw = "  a, b ,c ,, d "
    parts = split_tags(raw)
    assert parts == ["a", "b", "c", "d"]
    assert join_tags(parts) == "a, b, c, d"


def test_is_score_tag_and_year_tag() -> None:
    assert is_score_tag("score_7")
    assert is_score_tag("score_9_up")
    assert not is_score_tag("score_seven")
    assert not is_score_tag("masterpiece")
    assert is_year_tag("year 2024")
    assert not is_year_tag("year tag")


# --------------------------------------------------------------------------- #
# Random-modifying transformations
# --------------------------------------------------------------------------- #


def test_shuffle_keep_n_anchors_prefix_and_is_reproducible() -> None:
    """First `keep_n` tags stay; same seed -> same output."""
    tags = ["trigger", "1girl", "blue hair", "smile", "outdoors", "sky"]
    a = shuffle_tags(tags, keep_n=2, seed=42)
    b = shuffle_tags(tags, keep_n=2, seed=42)
    assert a[:2] == ["trigger", "1girl"]
    assert sorted(a[2:]) == sorted(tags[2:])
    assert a == b
    # Different seed should (almost certainly) produce a different ordering
    # for the suffix; we don't assert that to avoid theoretical flakes.


def test_drop_tags_zero_is_identity() -> None:
    """drop_rate=0 must round-trip the input unchanged."""
    tags = ["1girl", "blue hair", "smile"]
    assert drop_tags(tags, drop_rate=0.0, seed=1) == tags


def test_drop_tags_preserves_quality_score_safety() -> None:
    """drop_rate=1.0 deletes everything *except* the anchored prefix tags."""
    tags = [
        "score_9",
        "score_8_up",
        "masterpiece",
        "safe",
        "1girl",
        "blue hair",
        "smile",
    ]
    out = drop_tags(tags, drop_rate=1.0, seed=0)
    # Only anchored tags survive; ordering preserved.
    assert out == ["score_9", "score_8_up", "masterpiece", "safe"]


# --------------------------------------------------------------------------- #
# Vocabulary edits
# --------------------------------------------------------------------------- #


def test_add_artist_prefix_idempotent() -> None:
    """Already-prefixed artist tags don't get a second @."""
    artists = {"wlop", "kantoku"}
    out = add_artist_prefix(["@wlop", "kantoku", "1girl"], known_artists=artists)
    assert out == ["@wlop", "@kantoku", "1girl"]
    # Second pass changes nothing.
    assert add_artist_prefix(out, known_artists=artists) == out


def test_inject_quality_skips_existing() -> None:
    """`masterpiece` already present? Don't add a duplicate copy."""
    tags = ["masterpiece", "1girl", "blue hair"]
    out = inject_quality(
        tags,
        quality=["masterpiece", "best quality"],
        score=None,
        safety="safe",
    )
    # `masterpiece` is preserved (not duplicated), `best quality` and `safe`
    # are prepended in score->quality->safety order.
    assert out == ["best quality", "safe", "masterpiece", "1girl", "blue hair"]


def test_inject_quality_with_pony_score_chain() -> None:
    """Score chain materialises before the quality + safety markers."""
    out = inject_quality(
        ["1girl", "blue hair"],
        quality=["masterpiece"],
        score=["score_9", "score_8_up", "score_7_up"],
        safety="safe",
    )
    assert out == [
        "score_9",
        "score_8_up",
        "score_7_up",
        "masterpiece",
        "safe",
        "1girl",
        "blue hair",
    ]


def test_filter_blacklist_case_insensitive() -> None:
    """`NSFW` and `nsfw` collide; both get filtered."""
    out = filter_blacklist(
        ["1girl", "NSFW", "blue hair", "explicit"], blacklist={"nsfw", "explicit"}
    )
    assert out == ["1girl", "blue hair"]


def test_remap_delete_with_empty_target() -> None:
    """Empty target string deletes the key; other rules expand or replace."""
    out = remap_tags(
        ["1girl", "old_tag", "blue hair"],
        rules={"old_tag": "", "1girl": "solo, 1girl"},
    )
    assert out == ["solo", "1girl", "blue hair"]


# --------------------------------------------------------------------------- #
# Pipeline + directory batch
# --------------------------------------------------------------------------- #


def test_pipeline_apply_order() -> None:
    """End-to-end: filter -> remap -> normalise -> artist -> quality -> shuffle."""
    pipeline = CaptionPipeline(
        blacklist={"nsfw"},
        remap={"old_tag": "1girl"},
        known_artists={"wlop"},
        quality=["masterpiece"],
        score=["score_9", "score_8_up"],
        safety="safe",
        shuffle=True,
        keep_n=4,  # anchor the score chain + masterpiece during shuffle
        seed=0,
    )

    out = pipeline.transform_text(
        "Old_Tag, NSFW, BLUE_HAIR, wlop, smile, outdoors"
    )

    parts = split_tags(out)
    # Score chain + masterpiece + safety form the prefix (4 tags).
    assert parts[:4] == ["score_9", "score_8_up", "masterpiece", "safe"]
    # The remaining tags are the post-normalise body — order varies due to
    # the shuffle, but the *set* must match.
    assert set(parts[4:]) == {"1girl", "blue hair", "@wlop", "smile", "outdoors"}
    # NSFW was filtered, old_tag was remapped to 1girl, blue_hair lost its
    # underscore, and wlop got the @ artist prefix.
    assert "nsfw" not in parts


def test_pipeline_default_is_identity_on_clean_text() -> None:
    """A bare CaptionPipeline shouldn't mangle already-clean captions."""
    pipeline = CaptionPipeline()
    assert pipeline.transform_text("1girl, blue hair, smile") == (
        "1girl, blue hair, smile"
    )


def test_transform_directory_writes_back(tmp_path: Path) -> None:
    """Two .txt files in tmp_path get rewritten in place; progress fires."""
    (tmp_path / "a.txt").write_text("Blue_Hair, NSFW, 1girl", encoding="utf-8")
    (tmp_path / "b.txt").write_text("blue hair, 1girl", encoding="utf-8")

    pipeline = CaptionPipeline(blacklist={"nsfw"})
    seen: list[tuple[str, int, int]] = []

    def progress(p: Path, done: int, total: int) -> None:
        seen.append((p.name, done, total))

    written = pipeline.transform_directory(tmp_path, progress=progress)

    # `a.txt` had blue_hair + nsfw -> changed; `b.txt` was already clean -> skipped.
    assert written == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "blue hair, 1girl"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "blue hair, 1girl"
    # Progress fires once per caption file regardless of whether it changed.
    assert [name for name, _, _ in seen] == ["a.txt", "b.txt"]
    assert seen[-1][1:] == (2, 2)


def test_transform_directory_recursive_picks_up_subdirs(tmp_path: Path) -> None:
    """`recursive=True` walks subfolders; `recursive=False` skips them."""
    (tmp_path / "top.txt").write_text("BLUE_HAIR", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "deep.txt").write_text("BLUE_HAIR", encoding="utf-8")

    pipeline = CaptionPipeline()

    flat = pipeline.transform_directory(tmp_path, recursive=False)
    assert flat == 1
    assert (sub / "deep.txt").read_text(encoding="utf-8") == "BLUE_HAIR"

    deep = pipeline.transform_directory(tmp_path, recursive=True)
    assert deep == 1  # only the nested file changed; top.txt was already done.
    assert (sub / "deep.txt").read_text(encoding="utf-8") == "blue hair"


def test_vocab_constants_are_frozen_and_populated() -> None:
    """Cheap sanity: the curated vocabularies are non-empty + immutable."""
    for vocab in (QUALITY_TAGS, SCORE_TAGS, SAFETY_TAGS, META_TAGS, TIME_TAGS):
        assert isinstance(vocab, frozenset)
        assert len(vocab) > 0
    assert "masterpiece" in QUALITY_TAGS
    assert "score_7" in SCORE_TAGS
    assert "score_8_up" in SCORE_TAGS
    assert "safe" in SAFETY_TAGS
