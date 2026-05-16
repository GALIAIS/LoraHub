"""Tests for the Anima caption formatter and dataset transformer."""

from __future__ import annotations

from pathlib import Path

from lorahub.core.dataset.anima import (
    AnimaCaptionFormatter,
    AnimaDatasetTransformer,
    parse_caption,
)


# --------------------------------------------------------------------------- #
# AnimaCaptionFormatter.format()
# --------------------------------------------------------------------------- #


def test_format_basic_tag_order() -> None:
    """Sections render in Anima's documented order regardless of construction order."""
    fmt = AnimaCaptionFormatter(
        general=["smile", "brown hair", "solo"],
        artist=["nnn yryr"],
        series=["yuru yuri"],
        character=["oomuro sakurako"],
        subject=["1girl"],
        safety=["safe"],
        meta=["highres"],
        year=["year 2025", "newest"],
        score=["score_5"],
        quality=["normal quality"],
    )
    out = fmt.format()
    expected = (
        "normal quality, score_5, year 2025, newest, highres, safe, 1girl, "
        "oomuro sakurako, yuru yuri, @nnn yryr, smile, brown hair, solo"
    )
    assert out == expected


def test_format_underscore_to_space_except_score() -> None:
    """Tags get ``_`` -> space, but ``score_N`` keeps its underscore intact."""
    fmt = AnimaCaptionFormatter(
        score=["score_7"],
        character=["oomuro_sakurako"],
        general=["brown_hair"],
    )
    out = fmt.format()
    assert "oomuro sakurako" in out
    assert "brown hair" in out
    assert "score_7" in out
    assert "oomuro_sakurako" not in out
    assert "brown_hair" not in out


def test_format_artist_at_prefix_auto() -> None:
    """Artist tags pick up an ``@`` prefix exactly once."""
    fmt = AnimaCaptionFormatter(artist=["nnn yryr", "@kantoku"])
    out = fmt.format()
    assert "@nnn yryr" in out
    assert "@kantoku" in out
    # No double-prefix.
    assert "@@" not in out


def test_format_dataset_tag_layout() -> None:
    """Non-anime subsets get a multi-line layout: dataset / NL / tags."""
    fmt = AnimaCaptionFormatter(
        dataset_tag="ye-pop",
        natural_language="A girl in a yellow dress walking in the park.",
        subject=["1girl"],
        general=["yellow dress", "park"],
        safety=["safe"],
    )
    out = fmt.format()
    lines = out.split("\n")
    assert lines[0] == "ye-pop"
    assert lines[1] == "A girl in a yellow dress walking in the park."
    assert lines[2].startswith("safe, 1girl,")


# --------------------------------------------------------------------------- #
# parse_caption()
# --------------------------------------------------------------------------- #


_FULL_EXAMPLE = (
    "year 2025, newest, normal quality, score_5, highres, safe, 1girl, "
    "oomuro sakurako, @nnn yryr, smile, brown hair, hat, solo"
)


def test_parse_caption_round_trip() -> None:
    """parse → format yields the same tag set, ordered by Anima layout."""
    parsed = parse_caption(_FULL_EXAMPLE)
    rendered = parsed.format()

    # Tag set is preserved (order may legitimately change because `parse_caption`
    # cannot tell character from series without a vocab).
    original_tags = {t.strip() for t in _FULL_EXAMPLE.split(",")}
    rendered_tags = {t.strip() for t in rendered.split(",")}
    assert original_tags == rendered_tags

    # Quality / safety / score / year all hop to the front of the line.
    head = rendered.split(",")[:6]
    head = [t.strip() for t in head]
    assert head[0] == "normal quality"
    assert head[1] == "score_5"
    # Year tokens come next (order within a section preserved).
    assert "year 2025" in head
    assert "newest" in head
    assert "highres" in head
    assert "safe" in head


def test_parse_classifies_quality_safety_year() -> None:
    """parse_caption assigns each known token to the correct bucket."""
    parsed = parse_caption(
        "masterpiece, year 2025, safe, 1girl, blue hair, score_9, official art"
    )
    assert parsed.quality == ["masterpiece"]
    assert parsed.year == ["year 2025"]
    assert parsed.safety == ["safe"]
    assert parsed.score == ["score_9"]
    assert parsed.meta == ["official art"]
    assert parsed.subject == ["1girl"]
    assert "blue hair" in parsed.general


# --------------------------------------------------------------------------- #
# AnimaDatasetTransformer
# --------------------------------------------------------------------------- #


def test_transform_directory_writes_back(tmp_path: Path) -> None:
    """With overwrite=True, every .txt caption is rewritten in Anima layout."""
    (tmp_path / "a.txt").write_text("1girl, blue_hair, masterpiece", encoding="utf-8")
    (tmp_path / "b.txt").write_text("safe, 1boy, smile", encoding="utf-8")

    tx = AnimaDatasetTransformer(
        default_quality=["best quality"],
        default_safety="safe",
    )
    written = tx.transform_directory(tmp_path, overwrite=True)
    assert written == 2

    a = (tmp_path / "a.txt").read_text(encoding="utf-8")
    # Quality default merges in alongside the original quality tag.
    assert a.startswith("masterpiece, best quality")
    # Default safety injected since the source had none.
    assert "safe" in a
    # Underscore normalised on a non-score tag.
    assert "blue hair" in a
    assert "blue_hair" not in a

    b = (tmp_path / "b.txt").read_text(encoding="utf-8")
    # Existing safety wasn't duplicated; subject still present.
    assert b.count("safe") == 1
    assert "1boy" in b


def test_transform_directory_skip_existing(tmp_path: Path) -> None:
    """Default overwrite=False leaves existing captions untouched."""
    original = "1girl, blue_hair"
    (tmp_path / "a.txt").write_text(original, encoding="utf-8")

    tx = AnimaDatasetTransformer(default_quality=["masterpiece"])
    written = tx.transform_directory(tmp_path)  # overwrite default = False
    assert written == 0
    # File contents intact, including the underscore that would normally go.
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == original


def test_transform_directory_handles_score_underscore(tmp_path: Path) -> None:
    """score_N preserves its underscore through a full round-trip."""
    (tmp_path / "a.txt").write_text("score_5, 1girl, blue_hair", encoding="utf-8")

    tx = AnimaDatasetTransformer(default_safety=None, default_quality=None)
    written = tx.transform_directory(tmp_path, overwrite=True)
    assert written == 1

    out = (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert "score_5" in out
    # General tags got their underscore stripped, but score_N didn't.
    assert "blue hair" in out
    assert "blue_hair" not in out


def test_transform_directory_recursive(tmp_path: Path) -> None:
    """Recursive walk picks up captions in subdirectories."""
    sub = tmp_path / "char_a"
    sub.mkdir()
    (sub / "img.txt").write_text("1girl, blue_hair", encoding="utf-8")

    tx = AnimaDatasetTransformer(default_safety="safe")
    # Non-recursive misses the file entirely.
    assert tx.transform_directory(tmp_path, overwrite=True) == 0
    assert tx.transform_directory(tmp_path, recursive=True, overwrite=True) == 1

    out = (sub / "img.txt").read_text(encoding="utf-8")
    assert "blue hair" in out
    assert "safe" in out


def test_transform_directory_progress_callback(tmp_path: Path) -> None:
    """Per-file progress hook fires once per written caption."""
    (tmp_path / "a.txt").write_text("1girl", encoding="utf-8")
    (tmp_path / "b.txt").write_text("1boy", encoding="utf-8")

    seen: list[str] = []
    tx = AnimaDatasetTransformer(default_safety="safe")
    tx.transform_directory(
        tmp_path, overwrite=True, progress=lambda p: seen.append(p.name)
    )
    assert sorted(seen) == ["a.txt", "b.txt"]
