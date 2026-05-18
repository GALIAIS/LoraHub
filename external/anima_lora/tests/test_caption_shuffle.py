"""Caption-shuffle boundary tests.

Pins down the three corner cases the inline ``tag.startswith("@")`` predicate
got wrong:

1. ``@ @`` (booru ``@_@`` eye-shape, space-form) must not trigger the artist
   boundary.
2. Multi-artist captions (``@artist1, @artist2, …``) must protect the full
   leading handle run.
3. The ``@no-artist`` sentinel must participate in the boundary but be
   stripped from every cache variant (including v0).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.anima.training import (  # noqa: E402
    NO_ARTIST_SENTINEL,
    _is_artist_tag,
    anima_smart_shuffle_caption,
    find_anima_prefix_end,
    strip_no_artist_sentinel,
)


# ----- predicate ----------------------------------------------------------


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("@sincos", True),
        ("@sumiyao (amam)", True),
        ("@no-artist", True),
        ("@", False),  # one char, no handle body
        ("@ @", False),  # booru @_@ eye-shape, space-form
        ("@ ", False),  # trailing space → not artist
        ("blue hair", False),
        ("1girl", False),
        ("", False),
    ],
)
def test_is_artist_tag(tag, expected):
    assert _is_artist_tag(tag) is expected


# ----- prefix-end walk ----------------------------------------------------


def test_prefix_end_single_artist_first():
    assert find_anima_prefix_end(["@sincos", "blue hair", "1girl"]) == 1


def test_prefix_end_no_artist():
    # The case @no-artist exists to fix: zero protection without a sentinel.
    assert find_anima_prefix_end(["blue hair", "1girl"]) == 0


def test_prefix_end_multi_artist_collab():
    assert find_anima_prefix_end(["@artist1", "@artist2", "@artist3", "blue hair"]) == 3


def test_prefix_end_leading_content_then_artist():
    # Old behavior preserved: leading non-@ tags extend into the prefix.
    assert find_anima_prefix_end(["solo", "1girl", "@sincos", "blue hair"]) == 3


def test_prefix_end_eye_shape_no_artist():
    # @ @ alone must NOT trigger the boundary.
    assert find_anima_prefix_end(["solo", "@ @", "blue hair"]) == 0


def test_prefix_end_eye_shape_before_real_artist():
    # @ @ falls through; @sincos is the real boundary, and @ @ rides along
    # in the prefix as a leading-content tag (same as any other non-@ tag).
    assert find_anima_prefix_end(["@ @", "solo", "@sincos", "blue hair"]) == 3


def test_prefix_end_sentinel_acts_as_artist():
    assert find_anima_prefix_end([NO_ARTIST_SENTINEL, "blue hair"]) == 1


# ----- strip helper -------------------------------------------------------


def test_strip_no_artist_sentinel_removes_all_occurrences():
    tags = ["a", NO_ARTIST_SENTINEL, "b", NO_ARTIST_SENTINEL, "c"]
    assert strip_no_artist_sentinel(tags) == ["a", "b", "c"]


def test_strip_no_artist_sentinel_no_op_when_absent():
    tags = ["@sincos", "blue hair"]
    assert strip_no_artist_sentinel(tags) == tags


# ----- shuffle integration -----------------------------------------------


def test_shuffle_preserves_prefix_order_with_multi_artist():
    random.seed(0)
    tags = ["@artist1", "@artist2", "@artist3", "a", "b", "c", "d"]
    out = anima_smart_shuffle_caption(tags.copy())
    # Prefix run preserved in order.
    assert out[:3] == ["@artist1", "@artist2", "@artist3"]
    # Suffix has same multiset, possibly reordered.
    assert sorted(out[3:]) == sorted(["a", "b", "c", "d"])


def test_shuffle_eye_shape_does_not_anchor_prefix():
    random.seed(0)
    tags = ["solo", "@ @", "blue hair", "red eyes"]
    out = anima_smart_shuffle_caption(tags.copy())
    # No real artist → split_idx=0 → everything is shuffleable.
    # We can't assert a specific order (random), only that the multiset is
    # preserved and the input wasn't accidentally locked into the prefix.
    assert sorted(out) == sorted(tags)


def test_shuffle_keeps_sentinel_in_output_for_caller_strip():
    # Contract: shuffle does NOT strip the sentinel (so split_idx stays
    # meaningful for the caller's dropout protection). The caller must strip
    # before tokenization.
    random.seed(0)
    tags = [NO_ARTIST_SENTINEL, "blue hair", "1girl"]
    out = anima_smart_shuffle_caption(tags.copy())
    assert NO_ARTIST_SENTINEL in out
    assert out[0] == NO_ARTIST_SENTINEL  # prefix order preserved


# ----- variant generator (TE cache path) ---------------------------------


def _gen_variants(*args, **kwargs):
    from preprocess.cache_text_embeddings import _generate_caption_variants

    return _generate_caption_variants(*args, **kwargs)


def test_variants_strip_sentinel_from_v0():
    random.seed(0)
    out = _gen_variants(
        f"{NO_ARTIST_SENTINEL}, blue hair, 1girl", num_variants=3, tag_dropout_rate=0.0
    )
    assert all(NO_ARTIST_SENTINEL not in v for v in out)
    # v0 retains the original tag order (sentinel removed).
    assert out[0] == "blue hair, 1girl"


def test_variants_v0_byte_identical_when_no_sentinel():
    # Existing datasets must not see whitespace renormalization in v0.
    raw = "@sincos,blue hair  ,1girl"
    out = _gen_variants(raw, num_variants=1, tag_dropout_rate=0.0)
    assert out[0] == raw


def test_variants_strip_sentinel_after_dropout():
    random.seed(0)
    # High dropout rate to exercise the kept-list path.
    out = _gen_variants(
        f"{NO_ARTIST_SENTINEL}, a, b, c, d, e",
        num_variants=8,
        tag_dropout_rate=0.5,
    )
    for v in out:
        assert NO_ARTIST_SENTINEL not in v


def test_variants_multi_artist_protected_from_dropout():
    random.seed(0)
    # Force every dropable tag to roll the dice; with rate=1.0 every
    # non-prefix tag is dropped. All three artist handles must survive.
    out = _gen_variants(
        "@artist1, @artist2, @artist3, a, b, c",
        num_variants=4,
        tag_dropout_rate=1.0,
    )
    # v0 untouched.
    assert out[0] == "@artist1, @artist2, @artist3, a, b, c"
    for v in out[1:]:
        toks = [t.strip() for t in v.split(",")]
        assert "@artist1" in toks
        assert "@artist2" in toks
        assert "@artist3" in toks
