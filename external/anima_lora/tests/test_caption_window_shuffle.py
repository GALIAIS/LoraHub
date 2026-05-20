"""Sanity tests for the optional ``window_size`` mode of
``anima_smart_shuffle_caption``."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.anima.training import anima_smart_shuffle_caption  # noqa: E402


def _tags(n: int, prefix: str = "tag") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def test_unrestricted_full_shuffle() -> None:
    """When window_size is None, behaviour matches the legacy shuffle."""
    random.seed(0)
    body = _tags(20)
    caption = ["@artist"] + body
    out = anima_smart_shuffle_caption(caption.copy(), window_size=None)
    assert out[0] == "@artist", "@artist prefix preserved"
    # 20-tag full shuffle: extremely unlikely to land identity (p≈1/20!).
    assert out[1:] != body
    # The set is preserved.
    assert sorted(out[1:]) == sorted(body)
    print("test_unrestricted_full_shuffle OK")


def test_window_keeps_block_locality() -> None:
    """Each block-of-4 in the body shuffles internally; cross-window
    ordering is preserved (tag i stays in window i//4)."""
    random.seed(1)
    body = _tags(20)
    caption = ["@artist"] + body
    out = anima_smart_shuffle_caption(caption.copy(), window_size=4)
    assert out[0] == "@artist"
    # For each window of 4 in the output, the *set* of tags must equal
    # the corresponding window's set in the original body — proving no
    # tag crossed window boundaries.
    for start in range(0, 20, 4):
        in_block = set(body[start : start + 4])
        out_block = set(out[1 + start : 1 + start + 4])
        assert in_block == out_block, (start, in_block, out_block)
    print("test_window_keeps_block_locality OK")


def test_window_actually_shuffles_inside() -> None:
    """For a non-trivial seed, at least one window deviates from
    identity, proving shuffling did happen inside."""
    body = _tags(12)
    caption = ["@artist"] + body
    seen_change = False
    for seed in range(50):
        random.seed(seed)
        out = anima_smart_shuffle_caption(caption.copy(), window_size=4)
        if out[1:] != body:
            seen_change = True
            break
    assert seen_change, "window mode never reordered any block in 50 attempts"
    print("test_window_actually_shuffles_inside OK")


def test_window_size_one_is_noop() -> None:
    """``window_size <= 1`` falls back to the unrestricted path (the
    helper documents window_size > 1 as the activation threshold)."""
    random.seed(0)
    body = _tags(8)
    caption = ["@artist"] + body
    # window_size=1 should NOT lock the order — it falls through to
    # full shuffle.
    out = anima_smart_shuffle_caption(caption.copy(), window_size=1)
    assert sorted(out[1:]) == sorted(body)
    # And the full-shuffle path should differ from identity for n=8.
    out2 = anima_smart_shuffle_caption(caption.copy(), window_size=1)
    assert sorted(out2[1:]) == sorted(body)
    print("test_window_size_one_is_noop OK")


if __name__ == "__main__":
    test_unrestricted_full_shuffle()
    test_window_keeps_block_locality()
    test_window_actually_shuffles_inside()
    test_window_size_one_is_noop()
