"""Unit tests for the tag-groups loader + resolver.

Smoke-coverage of the YAML schema, the cross-group uniqueness check,
and the vocab-resolution pass that drops tags below ``min_freq``.
"""

from __future__ import annotations

import textwrap

import pytest

from library.captioning import tag_groups as tg


def _write(tmp_path, body: str):
    p = tmp_path / "tag_groups.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_minimal(tmp_path):
    p = _write(tmp_path, """
        version: 1
        eye_color:
          mode: softmax_when_solo
          tags:
            - blue eyes
            - red eyes
    """)
    g = tg.load_groups(p)
    assert len(g.groups) == 1
    assert g.tag_to_group == {"blue eyes": "eye_color", "red eyes": "eye_color"}


def test_dup_tag_rejected(tmp_path):
    p = _write(tmp_path, """
        version: 1
        a:
          mode: softmax
          tags: [x]
        b:
          mode: softmax
          tags: [x]
    """)
    with pytest.raises(ValueError, match="listed under both"):
        tg.load_groups(p)


def test_bad_mode_rejected(tmp_path):
    p = _write(tmp_path, """
        version: 1
        a:
          mode: nope
          tags: [x]
    """)
    with pytest.raises(ValueError, match="mode='nope'"):
        tg.load_groups(p)


def test_resolve_drops_missing(tmp_path):
    """Tags below min_freq disappear from vocab; resolve_groups drops them silently."""
    p = _write(tmp_path, """
        version: 1
        eye_color:
          mode: softmax_when_solo
          escape: [heterochromia]
          tags: [blue eyes, red eyes, fictional eyes]
    """)
    g = tg.load_groups(p)
    # Mock vocab: only "blue eyes" + "red eyes" survived (no escape, no
    # fictional eyes). The resolver must not fail; both unknown names go
    # into ``dropped``.
    vocab = {"blue eyes": 0, "red eyes": 1}
    resolved, dropped = tg.resolve_groups(g, vocab)
    assert len(resolved) == 1
    eye = resolved[0]
    assert eye.tag_indices == (0, 1)
    assert eye.tag_names == ("blue eyes", "red eyes")
    assert eye.escape_indices == ()
    assert dropped == {"fictional eyes": "not_in_vocab", "heterochromia": "not_in_vocab"}


def test_round_trip(tmp_path):
    p = _write(tmp_path, """
        version: 1
        eye_color:
          mode: softmax_when_solo
          description: "test"
          escape: [heterochromia]
          tags: [blue eyes, red eyes]
    """)
    g = tg.load_groups(p)
    g2 = tg.from_dict(g.to_dict())
    assert len(g2.groups) == len(g.groups)
    eye1 = g.by_name("eye_color")
    eye2 = g2.by_name("eye_color")
    assert eye1.tags == eye2.tags
    assert eye1.escape == eye2.escape
    assert eye1.mode == eye2.mode
    assert eye1.description == eye2.description


def test_empty_tags_allowed(tmp_path):
    """Group with no listed tags still parses — useful for committing the
    YAML schema before the corpus catches up."""
    p = _write(tmp_path, """
        version: 1
        future_group:
          mode: softmax_when_solo
          tags: []
    """)
    g = tg.load_groups(p)
    assert len(g.groups) == 1
    assert g.by_name("future_group").tags == ()
