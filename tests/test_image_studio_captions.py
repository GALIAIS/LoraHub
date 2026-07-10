"""Tests for the captions router — vocab / find-replace / inject-trigger / blacklist."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

# Bootstrap app namespace before touching sub-routers.
from lorahub.api import app as _app_module  # noqa: F401
from lorahub.api.routers.image_studio.captions import (
    BlacklistRequest,
    FindReplaceRequest,
    InjectTriggerRequest,
    captions_blacklist,
    captions_find_replace,
    captions_inject_trigger,
    captions_vocab,
)


@pytest.fixture(autouse=True)
def _allow_test_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))


def _seed(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    fixtures = {
        "a": "1girl, looking at viewer, smile",
        "b": "1girl, Looking At Viewer, long hair",
        "c": "smile, blush, blue eyes",
    }
    for stem, cap in fixtures.items():
        Image.new("RGB", (1024, 1024)).save(d / f"{stem}.png")
        (d / f"{stem}.txt").write_text(cap, encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #


def test_vocab_lowercases_and_counts(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    res = captions_vocab(str(d), recursive=False, limit=50)
    vocab = {r["tag"]: r["count"] for r in res["vocab"]}
    # Lowercased — `Looking At Viewer` and `looking at viewer` merge.
    assert vocab["looking at viewer"] == 2
    assert vocab["1girl"] == 2
    assert vocab["smile"] == 2
    assert vocab["long hair"] == 1
    assert res["files_seen"] == 3


def test_caption_workflows_skip_linked_sidecars(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private, smile", encoding="utf-8")
    (d / "a.txt").unlink()
    try:
        (d / "a.txt").symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = captions_blacklist(
        BlacklistRequest(dataset_path=str(d), tags=["smile"]),
    )

    assert result["removed_count"] == 1
    assert outside.read_text(encoding="utf-8") == "private, smile"


# --------------------------------------------------------------------------- #
# Find-replace
# --------------------------------------------------------------------------- #


def test_find_replace_dry_run_returns_diffs_no_writes(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    res = captions_find_replace(
        FindReplaceRequest(
            dataset_path=str(d),
            pattern="smile",
            replacement="grin",
            dry_run=True,
        ),
    )
    assert res["dry_run"] is True
    assert res["matched_files"] == 2
    assert res["matched_count"] == 2
    # Disk unchanged.
    assert "smile" in (d / "a.txt").read_text(encoding="utf-8")
    assert "grin" not in (d / "a.txt").read_text(encoding="utf-8")


def test_find_replace_apply_writes_changes(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    res = captions_find_replace(
        FindReplaceRequest(
            dataset_path=str(d),
            pattern="smile",
            replacement="grin",
            dry_run=False,
        ),
    )
    assert res["dry_run"] is False
    assert "grin" in (d / "a.txt").read_text(encoding="utf-8")
    assert "smile" not in (d / "a.txt").read_text(encoding="utf-8")


def test_find_replace_empty_replacement_drops_tag(tmp_path: Path) -> None:
    """Empty replacement = blacklist single tag via find-replace."""
    d = _seed(tmp_path)
    captions_find_replace(
        FindReplaceRequest(
            dataset_path=str(d),
            pattern="smile",
            replacement="",
            dry_run=False,
        ),
    )
    text_a = (d / "a.txt").read_text(encoding="utf-8")
    assert "smile" not in text_a
    assert "1girl" in text_a  # other tags preserved


def test_find_replace_regex_partial_match(tmp_path: Path) -> None:
    """Regex mode lets us replace inside tags ('1girl' → '1 girl')."""
    d = _seed(tmp_path)
    captions_find_replace(
        FindReplaceRequest(
            dataset_path=str(d),
            pattern=r"^1girl$",
            replacement="1 girl",
            is_regex=True,
            dry_run=False,
        ),
    )
    assert "1 girl" in (d / "a.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Inject trigger
# --------------------------------------------------------------------------- #


def test_inject_trigger_prepend_skips_existing(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    # First inject: all 3 captions get the trigger.
    res = captions_inject_trigger(
        InjectTriggerRequest(
            dataset_path=str(d),
            trigger_word="@charA",
        ),
    )
    assert res["injected_count"] == 3
    assert (d / "a.txt").read_text(encoding="utf-8").startswith("@charA,")

    # Second inject is a no-op (skip_existing default).
    res2 = captions_inject_trigger(
        InjectTriggerRequest(
            dataset_path=str(d),
            trigger_word="@charA",
        ),
    )
    assert res2["injected_count"] == 0
    assert res2["skipped_count"] == 3


def test_inject_trigger_append(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    captions_inject_trigger(
        InjectTriggerRequest(
            dataset_path=str(d),
            trigger_word="trigger",
            position="append",
        ),
    )
    assert (d / "a.txt").read_text(encoding="utf-8").rstrip().endswith("trigger")


# --------------------------------------------------------------------------- #
# Blacklist
# --------------------------------------------------------------------------- #


def test_blacklist_removes_listed_tags(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    res = captions_blacklist(
        BlacklistRequest(
            dataset_path=str(d),
            tags=["smile", "looking at viewer"],
        ),
    )
    assert res["removed_count"] == 4  # 2 smile + 2 looking-at-viewer
    assert "smile" not in (d / "a.txt").read_text(encoding="utf-8")
    assert "1girl" in (d / "a.txt").read_text(encoding="utf-8")


def test_blacklist_does_not_partial_match(tmp_path: Path) -> None:
    """Blacklisting "smile" must not strip "smiley"."""
    d = tmp_path / "ds"
    d.mkdir()
    Image.new("RGB", (1024, 1024)).save(d / "x.png")
    (d / "x.txt").write_text("smile, smiley face", encoding="utf-8")

    captions_blacklist(
        BlacklistRequest(dataset_path=str(d), tags=["smile"]),
    )
    assert (d / "x.txt").read_text(encoding="utf-8").strip() == "smiley face"
