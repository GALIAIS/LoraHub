from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.config.caption_filter import sanitise_dataset


def test_caption_mirror_rejects_linked_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("keep, remove", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("must survive", encoding="utf-8")
    try:
        (workspace / "captions_sanitized").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(RuntimeError, match="cannot be a link"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=workspace,
        )

    assert marker.read_text(encoding="utf-8") == "must survive"


def test_caption_mirror_failure_preserves_previous_complete_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("keep, remove", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "captions_sanitized"
    target.mkdir(parents=True)
    previous = target / "previous.txt"
    previous.write_text("complete", encoding="utf-8")

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=workspace,
        )

    assert previous.read_text(encoding="utf-8") == "complete"


def test_caption_mirror_rejects_target_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    marker = source / "keep.txt"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be separate"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=source,
        )

    assert marker.read_text(encoding="utf-8") == "must survive"


def test_caption_mirror_rejects_source_inside_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "captions_sanitized" / "nested-source"
    source.mkdir(parents=True)
    marker = source / "keep.txt"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be separate"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=workspace,
        )

    assert marker.read_text(encoding="utf-8") == "must survive"
