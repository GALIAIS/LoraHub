from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lorahub.core.backends._common.dataset_prep import apply_caption_dropouts
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


def test_caption_dropouts_mirror_every_active_subset(tmp_path: Path) -> None:
    """Subset-driven backends must not silently train unfiltered captions."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    for source, caption in ((first, "keep, remove first"), (second, "remove second, keep")):
        source.mkdir()
        (source / "sample.png").write_bytes(b"image")
        (source / "sample.txt").write_text(caption, encoding="utf-8")

    cfg = SimpleNamespace(
        backend=SimpleNamespace(type="ai_toolkit"),
        dataset=SimpleNamespace(
            source=tmp_path / "unused",
            subsets=[
                SimpleNamespace(path=first),
                SimpleNamespace(path=second),
            ],
            caption=SimpleNamespace(drop_tokens=["remove"]),
        ),
    )
    workspace = tmp_path / "workspace"

    apply_caption_dropouts(cfg, workspace)

    first_target = workspace / "captions_sanitized" / "subset-1"
    second_target = workspace / "captions_sanitized" / "subset-2"
    assert cfg.dataset.subsets[0].path == first_target
    assert cfg.dataset.subsets[1].path == second_target
    assert (first_target / "sample.txt").read_text(encoding="utf-8") == "keep, first"
    assert (second_target / "sample.txt").read_text(encoding="utf-8") == "second, keep"
    assert (first / "sample.txt").read_text(encoding="utf-8") == "keep, remove first"
    assert (second / "sample.txt").read_text(encoding="utf-8") == "remove second, keep"


def test_caption_mirror_rejects_custom_target_outside_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("keep, remove", encoding="utf-8")
    workspace = tmp_path / "workspace"

    with pytest.raises(RuntimeError, match="must stay under workspace"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=workspace,
            target_dir=tmp_path / "outside",
        )


def test_caption_mirror_rejects_workspace_as_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("keep, remove", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "keep.txt"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a child"):
        sanitise_dataset(
            source=source,
            drop_tokens=["remove"],
            workspace=workspace,
            target_dir=workspace,
        )

    assert marker.read_text(encoding="utf-8") == "must survive"
