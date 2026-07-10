"""Tests for the ship router — lint / export / save-as."""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from lorahub.api import app as _app_module  # noqa: F401
from lorahub.api.routers.image_studio import ship as ship_module
from lorahub.api.routers.image_studio.audit import ScanRequest, audit_scan
from lorahub.api.routers.image_studio.ship import (
    ExportRequest,
    SaveAsRequest,
    ship_export,
    ship_lint,
    ship_save_as,
)


@pytest.fixture(autouse=True)
def _allow_test_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))


def _drain_streaming(resp) -> bytes:
    """Collect bytes from a StreamingResponse body iterator (sync wrapper)."""
    async def _consume():
        chunks = []
        async for c in resp.body_iterator:
            if isinstance(c, str):
                c = c.encode()
            chunks.append(c)
        return b"".join(chunks)
    return asyncio.run(_consume())


def _seed(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    for i, size in enumerate([(1024, 1024), (1024, 768), (768, 1024)]):
        Image.new("RGB", size).save(d / f"img{i}.png")
        (d / f"img{i}.txt").write_text(f"caption {i}", encoding="utf-8")
    return d


def _symlink_or_skip(target: Path, link: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Lint
# --------------------------------------------------------------------------- #


def test_lint_stale_when_no_audit_cache(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    res = ship_lint(str(d))
    assert res["stale"] is True
    assert res["ready"] is False
    assert "stale_reason" in res


def test_lint_ready_after_clean_scan(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    audit_scan(ScanRequest(dataset_path=str(d), recursive=False, blur_check=False))
    res = ship_lint(str(d))
    assert res["ready"] is True
    assert res["stale"] is False
    assert res["blockers"] == 0
    assert res["image_count"] == 3


def test_lint_warn_on_no_caption(tmp_path: Path) -> None:
    """Missing caption surfaces as a warning, not a blocker."""
    d = tmp_path / "ds"
    d.mkdir()
    Image.new("RGB", (1024, 1024)).save(d / "a.png")  # no .txt
    audit_scan(ScanRequest(dataset_path=str(d), recursive=False, blur_check=False))
    res = ship_lint(str(d))
    assert res["ready"] is True  # missing caption is warn, not block
    assert any(i["code"] == "no_caption" for i in res["issues"])
    assert res["warnings"] >= 1


def test_lint_stale_when_image_added_after_scan(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    audit_scan(ScanRequest(dataset_path=str(d), recursive=False, blur_check=False))
    # Add a new image after the audit.
    Image.new("RGB", (1024, 1024)).save(d / "new.png")
    res = ship_lint(str(d))
    assert res["stale"] is True
    assert res["ready"] is False


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_streams_zip(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    resp = ship_export(ExportRequest(dataset_path=str(d)))
    blob = _drain_streaming(resp)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
    # Three images + three sidecars.
    assert "img0.png" in names
    assert "img0.txt" in names
    assert len([n for n in names if n.endswith(".png")]) == 3


def test_export_excludes_workbench_by_default(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    # Create some workbench junk.
    (d / ".workbench" / "audit.json").parent.mkdir(parents=True)
    (d / ".workbench" / "audit.json").write_text("{}", encoding="utf-8")
    (d / ".workbench" / "backups" / "img0.png").parent.mkdir(parents=True)
    (d / ".workbench" / "backups" / "img0.png").write_bytes(b"backup")

    resp = ship_export(ExportRequest(dataset_path=str(d)))
    blob = _drain_streaming(resp)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
    # No .workbench paths in the archive.
    assert not any(".workbench" in n for n in names)


def test_export_include_backups_flag(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    (d / ".workbench" / "backups").mkdir(parents=True)
    (d / ".workbench" / "backups" / "img0.png").write_bytes(b"old")

    resp = ship_export(
        ExportRequest(dataset_path=str(d), include_backups=True),
    )
    blob = _drain_streaming(resp)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
    assert any("backups/img0.png" in n.replace("\\", "/") for n in names)


def test_export_excludes_metadata_when_requested(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    (d / "dataset.json").write_text('{"name":"ds"}', encoding="utf-8")

    resp = ship_export(ExportRequest(dataset_path=str(d), include_meta=False))
    blob = _drain_streaming(resp)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert "dataset.json" not in zf.namelist()


def test_export_skips_linked_files(tmp_path: Path) -> None:
    d = _seed(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    _symlink_or_skip(outside, d / "linked.png")

    resp = ship_export(ExportRequest(dataset_path=str(d)))
    blob = _drain_streaming(resp)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert "linked.png" not in zf.namelist()


# --------------------------------------------------------------------------- #
# Save-as
# --------------------------------------------------------------------------- #


def test_save_as_copies_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(registry))
    d = _seed(registry)
    res = ship_save_as(
        SaveAsRequest(source_path=str(d), new_name="my-copy"),
    )
    assert res["images_copied"] == 3
    assert (Path(res["path"]) / "img0.png").is_file()
    assert (Path(res["path"]) / "img0.txt").is_file()
    assert (Path(res["path"]) / "dataset.json").is_file()


def test_save_as_rejects_existing_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(registry))
    d = _seed(registry)
    ship_save_as(SaveAsRequest(source_path=str(d), new_name="dup"))
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        ship_save_as(SaveAsRequest(source_path=str(d), new_name="dup"))
    assert exc.value.status_code == 409


def test_save_as_filters_by_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(registry))
    d = _seed(registry)
    res = ship_save_as(
        SaveAsRequest(
            source_path=str(d),
            new_name="subset",
            paths=[str(d / "img0.png"), str(d / "img2.png")],
        ),
    )
    assert res["images_copied"] == 2
    assert (Path(res["path"]) / "img0.png").is_file()
    assert (Path(res["path"]) / "img2.png").is_file()
    assert not (Path(res["path"]) / "img1.png").exists()
    # Sidecars came along with the matched images.
    assert (Path(res["path"]) / "img0.txt").is_file()


def test_save_as_preserves_canonical_metadata(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(registry))
    d = _seed(registry)
    (d / "dataset.json").write_text(
        '{"name":"ds","description":"source","triggerWord":"hero"}',
        encoding="utf-8",
    )

    result = ship_save_as(SaveAsRequest(source_path=str(d), new_name="copy"))

    metadata = (Path(result["path"]) / "dataset.json").read_text(encoding="utf-8")
    assert '"description": "source (copied from ds)"' in metadata
    assert '"triggerWord": "hero"' in metadata


def test_save_as_rolls_back_partial_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(registry))
    d = _seed(registry)
    real_copy = ship_module.shutil.copy2
    calls = 0

    def fail_second_copy(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        return real_copy(source, target)

    monkeypatch.setattr(ship_module.shutil, "copy2", fail_second_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        ship_save_as(SaveAsRequest(source_path=str(d), new_name="incomplete"))

    assert not (registry / "incomplete").exists()
    assert not list(registry.glob(".dataset-importing-*"))


def test_save_as_rejects_cross_platform_reserved_name(tmp_path: Path) -> None:
    from fastapi import HTTPException

    d = _seed(tmp_path)
    with pytest.raises(HTTPException) as exc:
        ship_save_as(SaveAsRequest(source_path=str(d), new_name="CON"))
    assert exc.value.status_code == 400
