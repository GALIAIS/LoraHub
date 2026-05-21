"""Tests for the intake router — preflight, local-path, from-dataset."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from lorahub.api import app as _app_module  # noqa: F401

from lorahub.api.routers.image_studio.intake import (
    FromDatasetRequest,
    LocalPathRequest,
    PreflightRequest,
    intake_from_dataset,
    intake_local_path,
    intake_preflight,
)


def _make_dataset(tmp_path: Path, name: str = "ds") -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


def _save_img(d: Path, name: str, *, color: tuple[int, int, int] = (200, 100, 50), size=(512, 512)):
    p = d / name
    Image.new("RGB", size, color=color).save(p)
    return p


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def test_preflight_classifies_new_files(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(10, 20, 30))
    _save_img(src, "b.png", color=(200, 100, 50))
    dst = _make_dataset(tmp_path, "dst")  # empty

    res = intake_preflight(
        PreflightRequest(dataset_path=str(dst), source_path=str(src)),
    )
    assert res["candidate_count"] == 2
    assert res["new_count"] == 2
    assert res["duplicate_existing_count"] == 0


def test_preflight_detects_existing_duplicates(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    img_src = _save_img(src, "a.png", color=(80, 90, 100))
    dst = _make_dataset(tmp_path, "dst")
    # Same image already in dst.
    Image.open(img_src).save(dst / "already.png")

    res = intake_preflight(
        PreflightRequest(dataset_path=str(dst), source_path=str(src)),
    )
    assert res["new_count"] == 0
    assert res["duplicate_existing_count"] == 1


def test_preflight_detects_within_batch_duplicates(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(33, 44, 55))
    # b is a re-save of the same pixels → same phash.
    Image.open(src / "a.png").save(src / "b.png")
    dst = _make_dataset(tmp_path, "dst")

    res = intake_preflight(
        PreflightRequest(dataset_path=str(dst), source_path=str(src)),
    )
    assert res["candidate_count"] == 2
    assert res["new_count"] == 1
    assert res["duplicate_within_batch_count"] == 1


# --------------------------------------------------------------------------- #
# Local path
# --------------------------------------------------------------------------- #


def test_local_path_imports_with_sidecars(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(10, 20, 30))
    (src / "a.txt").write_text("caption a", encoding="utf-8")
    _save_img(src, "b.png", color=(200, 100, 50))
    # b has no sidecar — must still import.
    dst = _make_dataset(tmp_path, "dst")

    res = intake_local_path(
        LocalPathRequest(dataset_path=str(dst), source_path=str(src)),
    )
    assert res["imported_count"] == 2
    assert (dst / "a.png").is_file()
    assert (dst / "a.txt").is_file()
    assert (dst / "b.png").is_file()
    # Source untouched.
    assert (src / "a.png").is_file()


def test_local_path_disambiguates_collisions(tmp_path: Path) -> None:
    """A file already named ``a.png`` in dst shouldn't be overwritten."""
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(10, 20, 30))
    dst = _make_dataset(tmp_path, "dst")
    _save_img(dst, "a.png", color=(99, 99, 99))  # different image, same name

    res = intake_local_path(
        LocalPathRequest(
            dataset_path=str(dst),
            source_path=str(src),
            skip_duplicates=False,  # otherwise the import is skipped on phash mismatch
        ),
    )
    assert res["imported_count"] == 1
    # Dst now has both.
    assert (dst / "a.png").is_file()
    assert (dst / "a-2.png").is_file()


def test_local_path_skips_duplicates(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    img_src = _save_img(src, "a.png", color=(50, 60, 70))
    dst = _make_dataset(tmp_path, "dst")
    # Image with same content already in dst.
    Image.open(img_src).save(dst / "different_name.png")

    res = intake_local_path(
        LocalPathRequest(
            dataset_path=str(dst),
            source_path=str(src),
            skip_duplicates=True,
        ),
    )
    assert res["imported_count"] == 0
    assert res["skipped_count"] == 1


def test_local_path_move_semantics(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(11, 22, 33))
    (src / "a.txt").write_text("c", encoding="utf-8")
    dst = _make_dataset(tmp_path, "dst")

    intake_local_path(
        LocalPathRequest(
            dataset_path=str(dst),
            source_path=str(src),
            move=True,
            skip_duplicates=False,
        ),
    )
    assert (dst / "a.png").is_file()
    assert (dst / "a.txt").is_file()
    # Source removed.
    assert not (src / "a.png").exists()
    assert not (src / "a.txt").exists()


# --------------------------------------------------------------------------- #
# From-dataset
# --------------------------------------------------------------------------- #


def test_from_dataset_glob_filter(tmp_path: Path) -> None:
    """``pattern`` filters by relative path under the source dataset."""
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "portrait_1.png", color=(10, 20, 30))
    _save_img(src, "portrait_2.png", color=(50, 60, 70))
    _save_img(src, "landscape_1.png", color=(90, 100, 110))
    dst = _make_dataset(tmp_path, "dst")

    res = intake_from_dataset(
        FromDatasetRequest(
            dataset_path=str(dst),
            source_dataset_path=str(src),
            pattern="portrait*",
            skip_duplicates=False,
        ),
    )
    assert res["candidate_count"] == 2
    assert res["imported_count"] == 2
    assert (dst / "portrait_1.png").is_file()
    assert not (dst / "landscape_1.png").exists()


def test_from_dataset_rejects_self_copy(tmp_path: Path) -> None:
    import pytest
    from fastapi import HTTPException

    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png")

    with pytest.raises(HTTPException) as exc:
        intake_from_dataset(
            FromDatasetRequest(
                dataset_path=str(src),
                source_dataset_path=str(src),
            ),
        )
    assert exc.value.status_code == 400


def test_from_dataset_brings_sidecars(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "src")
    _save_img(src, "a.png", color=(10, 20, 30))
    (src / "a.txt").write_text("caption", encoding="utf-8")
    dst = _make_dataset(tmp_path, "dst")

    intake_from_dataset(
        FromDatasetRequest(
            dataset_path=str(dst),
            source_dataset_path=str(src),
            pattern="*",
            skip_duplicates=False,
        ),
    )
    assert (dst / "a.txt").is_file()
    assert (dst / "a.txt").read_text(encoding="utf-8") == "caption"
