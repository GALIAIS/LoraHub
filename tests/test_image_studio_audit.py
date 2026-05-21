"""Smoke tests for the dataset audit router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Bootstrap the app namespace before touching any sub-router. The
# routers package's __init__ pulls backends.py which back-references
# lorahub.api.app; importing app first warms the chain so the routers
# package finishes loading without the circular import error.
from lorahub.api import app as _app_module  # noqa: F401

from lorahub.api.routers.image_studio.audit import (
    ScanRequest,
    audit_report,
    audit_scan,
)


def _make_dataset(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    # 4 valid images at varied sizes
    for i, size in enumerate([(1024, 1024), (768, 1024), (256, 256), (1536, 1024)]):
        img = Image.new("RGB", size, color=(i * 50, 100, 150))
        img.save(d / f"img{i}.png")
        # Caption for the first 3 only — the 4th tests "no_caption" issue.
        if i < 3:
            (d / f"img{i}.txt").write_text(
                "1girl, looking at viewer, smile",
                encoding="utf-8",
            )
    # Add a corrupt file (zero bytes pretending to be a png)
    (d / "broken.png").write_bytes(b"")
    return d


def test_audit_scan_basic_metrics(tmp_path: Path) -> None:
    """Scanning a small dataset yields a populated report cache."""
    from lorahub.api.routers.image_studio.audit import audit_scan, ScanRequest

    ds = _make_dataset(tmp_path)
    report = audit_scan(
        ScanRequest(dataset_path=str(ds), recursive=False, blur_check=False),
    )

    # 4 valid + 1 broken = 5 total scanned. Broken counted as image_count
    # via stat() then bumped back out by the verify(). The current impl
    # increments image_count first, then continues so corrupt rows still
    # land. Check both code paths.
    assert report["image_count"] >= 4
    assert report["captioned_count"] == 3
    # Corrupt issue surfaced.
    kinds = {iss["kind"] for iss in report["issues"]}
    assert "corrupt" in kinds
    # Tiny (256x256, long edge < 512) issue surfaced.
    assert "tiny" in kinds
    # 1 image without caption.
    assert "no_caption" in kinds

    # Histograms cover the expected long edges.
    res_labels = {b["bucket"] for b in report["resolution_histogram"]}
    assert "1024-1279" in res_labels
    assert "1536-2047" in res_labels

    # Cache landed on disk.
    cache = ds / ".workbench" / "audit.json"
    assert cache.is_file()
    parsed = json.loads(cache.read_text(encoding="utf-8"))
    assert parsed["image_count"] == report["image_count"]


def test_audit_scan_trigger_word(tmp_path: Path) -> None:
    """``trigger_word`` flags missing-trigger images."""
    from lorahub.api.routers.image_studio.audit import audit_scan, ScanRequest

    ds = tmp_path / "ds"
    ds.mkdir()
    Image.new("RGB", (1024, 1024)).save(ds / "a.png")
    (ds / "a.txt").write_text("@charA, 1girl, smile", encoding="utf-8")
    Image.new("RGB", (1024, 1024)).save(ds / "b.png")
    (ds / "b.txt").write_text("1girl, smile", encoding="utf-8")  # no trigger

    report = audit_scan(
        ScanRequest(
            dataset_path=str(ds),
            recursive=False,
            blur_check=False,
            trigger_word="@charA",
        ),
    )
    assert report["trigger_word_hits"] == 1
    miss = [i for i in report["issues"] if i["kind"] == "missing_trigger"]
    assert len(miss) == 1
    assert "b.png" in miss[0]["path"]


def test_audit_scan_blur_detection(tmp_path: Path) -> None:
    """Solid-color image has ~0 Laplacian variance → flagged blurry."""
    from lorahub.api.routers.image_studio.audit import audit_scan, ScanRequest

    ds = tmp_path / "ds"
    ds.mkdir()
    # Pure solid: Laplacian is trivially 0 → flagged.
    Image.new("RGB", (1024, 1024), color=(128, 128, 128)).save(ds / "solid.png")

    # Sharp image: high-contrast random noise → high Laplacian variance.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(ds / "noisy.png")

    report = audit_scan(
        ScanRequest(dataset_path=str(ds), recursive=False, blur_check=True),
    )
    blurry = [i for i in report["issues"] if i["kind"] == "blurry"]
    assert len(blurry) == 1
    assert "solid.png" in blurry[0]["path"]


def test_audit_report_404_when_no_cache(tmp_path: Path) -> None:
    """Asking for a report on an unscanned dataset returns 404."""
    from fastapi import HTTPException

    import pytest

    from lorahub.api.routers.image_studio.audit import audit_report

    ds = tmp_path / "fresh"
    ds.mkdir()
    with pytest.raises(HTTPException) as exc_info:
        audit_report(str(ds))
    assert exc_info.value.status_code == 404


def test_audit_report_returns_cache(tmp_path: Path) -> None:
    """After scan, report endpoint returns the cached payload."""
    from lorahub.api.routers.image_studio.audit import audit_scan, audit_report, ScanRequest

    ds = _make_dataset(tmp_path)
    scan = audit_scan(
        ScanRequest(dataset_path=str(ds), recursive=False, blur_check=False),
    )
    rep = audit_report(str(ds))
    assert rep["image_count"] == scan["image_count"]
    assert rep["captioned_count"] == scan["captioned_count"]


def test_audit_tag_vocab_top_50(tmp_path: Path) -> None:
    """Tag vocab is the top-50 most-frequent normalised tags."""
    from lorahub.api.routers.image_studio.audit import audit_scan, ScanRequest

    ds = tmp_path / "ds"
    ds.mkdir()
    Image.new("RGB", (1024, 1024)).save(ds / "a.png")
    (ds / "a.txt").write_text("1girl, looking at viewer, smile", encoding="utf-8")
    Image.new("RGB", (1024, 1024)).save(ds / "b.png")
    (ds / "b.txt").write_text("1girl, Looking At Viewer", encoding="utf-8")

    report = audit_scan(
        ScanRequest(dataset_path=str(ds), recursive=False, blur_check=False),
    )
    vocab = {row["tag"]: row["count"] for row in report["tag_vocab"]}
    # Lowercased + dedup'd.
    assert vocab.get("1girl") == 2
    assert vocab.get("looking at viewer") == 2
    assert vocab.get("smile") == 1
