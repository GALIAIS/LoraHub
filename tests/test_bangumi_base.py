"""Tests for the BangumiBase fetcher (network-mocked)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from lorahub.core.dataset.sources import bangumi_base


def _make_zip(tmp_path: Path, names: list[str]) -> Path:
    p = tmp_path / "dataset.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for n in names:
            zf.writestr(n, b"\x89PNG\r\n\x1a\nfake")
    return p


class _FakeHfApi:
    """Minimal stand-in for HfApi used by list_characters tests."""

    def __init__(self, files: list[str] | None = None, seen: dict | None = None):
        self._files = files or []
        self._seen = seen

    def list_repo_files(self, repo_id: str, repo_type: str | None = None) -> list[str]:
        if self._seen is not None:
            self._seen["repo_id"] = repo_id
        return self._files


def test_list_characters_filters_to_numeric_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    files = [
        "README.md",
        "0/dataset.zip",
        "0/preview_1.png",
        "1/dataset.zip",
        "12/dataset.zip",
        "all.zip",
        ".gitattributes",
    ]
    monkeypatch.setattr(
        bangumi_base, "_make_hf_api", lambda: _FakeHfApi(files=files)
    )

    chars = bangumi_base.list_characters("azurlaneanime")

    assert chars == ["0", "1", "12"]


def test_repo_id_short_form_is_namespaced(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        bangumi_base, "_make_hf_api", lambda: _FakeHfApi(seen=seen)
    )

    bangumi_base.list_characters("azurlaneanime")
    assert seen["repo_id"] == "BangumiBase/azurlaneanime"

    bangumi_base.list_characters("Custom/repo")
    assert seen["repo_id"] == "Custom/repo"


def test_fetch_character_unpacks_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_zip = _make_zip(tmp_path, ["a.png", "b.jpg", "c.webp", "notes.txt"])
    monkeypatch.setattr(
        bangumi_base, "hf_download", lambda **_: str(fake_zip)
    )
    monkeypatch.setattr(bangumi_base, "_read_dataset_license", lambda _r: "mit")

    out = tmp_path / "out"
    result = bangumi_base.fetch_character("azurlaneanime", "3", out)

    assert result.image_count == 3
    assert result.license == "mit"
    assert (out / "a.png").exists()
    assert (out / "b.jpg").exists()
    assert (out / "c.webp").exists()
    assert not (out / "notes.txt").exists()


def test_fetch_character_seeds_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_zip = _make_zip(tmp_path, ["a.png", "b.png"])
    monkeypatch.setattr(bangumi_base, "hf_download", lambda **_: str(fake_zip))
    monkeypatch.setattr(bangumi_base, "_read_dataset_license", lambda _r: None)

    out = tmp_path / "out"
    bangumi_base.fetch_character("azurlaneanime", "3", out, seed_captions=True)

    assert (out / "a.txt").exists()
    assert (out / "a.txt").read_text(encoding="utf-8") == ""
    assert (out / "b.txt").exists()


def test_fetch_character_respects_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_zip = _make_zip(tmp_path, [f"{i:03d}.png" for i in range(10)])
    monkeypatch.setattr(bangumi_base, "hf_download", lambda **_: str(fake_zip))
    monkeypatch.setattr(bangumi_base, "_read_dataset_license", lambda _r: None)

    out = tmp_path / "out"
    result = bangumi_base.fetch_character("x", "0", out, limit=4)

    assert result.image_count == 4
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["000.png", "001.png", "002.png", "003.png"]


def test_fetch_character_progress_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_zip = _make_zip(tmp_path, ["a.png"])
    monkeypatch.setattr(bangumi_base, "hf_download", lambda **_: str(fake_zip))
    monkeypatch.setattr(bangumi_base, "_read_dataset_license", lambda _r: None)

    msgs: list[str] = []
    bangumi_base.fetch_character(
        "x", "0", tmp_path / "out", on_progress=msgs.append
    )
    assert any("resolving" in m for m in msgs)
    assert any("unpacking" in m for m in msgs)


def test_download_failure_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_):  # type: ignore[no-untyped-def]
        raise RuntimeError("hf is down")

    monkeypatch.setattr(bangumi_base, "hf_download", boom)
    with pytest.raises(bangumi_base.BangumiBaseError, match="failed to download"):
        bangumi_base.fetch_character("x", "0", tmp_path / "out")
