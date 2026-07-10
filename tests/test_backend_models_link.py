from pathlib import Path

import pytest

from lorahub.core.backends._common.bootstrap import ensure_models_link
from lorahub.core.backends.anima_lora import models as anima_models


def test_ensure_models_link_points_backend_models_at_root(tmp_path: Path) -> None:
    repo = tmp_path / "backend"
    target = tmp_path / "models"
    repo.mkdir()

    link = ensure_models_link(repo, target)

    assert link.samefile(target)


def test_ensure_models_link_keeps_populated_backend_models(tmp_path: Path) -> None:
    repo = tmp_path / "backend"
    existing = repo / "models"
    existing.mkdir(parents=True)
    (existing / "local.safetensors").write_text("keep", encoding="utf-8")

    link = ensure_models_link(repo, tmp_path / "root-models")

    assert link == existing
    assert (existing / "local.safetensors").is_file()


def test_link_anima_models_dir_raises_when_link_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ensure_models_link returns the *target* (not the link) when link
    creation fails. _link_anima_models_dir must surface that as an OSError
    so download_models doesn't report success while anima's hardcoded
    models/... paths can't resolve (Docker --user / read-only checkout)."""
    repo = tmp_path / "anima_lora"
    repo.mkdir()
    target = tmp_path / "models"
    target.mkdir()

    monkeypatch.setattr(anima_models, "default_repo_path", lambda: repo)
    # Simulate the failure fallback: returns target, not the link.
    monkeypatch.setattr(anima_models, "ensure_models_link", lambda _repo: target)

    with pytest.raises(OSError, match="Failed to link"):
        anima_models._link_anima_models_dir()


def test_link_anima_models_dir_silent_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ensure_models_link returns the link (success path), no raise."""
    repo = tmp_path / "anima_lora"
    repo.mkdir()
    link = repo / "models"
    for subdir, filename, _repo_path in anima_models._TARGETS:
        destination = link / subdir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checkpoint")

    monkeypatch.setattr(anima_models, "default_repo_path", lambda: repo)
    monkeypatch.setattr(anima_models, "ensure_models_link", lambda _repo: link)

    anima_models._link_anima_models_dir()  # must not raise


def test_anima_download_cleanup_preserves_unknown_split_files(tmp_path: Path) -> None:
    split_root = tmp_path / "split_files"
    custom = split_root / "custom" / "user-model.safetensors"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"user data")
    for subdir, _filename, _repo_path in anima_models._TARGETS:
        (split_root / subdir).mkdir(parents=True, exist_ok=True)

    anima_models._remove_empty_download_dirs(tmp_path)

    assert custom.read_bytes() == b"user data"
