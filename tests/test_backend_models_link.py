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
    link.mkdir()

    monkeypatch.setattr(anima_models, "default_repo_path", lambda: repo)
    monkeypatch.setattr(anima_models, "models_root", lambda: link)
    monkeypatch.setattr(anima_models, "ensure_models_link", lambda _repo: link)

    anima_models._link_anima_models_dir()  # must not raise


def test_link_anima_models_dir_rejects_backend_local_models_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "anima_lora"
    link = repo / "models"
    root_models = tmp_path / "root-models"
    link.mkdir(parents=True)
    root_models.mkdir()
    (link / "local.safetensors").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(anima_models, "default_repo_path", lambda: repo)
    monkeypatch.setattr(anima_models, "models_root", lambda: root_models)
    monkeypatch.setattr(anima_models, "ensure_models_link", lambda _repo: link)

    with pytest.raises(OSError, match="does not point"):
        anima_models._link_anima_models_dir()
