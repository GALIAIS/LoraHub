from pathlib import Path

from lorahub.core.backends._common.bootstrap import ensure_models_link


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
