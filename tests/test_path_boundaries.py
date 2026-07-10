from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from lorahub.api import paths
from lorahub.api.helpers import _config_path
from lorahub.api.dataset_files import (
    resolve_caption_path,
    resolve_dataset_directory,
    resolve_dataset_file,
)
from lorahub.api.jobs_helpers.resume_dispatch import (
    ResumeTargetInvalid,
    _validate_resume_target,
)
from lorahub.core.config.schema import TrainingConfig


def test_dataset_writes_require_registered_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    dataset = allowed / "dataset"
    dataset.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(allowed))

    assert resolve_dataset_directory(str(dataset)) == dataset.resolve()
    with pytest.raises(ValueError, match="outside writable roots"):
        resolve_dataset_directory(str(outside))


def test_config_path_rejects_link_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    real = configs / "real.yaml"
    real.write_text("backend: {}\n", encoding="utf-8")
    alias = configs / "alias.yaml"
    try:
        alias.symlink_to(real)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    monkeypatch.setenv("LORAHUB_configs_dir", str(configs))

    with pytest.raises(HTTPException) as exc_info:
        _config_path("alias")

    assert exc_info.value.status_code == 400
    assert real.is_file()


def test_dataset_file_writes_require_registered_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "inside.png"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(allowed))

    assert resolve_dataset_file(str(inside)) == inside.resolve()
    with pytest.raises(ValueError, match="outside writable roots"):
        resolve_dataset_file(str(outside))


def test_dataset_file_writes_reject_link_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    real = allowed / "real.png"
    real.write_bytes(b"real")
    alias = allowed / "alias.png"
    try:
        alias.symlink_to(real)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(allowed))

    with pytest.raises(ValueError, match="cannot be a link"):
        resolve_dataset_file(str(alias))


def test_caption_writes_require_existing_image_in_writable_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "inside.png"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(allowed))

    assert resolve_caption_path(str(inside), writable=True) == allowed / "inside.txt"
    with pytest.raises(ValueError, match="outside writable roots"):
        resolve_caption_path(str(outside), writable=True)
    with pytest.raises(ValueError, match="not found"):
        resolve_caption_path(str(allowed / "missing.png"), writable=True)


def test_destructive_run_paths_cannot_escape_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    workspace = runs / "job"
    workspace.mkdir()
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    assert paths.resolve_run_path(workspace) == workspace.resolve()
    with pytest.raises(ValueError, match="must be under"):
        paths.resolve_run_path(tmp_path / "other")


@pytest.mark.parametrize(
    "variant_name",
    ["../other-job", "nested/variant", r"nested\variant", ".", "..", ""],
)
def test_sweep_variant_workspace_stays_below_sweep_root(
    variant_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    sweep_root = runs / "sweep"
    sweep_root.mkdir(parents=True)
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    with pytest.raises(ValueError, match="single path component"):
        paths.resolve_sweep_variant_path(sweep_root, variant_name)


def test_sweep_variant_workspace_accepts_direct_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    sweep_root = runs / "sweep"
    sweep_root.mkdir(parents=True)
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    assert paths.resolve_sweep_variant_path(sweep_root, "rank-32") == (
        sweep_root / "rank-32"
    ).resolve()


def test_resume_state_cannot_escape_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    state_dir = tmp_path / "outside-state"
    state_dir.mkdir()
    (state_dir / "optimizer.bin").write_bytes(b"state")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"model")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(checkpoint)},
            "dataset": {"source": str(dataset)},
            "resume": {"resume_from": str(state_dir)},
        }
    )

    with pytest.raises(ResumeTargetInvalid, match="not owned by the source job"):
        _validate_resume_target(cfg, source_roots=(runs,))


def test_model_paths_require_project_or_explicit_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-models"
    outside = tmp_path / "outside"
    monkeypatch.setattr(paths, "project_root", lambda: project)
    monkeypatch.setenv("LORAHUB_MODELS_ROOT", str(external))

    assert paths.resolve_model_path("vendor/model") == (
        project / "models" / "vendor" / "model"
    ).resolve()
    assert paths.resolve_model_path(external / "model") == (external / "model").resolve()
    with pytest.raises(ValueError, match="outside configured model roots"):
        paths.resolve_model_path(outside / "model")
