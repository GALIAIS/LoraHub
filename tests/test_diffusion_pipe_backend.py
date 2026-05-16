"""Tests for DiffusionPipeBackend (uses a stubbed diffusion-pipe checkout)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.backends.base import Severity
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import EventType, TrainingEvent


def _make_stub_repo(root: Path) -> Path:
    """Create a fake diffusion-pipe checkout with a no-op `train.py`."""
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        # Mimic the surface of train.py just enough for the parser to fire.
        print("loaded config", flush=True)
        print("Started new epoch: 1", flush=True)
        print("Saving model to directory epoch1", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    (root / "train.py").write_text(stub, encoding="utf-8")
    return root


def _make_recipe(tmp_path: Path, repo: Path, *, arch: str = "sdxl") -> RecipeConfig:
    ckpt = tmp_path / ("model.safetensors" if arch == "sdxl" else "diffusers")
    if arch == "sdxl":
        ckpt.write_bytes(b"")
    else:
        ckpt.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return RecipeConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "type": "diffusion-pipe",
                "sd_scripts_path": str(repo),
                "python_executable": sys.executable,
            },
        }
    )


@pytest.fixture
def backend() -> DiffusionPipeBackend:
    return DiffusionPipeBackend()


def test_supported_archs_excludes_sd15(backend: DiffusionPipeBackend) -> None:
    names = {a.value for a in backend.supported_archs}
    assert {"sdxl", "flux", "sd3"}.issubset(names)
    assert "sd15" not in names


def test_validate_passes_for_good_recipe(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    recipe = _make_recipe(tmp_path, repo, arch="flux")
    issues = backend.validate(recipe)
    errors = [i for i in issues if i.severity is Severity.error]
    assert errors == []


def test_validate_reports_missing_repo(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    recipe = _make_recipe(tmp_path, tmp_path / "missing")
    issues = backend.validate(recipe)
    assert any(
        i.severity is Severity.error and "repo_path" in i.field for i in issues
    )


def test_validate_rejects_sd15_with_pointer_to_kohya(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    ckpt = tmp_path / "sd15.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "d"
    data.mkdir()
    recipe = RecipeConfig.model_validate(
        {
            "base_model": {"arch": "sd15", "checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "type": "diffusion-pipe",
                "sd_scripts_path": str(repo),
                "python_executable": sys.executable,
            },
        }
    )
    issues = backend.validate(recipe)
    arch_errors = [
        i for i in issues if i.severity is Severity.error and i.field == "base_model.arch"
    ]
    assert len(arch_errors) == 1
    assert "kohya" in arch_errors[0].message.lower()


def test_estimate_vram_returns_sane_numbers(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    recipe = _make_recipe(tmp_path, repo, arch="sdxl")
    est = backend.estimate_vram(recipe)
    assert est.total_mib > 0


def test_launch_writes_toml_files_and_runs_subprocess(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    recipe = _make_recipe(tmp_path, repo, arch="sdxl")
    workspace = tmp_path / "ws"

    events: list[TrainingEvent] = []
    handle = backend.launch(recipe, workspace=workspace, on_event=events.append)
    assert handle.pid is not None
    rc = handle.wait(timeout=30)
    assert rc == 0

    # The two TOML files were materialised before launch.
    assert (workspace / "diffusion_pipe.toml").is_file()
    assert (workspace / "dataset.toml").is_file()
    main_toml = (workspace / "diffusion_pipe.toml").read_text(encoding="utf-8")
    assert "[model]" in main_toml
    assert "[adapter]" in main_toml

    # Parser surfaced the stub's epoch + save lines, and the runner
    # always emits a terminal `done` event.
    types = [e.type for e in events]
    assert EventType.epoch_end in types
    assert EventType.checkpoint_saved in types
    assert types[-1] is EventType.done
