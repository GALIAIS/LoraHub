"""Tests for DiffusionPipeBackend (uses a stubbed diffusion-pipe checkout)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.backends.base import Severity
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.config.schema import TrainingConfig
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


def _make_recipe(tmp_path: Path, repo: Path, *, arch: str = "sdxl") -> TrainingConfig:
    ckpt = tmp_path / ("model.safetensors" if arch == "sdxl" else "diffusers")
    if arch == "sdxl":
        ckpt.write_bytes(b"")
    else:
        ckpt.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return TrainingConfig.model_validate(
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
    assert "sd2" not in names


def test_supported_archs_cover_dp_only_models(backend: DiffusionPipeBackend) -> None:
    """The full dp matrix (Wan, HunyuanVideo, Chroma, ...) is reachable."""
    names = {a.value for a in backend.supported_archs}
    assert {
        "wan",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "anima",
        "hunyuan_image",
        "lumina",
        "flux2",
        "z_image",
        "ernie_image",
    }.issubset(names)


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
    recipe = TrainingConfig.model_validate(
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


@pytest.mark.parametrize(
    "arch",
    [
        # full 23-arch matrix; dp supports a superset for vram estimation
        # purposes even though sd15/sd2 fail validation downstream.
        "sd15",
        "sd2",
        "sdxl",
        "sd3",
        "flux",
        "flux2",
        "lumina",
        "anima",
        "hunyuan_image",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "wan",
        "z_image",
        "ernie_image",
    ],
)
def test_estimate_vram_covers_every_arch(
    tmp_path: Path, backend: DiffusionPipeBackend, arch: str
) -> None:
    """Every arch yields a positive estimate, even ones dp would refuse to launch."""
    from lorahub.core.backends.base import VRAMEstimate

    repo = _make_stub_repo(tmp_path / "dp")
    # `_make_recipe` creates either a flat .safetensors or a diffusers dir
    # depending on the arch token; reuse it so checkpoint shape matches what
    # the dp recipe expects.
    recipe = _make_recipe(tmp_path, repo, arch="sdxl")
    cfg = recipe.model_copy(
        update={
            "base_model": recipe.base_model.model_copy(update={"arch": arch}),
        }
    )
    est = backend.estimate_vram(cfg)
    assert isinstance(est, VRAMEstimate)
    assert est.total_mib > 0


def test_estimate_vram_activations_scale_with_batch_size(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    """Doubling batch_size doubles the activations component."""
    repo = _make_stub_repo(tmp_path / "dp")
    recipe = _make_recipe(tmp_path, repo, arch="sdxl")
    recipe = recipe.model_copy(update={"gradient_checkpointing": False})

    bs1 = backend.estimate_vram(
        recipe.model_copy(
            update={"schedule": recipe.schedule.model_copy(update={"batch_size": 1})}
        )
    )
    bs2 = backend.estimate_vram(
        recipe.model_copy(
            update={"schedule": recipe.schedule.model_copy(update={"batch_size": 2})}
        )
    )
    assert bs2.activations_mib == 2 * bs1.activations_mib


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
