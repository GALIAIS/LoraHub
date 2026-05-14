"""Tests for KohyaBackend (uses a stubbed sd-scripts checkout)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.backends.base import Severity
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import EventType, TrainingEvent


def _make_stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        print("loading model", flush=True)
        print("steps:   1%|          | 1/2 [00:01<00:01,  1.00s/it, avr_loss=0.5]", flush=True)
        print("epoch 1/1", flush=True)
        print("saving checkpoint: out.safetensors", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    (root / "train_network.py").write_text(stub, encoding="utf-8")
    (root / "sdxl_train_network.py").write_text(stub, encoding="utf-8")
    return root


def _make_recipe(tmp_path: Path, sd_scripts: Path) -> RecipeConfig:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return RecipeConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "sd_scripts_path": str(sd_scripts),
                "python_executable": sys.executable,
            },
        }
    )


@pytest.fixture
def backend() -> KohyaBackend:
    return KohyaBackend()


def test_supported_archs_cover_main_models(backend: KohyaBackend) -> None:
    names = {a.value for a in backend.supported_archs}
    assert {"sdxl", "sd15", "flux", "sd3"}.issubset(names)


def test_validate_passes_for_good_recipe(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe(tmp_path, sd)
    issues = backend.validate(recipe)
    errors = [i for i in issues if i.severity is Severity.error]
    assert errors == []


def test_validate_reports_missing_sd_scripts(tmp_path: Path, backend: KohyaBackend) -> None:
    recipe = _make_recipe(tmp_path, tmp_path / "missing")
    issues = backend.validate(recipe)
    assert any(
        i.severity is Severity.error and "sd_scripts" in i.field for i in issues
    )


def test_estimate_vram_returns_sane_numbers(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe(tmp_path, sd)
    est = backend.estimate_vram(recipe)
    assert est.total_mib > 0
    assert 1.0 <= est.total_gib <= 32.0


def test_launch_runs_to_completion(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe(tmp_path, sd)

    events: list[TrainingEvent] = []
    handle = backend.launch(recipe, workspace=tmp_path / "ws", on_event=events.append)
    assert handle.pid is not None
    rc = handle.wait(timeout=30)
    assert rc == 0

    types = [e.type for e in events]
    assert EventType.step in types
    assert EventType.epoch_end in types
    assert EventType.checkpoint_saved in types
    assert types[-1] is EventType.done
