"""Tests for the lorahub CLI."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import yaml
from typer.testing import CliRunner

from lorahub.cli.main import app

runner = CliRunner()


def _make_stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        print("epoch 1/1", flush=True)
        print("saving checkpoint: out.safetensors", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    (root / "train_network.py").write_text(stub, encoding="utf-8")
    (root / "sdxl_train_network.py").write_text(stub, encoding="utf-8")
    return root


def _make_recipe_yaml(tmp_path: Path, sd_scripts: Path) -> Path:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    recipe = {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {
            "sd_scripts_path": str(sd_scripts),
            "python_executable": sys.executable,
        },
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.dump(recipe), encoding="utf-8")
    return path


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "lorahub" in result.stdout


def test_validate_passes(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe_yaml(tmp_path, sd)
    result = runner.invoke(app, ["validate", str(recipe)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_fails_for_missing_sd_scripts(tmp_path: Path) -> None:
    recipe = _make_recipe_yaml(tmp_path, tmp_path / "missing")
    result = runner.invoke(app, ["validate", str(recipe)])
    assert result.exit_code == 1


def test_info_renders_summary(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe_yaml(tmp_path, sd)
    result = runner.invoke(app, ["info", str(recipe)])
    assert result.exit_code == 0
    assert "sdxl" in result.stdout
    assert "VRAM" in result.stdout


def test_train_runs_end_to_end(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    recipe = _make_recipe_yaml(tmp_path, sd)
    ws = tmp_path / "ws"
    result = runner.invoke(app, ["train", str(recipe), "--workspace", str(ws)])
    assert result.exit_code == 0, result.stdout
    assert "training complete" in result.stdout
    assert (ws / "events.jsonl").exists()


def test_init_scaffolds_recipe(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my_lora"])
    assert result.exit_code == 0
    assert (tmp_path / "my_lora.yaml").exists()


def test_init_rejects_unknown_template(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "x", "--template", "no_such_template"])
    assert result.exit_code == 1
