"""Tests for kohya bootstrap (path resolution + health checks)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lorahub.core.backends.kohya.bootstrap import (
    BootstrapError,
    KohyaEnv,
    default_sd_scripts_path,
    resolve,
)


def _make_fake_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # Stub every script the bootstrap probe checks for. Mirrors
    # `lorahub.core.backends.kohya.bootstrap._REQUIRED_SCRIPTS`; tests that
    # need a deliberately incomplete checkout build it inline instead of
    # using this helper.
    for name in (
        "train_network.py",
        "sdxl_train_network.py",
        "sd3_train_network.py",
        "flux_train_network.py",
        "lumina_train_network.py",
        "hunyuan_image_train_network.py",
        "anima_train_network.py",
    ):
        (root / name).write_text("# stub\n", encoding="utf-8")
    return root


def test_resolve_with_explicit_config_path(tmp_path: Path) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    env = resolve(config_path=sd)
    assert isinstance(env, KohyaEnv)
    assert env.sd_scripts_path == sd.resolve()
    assert env.python_executable == Path(sys.executable).absolute()


def test_resolve_with_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    monkeypatch.setenv("LORAHUB_KOHYA_SD_SCRIPTS", str(sd))
    env = resolve()
    assert env.sd_scripts_path == sd.resolve()


def test_config_path_overrides_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd_a = _make_fake_sd_scripts(tmp_path / "a")
    sd_b = _make_fake_sd_scripts(tmp_path / "b")
    monkeypatch.setenv("LORAHUB_KOHYA_SD_SCRIPTS", str(sd_a))
    env = resolve(config_path=sd_b)
    assert env.sd_scripts_path == sd_b.resolve()


def test_missing_path_gives_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LORAHUB_KOHYA_SD_SCRIPTS", raising=False)
    nonexistent = tmp_path / "nope"
    with pytest.raises(BootstrapError, match="not found"):
        resolve(config_path=nonexistent)


def test_incomplete_checkout_rejected(tmp_path: Path) -> None:
    sd = tmp_path / "sd-scripts"
    sd.mkdir()
    (sd / "train_network.py").write_text("# stub", encoding="utf-8")
    with pytest.raises(BootstrapError, match="missing required files"):
        resolve(config_path=sd)


def test_invalid_python_executable_rejected(tmp_path: Path) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    bogus = tmp_path / "no_python"
    with pytest.raises(BootstrapError, match="not found"):
        resolve(config_path=sd, config_python=bogus)


def test_kohya_env_script_returns_absolute_path(tmp_path: Path) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    env = resolve(config_path=sd)
    script = env.script("sdxl_train_network.py")
    assert script.is_absolute()
    assert script.exists()


def test_resolve_picks_up_local_venv_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    venv_python = sd / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("# pretend interpreter\n", encoding="utf-8")
    monkeypatch.setattr(
        "lorahub.core.backends._common.bootstrap.check_python", lambda _python: None
    )

    env = resolve(config_path=sd)
    assert env.python_executable == venv_python.absolute()


def test_resolve_does_not_follow_venv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """venv `bin/python` is usually a symlink to /usr/bin/python — we must
    keep the venv path or the spawned process loses access to the venv's
    site-packages (e.g. `import wandb` fails).
    """
    if sys.platform == "win32":
        pytest.skip("symlink semantics differ on Windows")
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    real_python = tmp_path / "real_python"
    real_python.write_text("# real\n", encoding="utf-8")
    venv_python = sd / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.symlink_to(real_python)
    monkeypatch.setattr(
        "lorahub.core.backends._common.bootstrap.check_python", lambda _python: None
    )

    env = resolve(config_path=sd)
    # We should keep the venv path, not collapse it onto real_python.
    assert env.python_executable == venv_python.absolute()
    assert env.python_executable != real_python.resolve()


def test_config_python_overrides_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    venv_python = sd / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("# venv\n", encoding="utf-8")

    other = tmp_path / "other_python"
    other.write_text("# other\n", encoding="utf-8")
    monkeypatch.setattr(
        "lorahub.core.backends._common.bootstrap.check_python", lambda _python: None
    )

    env = resolve(config_path=sd, config_python=other)
    assert env.python_executable == other.absolute()


def test_default_path_prefers_cwd_local_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd_local = _make_fake_sd_scripts(tmp_path / "sd-scripts")
    monkeypatch.chdir(tmp_path)
    p = default_sd_scripts_path()
    assert p.resolve() == cwd_local.resolve()


def test_default_path_falls_back_to_cwd_local_name_when_neither_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    p = default_sd_scripts_path()
    assert p.name == "sd-scripts"
    assert (tmp_path / "sd-scripts").resolve() == p.resolve()
