"""Tests for the lorahub CLI."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lorahub.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _force_english_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CLI display language to English for these tests.

    The user-facing default is Simplified Chinese, but these test
    assertions match against the English wording (``"valid"``,
    ``"no jobs"``, ``"host:"`` …). Setting ``LORAHUB_LANG=en`` covers
    both the import-time language pin (``_pre_parse_lang``) and any
    subsequent ``set_lang`` calls.
    """
    monkeypatch.setenv("LORAHUB_LANG", "en")
    from lorahub.cli._i18n import set_lang  # noqa: PLC0415

    set_lang("en")


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
    for name in (
        "train_network.py",
        "sdxl_train_network.py",
        "sd3_train_network.py",
        "flux_train_network.py",
        "lumina_train_network.py",
        "hunyuan_image_train_network.py",
        "anima_train_network.py",
    ):
        (root / name).write_text(stub, encoding="utf-8")
    return root


def _make_config_yaml(tmp_path: Path, sd_scripts: Path) -> Path:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    config = {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {
            "sd_scripts_path": str(sd_scripts),
            "python_executable": sys.executable,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "lorahub" in result.stdout


def test_validate_passes(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config_yaml(tmp_path, sd)
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_fails_for_missing_sd_scripts(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path, tmp_path / "missing")
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 1


def test_validate_uses_configured_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "anima.yaml"
    config.write_text(
        yaml.dump(
            {
                "base_model": {"checkpoint": str(ckpt), "arch": "anima"},
                "dataset": {"source": str(data)},
                "schedule": {"epochs": 1, "batch_size": 1},
                "sampling": {"enabled": False},
                "backend": {"type": "anima_lora", "animaLora": {}},
            }
        ),
        encoding="utf-8",
    )

    from lorahub.core.backends.anima_lora import backend as anima_backend

    monkeypatch.setattr(anima_backend.AnimaLoraBackend, "validate", lambda *_: [])
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 0, result.stdout
    assert "sd-scripts" not in result.stdout


def test_info_renders_summary(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config_yaml(tmp_path, sd)
    result = runner.invoke(app, ["info", str(config)])
    assert result.exit_code == 0
    assert "sdxl" in result.stdout
    assert "VRAM" in result.stdout


def test_train_runs_end_to_end(tmp_path: Path) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config_yaml(tmp_path, sd)
    ws = tmp_path / "ws"
    result = runner.invoke(app, ["train", str(config), "--workspace", str(ws)])
    assert result.exit_code == 0, result.stdout
    assert "training complete" in result.stdout
    assert (ws / "events.jsonl").exists()


def test_init_scaffolds_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my_lora"])
    assert result.exit_code == 0
    assert (tmp_path / "my_lora.yaml").exists()


def test_init_rejects_unknown_template(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "x", "--template", "no_such_template"])
    assert result.exit_code == 1


def test_sweep_dry_run_lists_variants(tmp_path: Path) -> None:
    """`lorahub sweep ... --dry-run` prints each variant name and its diff
    without touching disk."""
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config_yaml(tmp_path, sd)
    output_root = tmp_path / "configs-out"

    result = runner.invoke(
        app,
        [
            "sweep",
            str(config),
            "--axis",
            "network.rank=16,32",
            "--axis",
            "schedule.epochs=1,2",
            "--output-dir",
            str(output_root),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "4 variant" in result.stdout
    # Default base output.name is "lora_output"; template renders {base}-{i:03d}.
    assert "lora_output-001" in result.stdout
    assert "lora_output-004" in result.stdout
    # Dry-run must not write any files.
    assert not output_root.exists()


# --------------------------------------------------------------------------- #
# B9 — sub-app surface (jobs / sweeps / system)
# --------------------------------------------------------------------------- #


def test_jobs_help_lists_subcommands() -> None:
    """`lorahub jobs --help` must surface ls/cancel/kill/resume/rerun/show."""
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("ls", "cancel", "kill", "resume", "rerun", "show"):
        assert cmd in result.stdout, f"missing subcommand {cmd!r}"


def test_jobs_ls_empty_store(tmp_path: Path, monkeypatch) -> None:
    """`jobs ls` against an empty store prints `no jobs` and exits 0."""
    # Point the store at a fresh dir so the test doesn't touch the user's
    # actual jobs.sqlite. ``paths.py`` honours LORAHUB_HOME first, so
    # pinning that to ``tmp_path`` redirects ``runs_dir()`` to the
    # tmp tree — the historical LORAHUB_DATA_DIR has no consumer.
    monkeypatch.setenv("LORAHUB_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from lorahub.api import paths as paths_module  # noqa: PLC0415

    paths_module._resolved = None  # type: ignore[attr-defined]
    try:
        result = runner.invoke(app, ["jobs", "ls"])
    finally:
        paths_module._resolved = None  # type: ignore[attr-defined]
    assert result.exit_code == 0, result.stdout
    assert "no jobs" in result.stdout


def test_sweeps_help_mentions_submit() -> None:
    result = runner.invoke(app, ["sweeps", "--help"])
    assert result.exit_code == 0
    assert "submit" in result.stdout
    assert "ls" in result.stdout


def test_system_help_mentions_gpu() -> None:
    result = runner.invoke(app, ["system", "--help"])
    assert result.exit_code == 0
    assert "gpu" in result.stdout
    assert "info" in result.stdout


def test_system_info_runs() -> None:
    """`system info` should print without needing GPUs or external state."""
    result = runner.invoke(app, ["system", "info"])
    assert result.exit_code == 0, result.stdout
    assert "host:" in result.stdout
    assert "CPU:" in result.stdout


def test_service_restart_reuses_previous_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """`service restart` without --port should keep the last daemon bind."""
    from lorahub.api.runtime_bind import RuntimeBind
    from lorahub.cli import service as service_mod

    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(service_mod, "read_runtime_bind", lambda: RuntimeBind("0.0.0.0", 19090, 123))
    monkeypatch.setattr(service_mod, "_read_pid", lambda: None)
    monkeypatch.setattr(
        service_mod,
        "start",
        lambda *, host, port, foreground: calls.append((host, port)),
    )

    result = runner.invoke(app, ["service", "restart"])

    assert result.exit_code == 0, result.stdout
    assert calls == [("0.0.0.0", 19090)]


def test_service_daemon_uses_pythonw_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorahub.cli import service as service_mod

    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_text("", encoding="utf-8")
    pythonw.write_text("", encoding="utf-8")

    monkeypatch.setattr(service_mod.sys, "platform", "win32")
    monkeypatch.setattr(service_mod, "_venv_python", lambda: python)

    assert service_mod._daemon_python() == pythonw


def test_service_resolves_windows_listener_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from lorahub.cli import service as service_mod

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def cmdline(self) -> list[str]:
            return ["pythonw.exe", "-m", "uvicorn", "lorahub.api.app:app"]

    fake_psutil = SimpleNamespace(
        net_connections=lambda kind: [
            SimpleNamespace(laddr=SimpleNamespace(port=18765), pid=456)
        ],
        Process=FakeProcess,
    )

    monkeypatch.setattr(service_mod.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert service_mod._resolve_daemon_pid(123, 18765) == 456


def test_service_stop_terminates_windows_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.cli import service as service_mod

    calls: list[tuple[int, float]] = []

    monkeypatch.setattr(service_mod.sys, "platform", "win32")
    monkeypatch.setattr(service_mod, "_read_pid", lambda: 123)
    monkeypatch.setattr(
        service_mod,
        "_terminate_windows_process_tree",
        lambda pid, timeout: calls.append((pid, timeout)),
    )
    monkeypatch.setattr(service_mod, "clear_runtime_bind", lambda *, keep_bind: None)

    service_mod.stop(timeout=1.5)

    assert calls == [(123, 1.5)]


# --------------------------------------------------------------------------- #
# Localisation — verify zh mode actually swaps the strings.
# --------------------------------------------------------------------------- #


def test_zh_locale_renders_chinese_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LORAHUB_LANG=zh`` must produce the Simplified Chinese help text.

    Re-invokes the CLI in a fresh subprocess-style runner — the autouse
    fixture above pins ``LORAHUB_LANG=en``, so we override it here and
    re-pin via ``set_lang('zh')`` to flip the dictionary back to zh
    for this single test. Sanity-check by asserting both a Commands-
    header phrase ("自检") and a sub-command phrase ("校验 config").
    """
    monkeypatch.setenv("LORAHUB_LANG", "zh")
    from lorahub.cli._i18n import set_lang  # noqa: PLC0415

    set_lang("zh")
    try:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.stdout
        # Top-level description (rendered via t("app.help")).
        assert "LoRA 训练工作台" in result.stdout
        # Sub-command short helps (rendered via t("validate.help") etc.).
        assert "校验 config" in result.stdout
        assert "守护进程" in result.stdout
    finally:
        # Don't leak the locale flip into other tests in the module.
        set_lang("en")


def test_zh_locale_renders_chinese_jobs_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``jobs ls`` against an empty store says "暂无任务" in zh mode."""
    monkeypatch.setenv("LORAHUB_HOME", str(tmp_path))
    monkeypatch.setenv("LORAHUB_LANG", "zh")
    monkeypatch.chdir(tmp_path)
    from lorahub.api import paths as paths_module  # noqa: PLC0415
    from lorahub.cli._i18n import set_lang  # noqa: PLC0415

    set_lang("zh")
    paths_module._resolved = None  # type: ignore[attr-defined]
    try:
        result = runner.invoke(app, ["jobs", "ls"])
    finally:
        paths_module._resolved = None  # type: ignore[attr-defined]
        set_lang("en")
    assert result.exit_code == 0, result.stdout
    assert "暂无任务" in result.stdout
