"""Tests for the lorahub CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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


def test_root_no_args_prints_help_in_non_terminal() -> None:
    env = os.environ.copy()
    env["LORAHUB_LANG"] = "en"
    result = subprocess.run(
        [sys.executable, "-m", "lorahub"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "Open-source LoRA training workbench" in result.stdout
    assert "Commands" in result.stdout


def test_root_no_tui_prints_help() -> None:
    env = os.environ.copy()
    env["LORAHUB_LANG"] = "en"
    result = subprocess.run(
        [sys.executable, "-m", "lorahub", "--no-tui"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "Open-source LoRA training workbench" in result.stdout
    assert "Commands" in result.stdout


def test_manage_install_windows_shim_runs_module_from_checkout() -> None:
    from lorahub.cli import manage_cmd

    body = manage_cmd._windows_shim_body(
        Path(r"E:\AIGC\LoraHub\.venv\Scripts\python.exe"),
        Path(r"E:\AIGC\LoraHub"),
    )

    assert r'set "PYTHONPATH=E:\AIGC\LoraHub;%PYTHONPATH%"' in body
    assert r'call "E:\AIGC\LoraHub\.venv\Scripts\python.exe" -m lorahub %*' in body
    assert "lorahub.exe" not in body


def test_manage_install_posix_shim_runs_module_from_checkout() -> None:
    from lorahub.cli import manage_cmd

    body = manage_cmd._posix_shim_body(
        PurePosixPath("/opt/LoraHub/.venv/bin/python"),
        PurePosixPath("/opt/LoraHub"),
    )

    assert "PYTHONPATH=/opt/LoraHub:${PYTHONPATH:-}" in body
    assert 'exec /opt/LoraHub/.venv/bin/python -m lorahub "$@"' in body
    assert "/bin/lorahub" not in body


def test_manage_build_uses_quiet_npm_and_exports_resolved_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorahub.cli import manage_cmd

    root = tmp_path / "repo"
    (root / "web").mkdir(parents=True)
    npm = tmp_path / ("npm.cmd" if sys.platform == "win32" else "npm")
    npm.write_text("", encoding="utf-8")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(manage_cmd, "_find_npm", lambda _root: npm)
    monkeypatch.setattr(manage_cmd, "_resolve_build_version", lambda: "1.1.0")
    monkeypatch.setattr("lorahub.core.paths.project_root", lambda: root)

    def fake_run(cmd, *, cwd, check, env):  # type: ignore[no-untyped-def]
        calls.append({"cmd": cmd, "cwd": cwd, "check": check, "env": env})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(manage_cmd.subprocess, "run", fake_run)

    result = runner.invoke(app, ["manage", "build"])

    assert result.exit_code == 0, result.stdout
    assert calls
    assert calls[0]["cmd"] == [str(npm), "--silent", "run", "build"]
    assert calls[0]["cwd"] == root / "web"
    assert calls[0]["env"]["LORAHUB_APP_VERSION"] == "1.1.0"  # type: ignore[index]
    assert "0.0.0-dev" not in result.stdout
    assert "1.1.0" in result.stdout


def test_root_tui_detection_ignores_global_options() -> None:
    from lorahub.cli.main import _argv_has_flag, _interactive_no_command, _should_open_tui

    assert _interactive_no_command(["--lang", "zh", "--no-tui"])
    assert _interactive_no_command(["--lang=en", "--tui"])
    assert _argv_has_flag("--no-tui", ["--lang", "zh", "--no-tui"])
    assert _should_open_tui(
        no_tui=False,
        force_tui=False,
        argv=["--lang", "zh"],
        is_terminal=True,
        stdio_terminal=True,
    )
    assert not _should_open_tui(
        no_tui=False,
        force_tui=False,
        argv=["version"],
        is_terminal=True,
    )


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


def test_train_keeps_events_when_console_render_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config_yaml(tmp_path, sd)
    ws = tmp_path / "ws"

    import lorahub.cli.main as cli_main  # noqa: PLC0415

    def broken_render(_ev):  # type: ignore[no-untyped-def]
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(cli_main, "_render_event", broken_render)
    result = runner.invoke(app, ["train", str(config), "--workspace", str(ws)])

    assert result.exit_code == 0, result.stdout
    body = (ws / "events.jsonl").read_text(encoding="utf-8")
    assert "console render failed: OSError(22, 'Invalid argument')" in body
    assert '"type":"done"' in body


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


def test_service_read_pid_recovers_listener_when_pid_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorahub.api.runtime_bind import RuntimeBind
    from lorahub.cli import service as service_mod

    monkeypatch.setattr(service_mod, "_pid_file", lambda: tmp_path / "missing.pid")
    monkeypatch.setattr(
        service_mod,
        "read_runtime_bind",
        lambda: RuntimeBind("0.0.0.0", 19090, None),
    )
    monkeypatch.setattr(service_mod, "_find_lorahub_uvicorn_pid", lambda port: 456)

    assert service_mod._read_pid() == 456


def test_service_health_requires_matching_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.cli import service as service_mod

    class FakeResponse:
        status = 200

        def __init__(self, token: str) -> None:
            self.token = token

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return f'{{"service_token": "{self.token}"}}'.encode()

    responses = [FakeResponse("old"), FakeResponse("new")]
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(service_mod.time, "sleep", lambda _seconds: None)

    assert service_mod._wait_for_health(19090, timeout_s=10, service_token="new") is True


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


def test_windows_daemon_creationflags_break_away_from_ssh_job() -> None:
    from lorahub.cli import service as service_mod

    flags = service_mod._windows_daemon_creationflags()

    assert flags & 0x00000008  # DETACHED_PROCESS
    assert flags & 0x08000000  # CREATE_NO_WINDOW
    assert flags & 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
    assert not service_mod._windows_daemon_creationflags(allow_breakaway=False) & 0x01000000


def test_service_child_env_prepends_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.cli import service as service_mod

    monkeypatch.setenv("PYTHONPATH", "old")

    env = service_mod._pythonpath_env(Path(r"E:\AIGC\LoraHub"))

    assert env["PYTHONPATH"].startswith(r"E:\AIGC\LoraHub")
    assert env["PYTHONPATH"].endswith("old")


def test_service_units_export_pythonpath_for_checkout() -> None:
    from lorahub.cli import service as service_mod

    systemd = service_mod._render_systemd_unit(
        py="/opt/LoraHub/.venv/bin/python",
        repo=PurePosixPath("/opt/LoraHub"),
        host="0.0.0.0",
        port=18765,
    )
    launchd = service_mod._render_launchd_plist(
        py="/opt/LoraHub/.venv/bin/python",
        repo=PurePosixPath("/opt/LoraHub"),
        host="0.0.0.0",
        port=18765,
    )

    assert "Environment=PYTHONPATH=/opt/LoraHub" in systemd
    assert "<key>PYTHONPATH</key><string>/opt/LoraHub</string>" in launchd


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
