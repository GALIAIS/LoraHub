"""anima_lora cut2 backend tests — bootstrap + parser + validate / launch shape.

We don't spawn a real training subprocess in CI: that would need a
Python venv with torch 2.11 nightly + accelerate, plus model weights
on disk, and would take minutes. Tests here cover the contract layer:

- Bootstrap finds the vendored copy by default; honours env var
  override; rejects a corrupted / missing repo with a clear message.
- Parser maps each recognised line shape to the right TrainingEvent.
- AnimaLoraBackend implements TrainingBackend cleanly: validate
  surfaces errors for wrong arch / missing animaLora field; launch
  builds the right argv shape (we patch the Runner's start() method
  to capture the command without actually running it).

A tiny end-to-end "the dispatch picks AnimaLoraBackend and launch
returns a TrainingHandle without crashing" smoke is in
test_anima_lora_schema.py (cut0 file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lorahub.core.backends.anima_lora import AnimaLoraBackend
from lorahub.core.backends.anima_lora import bootstrap as al_bootstrap
from lorahub.core.backends.anima_lora import installer as al_installer
from lorahub.core.backends.anima_lora import models as al_models
from lorahub.core.backends.anima_lora.backend import _prepare_sample_prompts_file
from lorahub.core.backends.anima_lora.parser import parse_line
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.config.schema import (
    TrainingConfig,
)
from lorahub.core.events import EventType
from lorahub.core.models.downloader import DownloadProgress

# --------------------------------------------------------------------------- #
# Bootstrap — vendored discovery + env var override + corruption detection
# --------------------------------------------------------------------------- #


def test_bootstrap_default_repo_resolves_to_vendored() -> None:
    """`default_repo_path()` lands on the vendored copy regardless of cwd."""
    p = al_bootstrap.default_repo_path()
    assert p.is_absolute()
    assert p.name == "anima_lora"
    assert p.parent.name == "external"
    # Vendored copy must contain the trainer.
    assert (p / "train.py").is_file()


def test_bootstrap_resolve_returns_env_with_required_files() -> None:
    """Resolving against the vendored copy succeeds without override."""
    env = al_bootstrap.resolve()
    assert env.repo_path.is_dir()
    assert env.python_executable.exists()
    # All required runtime files present.
    for name in ("train.py", "inference.py"):
        assert env.script(name).is_file()


def test_bootstrap_rejects_missing_repo(tmp_path: Path) -> None:
    """A user-pointed repo that doesn't exist surfaces as BootstrapError."""
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(BootstrapError, match="anima_lora"):
        al_bootstrap.resolve(config_path=bogus)


def test_bootstrap_rejects_repo_missing_train_py(tmp_path: Path) -> None:
    """A directory that exists but lacks train.py is "not anima_lora"."""
    fake = tmp_path / "fake-anima"
    fake.mkdir()
    # Missing train.py / inference.py / library/anima.
    with pytest.raises(BootstrapError, match="missing required files"):
        al_bootstrap.resolve(config_path=fake)


def test_bootstrap_env_var_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`LORAHUB_ANIMA_LORA_REPO` env var picks an alternate checkout."""
    fake = tmp_path / "alt-anima"
    fake.mkdir()
    (fake / "train.py").write_text("# stub", encoding="utf-8")
    (fake / "inference.py").write_text("# stub", encoding="utf-8")
    (fake / "library").mkdir()
    (fake / "library" / "anima").mkdir()
    (fake / "library" / "anima" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("LORAHUB_ANIMA_LORA_REPO", str(fake))

    env = al_bootstrap.resolve()
    # Env var wins over the vendored default.
    assert env.repo_path == fake.resolve()


def test_installer_uv_sync_env_defaults_to_project_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(al_installer, "project_root", lambda: tmp_path)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)

    plan = al_installer.BootstrapPlan(target=tmp_path / "external" / "anima_lora")
    env = al_installer._uv_sync_env(plan)

    assert env["UV_CACHE_DIR"] == str(tmp_path / ".cache" / "uv")
    if sys.platform == "win32":
        assert env["TEMP"] == str(tmp_path / ".cache" / "tmp")
        assert env["TMP"] == str(tmp_path / ".cache" / "tmp")
    else:
        assert env["TMPDIR"] == str(tmp_path / ".cache" / "tmp")


def test_installer_uv_sync_env_preserves_user_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(al_installer, "project_root", lambda: tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", "/custom/uv-cache")
    if sys.platform == "win32":
        monkeypatch.setenv("TEMP", "C:\\custom\\tmp")
        monkeypatch.setenv("TMP", "C:\\custom\\tmp")
    else:
        monkeypatch.setenv("TMPDIR", "/custom/tmp")

    plan = al_installer.BootstrapPlan(target=tmp_path / "external" / "anima_lora")
    env = al_installer._uv_sync_env(plan)

    assert "UV_CACHE_DIR" not in env
    if sys.platform == "win32":
        assert "TEMP" not in env
        assert "TMP" not in env
    else:
        assert "TMPDIR" not in env


def test_installer_deepspeed_skips_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(al_installer.sys, "platform", "win32")
    plan = al_installer.BootstrapPlan(target=tmp_path / "external" / "anima_lora")

    al_installer.install_deepspeed(plan, progress=seen.append)

    assert seen
    assert "skip deepspeed" in seen[0]


def test_installer_deepspeed_runs_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, list[str], str]] = []
    monkeypatch.setattr(al_installer.sys, "platform", "linux")

    def fake_pip_install(
        venv_py: Path,
        args: list[str],
        *,
        step: str,
        progress=None,
        pypi_index=None,
    ) -> None:
        calls.append((venv_py, args, step))

    monkeypatch.setattr(al_installer._uv, "pip_install", fake_pip_install)
    plan = al_installer.BootstrapPlan(
        target=tmp_path / "external" / "anima_lora",
        pypi_index="https://pypi.tuna.tsinghua.edu.cn/simple",
    )

    al_installer.install_deepspeed(plan)

    assert calls == [(plan.venv_python, ["deepspeed"], "install anima_lora deepspeed")]


def test_installer_bitsandbytes_runs_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, list[str], str]] = []
    monkeypatch.setattr(al_installer.sys, "platform", "linux")

    def fake_pip_install(
        venv_py: Path,
        args: list[str],
        *,
        step: str,
        progress=None,
        pypi_index=None,
    ) -> None:
        calls.append((venv_py, args, step))

    monkeypatch.setattr(al_installer._uv, "pip_install", fake_pip_install)
    plan = al_installer.BootstrapPlan(target=tmp_path / "external" / "anima_lora")

    al_installer.install_bitsandbytes(plan)

    assert calls == [
        (plan.venv_python, ["bitsandbytes"], "install anima_lora bitsandbytes")
    ]


def test_installer_bitsandbytes_skips_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(al_installer.sys, "platform", "win32")
    monkeypatch.setattr(al_installer._uv, "pip_install", lambda *a, **k: calls.append(a))
    plan = al_installer.BootstrapPlan(target=tmp_path / "external" / "anima_lora")

    al_installer.install_bitsandbytes(plan)

    assert calls == []


def test_anima_model_download_uses_env_hf_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    monkeypatch.setattr(al_models, "models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(al_models, "_link_anima_models_dir", lambda: None)
    endpoints: list[str | None] = []

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            endpoints.append(endpoint)

        def model_info(self, *_args: Any, **_kwargs: Any):
            siblings = [
                type("Sibling", (), {"rfilename": repo_path, "size": 1})()
                for _, _, repo_path in al_models._TARGETS
            ]
            return type("Info", (), {"siblings": siblings})()

    def fake_hf_hub_download(**kw: Any) -> str:
        endpoints.append(kw.get("endpoint"))
        cached = Path(kw["local_dir"]) / kw["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"model")
        return str(cached)

    fake_hub = type(
        "FakeHub",
        (),
        {"HfApi": FakeApi, "hf_hub_download": staticmethod(fake_hf_hub_download)},
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    al_models.download_models(source="huggingface", threads=1)

    assert endpoints == ["https://hf-mirror.com"] * 4
    for path in al_models.expected_files():
        assert path.is_file()


def test_anima_model_download_defaults_to_modelscope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(al_models, "models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(al_models, "_link_anima_models_dir", lambda: None)
    seen: dict[str, Any] = {}

    def fake_download(req: Any, progress: Any = None) -> None:
        seen["source"] = req.source
        seen["repo_id"] = req.repo_id
        seen["paths"] = req.paths
        for path in req.paths:
            out = req.target_dir / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"model")
        if progress:
            progress(DownloadProgress("done", 100, 3, 3, 0, 0))

    monkeypatch.setattr(al_models, "download", fake_download)

    al_models.download_models(threads=1)

    assert seen["source"] == "modelscope"
    assert seen["repo_id"] == al_models.ANIMA_REPO_ID
    assert set(seen["paths"]) == {target[2] for target in al_models._TARGETS}
    for path in al_models.expected_files():
        assert path.is_file()


# --------------------------------------------------------------------------- #
# Parser — every recognised line shape produces the right event type
# --------------------------------------------------------------------------- #


def test_parser_tqdm_steps_emits_step_event() -> None:
    line = (
        "steps:  17%|##2       | 51/300 [00:30<02:30,  1.67it/s, "
        "avr_loss=0.243, lr=5e-05]"
    )
    ev = parse_line(line, job_id="job-1")
    assert ev is not None
    assert ev.type == EventType.step
    assert ev.payload["step"] == 51
    assert ev.payload["total_steps"] == 300
    assert ev.payload["loss"] == pytest.approx(0.243)
    assert ev.payload["lr"] == pytest.approx(5e-5)
    assert ev.job_id == "job-1"


def test_parser_tqdm_nan_loss_keeps_step_progress() -> None:
    line = (
        "steps:   3%|3         | 55/2000 [00:28<16:48,  1.93it/s, "
        "avr_loss=nan, lr=2e-05]"
    )
    ev = parse_line(line, job_id="job-1")
    assert ev is not None
    assert ev.type == EventType.step
    assert ev.payload["step"] == 55
    assert ev.payload["total_steps"] == 2000
    assert "loss" not in ev.payload
    assert ev.payload["lr"] == pytest.approx(2e-5)


def test_parser_nan_guard_emits_diagnostic_warning() -> None:
    ev = parse_line("WARNING nan_guard recovery: halved LR (now 1e-05)")
    assert ev is not None
    assert ev.type == EventType.diagnostic_warning
    assert ev.payload["category"] == "nan_loss"


def test_parser_epoch_increment_emits_epoch_end() -> None:
    line = "epoch is incremented. current_epoch: 1, epoch: 2"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.epoch_end
    assert ev.payload["epoch"] == 2


def test_parser_save_checkpoint_emits_checkpoint_saved() -> None:
    line = "saving checkpoint: /abs/path/anima_lora-000002.safetensors"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.checkpoint_saved
    assert "anima_lora-000002.safetensors" in ev.payload["path"]


def test_parser_save_model_emits_checkpoint_saved() -> None:
    """Both ``saving checkpoint:`` and ``saving model:`` map to checkpoint_saved."""
    line = "saving model: /abs/path/anima_lora-final"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.checkpoint_saved
    assert "anima_lora-final" in ev.payload["path"]


def test_parser_validation_loss_emits_validation_event() -> None:
    line = "validation loss=0.187 epoch=3 step=512"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.validation
    assert ev.payload["val_loss"] == pytest.approx(0.187)
    assert ev.payload["epoch"] == 3
    assert ev.payload["step"] == 512


def test_parser_nan_validation_loss_emits_validation_event() -> None:
    line = "validation loss=nan epoch=1 step=113"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.validation
    assert "val_loss" not in ev.payload
    assert ev.payload["epoch"] == 1
    assert ev.payload["step"] == 113


def test_parser_eval_loss_emits_validation_event() -> None:
    line = "eval_loss=0.231 epoch=4 step=768"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type == EventType.validation
    assert ev.payload["val_loss"] == pytest.approx(0.231)
    assert ev.payload["epoch"] == 4
    assert ev.payload["step"] == 768


def test_parser_unknown_line_falls_to_log() -> None:
    """Anything we don't recognise lands on `log` so nothing's silently dropped."""
    ev = parse_line("loaded 1024 captions from image_dataset/")
    assert ev is not None
    assert ev.type == EventType.log
    assert ev.payload["level"] == "info"
    assert "loaded 1024 captions" in ev.payload["message"]


def test_parser_error_line_marked_red() -> None:
    """`Error:` / `Traceback` patterns escalate to level=error."""
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type == EventType.log
    assert ev.payload["level"] == "error"


def test_parser_keyboard_interrupt_not_an_error() -> None:
    """Clean cancel artefacts must not paint the timeline red."""
    ev = parse_line("KeyboardInterrupt")
    assert ev is not None
    assert ev.payload["level"] == "info"


def test_parser_broken_pipe_not_an_error() -> None:
    ev = parse_line("BrokenPipeError: [Errno 32] Broken pipe")
    assert ev is not None
    assert ev.type == EventType.log
    assert ev.payload["level"] == "info"


def test_parser_empty_line_returns_none() -> None:
    """Whitespace-only input is dropped to save downstream allocations."""
    assert parse_line("") is None
    assert parse_line("   \n") is None
    assert parse_line("\r\n") is None


# --------------------------------------------------------------------------- #
# Backend — validate / launch shape (no real subprocess)
# --------------------------------------------------------------------------- #


def _config(tmp_path: Path, **backend_extras: Any) -> TrainingConfig:
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    # Drop a dummy image + matching TE/latent cache so the auto-preprocess
    # short-circuits ("everything cached") and tests can focus on
    # validate / launch shape without spawning preprocess subprocesses.
    (data / "img1.jpg").write_bytes(b"")
    (data / "img1.txt").write_text("a tag", encoding="utf-8")
    cache = tmp_path / "ws" / "post_image_dataset" / "lora"
    cache.mkdir(parents=True)
    (cache / "img1_anima_te.safetensors").write_bytes(b"")
    (cache / "img1_1024x1024_anima.npz").write_bytes(b"")
    backend = {"type": "anima_lora", "animaLora": {}}
    backend.update(backend_extras)
    return TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt), "arch": "anima"},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "optimizer": {"lr": {"unet": 1e-4, "text_encoder": 5e-5}},
            "network": {"rank": 16, "alpha": 8},
            "output": {"name": "demo"},
            "backend": backend,
        }
    )


def _sample_tokens(width: int, height: int) -> int:
    return (width // 16) * (height // 16)


def test_backend_name_and_supported_archs() -> None:
    b = AnimaLoraBackend()
    assert b.name == "anima_lora"
    archs = {a.value for a in b.supported_archs}
    assert archs == {"anima"}


def test_sample_prompts_clamp_912x1632_to_static_token_budget(
    tmp_path: Path,
) -> None:
    """Regression for DiT sample unpad crash: 912x1632 is 5814 tokens."""
    cfg = _config(tmp_path)
    cfg.sampling.enabled = True
    cfg.sampling.resolution = [912, 1632]
    cfg.sampling.seed = 123

    workspace = tmp_path / "ws"
    _prepare_sample_prompts_file(cfg, workspace)

    body = (workspace / "_lorahub_sample_prompts.txt").read_text(encoding="utf-8")
    assert "--w 768 --h 1360" in body
    assert _sample_tokens(768, 1360) <= 4096


def test_existing_sample_prompts_are_rewritten_when_over_static_budget(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.sampling.enabled = True
    cfg.sampling.resolution = [912, 1632]
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "character portrait --w 912 --h 1632 --s 24 --l 5.0\n",
        encoding="utf-8",
    )
    cfg.sampling.prompts_file = prompts

    workspace = tmp_path / "ws"
    _prepare_sample_prompts_file(cfg, workspace)

    assert cfg.sampling.prompts_file == workspace / "_lorahub_anima_sample_prompts.txt"
    body = cfg.sampling.prompts_file.read_text(encoding="utf-8")
    assert "--w 768 --h 1360" in body
    assert "--s 24 --l 5.0" in body


def test_validate_anima_arch_clean(tmp_path: Path) -> None:
    """An anima config with vendored copy reachable returns no error issues."""
    cfg = _config(tmp_path)
    issues = AnimaLoraBackend().validate(cfg)
    errors = [i for i in issues if i.severity.value == "error"]
    assert errors == [], f"unexpected errors: {errors}"


def test_validate_warns_on_v100_risky_fp16_combo(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        animaLora={
            "mixedPrecision": "fp16",
            "networkDim": 32,
            "networkAlpha": 32,
            "lora": {"algorithm": "loha"},
            "useCustomDownAutograd": True,
        },
    )
    cfg.precision = "fp16"
    cfg.sampling.enabled = True
    cfg.sampling.at_first = True

    issues = AnimaLoraBackend().validate(cfg)
    fields = {i.field for i in issues if i.severity.value == "warning"}
    assert "backend.animaLora.networkDim" in fields
    assert "backend.animaLora.useCustomDownAutograd" in fields
    assert "sampling.atFirst" in fields


def test_validate_wrong_arch_errors(tmp_path: Path) -> None:
    """Config targets sdxl but type=anima_lora — clear error pointing back to kohya."""
    cfg = _config(tmp_path)
    # Force-bypass the schema-level arch check for this defence-in-depth test:
    # we want the backend's own validator to surface its message even if the
    # schema accepts the value.
    cfg.base_model.__dict__["arch"] = "sdxl"
    issues = AnimaLoraBackend().validate(cfg)
    arch_errors = [
        i for i in issues
        if i.severity.value == "error" and i.field == "base_model.arch"
    ]
    assert arch_errors
    assert "anima_lora does not support" in arch_errors[0].message


def test_launch_builds_accelerate_argv_without_running(tmp_path: Path) -> None:
    """`launch` constructs `<python> -m accelerate ... train.py <args>`.

    Patches the AnimaLoraRunner's start so we capture what would be
    spawned without actually paying the cost. The argv first three
    tokens must always be ``[python, '-m', 'accelerate.commands.accelerate_cli']``
    so a future change to the accelerate launcher entry point gets
    caught.
    """
    from types import SimpleNamespace

    cfg = _config(tmp_path)
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        # Stash a stub so the .pid property reads back something useful
        # for the TrainingHandle wiring below.
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    def fake_stop(self, *, graceful=True):  # type: ignore[no-untyped-def]
        return None

    def fake_wait(self, *, timeout=None):  # type: ignore[no-untyped-def]
        from lorahub.core.backends._common.runner import RunResult

        return RunResult(returncode=0, killed=False)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ), patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.stop", fake_stop
    ), patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.wait", fake_wait
    ):
        handle = AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=2
        )

    assert handle.pid == 4242
    assert len(captured_argv) == 1
    argv = captured_argv[0]
    # Frame: <python> -m accelerate.commands.accelerate_cli launch
    #         --num_cpu_threads_per_process 3 --mixed_precision <config precision>
    #         <repo>/train.py ...
    assert argv[1] == "-m"
    assert argv[2] == "accelerate.commands.accelerate_cli"
    assert argv[3] == "launch"
    assert "--num_cpu_threads_per_process" in argv
    assert "--mixed_precision" in argv
    assert argv[argv.index("--mixed_precision") + 1] == "bf16"
    assert "--multi_gpu" not in argv
    assert argv[argv.index("--num_processes") + 1] == "1"
    # train.py path must end the launcher prefix.
    train_py_idx = next(
        i for i, x in enumerate(argv) if x.endswith("train.py")
    )
    assert train_py_idx > 4
    # --config_file comes from the compiler payload after train.py
    # (LoraHub now drives every training knob through the generated
    # _lorahub_anima_config.toml; --method/--preset are gone).
    assert "--config_file" in argv[train_py_idx + 1 :]


def test_launch_distributed_ddp_uses_multi_gpu(tmp_path: Path) -> None:
    from types import SimpleNamespace

    cfg = _config(
        tmp_path,
        gpuDispatch={"mode": "distributed", "numGpus": 2},
        distributed={"strategy": "ddp"},
        animaLora={"mixedPrecision": "fp16"},
    )
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=4
        )

    argv = captured_argv[0]
    assert argv[argv.index("--mixed_precision") + 1] == "fp16"
    assert "--multi_gpu" in argv
    assert argv[argv.index("--num_processes") + 1] == "2"


def test_launch_uses_anima_mixed_precision_for_accelerate(tmp_path: Path) -> None:
    from types import SimpleNamespace

    cfg = _config(tmp_path, animaLora={"mixedPrecision": "fp16"})
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=1
        )

    argv = captured_argv[0]
    assert argv[argv.index("--mixed_precision") + 1] == "fp16"


def test_launch_maps_fp32_to_accelerate_no_precision(tmp_path: Path) -> None:
    from types import SimpleNamespace

    cfg = _config(tmp_path, animaLora={"mixedPrecision": "fp32"})
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=1
        )

    argv = captured_argv[0]
    assert argv[argv.index("--mixed_precision") + 1] == "no"


def test_launch_fsdp_writes_accelerate_config(tmp_path: Path) -> None:
    from types import SimpleNamespace

    cfg = _config(
        tmp_path,
        gpuDispatch={"mode": "distributed", "numGpus": 2},
        distributed={
            "strategy": "fsdp",
            "fsdp": {
                "autoWrapPolicy": "size_based",
                "minNumParams": 123456,
                "shardingStrategy": "full_shard",
            },
        },
    )
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=2
        )

    argv = captured_argv[0]
    assert "--multi_gpu" not in argv
    config_path = Path(argv[argv.index("--config_file") + 1])
    assert config_path.name == "_lorahub_accelerate.yaml"
    body = config_path.read_text(encoding="utf-8")
    assert "distributed_type: FSDP" in body
    assert "num_processes: 2" in body
    assert "fsdp_auto_wrap_policy: SIZE_BASED_WRAP" in body
    assert "fsdp_sharding_strategy: FULL_SHARD" in body
    assert "fsdp_backward_prefetch_policy: NO_PREFETCH" in body
    assert "fsdp_min_num_params: 123456" in body


def test_launch_deepspeed_zero_writes_accelerate_and_zero_config(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    cfg = _config(
        tmp_path,
        gpuDispatch={"mode": "distributed", "numGpus": 2},
        animaLora={"mixedPrecision": "fp16"},
        distributed={
            "strategy": "deepspeed_zero",
            "zero": {
                "stage": 3,
                "offloadOptimizer": "cpu",
                "offloadParam": "none",
                "overlapComm": False,
            },
        },
    )
    captured_argv: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured_argv.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ), patch(
        "lorahub.core.backends.anima_lora.backend._ensure_deepspeed_available",
        lambda _python: None,
    ):
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None, gpu_count=2
        )

    argv = captured_argv[0]
    assert "--multi_gpu" not in argv
    accelerate_path = Path(argv[argv.index("--config_file") + 1])
    accelerate_body = accelerate_path.read_text(encoding="utf-8")
    assert "distributed_type: DEEPSPEED" in accelerate_body
    assert "num_processes: 2" in accelerate_body
    assert "_lorahub_deepspeed_zero.json" in accelerate_body

    zero_path = tmp_path / "ws" / "_lorahub_deepspeed_zero.json"
    zero_body = json.loads(zero_path.read_text(encoding="utf-8"))
    assert zero_body["bf16"]["enabled"] is False
    assert zero_body["fp16"]["enabled"] is True
    assert zero_body["train_batch_size"] == "auto"
    assert zero_body["zero_optimization"]["stage"] == 3
    assert zero_body["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert zero_body["zero_optimization"]["offload_param"]["device"] == "none"
    assert zero_body["zero_optimization"]["overlap_comm"] is False
