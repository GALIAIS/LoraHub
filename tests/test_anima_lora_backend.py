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

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lorahub.core.backends.anima_lora import AnimaLoraBackend
from lorahub.core.backends.anima_lora import bootstrap as al_bootstrap
from lorahub.core.backends.anima_lora.parser import parse_line
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.config.schema import (
    TrainingConfig,
)
from lorahub.core.events import EventType

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
    # Drop a dummy image + matching TE cache so the auto-preprocess
    # short-circuits ("everything cached") and tests can focus on
    # validate / launch shape without spawning preprocess subprocesses.
    (data / "img1.jpg").write_bytes(b"")
    (data / "img1.txt").write_text("a tag", encoding="utf-8")
    cache = tmp_path / "ws" / "post_image_dataset" / "lora"
    cache.mkdir(parents=True)
    (cache / "img1_anima_te.safetensors").write_bytes(b"")
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


def test_backend_name_and_supported_archs() -> None:
    b = AnimaLoraBackend()
    assert b.name == "anima_lora"
    archs = {a.value for a in b.supported_archs}
    assert archs == {"anima"}


def test_validate_anima_arch_clean(tmp_path: Path) -> None:
    """An anima config with vendored copy reachable returns no error issues."""
    cfg = _config(tmp_path)
    issues = AnimaLoraBackend().validate(cfg)
    errors = [i for i in issues if i.severity.value == "error"]
    assert errors == [], f"unexpected errors: {errors}"


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
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None
        )

    assert handle.pid == 4242
    assert len(captured_argv) == 1
    argv = captured_argv[0]
    # Frame: <python> -m accelerate.commands.accelerate_cli launch
    #         --num_cpu_threads_per_process 3 --mixed_precision bf16
    #         <repo>/train.py ...
    assert argv[1] == "-m"
    assert argv[2] == "accelerate.commands.accelerate_cli"
    assert argv[3] == "launch"
    assert "--num_cpu_threads_per_process" in argv
    assert "--mixed_precision" in argv
    # train.py path must end the launcher prefix.
    train_py_idx = next(
        i for i, x in enumerate(argv) if x.endswith("train.py")
    )
    assert train_py_idx > 4
    # --config_file comes from the compiler payload after train.py
    # (LoraHub now drives every training knob through the generated
    # _lorahub_anima_config.toml; --method/--preset are gone).
    assert "--config_file" in argv[train_py_idx + 1 :]
