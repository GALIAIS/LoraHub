"""cut4 — Turbo distillation tests.

Three layers:
1. Schema — AnimaLoraTurboConfig defaults match upstream turbo.toml.
2. Compiler — `compile_turbo_config` emits the right CLI shape;
   `compile_config` rejects when turbo is set; backend.launch picks
   the right runner.
3. Parser — turbo's tqdm postfix (g/dca/ddm/xp/vs/fake) maps to a
   `step` event with `loss = grad_rms`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lorahub.core.backends.anima_lora import AnimaLoraBackend
from lorahub.core.backends.anima_lora.compiler import (
    CompilationError,
    compile_config,
    compile_turbo_config,
)
from lorahub.core.backends.anima_lora.turbo_parser import parse_line
from lorahub.core.config.schema import (
    AnimaLoraOptions,
    AnimaLoraTurboConfig,
    TrainingConfig,
)
from lorahub.core.events import EventType


def _recipe(tmp_path: Path, *, with_turbo: bool) -> TrainingConfig:
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    # Drop a dummy image + matching TE cache so the auto-preprocess
    # short-circuits ("everything cached"); these turbo tests only care
    # about runner / argv selection, not the preprocess wiring.
    (data / "img1.jpg").write_bytes(b"")
    (data / "img1.txt").write_text("a tag", encoding="utf-8")
    cache = tmp_path / "ws" / "post_image_dataset" / "lora"
    cache.mkdir(parents=True)
    (cache / "img1_anima_te.safetensors").write_bytes(b"")
    anima_lora_payload: dict = {}
    if with_turbo:
        anima_lora_payload["turbo"] = {}
    return TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt), "arch": "anima"},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "optimizer": {"lr": {"unet": 1e-4, "text_encoder": 5e-5}},
            "network": {"rank": 16, "alpha": 8},
            "output": {"name": "x"},
            "backend": {"type": "anima_lora", "animaLora": anima_lora_payload},
        }
    )


def _argv_pairs(argv: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        assert flag.startswith("--"), f"expected flag at {i}, got {flag!r}"
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.setdefault(flag, []).append(argv[i + 1])
            i += 2
        else:
            out.setdefault(flag, []).append("")
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Schema — defaults match upstream turbo.toml
# --------------------------------------------------------------------------- #


def test_turbo_defaults_match_upstream() -> None:
    """Schema defaults track configs/methods/turbo.toml in vendored copy."""
    t = AnimaLoraTurboConfig()
    assert t.iterations == 1000
    assert t.batch_size == 1
    # ComfyUI-style sentinel: -1 = pick a fresh random seed at job start.
    assert t.seed == -1
    assert t.student_rank == 48
    assert t.fake_rank == 64
    assert t.student_steps == 4
    assert t.teacher_cfg == 4.0
    assert t.student_lr == pytest.approx(5e-6)
    assert t.fake_lr == pytest.approx(5e-5)
    assert t.fake_steps_per_student_step == 2
    assert t.alpha_warmup_steps == 100
    assert t.tau_ca_strategy == "above_t"
    assert t.tau_dm_strategy == "uniform"
    assert t.save_every == 250
    assert t.log_interval == 5


def test_turbo_field_attachment_optional() -> None:
    """`AnimaLoraOptions.turbo` defaults None — turbo opt-in only."""
    o = AnimaLoraOptions()
    assert o.turbo is None
    o2 = AnimaLoraOptions(turbo=AnimaLoraTurboConfig())
    assert o2.turbo is not None
    assert o2.turbo.iterations == 1000


# --------------------------------------------------------------------------- #
# Compiler — turbo branch routes through distill_turbo, regular path rejects
# --------------------------------------------------------------------------- #


def test_compile_config_rejects_recipe_with_turbo_set(tmp_path: Path) -> None:
    """compile_config (train.py path) must refuse when turbo is set.

    backend.launch uses the presence of opts.turbo to pick which
    compiler to call. If a caller bypasses launch and invokes
    compile_config directly with turbo set, surface a clear error.
    """
    cfg = _recipe(tmp_path, with_turbo=True)
    with pytest.raises(CompilationError, match="turbo"):
        compile_config(cfg, tmp_path / "ws")


def test_compile_turbo_config_rejects_when_turbo_unset(tmp_path: Path) -> None:
    """Inverse — compile_turbo_config refuses recipes without turbo."""
    cfg = _recipe(tmp_path, with_turbo=False)
    with pytest.raises(CompilationError, match="turbo"):
        compile_turbo_config(cfg, tmp_path / "ws")


def test_compile_turbo_emits_distill_turbo_argv(tmp_path: Path) -> None:
    """Standard turbo recipe produces a clean distill_turbo argv set."""
    cfg = _recipe(tmp_path, with_turbo=True)
    argv, files = compile_turbo_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert files == {}, "turbo compiler must not emit any files"
    # Critical turbo-only flags must be present.
    assert pairs["--iterations"] == ["1000"]
    assert pairs["--batch_size"] == ["1"]
    assert pairs["--student_rank"] == ["48"]
    assert pairs["--fake_rank"] == ["64"]
    assert pairs["--student_steps"] == ["4"]
    # `--alpha` overrides dmd.teacher_cfg per upstream argparse.
    assert pairs["--alpha"] == [repr(4.0)]
    assert pairs["--student_lr"] == [repr(5e-6)]
    assert pairs["--fake_lr"] == [repr(5e-5)]
    assert pairs["--fake_steps_per_student_step"] == ["2"]
    assert pairs["--alpha_warmup_steps"] == ["100"]
    assert pairs["--save_every"] == ["250"]
    assert pairs["--log_interval"] == ["5"]
    # Output dir routes into the workspace.
    assert pairs["--output_dir"][0].endswith("ckpt")


def test_compile_turbo_no_method_or_preset_flags(tmp_path: Path) -> None:
    """Turbo path must NOT emit `--method` / `--preset` — that's train.py only."""
    cfg = _recipe(tmp_path, with_turbo=True)
    argv, _ = compile_turbo_config(cfg, tmp_path / "ws")
    assert "--method" not in argv, "turbo path must not pass --method"
    assert "--preset" not in argv, "turbo path must not pass --preset"


def test_compile_turbo_use_custom_down_autograd_off(tmp_path: Path) -> None:
    """`use_custom_down_autograd=False` emits the `--no_use_custom_down_autograd` flag."""
    cfg = _recipe(tmp_path, with_turbo=True)
    cfg.backend.anima_lora.turbo.__dict__["use_custom_down_autograd"] = False
    argv, _ = compile_turbo_config(cfg, tmp_path / "ws")
    assert "--no_use_custom_down_autograd" in argv
    assert "--use_custom_down_autograd" not in argv


# --------------------------------------------------------------------------- #
# backend.launch — turbo recipe picks AnimaLoraTurboRunner not AnimaLoraRunner
# --------------------------------------------------------------------------- #


def test_launch_picks_turbo_runner_when_turbo_set(tmp_path: Path) -> None:
    """backend.launch dispatches to TurboRunner (no accelerate prefix)."""
    cfg = _recipe(tmp_path, with_turbo=True)
    captured: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured.append(list(self._argv))
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
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None
        )

    assert len(captured) == 1
    argv = captured[0]
    # Turbo path: <python> <repo>/scripts/distill_turbo.py <args>
    # No `-m accelerate.commands.accelerate_cli launch` prefix.
    assert "accelerate.commands.accelerate_cli" not in argv
    assert any(x.endswith("distill_turbo.py") for x in argv)
    # Turbo-specific flags survive.
    assert "--iterations" in argv
    assert "--student_rank" in argv


def test_launch_picks_regular_runner_when_turbo_unset(tmp_path: Path) -> None:
    """Without turbo, backend.launch still routes through accelerate launch."""
    cfg = _recipe(tmp_path, with_turbo=False)
    captured: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured.append(list(self._argv))
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
        AnimaLoraBackend().launch(
            cfg, workspace=tmp_path / "ws", on_event=lambda _e: None
        )

    argv = captured[0]
    # Regular path: accelerate launcher present, distill_turbo.py absent.
    assert "accelerate.commands.accelerate_cli" in argv
    assert not any(x.endswith("distill_turbo.py") for x in argv)


# --------------------------------------------------------------------------- #
# Turbo parser — tqdm postfix maps to step event with loss = grad_rms
# --------------------------------------------------------------------------- #


def test_turbo_parser_step_event_with_grad_as_loss() -> None:
    line = (
        "turbo:  17%|##2       | 51/300 [00:30<02:30,  1.67it/s, "
        "g=1.23e-03, dca=4.56e-04, ddm=7.89e-05, xp=0.421, vs=0.317, "
        "fake=2.10e-05]"
    )
    ev = parse_line(line, job_id="turbo-1")
    assert ev is not None
    assert ev.type == EventType.step
    assert ev.payload["step"] == 51
    assert ev.payload["total_steps"] == 300
    # Grad-RMS becomes the canonical loss signal.
    assert ev.payload["loss"] == pytest.approx(1.23e-3)
    assert ev.payload["grad_rms"] == pytest.approx(1.23e-3)
    # All RMS metrics surface separately for the analytics tab.
    assert ev.payload["dca_rms"] == pytest.approx(4.56e-4)
    assert ev.payload["ddm_rms"] == pytest.approx(7.89e-5)
    assert ev.payload["xpred_rms"] == pytest.approx(0.421)
    assert ev.payload["vstudent_rms"] == pytest.approx(0.317)
    assert ev.payload["fake_loss"] == pytest.approx(2.10e-5)
    assert ev.job_id == "turbo-1"


def test_turbo_parser_unknown_line_falls_to_log() -> None:
    """Anything we don't recognise lands as info log, not silently dropped."""
    ev = parse_line("[turbo] loaded 1024 captions")
    assert ev is not None
    assert ev.type == EventType.log
    assert ev.payload["level"] == "info"


def test_turbo_parser_error_line_marked_red() -> None:
    """`Error:` / `out of memory` patterns escalate to level=error."""
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type == EventType.log
    assert ev.payload["level"] == "error"


def test_turbo_parser_keyboard_interrupt_not_an_error() -> None:
    ev = parse_line("KeyboardInterrupt")
    assert ev is not None
    assert ev.payload["level"] == "info"


def test_turbo_parser_empty_line_returns_none() -> None:
    assert parse_line("") is None
    assert parse_line("\r\n") is None
