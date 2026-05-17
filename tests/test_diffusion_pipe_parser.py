"""Tests for the diffusion-pipe stdout parser.

The parser is stateless: each line maps to at most one TrainingEvent. We
exercise the recognised patterns (step, loss, save, epoch) plus the catch-all
log/error fallback.
"""

from __future__ import annotations

from lorahub.core.backends.diffusion_pipe.parser import parse_line
from lorahub.core.events import EventType


def test_blank_line_returns_none() -> None:
    assert parse_line("") is None
    assert parse_line("   \n") is None


def test_step_event() -> None:
    line = "[INFO] [engine.py:123] [Rank 0] step=42, skipped=0, lr=[0.0001], mom=[(0.9, 0.99)]"
    ev = parse_line(line, job_id="J1")
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload["step"] == 42
    assert ev.payload["lr"] == 0.0001
    assert ev.job_id == "J1"


def test_step_event_with_loss() -> None:
    line = "[INFO] step=10 lr=[1e-4] loss=0.5237"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload["step"] == 10
    assert ev.payload["loss"] == 0.5237


def test_loss_only_line_emits_log_event() -> None:
    # A bare ``loss=`` line without a step is treated as a log line; the
    # parser only emits step events when a step number is present.
    ev = parse_line("loss=0.42")
    assert ev is not None
    assert ev.type is EventType.log


def test_dp_steps_loss_summary_emits_step_event() -> None:
    # diffusion-pipe (newer releases) prints its own per-step summary on
    # a separate line. This is the only line that carries the loss, so
    # the parser MUST recognise it or the metrics endpoint never sees a
    # loss series.
    line = "steps: 30 loss: 0.1808 iter time (s): 3.662 samples/sec: 1.092"
    ev = parse_line(line, job_id="J9")
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.job_id == "J9"
    assert ev.payload["step"] == 30
    assert ev.payload["loss"] == 0.1808
    assert ev.payload["iter_time_s"] == 3.662
    assert ev.payload["samples_per_sec"] == 1.092


def test_dp_steps_loss_minimal() -> None:
    # The trailing iter time / samples/sec are optional — older or
    # patched dp builds may omit them. Loss must still be captured.
    ev = parse_line("steps: 7 loss: 0.42")
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload["step"] == 7
    assert ev.payload["loss"] == 0.42
    assert "iter_time_s" not in ev.payload
    assert "samples_per_sec" not in ev.payload


def test_epoch_event() -> None:
    ev = parse_line("Started new epoch: 3")
    assert ev is not None
    assert ev.type is EventType.epoch_end
    assert ev.payload == {"epoch": 3}


def test_save_event_uppercase() -> None:
    ev = parse_line("Saving model to directory /runs/demo/epoch5")
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload == {"path": "/runs/demo/epoch5"}


def test_save_event_past_tense() -> None:
    ev = parse_line("Saved model to /runs/demo/epoch5")
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload == {"path": "/runs/demo/epoch5"}


def test_save_event_with_spaces_in_path() -> None:
    line = r"Saving model to directory C:\My Models\epoch5"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload["path"] == r"C:\My Models\epoch5"


def test_irrelevant_lines_kept_as_log() -> None:
    ev = parse_line("Loading checkpoint shards: 100%|##########| 7/7 [00:01<00:00]")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_traceback_banner_stays_info() -> None:
    # `Traceback (most recent call last):` is just a banner — it appears
    # for both genuine crashes and clean Ctrl-C shutdowns, so the parser
    # must NOT auto-redden it. The exception summary line that closes
    # the traceback is the real error signal.
    ev = parse_line("Traceback (most recent call last):")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_runtime_error_flagged() -> None:
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "error"


def test_keyboard_interrupt_stays_info() -> None:
    # User-cancel artefact, must not be flagged as error.
    ev = parse_line("KeyboardInterrupt")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_killing_subprocess_stays_info() -> None:
    line = "[2026-05-18 03:37:03,778] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 64724"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_cancel_returncode_stays_info() -> None:
    line = (
        "[2026-05-18 03:37:03,778] [ERROR] [launch.py:341:sigkill_handler] "
        "[...] exits with return code = -2"
    )
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.log
    # Despite the embedded "[ERROR]" tag, this is a SIGINT-driven
    # cancellation — keep it info so a clean cancel doesn't render red.
    assert ev.payload["level"] == "info"
