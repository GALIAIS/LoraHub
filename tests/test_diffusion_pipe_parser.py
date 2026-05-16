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


def test_traceback_first_line_flagged_as_error() -> None:
    ev = parse_line("Traceback (most recent call last):")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "error"


def test_runtime_error_flagged() -> None:
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "error"
