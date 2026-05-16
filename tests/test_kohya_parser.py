"""Tests for the kohya stdout parser."""

from __future__ import annotations

from lorahub.core.backends.kohya.parser import parse_line
from lorahub.core.events import EventType


def test_blank_line_returns_none() -> None:
    assert parse_line("") is None
    assert parse_line("   \n") is None


def test_progress_line_parsed() -> None:
    line = "steps:   2%|▏         | 5/200 [00:30<19:30,  6.00s/it, avr_loss=0.123]"
    ev = parse_line(line, job_id="J1")
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload == {"step": 5, "total_steps": 200, "loss": 0.123}
    assert ev.job_id == "J1"


def test_progress_line_without_loss() -> None:
    line = "steps:   1%|          | 1/200 [00:06<19:30,  6.00s/it]"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload["step"] == 1
    assert ev.payload["total_steps"] == 200
    assert "loss" not in ev.payload


def test_epoch_line_parsed() -> None:
    ev = parse_line("epoch 3/10")
    assert ev is not None
    assert ev.type is EventType.epoch_end
    assert ev.payload == {"epoch": 3, "total_epochs": 10}


def test_epoch_case_insensitive() -> None:
    ev = parse_line("Epoch 1/5")
    assert ev is not None
    assert ev.type is EventType.epoch_end


def test_checkpoint_saved_with_spaces_in_path() -> None:
    line = r"saving checkpoint: E:\WorkSpace\Lora Scripts\out\smoke.safetensors"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload["path"] == r"E:\WorkSpace\Lora Scripts\out\smoke.safetensors"


def test_sample_saved_with_spaces_in_path() -> None:
    line = "sample saved at: C:\\My Models\\sample.png"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.sample_ready
    assert ev.payload["path"] == "C:\\My Models\\sample.png"


def test_checkpoint_saved_parsed() -> None:
    ev = parse_line("saving checkpoint: /out/my_lora-000003.safetensors")
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload == {"path": "/out/my_lora-000003.safetensors"}


def test_checkpoint_saved_alt_phrasing() -> None:
    ev = parse_line("model saved as safetensors: /out/foo.safetensors")
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved


def test_sample_ready_parsed() -> None:
    ev = parse_line("sample saved at /out/sample/my_lora_1_000001.png")
    assert ev is not None
    assert ev.type is EventType.sample_ready
    assert ev.payload["path"].endswith(".png")


def test_arbitrary_log_line_kept_as_log() -> None:
    ev = parse_line("loading model from sdxl_base_1.0.safetensors")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"
    assert "loading model" in ev.payload["message"]


def test_error_line_flagged() -> None:
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "error"


def test_traceback_line_flagged() -> None:
    ev = parse_line("Traceback (most recent call last):")
    assert ev is not None
    assert ev.payload["level"] == "error"


def test_validation_loss_emits_validation_event() -> None:
    """sd-scripts' eval print becomes a structured `validation` event."""
    ev = parse_line("epoch 3 validation loss: 0.5237", job_id="J1")
    assert ev is not None
    assert ev.type is EventType.validation
    assert ev.payload["val_loss"] == 0.5237
    assert ev.payload.get("epoch") == 3
    assert ev.job_id == "J1"
