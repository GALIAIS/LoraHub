"""Tests for the kohya stdout parser."""

from __future__ import annotations

from lorahub.core.backends.kohya.parser import KohyaLineParser, parse_line
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


def test_sample_ready_from_banner_with_step() -> None:
    # sd-scripts doesn't print the saved sample path, only a banner
    # ahead of generation. We surface that as a sample_ready event so
    # the timeline UI gets a milestone marker; the actual image path
    # comes from /api/jobs/{id}/files.
    ev = parse_line(
        "generating sample images at step / サンプル画像生成 ステップ: 200",
    )
    assert ev is not None
    assert ev.type is EventType.sample_ready
    assert ev.payload["step"] == 200
    assert "path" not in ev.payload


def test_save_phrasing_full_model_checkpoint() -> None:
    # `train.py` / `fine_tune.py` use a different phrasing than the
    # LoRA-specific `saving checkpoint: <path>` line, so the SAVE
    # regex must accept the verb `save` (not just saving/saved).
    ev = parse_line(
        "save trained model as StableDiffusion checkpoint to "
        "/runs/foo/model_final.safetensors",
    )
    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload["path"].endswith("model_final.safetensors")


def test_keyboard_interrupt_traceback_is_cancel_not_error() -> None:
    """User cancel: KeyboardInterrupt closes the traceback as a log,
    not as EventType.error, so the UI doesn't flag a clean stop red."""
    p = KohyaLineParser()
    lines = [
        "Traceback (most recent call last):",
        '  File "/foo/deepspeed/launcher/runner.py", line 646, in main',
        "    result.wait()",
        '  File "/foo/subprocess.py", line 1266, in wait',
        "    return self._wait(timeout=timeout)",
        "KeyboardInterrupt",
    ]
    events = [p.parse_line(line, job_id="J") for line in lines]
    closing = events[-1]
    assert closing is not None
    assert closing.type is EventType.log, (
        f"expected log (cancel), got {closing.type}"
    )
    assert closing.payload.get("level") == "info"
    assert closing.payload.get("kind") == "cancel"
    # Body still contains the full traceback for forensics.
    assert "KeyboardInterrupt" in closing.payload["traceback"]


def test_real_exception_traceback_still_error() -> None:
    """A genuine RuntimeError must still close the traceback as
    EventType.error so failures stay visually distinct from cancels."""
    p = KohyaLineParser()
    lines = [
        "Traceback (most recent call last):",
        '  File "/foo/train.py", line 1, in <module>',
        "    raise RuntimeError('boom')",
        "RuntimeError: boom",
    ]
    events = [p.parse_line(line, job_id="J") for line in lines]
    closing = events[-1]
    assert closing is not None
    assert closing.type is EventType.error
    assert closing.payload["summary"] == "RuntimeError: boom"


def test_killing_subprocess_log_is_info() -> None:
    line = "[2026-05-18 03:37:03,778] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 64724"
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_cancel_returncode_log_is_info() -> None:
    # The line literally contains [ERROR] but it's a SIGINT cancel — keep info.
    line = (
        "[2026-05-18 03:37:03,778] [ERROR] [launch.py:341:sigkill_handler] "
        "[...] exits with return code = -2"
    )
    ev = parse_line(line)
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"


def test_arbitrary_log_line_kept_as_log() -> None:
    ev = parse_line("loading model from sdxl_base_1.0.safetensors")
    assert ev is not None
    assert ev.type is EventType.log
    assert ev.payload["level"] == "info"
    assert "loading model" in ev.payload["message"]


def test_error_line_flagged() -> None:
    """OOM lines now go to a dedicated `oom` event, not generic log/error."""
    ev = parse_line("RuntimeError: CUDA out of memory")
    assert ev is not None
    assert ev.type is EventType.oom
    assert "out of memory" in ev.payload["message"].lower()


def test_traceback_opener_buffered() -> None:
    """A bare `Traceback (...)` opener is buffered, not emitted as a log."""
    parser = KohyaLineParser()
    ev = parser.parse_line("Traceback (most recent call last):")
    assert ev is None


def test_validation_loss_emits_validation_event() -> None:
    """sd-scripts' eval print becomes a structured `validation` event."""
    ev = parse_line("epoch 3 validation loss: 0.5237", job_id="J1")
    assert ev is not None
    assert ev.type is EventType.validation
    assert ev.payload["val_loss"] == 0.5237
    assert ev.payload.get("epoch") == 3
    assert ev.job_id == "J1"


def test_eval_loss_emits_validation_event() -> None:
    """Some trainers report held-out loss as eval_loss instead of val_loss."""
    ev = parse_line("epoch 4 step 768 eval_loss=0.412", job_id="J1")
    assert ev is not None
    assert ev.type is EventType.validation
    assert ev.payload["val_loss"] == 0.412
    assert ev.payload.get("epoch") == 4
    assert ev.payload.get("step") == 768


def test_oom_runtime_error_emits_oom_event() -> None:
    """Legacy `RuntimeError: CUDA out of memory` becomes an `oom` event."""
    ev = parse_line("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert ev is not None
    assert ev.type is EventType.oom
    assert "CUDA out of memory" in ev.payload["message"]


def test_oom_torch_class_name_emits_oom_event() -> None:
    """Modern `torch.cuda.OutOfMemoryError` form is also recognised."""
    ev = parse_line("torch.cuda.OutOfMemoryError: CUDA out of memory.")
    assert ev is not None
    assert ev.type is EventType.oom


def test_traceback_aggregated_into_single_error_event() -> None:
    """Traceback opener + body + summary line collapses into one error event."""
    parser = KohyaLineParser()
    lines = [
        "Traceback (most recent call last):",
        '  File "/sd-scripts/train_network.py", line 42, in <module>',
        "    main()",
        '  File "/sd-scripts/train_network.py", line 30, in main',
        "    raise ValueError('boom')",
        "ValueError: boom",
    ]
    events = [parser.parse_line(ln, job_id="J7") for ln in lines]
    # Only the closing summary line yields an event; the rest buffer.
    assert events[:-1] == [None] * (len(lines) - 1)
    flush = events[-1]
    assert flush is not None
    assert flush.type is EventType.error
    assert flush.job_id == "J7"
    assert flush.payload["summary"] == "ValueError: boom"
    assert flush.payload["traceback"].startswith("Traceback (")
    assert flush.payload["traceback"].endswith("ValueError: boom")
    # No truncation on a normal flush.
    assert "truncated" not in flush.payload


def test_traceback_truncates_after_max_lines() -> None:
    """An adversarial traceback longer than the cap is force-flushed."""
    parser = KohyaLineParser()
    feed = ["Traceback (most recent call last):"] + [
        f"  File \"x.py\", line {i}, in fn{i}" for i in range(80)
    ]
    error_events = []
    for line in feed:
        ev = parser.parse_line(line)
        if ev is not None and ev.type is EventType.error:
            error_events.append(ev)

    assert len(error_events) == 1
    ev = error_events[0]
    assert ev.payload.get("truncated") is True
    # 50 line cap → traceback text contains exactly 50 buffered lines.
    assert ev.payload["traceback"].count("\n") == 49


def test_cache_progress_throttled() -> None:
    """30 progress lines should collapse to a small number of events."""
    parser = KohyaLineParser()
    events = []
    for i in range(1, 31):
        line = f"caching latents:  {i*3}%|##        | {i}/30 [00:01<00:30,  1.00it/s]"
        ev = parser.parse_line(line)
        if ev is not None:
            events.append(ev)

    # First emit (i=1) + roughly every 10% step (~10/20/30 done) + terminal.
    assert 3 <= len(events) <= 5
    assert all(e.type is EventType.cache_progress for e in events)
    assert all(e.payload["phase"] == "latents" for e in events)
    # Terminal emit pinned to the final tick.
    assert events[-1].payload["done"] == 30
    assert events[-1].payload["total"] == 30


def test_cache_progress_phases_independent() -> None:
    """Latents and text-encoder phases keep independent throttle state."""
    parser = KohyaLineParser()
    a = parser.parse_line("caching latents: 5%| | 1/20")
    b = parser.parse_line("caching text encoder outputs: 5%| | 1/20")
    assert a is not None and a.payload["phase"] == "latents"
    assert b is not None and b.payload["phase"] == "text_encoder"
