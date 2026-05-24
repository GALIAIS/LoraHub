"""Tests for the live-preview module — prompt parsing + worker plumbing."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from lorahub.core.events import EventType, TrainingEvent
from lorahub.core.inference import (
    PreviewConfig,
    PreviewWorker,
    PromptSpec,
    StubInference,
    parse_prompts_file,
)


# --------------------------------------------------------------------------- #
# parse_prompts_file
# --------------------------------------------------------------------------- #


def test_parse_blank_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert parse_prompts_file(f) == []


def test_parse_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / "p.txt"
    f.write_text(
        "# header\n"
        "\n"
        "first prompt --w 768 --h 1024 --d 7 --s 30 --l 4.5 --n bad anatomy\n"
        "  \n"
        "# inline comment\n"
        "second prompt only\n"
    )
    out = parse_prompts_file(f)
    assert len(out) == 2
    a, b = out
    assert a.prompt == "first prompt"
    assert (a.width, a.height) == (768, 1024)
    assert a.seed == 7 and a.steps == 30 and a.cfg == 4.5
    assert a.negative == "bad anatomy"
    assert a.index == 0
    assert b.prompt == "second prompt only"
    # Defaults for unset fields:
    assert (b.width, b.height) == (1024, 1024)
    assert b.seed is None and b.steps is None and b.cfg is None and b.negative is None


def test_parse_negative_prompt_consumes_until_next_flag(tmp_path: Path) -> None:
    """`--n` is a free-form string; it must capture everything up to the
    next `--<flag>` rather than just the first word."""
    f = tmp_path / "p.txt"
    f.write_text(
        "scene description --n lowres, blurry, watermark, text --w 1280 --h 768\n"
    )
    [spec] = parse_prompts_file(f)
    assert spec.negative == "lowres, blurry, watermark, text"
    assert (spec.width, spec.height) == (1280, 768)


def test_parse_bad_int_kept_as_default(tmp_path: Path) -> None:
    """A malformed flag value is logged and dropped, not fatal."""
    f = tmp_path / "p.txt"
    f.write_text("anything --w not_a_number\n")
    [spec] = parse_prompts_file(f)
    assert spec.width == 1024  # default kept


# --------------------------------------------------------------------------- #
# StubInference
# --------------------------------------------------------------------------- #


def test_stub_writes_png(tmp_path: Path) -> None:
    lora = tmp_path / "step100" / "lora.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"fake")
    out = tmp_path / "samples" / "step100_00.png"
    spec = PromptSpec(prompt="hello", index=0, width=256, height=256, seed=42)

    StubInference().render(
        lora_path=lora,
        spec=spec,
        out_path=out,
        default_steps=24,
        default_cfg=5.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    img = Image.open(out)
    assert img.size == (256, 256)


# --------------------------------------------------------------------------- #
# PreviewWorker — end-to-end with stub
# --------------------------------------------------------------------------- #


def _make_worker(
    tmp_path: Path,
    *,
    prompts: str = "p1 --w 128 --h 128 --d 1\np2 --w 128 --h 128 --d 2\n",
) -> tuple[PreviewWorker, threading.Event, list[TrainingEvent]]:
    """Spin up a worker rooted at tmp_path with two stub prompts.

    Returns (worker, stop_evt, events_collected).
    """
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(prompts)
    output_dir = tmp_path / "output"
    samples_dir = tmp_path / "samples"
    output_dir.mkdir()
    samples_dir.mkdir()
    cfg = PreviewConfig(
        enabled=True,
        prompts_file=prompts_file,
        default_steps=4,
        default_cfg=5.0,
        samples_dir=samples_dir,
        output_dir=output_dir,
        poll_interval_s=0.1,
        # These tests assert the legacy "one sample_ready per prompt"
        # contract; the post-render artefacts (grid / animation /
        # png-metadata re-save) are exercised in their own dedicated
        # tests below. Pin them off so the assertions stay tight.
        grid_stitching=False,
        cross_ckpt_animation=False,
        png_metadata=False,
    )
    events: list[TrainingEvent] = []
    stop = threading.Event()
    worker = PreviewWorker(
        config=cfg,
        inference=StubInference(),
        on_event=events.append,
        job_id="J9",
        stop_evt=stop,
    )
    return worker, stop, events


def _drop_checkpoint(output_dir: Path, name: str) -> Path:
    d = output_dir / name
    d.mkdir()
    f = d / "adapter.safetensors"
    f.write_bytes(b"fake-lora-bytes")
    return f


def test_worker_renders_each_new_checkpoint_once(tmp_path: Path) -> None:
    worker, stop, events = _make_worker(tmp_path)

    # Pre-create one checkpoint so the first tick has work.
    _drop_checkpoint(worker.config.output_dir, "step100")

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        # 2 prompts × 1 ckpt → 2 PNGs and 2 sample_ready events.
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                len([e for e in events if e.type is EventType.sample_ready]) >= 2
            ):
                break
            time.sleep(0.1)

        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert len(sample_evs) == 2, f"got {len(sample_evs)} sample events"
        assert {e.payload["checkpoint"] for e in sample_evs} == {"step100"}
        assert {e.payload["prompt_index"] for e in sample_evs} == {0, 1}
        # PNG files actually exist:
        for ev in sample_evs:
            assert Path(ev.payload["path"]).is_file()

        # Hold steady — the worker must NOT re-render the same checkpoint
        # on subsequent ticks.
        before = len(sample_evs)
        time.sleep(0.5)
        sample_evs_after = [e for e in events if e.type is EventType.sample_ready]
        assert len(sample_evs_after) == before
    finally:
        stop.set()
        t.join(timeout=2)


def test_worker_finds_dp_nested_run_dir_layout(tmp_path: Path) -> None:
    """diffusion-pipe writes ckpts under `<output>/<UTC>/step{N}/`.

    The worker has to peek through that timestamp dir; a flat scan of
    `output_dir` would miss every checkpoint. We assert the worker also
    picks the alphabetically-last subdir (matching dp's own
    `get_most_recent_run_dir` selection).
    """
    worker, stop, events = _make_worker(tmp_path)
    # Stale older run that should NOT be picked up.
    older = worker.config.output_dir / "20250101_00-00-00"
    older.mkdir()
    _drop_checkpoint(older, "step50")
    # Active run — alphabetically later, so this is the one dp would
    # also resolve to via its `sorted([...])[-1]` rule.
    active = worker.config.output_dir / "20260518_05-37-00"
    active.mkdir()
    _drop_checkpoint(active, "step100")

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                len([e for e in events if e.type is EventType.sample_ready]) >= 2
            ):
                break
            time.sleep(0.1)
        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert len(sample_evs) == 2, f"got {len(sample_evs)} sample events"
        assert {e.payload["checkpoint"] for e in sample_evs} == {"step100"}
    finally:
        stop.set()
        t.join(timeout=2)


def test_worker_picks_up_new_checkpoint_added_after_start(tmp_path: Path) -> None:
    worker, stop, events = _make_worker(tmp_path)
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        # Drop a checkpoint after the worker is already running.
        time.sleep(0.2)
        _drop_checkpoint(worker.config.output_dir, "step200")
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                len([e for e in events if e.type is EventType.sample_ready]) >= 2
            ):
                break
            time.sleep(0.1)
        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert len(sample_evs) == 2
        assert sample_evs[0].payload["checkpoint"] == "step200"
    finally:
        stop.set()
        t.join(timeout=2)


def test_worker_no_op_when_disabled(tmp_path: Path) -> None:
    worker, stop, events = _make_worker(tmp_path)
    worker.config.enabled = False
    _drop_checkpoint(worker.config.output_dir, "step100")

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        time.sleep(0.4)
        assert events == []
    finally:
        stop.set()
        t.join(timeout=2)


def test_worker_no_op_when_prompts_file_missing(tmp_path: Path) -> None:
    worker, stop, events = _make_worker(tmp_path, prompts="")
    _drop_checkpoint(worker.config.output_dir, "step100")
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        time.sleep(0.4)
        assert events == []
    finally:
        stop.set()
        t.join(timeout=2)


# --------------------------------------------------------------------------- #
# Cut-3 features: notify wake, budget enforcement, skip semantics
# --------------------------------------------------------------------------- #


def test_notify_checkpoint_wakes_worker_before_poll(tmp_path: Path) -> None:
    """Even with a slow poll interval, a `notify_checkpoint` call from
    the event sink should make the worker render within milliseconds
    rather than waiting for the next tick."""
    worker, stop, events = _make_worker(tmp_path)
    # Bump poll interval far above the test's deadline so the only way
    # work happens within the window is via notify.
    worker.config.poll_interval_s = 30.0

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        # Drop a checkpoint and poke the worker.
        _drop_checkpoint(worker.config.output_dir, "step777")
        worker.notify_checkpoint("step777")

        deadline = time.time() + 3
        while time.time() < deadline:
            if (
                len([e for e in events if e.type is EventType.sample_ready]) >= 2
            ):
                break
            time.sleep(0.05)

        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert len(sample_evs) == 2
        assert {e.payload["checkpoint"] for e in sample_evs} == {"step777"}
    finally:
        stop.set()
        t.join(timeout=2)


def test_budget_drops_remaining_prompts(tmp_path: Path) -> None:
    """When the per-checkpoint render budget is exhausted, the worker
    must skip the remaining prompts and emit a warning log instead of
    pushing through and stalling training."""
    # Force-feed a tight budget by spending real time inside the
    # inference render. Use a fake backend that sleeps deliberately.
    class _SlowInference:
        def render(self, *, lora_path, spec, out_path, default_steps, default_cfg):
            from PIL import Image  # noqa: PLC0415
            time.sleep(0.4)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(out_path)

    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(
        "p0 --w 128 --h 128\np1 --w 128 --h 128\np2 --w 128 --h 128\n"
    )
    output_dir = tmp_path / "output"
    samples_dir = tmp_path / "samples"
    output_dir.mkdir()
    samples_dir.mkdir()
    cfg = PreviewConfig(
        enabled=True,
        prompts_file=prompts_file,
        default_steps=4,
        default_cfg=5.0,
        samples_dir=samples_dir,
        output_dir=output_dir,
        poll_interval_s=0.1,
        # 0.5s budget — roughly enough for one 0.4s render, definitely
        # not three.
        max_render_time_per_ckpt_s=0.5,
        budget_fraction=1.0,
        grid_stitching=False,
        cross_ckpt_animation=False,
        png_metadata=False,
    )
    events: list[TrainingEvent] = []
    stop = threading.Event()
    worker = PreviewWorker(
        config=cfg,
        inference=_SlowInference(),
        on_event=events.append,
        job_id="J9",
        stop_evt=stop,
    )
    _drop_checkpoint(output_dir, "step100")

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        # Wait for either a budget-warning log or sample events to
        # stabilise. 2 seconds is plenty for the budget to expire.
        deadline = time.time() + 3
        while time.time() < deadline:
            if any(
                e.type is EventType.log
                and e.payload.get("level") == "warning"
                and "budget exceeded" in e.payload.get("message", "")
                for e in events
            ):
                break
            time.sleep(0.05)

        sample_count = sum(1 for e in events if e.type is EventType.sample_ready)
        warning_logs = [
            e for e in events
            if e.type is EventType.log and e.payload.get("level") == "warning"
        ]
        assert sample_count >= 1, "expected at least one render to land"
        assert sample_count < 3, f"expected fewer than 3 renders, got {sample_count}"
        assert warning_logs, "expected a budget-exceeded warning event"
    finally:
        stop.set()
        t.join(timeout=2)


def test_skipped_inference_does_not_emit_error(tmp_path: Path) -> None:
    """If the backend raises an `*Skipped` exception (e.g. low VRAM),
    the worker must NOT emit an error log — it's a soft fail by design."""

    class _AlwaysSkipped:
        def render(self, *, lora_path, spec, out_path, default_steps, default_cfg):
            class InferenceSkipped(Exception):
                pass

            raise InferenceSkipped("fake VRAM pressure")

    worker, stop, events = _make_worker(tmp_path)
    worker.inference = _AlwaysSkipped()
    _drop_checkpoint(worker.config.output_dir, "step100")
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        time.sleep(0.5)
        # No error logs even though render() always raised.
        error_logs = [
            e for e in events
            if e.type is EventType.log and e.payload.get("level") == "error"
        ]
        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert error_logs == [], f"unexpected error logs: {error_logs}"
        assert sample_evs == [], "no PNG should have been produced"
    finally:
        stop.set()
        t.join(timeout=2)


def test_genuine_inference_failure_does_emit_error(tmp_path: Path) -> None:
    """Counter-test: a non-Skipped exception should still surface as
    an error event so the user sees real failures in the events tab."""

    class _AlwaysCrash:
        def render(self, *, lora_path, spec, out_path, default_steps, default_cfg):
            raise RuntimeError("real boom")

    worker, stop, events = _make_worker(tmp_path)
    worker.inference = _AlwaysCrash()
    _drop_checkpoint(worker.config.output_dir, "step100")
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    try:
        deadline = time.time() + 2
        while time.time() < deadline:
            if any(
                e.type is EventType.log and e.payload.get("level") == "error"
                for e in events
            ):
                break
            time.sleep(0.05)
        error_logs = [
            e for e in events
            if e.type is EventType.log and e.payload.get("level") == "error"
        ]
        assert error_logs, "expected an error log for a real crash"
    finally:
        stop.set()
        t.join(timeout=2)
