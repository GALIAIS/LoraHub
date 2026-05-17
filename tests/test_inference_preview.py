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
