"""Tests for the kohya subprocess runner.

We don't have a real kohya checkout in CI, so these tests use the active
Python interpreter to run a small inline script that emits the same shape
of stdout kohya produces.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.backends.kohya.runner import KohyaRunner
from lorahub.core.events import EventType, TrainingEvent


@pytest.fixture
def fake_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_kohya.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, time
            print("loading model from sdxl_base.safetensors", flush=True)
            print("steps:   1%|          | 1/3 [00:01<00:02,  1.00s/it, avr_loss=0.42]", flush=True)
            print("steps:   2%|          | 2/3 [00:02<00:01,  1.00s/it, avr_loss=0.31]", flush=True)
            print("epoch 1/1", flush=True)
            print("saving checkpoint: out.safetensors", flush=True)
            sys.exit(0)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script


def _collect(script: Path) -> tuple[list[TrainingEvent], int]:
    events: list[TrainingEvent] = []
    runner = KohyaRunner(
        python=Path(sys.executable),
        script=script,
        argv=[],
        workspace=script.parent,
        on_event=events.append,
        job_id="J",
    )
    runner.start()
    result = runner.wait(timeout=30)
    return events, result.returncode


def test_runner_streams_events_in_order(fake_script: Path) -> None:
    events, rc = _collect(fake_script)
    assert rc == 0

    types = [e.type for e in events]
    assert EventType.step in types
    assert EventType.epoch_end in types
    assert EventType.checkpoint_saved in types
    assert types[-1] is EventType.done


def test_runner_emits_done_with_returncode(fake_script: Path) -> None:
    events, _ = _collect(fake_script)
    done = [e for e in events if e.type is EventType.done]
    assert len(done) == 1
    assert done[0].payload["returncode"] == 0
    assert "duration_s" in done[0].payload


def test_runner_propagates_job_id(fake_script: Path) -> None:
    events, _ = _collect(fake_script)
    assert all(e.job_id == "J" for e in events)


def test_runner_captures_nonzero_exit(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(7)\n", encoding="utf-8")
    events: list[TrainingEvent] = []
    runner = KohyaRunner(
        python=Path(sys.executable),
        script=script,
        argv=[],
        workspace=tmp_path,
        on_event=events.append,
    )
    runner.start()
    result = runner.wait(timeout=30)
    assert result.returncode == 7
    assert events[-1].type is EventType.done
    assert events[-1].payload["returncode"] == 7


def test_double_start_rejected(fake_script: Path) -> None:
    runner = KohyaRunner(
        python=Path(sys.executable),
        script=fake_script,
        argv=[],
        workspace=fake_script.parent,
        on_event=lambda _e: None,
    )
    runner.start()
    with pytest.raises(RuntimeError, match="already"):
        runner.start()
    runner.wait(timeout=30)
