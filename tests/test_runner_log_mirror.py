"""End-to-end tests for ``SubprocessRunner.training.log`` mirroring.

These spawn a real Python subprocess that prints to stdout + stderr and
then exits, so we exercise the full pump → mirror path that the
diagnostic flow ultimately depends on. Each test pins a small workspace
under ``tmp_path`` so concurrent runs don't share state.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.events import EventType, TrainingEvent
from lorahub.core.backends._common.runner import (
    SubprocessRunner,
    _TRAINING_LOG_FILENAME,
)


def _capture_listener() -> tuple[list[TrainingEvent], "list[TrainingEvent]"]:
    events: list[TrainingEvent] = []

    def on_event(ev: TrainingEvent) -> None:
        events.append(ev)

    return events, on_event  # type: ignore[return-value]


def _identity_parser(raw: str, *, job_id: str | None) -> TrainingEvent | None:
    """Re-emit the line as a log event so test bodies can also check the
    structured stream while the mirror file gets the raw bytes."""
    line = raw.rstrip("\r\n")
    if not line:
        return None
    return TrainingEvent(
        type=EventType.log,
        payload={"level": "info", "source": "stub", "message": line},
        job_id=job_id,
    )


def _stub_argv(body: str) -> list[str]:
    """Wrap ``body`` in a python -c invocation that flushes after each line.

    sd-scripts and accelerate are very chatty — block-buffered stdout
    would let lines arrive in giant gulps and mask interleaving bugs.
    Forcing line buffering matches the trainer's real behaviour.
    """
    return [sys.executable, "-u", "-c", textwrap.dedent(body)]


def test_training_log_records_stdout_and_stderr(tmp_path: Path) -> None:
    """Every trainer line should appear in workspace/training.log."""
    events, on_event = _capture_listener()
    runner = SubprocessRunner(
        argv=_stub_argv(
            """
            import sys
            print("hello stdout", flush=True)
            print("hello stderr", file=sys.stderr, flush=True)
            sys.exit(0)
            """
        ),
        workspace=tmp_path,
        on_event=on_event,
        parse_line=_identity_parser,
    )
    runner.start()
    rc = runner.wait().returncode
    assert rc == 0

    log_path = tmp_path / _TRAINING_LOG_FILENAME
    assert log_path.is_file(), "training.log should be created next to events"
    body = log_path.read_text(encoding="utf-8")
    assert "hello stdout" in body
    assert "hello stderr" in body
    # The spawn / exit banners frame the run so multi-spawn workspaces
    # stay navigable when grepped.
    assert "=== spawn " in body
    assert "=== exit " in body
    assert "returncode=0" in body


def test_training_log_appends_across_spawns(tmp_path: Path) -> None:
    """Resume / rerun-in-place must not nuke the prior run's log."""
    for marker in ("first-spawn", "second-spawn"):
        events, on_event = _capture_listener()
        runner = SubprocessRunner(
            argv=_stub_argv(f"print('{marker}', flush=True)"),
            workspace=tmp_path,
            on_event=on_event,
            parse_line=_identity_parser,
        )
        runner.start()
        runner.wait()

    body = (tmp_path / _TRAINING_LOG_FILENAME).read_text(encoding="utf-8")
    assert body.count("=== spawn ") == 2
    assert "first-spawn" in body and "second-spawn" in body


def test_training_log_disabled_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch keeps the file off disk for users who can't spare it."""
    monkeypatch.setenv("LORAHUB_DISABLE_TRAINING_LOG", "1")
    events, on_event = _capture_listener()
    runner = SubprocessRunner(
        argv=_stub_argv("print('should-not-be-logged')"),
        workspace=tmp_path,
        on_event=on_event,
        parse_line=_identity_parser,
    )
    runner.start()
    runner.wait()
    assert not (tmp_path / _TRAINING_LOG_FILENAME).exists()


def test_training_log_captures_stderr_traceback(tmp_path: Path) -> None:
    """The whole point: non-zero exit traceback must land in the file.

    Mirrors the anima_lora user report where a CalledProcessError
    bubbled up from accelerate's launcher; the diagnoser pointed at
    training.log but the file didn't exist. With the mirror in place,
    the file now contains the exact stderr that justified the diagnosis.
    """
    events, on_event = _capture_listener()
    runner = SubprocessRunner(
        argv=_stub_argv(
            """
            import sys, traceback
            try:
                raise RuntimeError("simulated failure for traceback capture")
            except RuntimeError:
                traceback.print_exc()
            sys.exit(1)
            """
        ),
        workspace=tmp_path,
        on_event=on_event,
        parse_line=_identity_parser,
    )
    runner.start()
    rc = runner.wait().returncode
    assert rc == 1

    body = (tmp_path / _TRAINING_LOG_FILENAME).read_text(encoding="utf-8")
    assert "RuntimeError: simulated failure for traceback capture" in body
    assert "Traceback (most recent call last)" in body
    assert "returncode=1" in body


def test_listener_failure_does_not_kill_pipe_pump(tmp_path: Path) -> None:
    """A broken terminal/UI listener must not strand the trainer process."""
    calls = 0

    def broken_listener(_ev: TrainingEvent) -> None:
        nonlocal calls
        calls += 1
        raise OSError(22, "Invalid argument")

    runner = SubprocessRunner(
        argv=_stub_argv(
            """
            import sys
            print("line-before-listener-error", flush=True)
            sys.exit(0)
            """
        ),
        workspace=tmp_path,
        on_event=broken_listener,
        parse_line=_identity_parser,
    )
    runner.start()
    rc = runner.wait().returncode

    assert rc == 0
    assert calls >= 1
    body = (tmp_path / _TRAINING_LOG_FILENAME).read_text(encoding="utf-8")
    assert "line-before-listener-error" in body
    assert "lorahub listener error: OSError(22, 'Invalid argument')" in body
    assert "returncode=0" in body
