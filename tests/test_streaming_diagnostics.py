"""StreamingDiagnosticWatcher unit tests.

We don't want to spin up real subprocesses just to test the regex
matching, so the watcher is exercised directly via ``feed()``. The
tests cover:

- A canonical OOM line emits one ``diagnostic_warning`` event with the
  oom category.
- The 60-second suppression window prevents double-fire of the same
  category on adjacent lines.
- Different categories on adjacent lines both emit (suppression is
  per-category, not global).
- Non-matching lines emit nothing.
- Severity arbitration: when one line matches multiple rules, the
  highest-severity category wins.
"""

from __future__ import annotations

from typing import Iterable

from lorahub.api.streaming_diagnostics import StreamingDiagnosticWatcher
from lorahub.core.events import EventType, TrainingEvent


class _Sink:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def __call__(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def categories(self) -> list[str]:
        return [
            e.payload["category"]
            for e in self.events
            if e.type is EventType.diagnostic_warning
        ]


def _feed(watcher: StreamingDiagnosticWatcher, lines: Iterable[str]) -> None:
    for ln in lines:
        watcher.feed(ln)


def test_oom_line_emits_one_event() -> None:
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink, job_id="job-1")
    watcher.feed(
        "RuntimeError: CUDA out of memory. Tried to allocate 1.20 GiB",
        source="stderr",
    )
    assert sink.categories() == ["oom"]
    payload = sink.events[0].payload
    assert payload["severity"] == "error"
    assert payload["source"] == "stderr"
    assert "out of memory" in payload["evidence"].lower()


def test_repeat_within_window_is_suppressed() -> None:
    sink = _Sink()
    fake_clock = [0.0]
    watcher = StreamingDiagnosticWatcher(
        on_event=sink,
        clock=lambda: fake_clock[0],
        suppress_window_s=60.0,
    )
    watcher.feed("RuntimeError: CUDA out of memory")
    fake_clock[0] = 30.0
    watcher.feed("RuntimeError: CUDA out of memory (second time)")
    assert sink.categories() == ["oom"]


def test_repeat_after_window_re_fires() -> None:
    sink = _Sink()
    fake_clock = [0.0]
    watcher = StreamingDiagnosticWatcher(
        on_event=sink,
        clock=lambda: fake_clock[0],
        suppress_window_s=60.0,
    )
    watcher.feed("RuntimeError: CUDA out of memory")
    fake_clock[0] = 120.0
    watcher.feed("RuntimeError: CUDA out of memory (much later)")
    assert sink.categories() == ["oom", "oom"]


def test_distinct_categories_both_fire_within_window() -> None:
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink)
    watcher.feed("RuntimeError: CUDA out of memory. Tried to allocate 1 GiB")
    watcher.feed(
        "OSError: [Errno 28] No space left on device",
        source="stderr",
    )
    assert sorted(sink.categories()) == ["disk_full", "oom"]


def test_no_match_emits_nothing() -> None:
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink)
    watcher.feed("epoch 3/10 - loss=0.1234 - 0.42 it/s")
    assert sink.events == []


def test_higher_severity_wins_when_a_line_matches_multiple_rules() -> None:
    """A line that matches both ``subprocess_returncode`` (warn) and
    ``cuda_driver_mismatch`` (error) should surface only the error."""
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink)
    watcher.feed(
        "subprocess.CalledProcessError: returned non-zero exit status 1: "
        "CUDA driver version is insufficient for CUDA runtime version"
    )
    cats = sink.categories()
    assert cats == ["cuda_driver_mismatch"], cats


def test_blank_lines_are_ignored() -> None:
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink)
    watcher.feed("")
    watcher.feed("\n")
    watcher.feed("\r\n")
    assert sink.events == []


def test_evidence_is_truncated_for_huge_lines() -> None:
    sink = _Sink()
    watcher = StreamingDiagnosticWatcher(on_event=sink)
    huge = "x" * 5000 + " RuntimeError: CUDA out of memory"
    watcher.feed(huge)
    assert sink.categories() == ["oom"]
    assert len(sink.events[0].payload["evidence"]) <= 500
