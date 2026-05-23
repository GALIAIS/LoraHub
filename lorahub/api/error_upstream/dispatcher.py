"""Async fan-out from the local store to a configured sink.

The dispatcher owns one background thread that picks up reports
needing upstream delivery and pushes them through the active sink.
``capture()`` enqueues a fingerprint-tagged job; the thread drains
the queue with exponential backoff per failure.

Failure semantics:

* ``retryable`` send results stay in the queue with an attempt
  counter; we back off 5s → 30s → 2 min → 10 min → 30 min capped at
  6 attempts. After the cap the report is marked
  ``upstream_status='failed'`` and the user can retry manually from
  the UI.
* Non-retryable failures (4xx config errors, sink not configured)
  short-circuit immediately to ``failed`` so the user sees the issue
  on the next list refresh rather than burning through the retry
  budget.

The dispatcher is **opt-in** — a None sink keeps the queue empty
forever, ensuring no traffic ever leaves the box without explicit
user action.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from lorahub.api.error_reports import ErrorReport, ErrorReportStore
from .sinks import SendResult, UpstreamSink

log = logging.getLogger(__name__)


_BACKOFF_SCHEDULE_S: tuple[float, ...] = (5.0, 30.0, 120.0, 600.0, 1800.0)
_MAX_ATTEMPTS = len(_BACKOFF_SCHEDULE_S) + 1


@dataclass
class _Pending:
    report: ErrorReport
    attempts: int = 0
    next_attempt_at: float = 0.0


@dataclass
class UpstreamDispatcher:
    """Single-thread upstream queue.

    ``store`` is mutated to reflect the upstream state of every
    processed report (``upstream_status`` / ``upstream_url`` /
    ``upstream_error`` / ``sent_at``). The store's schema needs the
    matching columns for those writes to be visible to the UI; the
    dispatcher itself doesn't ALTER tables.
    """

    store: ErrorReportStore
    sink_factory: Callable[[], UpstreamSink | None]
    poll_interval_s: float = 0.5
    name: str = "lorahub-upstream"

    _queue: list[_Pending] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    # ------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        t = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread = t
        t.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------ #

    def enqueue(self, report: ErrorReport) -> None:
        """Add a report to the outbound queue."""
        with self._lock:
            # De-dupe by id so a manual "retry" call doesn't double-enqueue.
            if any(p.report.id == report.id for p in self._queue):
                return
            self._queue.append(_Pending(report=report))
            self._wake.set()

    def send_now(self, report: ErrorReport) -> SendResult:
        """Synchronous send for the manual ``send_now`` button.

        Bypasses the queue. Updates the store with the outcome so the
        UI shows the same status whether a report came through the
        background thread or a one-shot user action.
        """
        sink = self.sink_factory()
        if sink is None:
            res = SendResult(ok=False, error="upstream channel disabled", retryable=False)
        else:
            res = _send_with_guard(sink, report)
        self._record_result(report, res)
        return res

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            ready: _Pending | None
            with self._lock:
                # Pop the first item whose backoff has expired.
                ready = next(
                    (p for p in self._queue if p.next_attempt_at <= now),
                    None,
                )
                if ready is not None:
                    self._queue.remove(ready)
            if ready is None:
                # Sleep until the next attempt is due (or until enqueue
                # wakes us). Bounded by ``poll_interval_s`` so a clock
                # jump can't strand us.
                self._wake.wait(timeout=self.poll_interval_s)
                self._wake.clear()
                continue

            sink = self.sink_factory()
            if sink is None:
                # Channel turned off mid-flight — record as 'skipped' and
                # don't retry. The user can re-enable + manually retry.
                self._record_result(
                    ready.report,
                    SendResult(ok=False, error="channel turned off", retryable=False),
                )
                continue

            ready.attempts += 1
            res = _send_with_guard(sink, ready.report)
            self._record_result(ready.report, res)
            if res.ok or not res.retryable or ready.attempts >= _MAX_ATTEMPTS:
                continue

            # Schedule the next attempt with exponential backoff.
            delay = _BACKOFF_SCHEDULE_S[
                min(ready.attempts - 1, len(_BACKOFF_SCHEDULE_S) - 1)
            ]
            ready.next_attempt_at = time.time() + delay
            with self._lock:
                self._queue.append(ready)

    def _record_result(self, report: ErrorReport, res: SendResult) -> None:
        try:
            status = (
                "sent"
                if res.ok
                else ("retrying" if res.retryable else "failed")
            )
            self.store.update_upstream(
                report.id,
                status=status,
                url=res.url or None,
                upstream_id=res.upstream_id or None,
                error=res.error or None,
            )
        except Exception:  # noqa: BLE001
            log.exception("could not record upstream result for %s", report.id)


def _send_with_guard(sink: UpstreamSink, report: ErrorReport) -> SendResult:
    """Wrap sink.send so a buggy custom sink can't crash the worker."""
    try:
        return sink.send(report)
    except Exception as exc:  # noqa: BLE001
        log.exception("sink.send raised for %s", report.id)
        return SendResult(ok=False, error=repr(exc)[:300], retryable=False)


__all__ = ["UpstreamDispatcher"]
