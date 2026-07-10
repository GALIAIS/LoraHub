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
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable

from lorahub.api.error_reports import ErrorReport, ErrorReportStore
from lorahub.core.redaction import redact_command_text
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
    _delivery_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    # ------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------ #

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.clear()
            sink, sink_error = self._resolve_sink()
            sink_enabled = sink is not None and sink_error is None
            if sink_enabled:
                try:
                    queued_ids = {pending.report.id for pending in self._queue}
                    for report in self.store.list_pending_upstream():
                        if report.id not in queued_ids:
                            self._queue.append(_Pending(report=report))
                            queued_ids.add(report.id)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "could not recover pending upstream reports: %s",
                        _exception_detail(exc),
                    )
            t = threading.Thread(target=self._run, name=self.name, daemon=True)
            self._thread = t
            t.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        with self._lock:
            if self._thread is t and (t is None or not t.is_alive()):
                self._thread = None
            elif t is not None and t.is_alive():
                log.warning("upstream dispatcher did not stop within %.1fs", timeout)

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
        with self._lock:
            self._queue = [p for p in self._queue if p.report.id != report.id]
        with self._delivery_lock:
            try:
                latest = self.store.get(report.id) or report
            except Exception as exc:  # noqa: BLE001
                detail = _exception_detail(exc)
                log.error("could not load report %s for delivery: %s", report.id, detail)
                return SendResult(
                    ok=False,
                    error=f"could not load local error report: {detail}",
                    retryable=True,
                )
            if latest.upstream_status == "sent":
                return SendResult(
                    ok=True,
                    upstream_id=latest.upstream_id or "",
                    url=latest.upstream_url or "",
                )
            sink, sink_error = self._resolve_sink()
            if sink_error:
                res = SendResult(
                    ok=False,
                    error=f"upstream channel configuration failed: {sink_error}",
                    retryable=False,
                )
            elif sink is None:
                res = SendResult(
                    ok=False,
                    error="upstream channel disabled",
                    retryable=False,
                )
            else:
                res = _send_with_guard(sink, latest)
            res = _sanitise_result(res)
            self._record_result(latest, res)
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

            try:
                self._deliver_pending(ready)
            except Exception as exc:  # noqa: BLE001
                # A local store or custom sink implementation must not kill
                # the only worker thread and strand every later report.
                ready.attempts += 1
                detail = _exception_detail(exc)
                log.error(
                    "unexpected upstream worker failure for %s: %s",
                    ready.report.id,
                    detail,
                )
                terminal = ready.attempts >= _MAX_ATTEMPTS
                self._record_result(
                    ready.report,
                    SendResult(
                        ok=False,
                        error=(
                            f"local upstream worker failure: {detail}"
                            + (
                                f" (retry limit reached after {ready.attempts} attempts)"
                                if terminal
                                else ""
                            )
                        ),
                        retryable=not terminal,
                    ),
                )
                if not terminal:
                    self._schedule_retry(ready)

    def _resolve_sink(self) -> tuple[UpstreamSink | None, str | None]:
        try:
            return self.sink_factory(), None
        except Exception as exc:  # noqa: BLE001
            detail = _exception_detail(exc)
            log.error("could not resolve error-report upstream sink: %s", detail)
            return None, detail

    def _deliver_pending(self, ready: _Pending) -> None:
        with self._delivery_lock:
            latest = self.store.get(ready.report.id)
            if latest is None or latest.upstream_status == "sent":
                return
            ready.report = latest
            sink, sink_error = self._resolve_sink()
            if sink_error:
                self._record_result(
                    ready.report,
                    SendResult(
                        ok=False,
                        error=f"upstream channel configuration failed: {sink_error}",
                        retryable=False,
                    ),
                )
                return
            if sink is None:
                # Re-enabling does not silently transmit old rows; the user
                # can explicitly retry reports after changing configuration.
                self._record_result(
                    ready.report,
                    SendResult(
                        ok=False,
                        error="channel turned off",
                        retryable=False,
                    ),
                )
                return

            ready.attempts += 1
            res = _sanitise_result(_send_with_guard(sink, ready.report))
            if not res.ok and res.retryable and ready.attempts >= _MAX_ATTEMPTS:
                detail = res.error or "upstream delivery failed"
                res = SendResult(
                    ok=False,
                    upstream_id=res.upstream_id,
                    url=res.url,
                    error=(
                        f"{detail} (retry limit reached after "
                        f"{ready.attempts} attempts)"
                    ),
                    retryable=False,
                )
            self._record_result(ready.report, res)
            if not res.ok and res.retryable and ready.attempts < _MAX_ATTEMPTS:
                self._schedule_retry(ready)

    def _schedule_retry(self, ready: _Pending) -> None:
        delay = _BACKOFF_SCHEDULE_S[
            min(max(ready.attempts - 1, 0), len(_BACKOFF_SCHEDULE_S) - 1)
        ]
        ready.next_attempt_at = time.time() + delay
        with self._lock:
            if not any(item.report.id == ready.report.id for item in self._queue):
                self._queue.append(ready)
                self._wake.set()

    def _record_result(self, report: ErrorReport, res: SendResult) -> None:
        safe_result = _sanitise_result(res)
        try:
            status = (
                "sent"
                if safe_result.ok
                else ("retrying" if safe_result.retryable else "failed")
            )
            self.store.update_upstream(
                report.id,
                status=status,
                url=safe_result.url or None,
                upstream_id=safe_result.upstream_id or None,
                error=safe_result.error or None,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "could not record upstream result for %s: %s",
                report.id,
                _exception_detail(exc),
            )


def _send_with_guard(sink: UpstreamSink, report: ErrorReport) -> SendResult:
    """Wrap sink.send so a buggy custom sink can't crash the worker."""
    try:
        return _sanitise_result(sink.send(report))
    except Exception as exc:  # noqa: BLE001
        detail = _exception_detail(exc)
        log.error("sink.send raised for %s: %s", report.id, detail)
        return SendResult(ok=False, error=detail, retryable=False)


def _exception_detail(exc: BaseException) -> str:
    return redact_command_text(f"{type(exc).__name__}: {exc}")[:500]


def _safe_upstream_url(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    try:
        parsed = urllib.parse.urlsplit(candidate)
        _ = parsed.port
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""
    return candidate


def _sanitise_result(res: SendResult) -> SendResult:
    return SendResult(
        ok=bool(res.ok),
        upstream_id=redact_command_text(str(res.upstream_id or ""))[:256],
        url=_safe_upstream_url(str(res.url or "")),
        error=redact_command_text(str(res.error or ""))[:2000],
        retryable=bool(res.retryable),
    )


__all__ = ["UpstreamDispatcher"]
