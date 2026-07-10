"""Central error reporter — single funnel into ``ErrorReportStore``.

Every code path that wants to surface a failure (FastAPI exception
handler, scheduler job-finished hook, training assistant, frontend
``POST /api/error-reports``) goes through ``capture(...)`` here so the
shape of the persisted record stays consistent.

The reporter resolves the active store via ``app.py``'s module-level
singleton (``_error_report_store``). Tests patch the singleton on the
app module, exactly as they do for ``_settings_store`` /
``_sweep_store``. Callers that need to skip persistence (a probe, a
unit test) can pass ``store=None`` to short-circuit.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from lorahub.api.error_reports import (
    MAX_ERROR_MESSAGE_CHARS,
    MAX_ERROR_STACK_CHARS,
    MAX_ERROR_TITLE_CHARS,
    ErrorReport,
    ErrorReportStore,
    Severity,
    normalise_error_context,
    truncate_error_text,
)
from lorahub.core.redaction import redact_command_text, redact_data

log = logging.getLogger(__name__)


def _active_store() -> ErrorReportStore | None:
    """Resolve the lifespan-managed singleton without forcing an import
    cycle. Tests reach the same attribute via monkeypatch."""
    try:
        from lorahub.api import app as app_module  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return getattr(app_module, "_error_report_store", None)


def _active_dispatcher_and_threshold() -> tuple[Any, str]:
    """Resolve the dispatcher + the auto-send threshold from settings.

    Returns ``(None, "off")`` when either piece isn't available — the
    caller treats that as "store-only, no fan-out".
    """
    try:
        from lorahub.api import app as app_module  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None, "off"
    dispatcher = getattr(app_module, "_error_upstream_dispatcher", None)
    settings_store = getattr(app_module, "_settings_store", None)
    if dispatcher is None or settings_store is None:
        return None, "off"
    try:
        settings = settings_store.load()
    except Exception:  # noqa: BLE001
        return dispatcher, "off"
    threshold = getattr(settings, "error_upstream_auto_severity", "off") or "off"
    return dispatcher, threshold


_SEVERITY_RANK = {"info": 1, "warn": 2, "error": 3, "fatal": 4}


def _meets_threshold(report_severity: str, threshold: str) -> bool:
    """``threshold`` of ``"error"`` means "auto-send severity ≥ error";
    ``"all"`` is unconditional; ``"off"`` blocks everything."""
    if threshold == "all":
        return True
    if threshold == "off":
        return False
    cutoff = _SEVERITY_RANK.get(threshold, _SEVERITY_RANK["error"])
    return _SEVERITY_RANK.get(report_severity, 0) >= cutoff


def capture(
    *,
    severity: Severity,
    source: str,
    category: str,
    title: str,
    message: str,
    stack: str | None = None,
    context: dict[str, Any] | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    request_path: str | None = None,
    store: ErrorReportStore | None | object = ...,
) -> ErrorReport | None:
    """Persist a single error event.

    Returns the saved ``ErrorReport`` (so the caller can surface its id
    in a 500 response or a toast), or ``None`` when the store is not
    available (early-boot, tests, explicit opt-out).

    The sentinel ``store=...`` means "use the global singleton".
    Pass ``store=None`` to skip persistence entirely (still returns
    the constructed report so logging / tests can introspect it).
    """
    bounded_context = normalise_error_context(context)
    safe_context = redact_data(bounded_context)
    safe_title = truncate_error_text(
        title, MAX_ERROR_TITLE_CHARS, preserve_tail=False
    )
    safe_message = truncate_error_text(message, MAX_ERROR_MESSAGE_CHARS)
    safe_stack = truncate_error_text(stack, MAX_ERROR_STACK_CHARS)
    report = ErrorReport.create(
        severity=severity,
        source=redact_command_text(source),
        category=redact_command_text(category),
        title=redact_command_text(safe_title or "Untitled error"),
        message=redact_command_text(safe_message or "Unknown error"),
        stack=redact_command_text(safe_stack) if safe_stack is not None else None,
        context=safe_context,
        job_id=redact_command_text(job_id) if job_id is not None else None,
        request_id=(
            redact_command_text(request_id) if request_id is not None else None
        ),
        request_path=(
            redact_command_text(request_path) if request_path is not None else None
        ),
    )
    # Stamp the fingerprint up-front so both the local row and any
    # outbound copy share the same hash. Cheap and pure — see
    # error_upstream.fingerprint for the inputs it considers.
    try:
        from lorahub.api.error_upstream import compute_fingerprint  # noqa: PLC0415

        report.fingerprint = compute_fingerprint(report)
    except Exception:  # noqa: BLE001
        report.fingerprint = None
    target: ErrorReportStore | None
    if store is ...:
        target = _active_store()
    elif store is None:
        target = None
    else:
        assert isinstance(store, ErrorReportStore)
        target = store
    if target is not None:
        try:
            target.insert(report)
        except Exception as exc:  # noqa: BLE001
            # Error reporting is a secondary path. A locked/corrupt report DB
            # must never replace the original application failure.
            log.error(
                "could not persist error report %s: %s",
                report.id,
                redact_command_text(str(exc)),
            )
        else:
            # Auto-fan-out: only if the user has opted in via settings AND
            # the severity meets their threshold. Manual ``send_now`` from
            # the UI bypasses this gate so users can always force a single
            # report through even when auto is off.
            dispatcher, threshold = _active_dispatcher_and_threshold()
            if dispatcher is not None and _meets_threshold(severity, threshold):
                try:
                    target.update_upstream(report.id, status="queued")
                    report.upstream_status = "queued"
                    dispatcher.enqueue(report)
                except Exception as exc:  # noqa: BLE001
                    # Reporter must never raise back into the caller —
                    # store-only is still better than not logging at all.
                    log.error(
                        "could not enqueue report %s for upstream: %s",
                        report.id,
                        redact_command_text(str(exc)),
                    )
    # Always also tee into stderr/log: even if persistence raced a boot
    # window or was disabled, the operator's terminal still surfaces the
    # signal. Severity → log level: fatal/error => error, warn => warn,
    # info => info.
    log_level = {
        "fatal": logging.ERROR,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "info": logging.INFO,
    }.get(severity, logging.ERROR)
    log.log(
        log_level,
        "[error-report:%s] %s — %s",
        report.source,
        report.title,
        truncate_error_text(report.message, 4_000),
    )
    return report


def capture_exception(
    exc: BaseException,
    *,
    source: str,
    category: str,
    title: str,
    severity: Severity = "error",
    context: dict[str, Any] | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    request_path: str | None = None,
) -> ErrorReport | None:
    """Convenience wrapper: pull message + traceback off ``exc``."""
    stack = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, limit=100)
    ).rstrip()
    return capture(
        severity=severity,
        source=source,
        category=category,
        title=title,
        message=f"{type(exc).__name__}: {exc}",
        stack=stack,
        context=context,
        job_id=job_id,
        request_id=request_id,
        request_path=request_path,
    )


__all__ = ["capture", "capture_exception"]
