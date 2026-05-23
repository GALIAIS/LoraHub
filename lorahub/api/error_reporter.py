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

from lorahub.api.error_reports import ErrorReport, ErrorReportStore, Severity

log = logging.getLogger(__name__)


def _active_store() -> ErrorReportStore | None:
    """Resolve the lifespan-managed singleton without forcing an import
    cycle. Tests reach the same attribute via monkeypatch."""
    try:
        from lorahub.api import app as app_module  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return getattr(app_module, "_error_report_store", None)


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
    report = ErrorReport.create(
        severity=severity,
        source=source,
        category=category,
        title=title,
        message=message,
        stack=stack,
        context=context,
        job_id=job_id,
        request_id=request_id,
        request_path=request_path,
    )
    target: ErrorReportStore | None
    if store is ...:
        target = _active_store()
    elif store is None:
        target = None
    else:
        assert isinstance(store, ErrorReportStore)
        target = store
    if target is not None:
        target.insert(report)
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
    log.log(log_level, "[error-report:%s] %s — %s", source, title, message)
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
        traceback.format_exception(type(exc), exc, exc.__traceback__)
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