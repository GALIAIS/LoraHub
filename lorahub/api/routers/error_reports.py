"""``/api/error-reports`` — read / submit / clear the local error log.

Backed by ``app_module._error_report_store`` so tests can swap it via
monkeypatch. The endpoints are deliberately small:

* ``GET /api/error-reports`` — paginated list, supports filter by
  ``severity`` / ``source`` / ``job_id`` / ``q`` (substring of title +
  message).
* ``GET /api/error-reports/{id}`` — full record incl. ``context`` blob.
* ``POST /api/error-reports`` — frontend-side capture entry. The body
  schema mirrors ``ErrorReport.create`` but the server stamps id /
  timestamp / version / platform — clients can't forge those.
* ``DELETE /api/error-reports/{id}`` — drop one row.
* ``POST /api/error-reports/clear`` — drop everything.
* ``GET /api/error-reports/export`` — return the full store as
  newline-delimited JSON for download.

No auth: this is a single-user local tool (same as the rest of the
LoraHub API). Egress to a remote service is an opt-in user action,
not a server feature.
"""

from __future__ import annotations

import io
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.error_reports import ErrorReport, Severity

router = APIRouter(prefix="/api")


class _ReportOut(BaseModel):
    id: str
    timestamp: str
    severity: Severity
    source: str
    category: str
    title: str
    message: str
    stack: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    job_id: str | None = None
    request_id: str | None = None
    request_path: str | None = None
    version: str
    platform: str

    @classmethod
    def from_report(cls, r: ErrorReport) -> "_ReportOut":
        return cls.model_validate(r.to_dict())


class _ListOut(BaseModel):
    items: list[_ReportOut]
    total: int
    limit: int
    offset: int


class _CreateBody(BaseModel):
    severity: Severity = "error"
    source: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=300)
    message: str = Field(min_length=1, max_length=20_000)
    stack: str | None = Field(default=None, max_length=200_000)
    context: dict[str, Any] | None = None
    job_id: str | None = None
    # Frontend-supplied request id / path so the report links back to
    # the API call that triggered it (e.g. the 500 response carries an
    # X-Request-ID header that the toast plumbs into here).
    request_id: str | None = None
    request_path: str | None = None


class _CreateResponse(BaseModel):
    id: str


def _store():
    store = getattr(app_module, "_error_report_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="error-report store not ready")
    return store


@router.get("/error-reports", response_model=_ListOut)
def list_reports(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    severity: Severity | None = Query(default=None),
    source: str | None = Query(default=None, max_length=64),
    job_id: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
) -> _ListOut:
    store = _store()
    items = store.list(limit=limit, offset=offset, severity=severity,
                       source=source, job_id=job_id)
    if q:
        needle = q.casefold()
        items = [
            r for r in items
            if needle in r.title.casefold() or needle in r.message.casefold()
        ]
    return _ListOut(
        items=[_ReportOut.from_report(r) for r in items],
        total=store.count(),
        limit=limit,
        offset=offset,
    )


@router.get("/error-reports/export")
def export_reports() -> StreamingResponse:
    """Stream the registry as newline-delimited JSON.

    Returned as a downloadable attachment so users can open it in a
    text editor or paste into a GitHub issue without going through the
    UI list.
    """
    import json

    store = _store()
    rows = store.list(limit=1000, offset=0)

    def _gen():  # type: ignore[no-untyped-def]
        for r in rows:
            yield json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(
        _gen(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": (
                'attachment; filename="lorahub-error-reports.ndjson"'
            ),
        },
    )


@router.get("/error-reports/{report_id}", response_model=_ReportOut)
def get_report(report_id: str) -> _ReportOut:
    store = _store()
    rec = store.get(report_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="report not found")
    return _ReportOut.from_report(rec)


@router.post("/error-reports", response_model=_CreateResponse, status_code=201)
def create_report(body: _CreateBody, request: Request) -> _CreateResponse:
    """Persist a frontend-side error.

    The frontend's ErrorBoundary, ``window.onerror`` listener, and the
    toast layer all funnel through this endpoint. We tag the request
    id from the middleware so a server-side 500 and the matching
    frontend toast end up linkable in the registry.
    """
    from lorahub.api.error_reporter import capture  # noqa: PLC0415

    rid = body.request_id or getattr(request.state, "request_id", None)
    report = capture(
        severity=body.severity,
        source=body.source,
        category=body.category,
        title=body.title,
        message=body.message,
        stack=body.stack,
        context=body.context or {},
        job_id=body.job_id,
        request_id=rid,
        request_path=body.request_path,
    )
    if report is None:
        # Reporter returns None when the singleton is unavailable; surface
        # that so the caller doesn't think the row is durable.
        raise HTTPException(
            status_code=503, detail="error-report store not ready",
        )
    return _CreateResponse(id=report.id)


@router.delete("/error-reports/{report_id}", status_code=204)
def delete_report(report_id: str) -> None:
    store = _store()
    if not store.delete(report_id):
        raise HTTPException(status_code=404, detail="report not found")


class _ClearResponse(BaseModel):
    deleted: int


@router.post("/error-reports/clear", response_model=_ClearResponse)
def clear_reports() -> _ClearResponse:
    store = _store()
    return _ClearResponse(deleted=store.clear())


# ---------------------------------------------------------------------- #
# Upstream fan-out (GitLab issues / webhook)
# ---------------------------------------------------------------------- #


class _UpstreamSendOut(BaseModel):
    """Outcome of one ``send_now`` call surfaced to the UI.

    ``status`` mirrors the ``upstream_status`` column the dispatcher
    writes. Successful sends carry ``url`` so the UI can deep-link to
    the GitLab issue / webhook receipt.
    """
    ok: bool
    status: str
    url: str | None = None
    upstream_id: str | None = None
    error: str | None = None


def _dispatcher():
    from lorahub.api import app as app_module  # noqa: PLC0415

    dispatcher = getattr(app_module, "_error_upstream_dispatcher", None)
    if dispatcher is None:
        raise HTTPException(
            status_code=503, detail="error-upstream dispatcher not ready",
        )
    return dispatcher


@router.post("/error-reports/{report_id}/send", response_model=_UpstreamSendOut)
def send_report_now(report_id: str) -> _UpstreamSendOut:
    """Fire one report through the configured sink synchronously.

    This bypasses the auto-send gate so the user can always push a
    warn / info row from the UI even when auto-send is set to
    error-only. Returns the resolved status so the caller can refresh
    the list after a 2xx without an extra round-trip.
    """
    store = _store()
    rec = store.get(report_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="report not found")
    dispatcher = _dispatcher()
    res = dispatcher.send_now(rec)
    status = "sent" if res.ok else "failed"
    return _UpstreamSendOut(
        ok=res.ok,
        status=status,
        url=res.url or None,
        upstream_id=res.upstream_id or None,
        error=res.error or None,
    )


class _UpstreamHealthOut(BaseModel):
    ok: bool
    channel: str
    url: str | None = None
    error: str | None = None


class _UpstreamHealthIn(BaseModel):
    """Optional ad-hoc config the UI can probe before saving.

    Empty body falls back to the persisted Settings — used by the
    "test connection" button on a saved configuration. When the body
    is populated, we build a one-off SinkConfig from the form draft
    and probe with that, so the user can validate a token before
    committing it to settings.json.
    """
    channel: str | None = None
    gitlab_base_url: str | None = None
    gitlab_repo: str | None = None
    gitlab_token: str | None = None
    webhook_url: str | None = None
    webhook_auth_header: str | None = None


@router.post("/error-reports/upstream/health", response_model=_UpstreamHealthOut)
def upstream_health(body: _UpstreamHealthIn | None = None) -> _UpstreamHealthOut:
    """Probe the configured sink end-to-end so the user can validate
    the token / repo path before accepting the settings.

    Two modes:

    * ``body`` is empty → probe whatever ``Settings`` currently holds.
      Used by a saved-config "re-check" button.
    * ``body`` carries a non-null ``channel`` → build a one-off
      ``SinkConfig`` from the form draft and probe that. Used by the
      Settings page so users don't have to commit an unverified token
      before testing it.

    No real report content leaves the box during the probe.
    """
    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.api.error_upstream import (  # noqa: PLC0415
        SinkConfig,
        build_sink_from_settings,
    )

    if body is not None and body.channel:
        # Ad-hoc config from the form draft. Token still falls back to
        # env vars when blank so users with LORAHUB_GITEA_TOKEN set can
        # leave the UI field empty.
        token = body.gitlab_token or ""
        if not token:
            import os  # noqa: PLC0415

            if body.channel == "gitea":
                token = os.environ.get("LORAHUB_GITEA_TOKEN", "")
            elif body.channel == "gitlab":
                token = os.environ.get("LORAHUB_GITLAB_TOKEN", "")
            if not token:
                token = os.environ.get("LORAHUB_REPORT_TOKEN", "")
        cfg = SinkConfig(
            channel=body.channel,  # type: ignore[arg-type]
            gitlab_base_url=body.gitlab_base_url or "",
            gitlab_repo=body.gitlab_repo or "",
            gitlab_token=token,
            webhook_url=body.webhook_url or "",
            webhook_auth_header=body.webhook_auth_header or "",
        )
        sink = build_sink_from_settings(cfg)
    else:
        settings_store = app_module._settings_store  # type: ignore[attr-defined]
        cfg = settings_store.load()
        sink = build_sink_from_settings(app_module._sink_config_from_settings(cfg))
    if sink is None:
        return _UpstreamHealthOut(
            ok=False,
            channel="off",
            error="upstream channel disabled — set Settings → 错误上报 first",
        )
    res = sink.health_check()
    return _UpstreamHealthOut(
        ok=res.ok,
        channel=sink.channel,
        url=res.url or None,
        error=res.error or None,
    )


class _UpstreamPreviewOut(BaseModel):
    """Redacted snapshot of what would be sent to the upstream sink.

    The fingerprint is also returned so the UI can show "this row will
    join issue #123 (12 prior occurrences)" before the user pushes
    send.
    """
    fingerprint: str
    body: dict[str, Any]


@router.get(
    "/error-reports/{report_id}/upstream-preview",
    response_model=_UpstreamPreviewOut,
)
def upstream_preview(report_id: str) -> _UpstreamPreviewOut:
    from lorahub.api.error_upstream import compute_fingerprint, redact_report

    store = _store()
    rec = store.get(report_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="report not found")
    redacted = redact_report(rec)
    return _UpstreamPreviewOut(
        fingerprint=compute_fingerprint(redacted),
        body=redacted.to_dict(),
    )


__all__ = ["router"]