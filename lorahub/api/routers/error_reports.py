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

The application-wide API authentication middleware protects these
routes when LoraHub binds beyond loopback. Egress remains opt-in.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.error_reports import (
    MAX_ERROR_CATEGORY_CHARS,
    MAX_ERROR_ID_CHARS,
    MAX_ERROR_MESSAGE_CHARS,
    MAX_ERROR_PATH_CHARS,
    MAX_ERROR_SOURCE_CHARS,
    MAX_ERROR_STACK_CHARS,
    MAX_ERROR_TITLE_CHARS,
    ErrorReport,
    ResolutionStatus,
    Severity,
)

router = APIRouter(prefix="/api")
_LIMIT_QUERY = Query(default=100, ge=1, le=1000)
_OFFSET_QUERY = Query(default=0, ge=0)
_SEVERITY_QUERY = Query(default=None)
_SOURCE_QUERY = Query(default=None, max_length=MAX_ERROR_SOURCE_CHARS)
_JOB_ID_QUERY = Query(default=None, max_length=MAX_ERROR_ID_CHARS)
_FINGERPRINT_QUERY = Query(default=None, max_length=128)
_RESOLUTION_QUERY = Query(default=None)
_SEARCH_QUERY = Query(default=None, max_length=200)


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
    fingerprint: str | None = None
    upstream_status: str | None = None
    upstream_url: str | None = None
    upstream_id: str | None = None
    upstream_error: str | None = None
    sent_at: str | None = None
    resolution_status: ResolutionStatus = "open"
    resolved_at: str | None = None
    resolution_note: str | None = None

    @classmethod
    def from_report(cls, r: ErrorReport) -> _ReportOut:
        return cls.model_validate(r.to_dict())


class _ListOut(BaseModel):
    items: list[_ReportOut]
    total: int
    limit: int
    offset: int


class _DuplicateGroupOut(BaseModel):
    fingerprint: str
    count: int
    latest_title: str
    latest_timestamp: str
    severity: Severity


class _SummaryOut(BaseModel):
    total: int
    by_severity: dict[str, int]
    by_source: dict[str, int]
    by_resolution: dict[str, int]
    upstream_attention: int
    duplicate_groups: list[_DuplicateGroupOut]


class _CreateBody(BaseModel):
    severity: Severity = "error"
    source: str = Field(min_length=1, max_length=MAX_ERROR_SOURCE_CHARS)
    category: str = Field(min_length=1, max_length=MAX_ERROR_CATEGORY_CHARS)
    title: str = Field(min_length=1, max_length=MAX_ERROR_TITLE_CHARS)
    message: str = Field(min_length=1, max_length=MAX_ERROR_MESSAGE_CHARS)
    stack: str | None = Field(default=None, max_length=MAX_ERROR_STACK_CHARS)
    context: dict[str, Any] | None = None
    job_id: str | None = Field(default=None, max_length=MAX_ERROR_ID_CHARS)
    # Frontend-supplied request id / path so the report links back to
    # the API call that triggered it (e.g. the 500 response carries an
    # X-Request-ID header that the toast plumbs into here).
    request_id: str | None = Field(default=None, max_length=MAX_ERROR_ID_CHARS)
    request_path: str | None = Field(default=None, max_length=MAX_ERROR_PATH_CHARS)


class _CreateResponse(BaseModel):
    id: str


def _store():
    store = getattr(app_module, "_error_report_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="error-report store not ready")
    return store


@router.get("/error-reports", response_model=_ListOut)
def list_reports(
    limit: int = _LIMIT_QUERY,
    offset: int = _OFFSET_QUERY,
    severity: Severity | None = _SEVERITY_QUERY,
    source: str | None = _SOURCE_QUERY,
    job_id: str | None = _JOB_ID_QUERY,
    fingerprint: str | None = _FINGERPRINT_QUERY,
    resolution_status: ResolutionStatus | None = _RESOLUTION_QUERY,
    q: str | None = _SEARCH_QUERY,
) -> _ListOut:
    store = _store()
    items = store.list(
        limit=limit,
        offset=offset,
        severity=severity,
        source=source,
        job_id=job_id,
        fingerprint=fingerprint,
        resolution_status=resolution_status,
        q=q,
    )
    return _ListOut(
        items=[_ReportOut.from_report(r) for r in items],
        total=store.count(
            severity=severity,
            source=source,
            job_id=job_id,
            fingerprint=fingerprint,
            resolution_status=resolution_status,
            q=q,
        ),
        limit=limit,
        offset=offset,
    )


@router.get("/error-reports/summary", response_model=_SummaryOut)
def reports_summary(
    severity: Severity | None = _SEVERITY_QUERY,
    source: str | None = _SOURCE_QUERY,
    job_id: str | None = _JOB_ID_QUERY,
    fingerprint: str | None = _FINGERPRINT_QUERY,
    resolution_status: ResolutionStatus | None = _RESOLUTION_QUERY,
    q: str | None = _SEARCH_QUERY,
) -> _SummaryOut:
    store = _store()
    return _SummaryOut.model_validate(
        store.summary(
            severity=severity,
            source=source,
            job_id=job_id,
            fingerprint=fingerprint,
            resolution_status=resolution_status,
            q=q,
        )
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
    def _gen():  # type: ignore[no-untyped-def]
        for r in store.iter_all():
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


class _ResolutionBody(BaseModel):
    status: ResolutionStatus
    note: str | None = Field(default=None, max_length=2000)


@router.post("/error-reports/{report_id}/resolution", response_model=_ReportOut)
def update_report_resolution(report_id: str, body: _ResolutionBody) -> _ReportOut:
    store = _store()
    rec = store.update_resolution(report_id, status=body.status, note=body.note)
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
        # Ad-hoc config from the form draft. Token resolution order:
        #   1. ``body.gitlab_token`` if it's neither blank nor the
        #      masked echo we serve in GET responses (``abcd...wxyz``).
        #      Without this guard the Settings UI's hydrated draft
        #      sends the masked preview straight back to us, we hand
        #      that literal string to Gitea, and the server rejects
        #      it with 401 even though the real token on disk is
        #      perfectly valid.
        #   2. Persisted ``Settings.error_upstream_gitlab_token`` —
        #      what the user already saved.
        #   3. Env vars (LORAHUB_GITEA_TOKEN / LORAHUB_GITLAB_TOKEN /
        #      LORAHUB_REPORT_TOKEN) — last-resort fallback so a
        #      first-time setup with token-via-env works.
        from lorahub.api.routers.settings_routes import _mask_secret  # noqa: PLC0415

        settings_store = app_module._settings_store  # type: ignore[attr-defined]
        persisted = settings_store.load()
        persisted_token = persisted.error_upstream_gitlab_token or ""
        masked_persisted = _mask_secret(persisted_token) or ""

        raw = (body.gitlab_token or "").strip()
        if raw and raw != masked_persisted:
            token = raw
        elif persisted_token:
            token = persisted_token
        else:
            import os  # noqa: PLC0415

            if body.channel == "gitea":
                token = os.environ.get("LORAHUB_GITEA_TOKEN", "")
            elif body.channel == "gitlab":
                token = os.environ.get("LORAHUB_GITLAB_TOKEN", "")
            else:
                token = ""
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
