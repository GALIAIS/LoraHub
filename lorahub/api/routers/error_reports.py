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


__all__ = ["router"]