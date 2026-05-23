"""End-to-end tests for the local error report registry.

Three layers under test:
    * ``ErrorReportStore`` — sqlite CRUD + bounding.
    * The reporter façade — ``capture()`` + ``capture_exception()``.
    * The HTTP surface (``/api/error-reports`` routes) and the FastAPI
      exception handler that auto-persists 500s.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api.error_reporter import capture, capture_exception
from lorahub.api.error_reports import (
    ErrorReport,
    ErrorReportStore,
    default_error_report_store_path,
)


@pytest.fixture
def store(tmp_path: Path) -> ErrorReportStore:
    return ErrorReportStore(tmp_path / "errors.sqlite", max_rows=10)


def test_store_round_trip(store: ErrorReportStore) -> None:
    rep = ErrorReport.create(
        severity="error",
        source="backend.job",
        category="oom",
        title="job xyz failed",
        message="CUDA out of memory",
        context={"workspace": "/tmp/runs/foo", "returncode": 1},
        job_id="job-xyz",
    )
    store.insert(rep)

    fetched = store.get(rep.id)
    assert fetched is not None
    assert fetched.title == "job xyz failed"
    assert fetched.severity == "error"
    assert fetched.context["returncode"] == 1
    assert fetched.job_id == "job-xyz"


def test_store_list_filters(store: ErrorReportStore) -> None:
    for sev, src in [
        ("error", "backend.job"),
        ("warn", "frontend.api"),
        ("info", "user.report"),
    ]:
        store.insert(
            ErrorReport.create(
                severity=sev,  # type: ignore[arg-type]
                source=src,
                category="x",
                title=f"{sev} entry",
                message="m",
            )
        )
    only_warn = store.list(severity="warn")
    assert len(only_warn) == 1 and only_warn[0].severity == "warn"
    only_backend = store.list(source="backend.job")
    assert len(only_backend) == 1 and only_backend[0].source == "backend.job"


def test_store_bounds_at_max_rows(store: ErrorReportStore) -> None:
    for i in range(20):
        store.insert(
            ErrorReport.create(
                severity="info",
                source="user.report",
                category="probe",
                title=f"item-{i}",
                message="x",
            )
        )
    # max_rows=10 in the fixture
    assert store.count() == 10


def test_capture_persists_via_module_singleton(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fresh = ErrorReportStore(tmp_path / "fresh.sqlite")
    monkeypatch.setattr(app_module, "_error_report_store", fresh, raising=False)

    rep = capture(
        severity="error",
        source="backend.job",
        category="cat",
        title="t",
        message="m",
    )
    assert rep is not None
    assert fresh.get(rep.id) is not None


def test_capture_exception_attaches_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fresh = ErrorReportStore(tmp_path / "ex.sqlite")
    monkeypatch.setattr(app_module, "_error_report_store", fresh, raising=False)

    try:
        raise ValueError("boom")
    except ValueError as exc:
        rep = capture_exception(
            exc,
            source="backend.test",
            category="probe",
            title="probe failure",
        )
    assert rep is not None
    persisted = fresh.get(rep.id)
    assert persisted is not None
    assert "ValueError" in (persisted.stack or "")
    assert "boom" in persisted.message


def test_capture_falls_back_when_store_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "_error_report_store", None, raising=False)
    # explicit ``store=None`` short-circuits without touching the singleton.
    rep = capture(
        severity="info",
        source="user.report",
        category="probe",
        title="t",
        message="m",
        store=None,
    )
    assert rep is not None  # constructed but not persisted


@pytest.fixture
def client_with_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> TestClient:
    """A TestClient bound to a per-test SQLite store.

    Use ``raising=False`` so this works whether or not the lifespan has
    populated ``_error_report_store`` yet — TestClient runs the lifespan
    when entered, but the monkeypatch needs to be in place beforehand
    so the lifespan picks it up. ``raise_server_exceptions=False`` lets
    us assert on the JSON the global exception handler renders rather
    than re-raising the original Python exception out of the test
    client (which would defeat the point of the handler).
    """
    monkeypatch.setattr(
        app_module,
        "_error_report_store",
        ErrorReportStore(tmp_path / "http.sqlite"),
        raising=False,
    )
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_http_create_and_list(client_with_store: TestClient) -> None:
    payload = {
        "severity": "warn",
        "source": "frontend.api",
        "category": "http_500",
        "title": "POST /api/jobs failed",
        "message": "internal server error",
    }
    resp = client_with_store.post("/api/error-reports", json=payload)
    assert resp.status_code == 201, resp.text
    rid = resp.json()["id"]

    listing = client_with_store.get("/api/error-reports").json()
    ids = [r["id"] for r in listing["items"]]
    assert rid in ids

    detail = client_with_store.get(f"/api/error-reports/{rid}").json()
    assert detail["title"] == "POST /api/jobs failed"
    assert detail["severity"] == "warn"


def test_http_filters_by_severity(client_with_store: TestClient) -> None:
    for sev in ("info", "warn", "error"):
        client_with_store.post(
            "/api/error-reports",
            json={
                "severity": sev,
                "source": "user.report",
                "category": "x",
                "title": f"t-{sev}",
                "message": "m",
            },
        )
    only_error = client_with_store.get(
        "/api/error-reports?severity=error",
    ).json()
    assert all(r["severity"] == "error" for r in only_error["items"])
    assert any(r["title"] == "t-error" for r in only_error["items"])


def test_http_clear_drops_everything(client_with_store: TestClient) -> None:
    for i in range(3):
        client_with_store.post(
            "/api/error-reports",
            json={
                "severity": "info",
                "source": "user.report",
                "category": "x",
                "title": f"t-{i}",
                "message": "m",
            },
        )
    cleared = client_with_store.post("/api/error-reports/clear").json()
    assert cleared["deleted"] == 3
    listing = client_with_store.get("/api/error-reports").json()
    assert listing["items"] == []


def test_http_unhandled_exception_is_persisted(
    client_with_store: TestClient,
) -> None:
    """The global handler should land 500s in the registry with the
    traceback intact and X-Request-ID linked through.

    POST under ``/api/`` so neither the SPA fallback nor any GET
    parameter route shadows our test endpoint.
    """
    # Register a one-off route that raises so the handler fires. We
    # POST so /api/error-reports/{report_id} doesn't capture the path
    # via its GET handler, and we pick a path that doesn't start with
    # ``error-reports`` to keep the routing decision unambiguous.
    test_path = "/api/_internal/test-explode"

    @app_module.app.post(test_path)
    def _boom() -> None:
        raise RuntimeError("synthetic explosion")

    try:
        resp = client_with_store.post(test_path)
    finally:
        # Tidy up so this route doesn't leak into other tests sharing the
        # module-level FastAPI instance.
        app_module.app.routes[:] = [
            r for r in app_module.app.routes
            if getattr(r, "path", None) != test_path
        ]
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["detail"]["request_id"]
    assert body["detail"]["report_id"]

    detail = client_with_store.get(
        f"/api/error-reports/{body['detail']['report_id']}",
    ).json()
    assert detail["source"] == "backend.exception"
    assert "synthetic explosion" in detail["message"]
    assert detail["stack"] and "RuntimeError" in detail["stack"]
    assert detail["request_path"] == test_path


def test_default_path_is_under_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults must land next to jobs.sqlite so users find them all
    in one place when archiving / migrating workspaces."""
    p = default_error_report_store_path()
    assert p.name == "error-reports.sqlite"
    assert p.parent.name == "runs"
