"""SQLite-backed persistence for the in-app error report log.

The app collects every interesting failure in one local-only registry so
users can inspect them after the fact: training-job failures, preflight
blockers, FastAPI 5xx exceptions, frontend render errors, and explicit
``Report bug`` button presses. Nothing leaves the machine without an
explicit user action — exporting to a file or copying a GitHub Issue
template are the only egress paths.

Layout decisions:

* SQLite, not JSON, because the list grows unbounded and we need
  bounded read costs (``LIMIT N`` + index). One file alongside the
  other lorahub stores under ``<project_root>/runs/error-reports.sqlite``.
* ``context`` is JSON text so we can attach arbitrary structured
  metadata (job id, request path, recent training events, log tail)
  without forcing a schema migration each time a producer adds a field.
* Severity / source / category are short strings indexed for the
  Settings → 错误上报 panel filters. They mirror what the existing
  diagnosis catalogue already uses, so the filter chips can share the
  category vocabulary.
* ``request_id`` is opaque, set by the FastAPI middleware so a server
  exception and the matching frontend toast can be cross-referenced.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

log = logging.getLogger(__name__)


Severity = Literal["info", "warn", "error", "fatal"]
"""``fatal`` is reserved for unrecoverable boot / lifespan failures.
Everything routine (job died, preflight blocked, 500 reply) is ``error``.
"""

# Bump together with a real ALTER path. Schema 1 = the columns below.
_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS error_reports (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    severity    TEXT NOT NULL,
    source      TEXT NOT NULL,
    category    TEXT NOT NULL,
    title       TEXT NOT NULL,
    message     TEXT NOT NULL,
    stack       TEXT,
    context     TEXT NOT NULL,
    job_id      TEXT,
    request_id  TEXT,
    request_path TEXT,
    version     TEXT NOT NULL,
    platform    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS error_reports_ts
    ON error_reports (timestamp DESC);
CREATE INDEX IF NOT EXISTS error_reports_severity
    ON error_reports (severity);
CREATE INDEX IF NOT EXISTS error_reports_source
    ON error_reports (source);
CREATE INDEX IF NOT EXISTS error_reports_job_id
    ON error_reports (job_id);

CREATE TABLE IF NOT EXISTS error_reports_schema_version (
    version INTEGER PRIMARY KEY
);
"""


@dataclass(slots=True)
class ErrorReport:
    """One captured error / failure event.

    All non-context fields are short strings so the list view doesn't
    need a join. ``context`` carries the long-form structured payload
    (recent events, traceback frames, training.log tail, request body).

    Construct via ``ErrorReport.create(...)`` to get an auto-generated
    id + timestamp + platform / version stamps; pass an explicit ``id``
    only when reconstructing from disk.
    """

    id: str
    timestamp: datetime
    severity: Severity
    # Where the report originated. Free-form but conventionally one of:
    #   ``backend.exception`` — uncaught FastAPI exception
    #   ``backend.job``       — training job failed / interrupted
    #   ``backend.preflight`` — preflight blockers (422)
    #   ``backend.bootstrap`` — kohya / anima install failure
    #   ``backend.update``    — system_update.apply raised
    #   ``frontend.render``   — React error boundary
    #   ``frontend.runtime``  — window.onerror / unhandledrejection
    #   ``frontend.api``      — non-2xx API response surfaced by toast
    #   ``user.report``       — explicit user-driven submission
    source: str
    category: str
    title: str
    message: str
    stack: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    request_id: str | None = None
    request_path: str | None = None
    version: str = ""
    platform: str = ""

    @classmethod
    def create(
        cls,
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
        version: str | None = None,
        platform: str | None = None,
        timestamp: datetime | None = None,
    ) -> ErrorReport:
        return cls(
            id=uuid.uuid4().hex,
            timestamp=timestamp or datetime.now(),
            severity=severity,
            source=source,
            category=category,
            title=title,
            message=message,
            stack=stack,
            context=dict(context or {}),
            job_id=job_id,
            request_id=request_id,
            request_path=request_path,
            version=version or _resolve_version(),
            platform=platform or _resolve_platform(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "source": self.source,
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "stack": self.stack,
            "context": self.context,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "request_path": self.request_path,
            "version": self.version,
            "platform": self.platform,
        }


class ErrorReportStore:
    """Thread-safe CRUD wrapper around the error-report SQLite file.

    Mirrors the shape of ``JobStore`` so the patterns stay consistent
    across stores (lifespan singleton, monkeypatchable from app.py,
    WAL + NORMAL synchronous tradeoff).
    """

    def __init__(self, path: Path, *, max_rows: int = 5000) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._max_rows = max_rows
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO error_reports_schema_version (version) "
                "VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def insert(self, report: ErrorReport) -> None:
        """Persist a report. Errors during write are swallowed and
        logged — losing one error log row must never raise from the
        reporter (we'd recursively try to log the failure-to-log)."""
        try:
            payload = (
                report.id,
                report.timestamp.isoformat(),
                report.severity,
                report.source,
                report.category,
                report.title,
                report.message,
                report.stack,
                json.dumps(report.context, ensure_ascii=False, default=str),
                report.job_id,
                report.request_id,
                report.request_path,
                report.version,
                report.platform,
            )
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO error_reports ("
                    "id, timestamp, severity, source, category, title, "
                    "message, stack, context, job_id, request_id, "
                    "request_path, version, platform"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    payload,
                )
                # Bound the table so a long-running install with a noisy
                # background loop can't fill the disk. We delete the
                # oldest rows past ``max_rows`` rather than truncating
                # the file because the front-end paginates from the
                # newest end anyway.
                conn.execute(
                    "DELETE FROM error_reports WHERE id IN ("
                    "  SELECT id FROM error_reports "
                    "  ORDER BY timestamp DESC LIMIT -1 OFFSET ?"
                    ")",
                    (self._max_rows,),
                )
                conn.commit()
        except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
            log.warning("could not persist error report %s: %r", report.id, exc)

    def get(self, report_id: str) -> ErrorReport | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM error_reports WHERE id = ?", (report_id,),
            ).fetchone()
        return _row_to_report(row) if row else None

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        severity: Severity | None = None,
        source: str | None = None,
        job_id: str | None = None,
    ) -> list[ErrorReport]:
        """Return reports newest-first. Filters compose with AND."""
        clauses: list[str] = []
        params: list[Any] = []
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        sql = "SELECT * FROM error_reports"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([max(1, min(1000, limit)), max(0, offset)])
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_report(r) for r in rows]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM error_reports").fetchone()
        return int(row["n"]) if row else 0

    def delete(self, report_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM error_reports WHERE id = ?", (report_id,),
            )
            conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM error_reports")
            conn.commit()
        return int(cur.rowcount or 0)

    def export_ndjson(self, dest: Path, *, items: Iterable[ErrorReport] | None = None) -> int:
        """Write reports as newline-delimited JSON.

        ``items`` lets callers pre-filter the export (e.g. last 7 days).
        Returns the number of rows written.
        """
        rows = list(items) if items is not None else self.list(limit=1000)
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with dest.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str))
                fh.write("\n")
                n += 1
        return n


def _row_to_report(row: sqlite3.Row) -> ErrorReport:
    raw_context = row["context"] or "{}"
    try:
        ctx = json.loads(raw_context)
        if not isinstance(ctx, dict):
            ctx = {"_raw": raw_context}
    except json.JSONDecodeError:
        ctx = {"_raw": raw_context}
    return ErrorReport(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        severity=row["severity"],  # type: ignore[arg-type]
        source=row["source"],
        category=row["category"],
        title=row["title"],
        message=row["message"],
        stack=row["stack"],
        context=ctx,
        job_id=row["job_id"],
        request_id=row["request_id"],
        request_path=row["request_path"],
        version=row["version"] or "",
        platform=row["platform"] or "",
    )


# ---------------------------------------------------------------------- #
# Defaults / helpers used by both the API and the CLI
# ---------------------------------------------------------------------- #


def default_error_report_store_path() -> Path:
    """Lives next to the other lorahub stores under ``runs/``."""
    from lorahub.api.paths import runs_dir  # noqa: PLC0415

    return runs_dir() / "error-reports.sqlite"


def _resolve_version() -> str:
    try:
        from lorahub import __version__  # noqa: PLC0415

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _resolve_platform() -> str:
    """Short identifier so a report from a Windows zh-CN box is
    distinguishable from one on Linux without dumping the whole
    ``platform.uname()`` tuple into every row."""
    import platform as _p  # noqa: PLC0415

    try:
        return f"{_p.system()} {_p.release()} / py{_p.python_version()}"
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = [
    "ErrorReport",
    "ErrorReportStore",
    "Severity",
    "default_error_report_store_path",
]