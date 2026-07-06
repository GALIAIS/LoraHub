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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


Severity = Literal["info", "warn", "error", "fatal"]
ResolutionStatus = Literal["open", "resolved", "ignored"]
"""``fatal`` is reserved for unrecoverable boot / lifespan failures.
Everything routine (job died, preflight blocked, 500 reply) is ``error``.
"""

# Bump together with a real ALTER path. v3 adds local resolution tracking.
_SCHEMA_VERSION = 3
# Tables-only DDL, run unconditionally on every open. ``CREATE TABLE IF
# NOT EXISTS`` won't reshape an existing table, which is exactly what
# we want — the ALTER pass below brings v1 stores up to v2 by adding
# the missing columns.
_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS error_reports (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    source          TEXT NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    message         TEXT NOT NULL,
    stack           TEXT,
    context         TEXT NOT NULL,
    job_id          TEXT,
    request_id      TEXT,
    request_path    TEXT,
    version         TEXT NOT NULL,
    platform        TEXT NOT NULL,
    fingerprint     TEXT,
    upstream_status TEXT,
    upstream_url    TEXT,
    upstream_id     TEXT,
    upstream_error  TEXT,
    sent_at         TEXT
);

CREATE TABLE IF NOT EXISTS error_reports_schema_version (
    version INTEGER PRIMARY KEY
);
"""

# Indexes are split out so they can run *after* the v1 → v2 column-add
# migration. Creating ``error_reports_fp ON (fingerprint)`` against a
# v1 database would fail with ``no such column: fingerprint`` because
# CREATE TABLE IF NOT EXISTS is a no-op on the existing legacy schema.
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS error_reports_ts
    ON error_reports (timestamp DESC);
CREATE INDEX IF NOT EXISTS error_reports_severity
    ON error_reports (severity);
CREATE INDEX IF NOT EXISTS error_reports_source
    ON error_reports (source);
CREATE INDEX IF NOT EXISTS error_reports_job_id
    ON error_reports (job_id);
CREATE INDEX IF NOT EXISTS error_reports_fp
    ON error_reports (fingerprint);
CREATE INDEX IF NOT EXISTS error_reports_resolution
    ON error_reports (resolution_status);
"""

_UPSTREAM_COLUMNS = (
    ("fingerprint", "TEXT"),
    ("upstream_status", "TEXT"),
    ("upstream_url", "TEXT"),
    ("upstream_id", "TEXT"),
    ("upstream_error", "TEXT"),
    ("sent_at", "TEXT"),
    ("resolution_status", "TEXT NOT NULL DEFAULT 'open'"),
    ("resolved_at", "TEXT"),
    ("resolution_note", "TEXT"),
)


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
    # Upstream delivery state — populated lazily by the dispatcher.
    # ``None`` everywhere means the report has never been considered for
    # remote delivery (channel was off, or the report only ever lived
    # locally). Status values: ``queued`` / ``retrying`` / ``sent`` /
    # ``failed`` / ``skipped``.
    fingerprint: str | None = None
    upstream_status: str | None = None
    upstream_url: str | None = None
    upstream_id: str | None = None
    upstream_error: str | None = None
    sent_at: datetime | None = None
    resolution_status: ResolutionStatus = "open"
    resolved_at: datetime | None = None
    resolution_note: str | None = None

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
            "fingerprint": self.fingerprint,
            "upstream_status": self.upstream_status,
            "upstream_url": self.upstream_url,
            "upstream_id": self.upstream_id,
            "upstream_error": self.upstream_error,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "resolution_status": self.resolution_status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_note": self.resolution_note,
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
            # 1. Create the table if missing. CREATE TABLE IF NOT EXISTS
            #    is a no-op on a legacy v1 store, leaving its column set
            #    intact for the ALTER pass below.
            conn.executescript(_SCHEMA_TABLES)
            conn.execute("DELETE FROM error_reports_schema_version")
            conn.execute(
                "INSERT INTO error_reports_schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            # 2. Idempotent column-add migration for stores predating the
            #    upstream fan-out. SQLite ``ALTER TABLE … ADD COLUMN``
            #    doesn't support IF NOT EXISTS so we sniff PRAGMA first.
            cur = conn.execute("PRAGMA table_info(error_reports)")
            cols = {row[1] for row in cur.fetchall()}
            for name, decl in _UPSTREAM_COLUMNS:
                if name not in cols:
                    conn.execute(
                        f"ALTER TABLE error_reports ADD COLUMN {name} {decl}",
                    )
            # 3. Indexes go last so ``error_reports_fp ON (fingerprint)``
            #    has a column to bind against. Creating these together
            #    with the table caused ``no such column: fingerprint``
            #    on legacy stores because CREATE TABLE IF NOT EXISTS
            #    didn't actually add the column.
            conn.executescript(_SCHEMA_INDEXES)
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
                report.fingerprint,
                report.upstream_status,
                report.upstream_url,
                report.upstream_id,
                report.upstream_error,
                report.sent_at.isoformat() if report.sent_at else None,
                report.resolution_status,
                report.resolved_at.isoformat() if report.resolved_at else None,
                report.resolution_note,
            )
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO error_reports ("
                    "id, timestamp, severity, source, category, title, "
                    "message, stack, context, job_id, request_id, "
                    "request_path, version, platform, "
                    "fingerprint, upstream_status, upstream_url, "
                    "upstream_id, upstream_error, sent_at, "
                    "resolution_status, resolved_at, resolution_note"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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

    def update_upstream(
        self,
        report_id: str,
        *,
        status: str,
        url: str | None = None,
        upstream_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Stamp the upstream delivery state on an existing row.

        ``status`` is one of ``queued`` / ``retrying`` / ``sent`` /
        ``failed`` / ``skipped``. ``sent_at`` is auto-stamped when
        status moves to ``sent`` so the UI can show "uploaded 14:32".
        """
        sent_at = (
            datetime.now().isoformat() if status == "sent" else None
        )
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE error_reports SET "
                    "  upstream_status = ?,"
                    "  upstream_url = COALESCE(?, upstream_url),"
                    "  upstream_id = COALESCE(?, upstream_id),"
                    "  upstream_error = ?,"
                    "  sent_at = COALESCE(?, sent_at) "
                    "WHERE id = ?",
                    (status, url, upstream_id, error, sent_at, report_id),
                )
                conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.warning(
                "could not update upstream state for %s: %r", report_id, exc,
            )

    def update_resolution(
        self,
        report_id: str,
        *,
        status: ResolutionStatus,
        note: str | None = None,
    ) -> ErrorReport | None:
        resolved_at = datetime.now().isoformat() if status != "open" else None
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "UPDATE error_reports SET "
                    "  resolution_status = ?,"
                    "  resolved_at = ?,"
                    "  resolution_note = ? "
                    "WHERE id = ?",
                    (status, resolved_at, note, report_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
        except (sqlite3.Error, OSError) as exc:
            log.warning("could not update resolution state for %s: %r", report_id, exc)
            return None
        return self.get(report_id)

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
        fingerprint: str | None = None,
        resolution_status: ResolutionStatus | None = None,
        q: str | None = None,
    ) -> list[ErrorReport]:
        """Return reports newest-first. Filters compose with AND."""
        clauses, params = self._filter_sql(
            severity=severity,
            source=source,
            job_id=job_id,
            fingerprint=fingerprint,
            resolution_status=resolution_status,
            q=q,
        )
        sql = "SELECT * FROM error_reports"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([max(1, min(1000, limit)), max(0, offset)])
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_report(r) for r in rows]

    def count(
        self,
        *,
        severity: Severity | None = None,
        source: str | None = None,
        job_id: str | None = None,
        fingerprint: str | None = None,
        resolution_status: ResolutionStatus | None = None,
        q: str | None = None,
    ) -> int:
        clauses, params = self._filter_sql(
            severity=severity,
            source=source,
            job_id=job_id,
            fingerprint=fingerprint,
            resolution_status=resolution_status,
            q=q,
        )
        sql = "SELECT COUNT(*) AS n FROM error_reports"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

    def summary(
        self,
        *,
        severity: Severity | None = None,
        source: str | None = None,
        job_id: str | None = None,
        fingerprint: str | None = None,
        resolution_status: ResolutionStatus | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate counts for the current filter.

        Kept in SQL so the Settings panel can track a large registry
        without pulling thousands of rows into React.
        """
        clauses, params = self._filter_sql(
            severity=severity,
            source=source,
            job_id=job_id,
            fingerprint=fingerprint,
            resolution_status=resolution_status,
            q=q,
        )
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self._connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM error_reports{where}",
                    params,
                ).fetchone()["n"]
            )
            by_severity = {
                row["severity"]: int(row["n"])
                for row in conn.execute(
                    f"SELECT severity, COUNT(*) AS n FROM error_reports{where} "
                    "GROUP BY severity",
                    params,
                ).fetchall()
            }
            by_source = {
                row["source"]: int(row["n"])
                for row in conn.execute(
                    f"SELECT source, COUNT(*) AS n FROM error_reports{where} "
                    "GROUP BY source ORDER BY n DESC LIMIT 8",
                    params,
                ).fetchall()
            }
            by_resolution = {
                row["resolution_status"]: int(row["n"])
                for row in conn.execute(
                    f"SELECT resolution_status, COUNT(*) AS n FROM error_reports{where} "
                    "GROUP BY resolution_status",
                    params,
                ).fetchall()
            }
            upstream_attention = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM error_reports{where}"
                    + (" AND " if where else " WHERE ")
                    + "(upstream_status IN ('failed', 'retrying', 'queued') "
                    "OR upstream_error IS NOT NULL)",
                    params,
                ).fetchone()["n"]
            )
            duplicate_groups = [
                {
                    "fingerprint": row["fingerprint"],
                    "count": int(row["n"]),
                    "latest_title": row["latest_title"],
                    "latest_timestamp": row["latest_ts"],
                    "severity": row["severity"],
                }
                for row in conn.execute(
                    f"""
                    SELECT
                        fingerprint,
                        COUNT(*) AS n,
                        MAX(timestamp) AS latest_ts,
                        (
                            SELECT title FROM error_reports e2
                            WHERE e2.fingerprint = e1.fingerprint
                            ORDER BY timestamp DESC
                            LIMIT 1
                        ) AS latest_title,
                        (
                            SELECT severity FROM error_reports e3
                            WHERE e3.fingerprint = e1.fingerprint
                            ORDER BY
                                CASE severity
                                    WHEN 'fatal' THEN 4
                                    WHEN 'error' THEN 3
                                    WHEN 'warn' THEN 2
                                    ELSE 1
                                END DESC,
                                timestamp DESC
                            LIMIT 1
                        ) AS severity
                    FROM error_reports e1
                    {where}
                    {"AND" if where else "WHERE"} fingerprint IS NOT NULL
                    GROUP BY fingerprint
                    HAVING n > 1
                    ORDER BY n DESC, latest_ts DESC
                    LIMIT 5
                    """,
                    params,
                ).fetchall()
            ]
        return {
            "total": total,
            "by_severity": by_severity,
            "by_source": by_source,
            "by_resolution": by_resolution,
            "upstream_attention": upstream_attention,
            "duplicate_groups": duplicate_groups,
        }

    @staticmethod
    def _filter_sql(
        *,
        severity: Severity | None = None,
        source: str | None = None,
        job_id: str | None = None,
        fingerprint: str | None = None,
        resolution_status: ResolutionStatus | None = None,
        q: str | None = None,
    ) -> tuple[list[str], list[Any]]:
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
        if fingerprint is not None:
            clauses.append("fingerprint = ?")
            params.append(fingerprint)
        if resolution_status is not None:
            clauses.append("resolution_status = ?")
            params.append(resolution_status)
        needle = (q or "").strip()
        if needle:
            like = f"%{needle.casefold()}%"
            clauses.append(
                "(lower(title) LIKE ? OR lower(message) LIKE ? OR lower(category) LIKE ?)"
            )
            params.extend([like, like, like])
        return clauses, params

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
    keys = row.keys()

    def _opt(name: str) -> str | None:
        # PRAGMA-driven fallback: an old DB that hasn't been ALTERed
        # yet will be missing some columns; the migration in __init__
        # handles that, but reads from a brand-new connection still
        # need to tolerate the column not being present in the row.
        return row[name] if name in keys else None

    sent_at_raw = _opt("sent_at")
    resolved_at_raw = _opt("resolved_at")
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
        fingerprint=_opt("fingerprint"),
        upstream_status=_opt("upstream_status"),
        upstream_url=_opt("upstream_url"),
        upstream_id=_opt("upstream_id"),
        upstream_error=_opt("upstream_error"),
        sent_at=datetime.fromisoformat(sent_at_raw) if sent_at_raw else None,
        resolution_status=(_opt("resolution_status") or "open"),  # type: ignore[arg-type]
        resolved_at=datetime.fromisoformat(resolved_at_raw) if resolved_at_raw else None,
        resolution_note=_opt("resolution_note"),
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
    "ResolutionStatus",
    "default_error_report_store_path",
]
