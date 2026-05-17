"""Per-provider AI API credentials, kept in their own SQLite file.

Lives at ``runs/ai_credentials.sqlite`` — separate from ``settings.json``
so that the auxiliary AI layer (chat completions, vision tagging,
training-failure diagnosis) can be enabled or wiped independently of the
core workbench config. The file mode is set to 0o600 on POSIX after
creation so other shell users on the same machine can't read it.

Plain text storage. Single-user product, single host, threat model
explicitly excludes "another user on the same box": if you're worried
about that, encrypt your home directory.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_credentials (
    provider     TEXT PRIMARY KEY,
    api_key      TEXT,
    base_url     TEXT,
    default_model TEXT,
    enabled      INTEGER NOT NULL DEFAULT 1,
    updated_at   TEXT NOT NULL
);
"""


@dataclass(slots=True)
class AICredential:
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    enabled: bool = True
    updated_at: datetime | None = None


class AICredentialStore:
    """CRUD around `runs/ai_credentials.sqlite`.

    All mutations bump `updated_at`. Reads return a fresh object each call;
    callers that cache should re-read after an upsert.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
        self._tighten_perms()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _tighten_perms(self) -> None:
        # Best-effort 0o600. No-op on Windows; on POSIX it stops other shell
        # users on the same machine from reading the keys.
        if os.name == "nt":
            return
        try:
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # If we can't chmod (network FS, immutable perms), don't crash;
            # the threat model already assumes the user owns the box.
            pass

    def upsert(self, cred: AICredential) -> None:
        row = {
            "provider": cred.provider,
            "api_key": cred.api_key,
            "base_url": cred.base_url,
            "default_model": cred.default_model,
            "enabled": 1 if cred.enabled else 0,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_SQL, row)

    def get(self, provider: str) -> AICredential | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_credentials WHERE provider = ?",
                (provider,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def list(self) -> list[AICredential]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_credentials ORDER BY provider"
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete(self, provider: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM ai_credentials WHERE provider = ?",
                (provider,),
            )
            return cur.rowcount > 0


_UPSERT_SQL = """
INSERT INTO ai_credentials
    (provider, api_key, base_url, default_model, enabled, updated_at)
VALUES
    (:provider, :api_key, :base_url, :default_model, :enabled, :updated_at)
ON CONFLICT(provider) DO UPDATE SET
    api_key       = excluded.api_key,
    base_url      = excluded.base_url,
    default_model = excluded.default_model,
    enabled       = excluded.enabled,
    updated_at    = excluded.updated_at
"""


def _row_to_record(row: sqlite3.Row) -> AICredential:
    return AICredential(
        provider=row["provider"],
        api_key=row["api_key"],
        base_url=row["base_url"],
        default_model=row["default_model"],
        enabled=bool(row["enabled"]),
        updated_at=datetime.fromisoformat(row["updated_at"])
        if row["updated_at"]
        else None,
    )


def default_ai_credentials_path() -> Path:
    """``<cwd>/runs/ai_credentials.sqlite``."""
    return Path.cwd() / "runs" / "ai_credentials.sqlite"


__all__ = [
    "AICredential",
    "AICredentialStore",
    "default_ai_credentials_path",
]
