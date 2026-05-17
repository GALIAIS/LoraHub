"""SQLite-backed AI subsystem store.

Mirrors ShiroManager's AI data model: a free-form provider catalogue
where every provider is OpenAI-compatible (custom base_url, headers,
optional org/project), each provider can hold *multiple* API keys with
runtime stats and cooldowns, plus model catalogue (manual or
auto-discovered) and per-task routes (taskId -> provider+model+sampling
parameters + system prompt).

Single SQLite file at ``runs/ai.sqlite``. Plain text storage; the file
mode is set to 0o600 on POSIX so other shell users on the same machine
can't read it. The product is single-user; if you need encryption,
encrypt your home directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ulid

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_providers (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'openai-compatible',
    base_url      TEXT NOT NULL DEFAULT '',
    organization  TEXT NOT NULL DEFAULT '',
    project       TEXT NOT NULL DEFAULT '',
    headers       TEXT NOT NULL DEFAULT '{}',
    enabled       INTEGER NOT NULL DEFAULT 1,
    selection_mode TEXT NOT NULL DEFAULT 'round_robin',
    last_key_index INTEGER NOT NULL DEFAULT -1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_provider_keys (
    id            TEXT PRIMARY KEY,
    provider_id   TEXT NOT NULL,
    api_key       TEXT NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_used_at      TEXT,
    last_succeeded_at TEXT,
    last_failed_at    TEXT,
    last_error        TEXT,
    cooldown_until    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    FOREIGN KEY (provider_id) REFERENCES ai_providers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ai_provider_keys_provider
    ON ai_provider_keys(provider_id);

CREATE TABLE IF NOT EXISTS ai_models (
    id           TEXT PRIMARY KEY,
    provider_id  TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'manual',
    enabled      INTEGER NOT NULL DEFAULT 1,
    raw          TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    FOREIGN KEY (provider_id) REFERENCES ai_providers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ai_models_provider
    ON ai_models(provider_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_models_unique
    ON ai_models(provider_id, model_id);

CREATE TABLE IF NOT EXISTS ai_routes (
    task_id              TEXT PRIMARY KEY,
    provider_id          TEXT,
    model_id             TEXT,
    system_prompt        TEXT NOT NULL DEFAULT '',
    stream               INTEGER,
    temperature          REAL,
    top_p                REAL,
    frequency_penalty    REAL,
    presence_penalty     REAL,
    max_output_tokens    INTEGER,
    seed                 INTEGER,
    reasoning_effort     TEXT,
    thinking_budget_tokens INTEGER,
    include_reasoning    INTEGER,
    stop_sequences       TEXT NOT NULL DEFAULT '[]',
    extra_body_json      TEXT NOT NULL DEFAULT '',
    enabled              INTEGER NOT NULL DEFAULT 1,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
"""


# --------------------------------------------------------------------------- #
# Dataclasses (mirror ShiroManager types/ai.ts shape)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AIProviderKeyRuntime:
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used_at: str | None = None
    last_succeeded_at: str | None = None
    last_failed_at: str | None = None
    last_error: str | None = None
    cooldown_until: str | None = None


@dataclass(slots=True)
class AIProviderKey:
    id: str
    provider_id: str
    api_key: str  # full text; preview is computed on serialise
    runtime: AIProviderKeyRuntime = field(default_factory=AIProviderKeyRuntime)
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class AIProvider:
    id: str
    name: str
    kind: str = "openai-compatible"
    base_url: str = ""
    organization: str = ""
    project: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    selection_mode: str = "round_robin"  # or "random"
    last_key_index: int = -1
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class AIModel:
    id: str
    provider_id: str
    model_id: str
    display_name: str
    source: str = "manual"  # or "discovered"
    enabled: bool = True
    raw: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class AIRoute:
    task_id: str
    provider_id: str | None = None
    model_id: str | None = None
    system_prompt: str = ""
    stream: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    include_reasoning: bool | None = None
    stop_sequences: list[str] = field(default_factory=list)
    extra_body_json: str = ""
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _bool(value: Any) -> int:
    return 1 if value else 0


def _opt_bool(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _row_runtime(row: sqlite3.Row) -> AIProviderKeyRuntime:
    return AIProviderKeyRuntime(
        request_count=row["request_count"],
        success_count=row["success_count"],
        failure_count=row["failure_count"],
        consecutive_failures=row["consecutive_failures"],
        last_used_at=row["last_used_at"],
        last_succeeded_at=row["last_succeeded_at"],
        last_failed_at=row["last_failed_at"],
        last_error=row["last_error"],
        cooldown_until=row["cooldown_until"],
    )


def _row_key(row: sqlite3.Row) -> AIProviderKey:
    return AIProviderKey(
        id=row["id"],
        provider_id=row["provider_id"],
        api_key=row["api_key"],
        runtime=_row_runtime(row),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_provider(row: sqlite3.Row) -> AIProvider:
    return AIProvider(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        base_url=row["base_url"],
        organization=row["organization"],
        project=row["project"],
        headers=json.loads(row["headers"] or "{}"),
        enabled=bool(row["enabled"]),
        selection_mode=row["selection_mode"],
        last_key_index=row["last_key_index"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_model(row: sqlite3.Row) -> AIModel:
    return AIModel(
        id=row["id"],
        provider_id=row["provider_id"],
        model_id=row["model_id"],
        display_name=row["display_name"],
        source=row["source"],
        enabled=bool(row["enabled"]),
        raw=json.loads(row["raw"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_route(row: sqlite3.Row) -> AIRoute:
    return AIRoute(
        task_id=row["task_id"],
        provider_id=row["provider_id"],
        model_id=row["model_id"],
        system_prompt=row["system_prompt"],
        stream=None if row["stream"] is None else bool(row["stream"]),
        temperature=row["temperature"],
        top_p=row["top_p"],
        frequency_penalty=row["frequency_penalty"],
        presence_penalty=row["presence_penalty"],
        max_output_tokens=row["max_output_tokens"],
        seed=row["seed"],
        reasoning_effort=row["reasoning_effort"],
        thinking_budget_tokens=row["thinking_budget_tokens"],
        include_reasoning=None if row["include_reasoning"] is None else bool(row["include_reasoning"]),
        stop_sequences=json.loads(row["stop_sequences"] or "[]"),
        extra_body_json=row["extra_body_json"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class AIStore:
    """CRUD for providers / keys / models / routes.

    All mutations bump `updated_at`. Concurrency-safe via an internal
    RLock; the underlying SQLite connection is also opened with a 10s
    busy timeout. Reads always return fresh dataclass objects.
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
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _tighten_perms(self) -> None:
        if os.name == "nt":
            return
        try:
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    # ------------------------------------------------------------- providers

    def list_providers(self) -> list[AIProvider]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_providers ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [_row_provider(r) for r in rows]

    def get_provider(self, provider_id: str) -> AIProvider | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_providers WHERE id = ?", (provider_id,)
            ).fetchone()
        return _row_provider(row) if row else None

    def upsert_provider(self, p: AIProvider) -> AIProvider:
        if not p.id:
            p.id = str(ulid.new())
        now = _now()
        if not p.created_at:
            p.created_at = now
        p.updated_at = now
        row = {
            "id": p.id,
            "name": p.name,
            "kind": p.kind,
            "base_url": p.base_url,
            "organization": p.organization,
            "project": p.project,
            "headers": _json(p.headers),
            "enabled": _bool(p.enabled),
            "selection_mode": p.selection_mode,
            "last_key_index": p.last_key_index,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_PROVIDER_SQL, row)
        return p

    def delete_provider(self, provider_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM ai_providers WHERE id = ?", (provider_id,))
            return cur.rowcount > 0

    # --------------------------------------------------------------- keys

    def list_keys(self, provider_id: str) -> list[AIProviderKey]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_provider_keys WHERE provider_id = ? ORDER BY created_at",
                (provider_id,),
            ).fetchall()
        return [_row_key(r) for r in rows]

    def upsert_key(self, k: AIProviderKey) -> AIProviderKey:
        if not k.id:
            k.id = str(ulid.new())
        now = _now()
        if not k.created_at:
            k.created_at = now
        k.updated_at = now
        row = {
            "id": k.id,
            "provider_id": k.provider_id,
            "api_key": k.api_key,
            "request_count": k.runtime.request_count,
            "success_count": k.runtime.success_count,
            "failure_count": k.runtime.failure_count,
            "consecutive_failures": k.runtime.consecutive_failures,
            "last_used_at": k.runtime.last_used_at,
            "last_succeeded_at": k.runtime.last_succeeded_at,
            "last_failed_at": k.runtime.last_failed_at,
            "last_error": k.runtime.last_error,
            "cooldown_until": k.runtime.cooldown_until,
            "created_at": k.created_at,
            "updated_at": k.updated_at,
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_KEY_SQL, row)
        return k

    def delete_key(self, key_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM ai_provider_keys WHERE id = ?", (key_id,))
            return cur.rowcount > 0

    def replace_keys(
        self, provider_id: str, drafts: list[AIProviderKey]
    ) -> list[AIProviderKey]:
        """Atomically replace the key set for a provider.

        Drafts whose ``id`` matches an existing key reuse that key's
        runtime stats; new drafts get fresh runtime; missing-id keys
        are deleted. Used by the "save provider" flow which submits
        the entire key list at once.
        """
        with self._lock, self._connect() as conn:
            existing = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT * FROM ai_provider_keys WHERE provider_id = ?",
                    (provider_id,),
                ).fetchall()
            }
            kept_ids: set[str] = set()
            now = _now()
            saved: list[AIProviderKey] = []
            for draft in drafts:
                kid = draft.id or str(ulid.new())
                if kid in existing:
                    prior = existing[kid]
                    runtime = _row_runtime(prior)
                    created_at = prior["created_at"]
                else:
                    runtime = AIProviderKeyRuntime()
                    created_at = now
                kept_ids.add(kid)
                row = {
                    "id": kid,
                    "provider_id": provider_id,
                    "api_key": draft.api_key,
                    "request_count": runtime.request_count,
                    "success_count": runtime.success_count,
                    "failure_count": runtime.failure_count,
                    "consecutive_failures": runtime.consecutive_failures,
                    "last_used_at": runtime.last_used_at,
                    "last_succeeded_at": runtime.last_succeeded_at,
                    "last_failed_at": runtime.last_failed_at,
                    "last_error": runtime.last_error,
                    "cooldown_until": runtime.cooldown_until,
                    "created_at": created_at,
                    "updated_at": now,
                }
                conn.execute(_UPSERT_KEY_SQL, row)
                saved.append(
                    AIProviderKey(
                        id=kid,
                        provider_id=provider_id,
                        api_key=draft.api_key,
                        runtime=runtime,
                        created_at=created_at,
                        updated_at=now,
                    )
                )
            for old_id in existing.keys() - kept_ids:
                conn.execute("DELETE FROM ai_provider_keys WHERE id = ?", (old_id,))
        return saved

    def update_key_runtime(
        self,
        key_id: str,
        *,
        success: bool,
        error: str | None = None,
        cooldown_until: str | None = None,
    ) -> None:
        """Bump the runtime counters after a request through this key.

        Called from the dispatcher whenever a chat completion finishes
        (success or failure). Idempotent: cooldown_until=None clears
        any prior cooldown.
        """
        now = _now()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_provider_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if row is None:
                return
            req = row["request_count"] + 1
            if success:
                succ = row["success_count"] + 1
                fail = row["failure_count"]
                consec = 0
                last_succ = now
                last_fail = row["last_failed_at"]
                last_err = None
            else:
                succ = row["success_count"]
                fail = row["failure_count"] + 1
                consec = row["consecutive_failures"] + 1
                last_succ = row["last_succeeded_at"]
                last_fail = now
                last_err = error
            conn.execute(
                """
                UPDATE ai_provider_keys SET
                    request_count = ?,
                    success_count = ?,
                    failure_count = ?,
                    consecutive_failures = ?,
                    last_used_at = ?,
                    last_succeeded_at = ?,
                    last_failed_at = ?,
                    last_error = ?,
                    cooldown_until = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (req, succ, fail, consec, now, last_succ, last_fail, last_err,
                 cooldown_until, now, key_id),
            )

    def reset_key_runtime(self, key_id: str) -> None:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE ai_provider_keys SET
                    request_count = 0,
                    success_count = 0,
                    failure_count = 0,
                    consecutive_failures = 0,
                    last_used_at = NULL,
                    last_succeeded_at = NULL,
                    last_failed_at = NULL,
                    last_error = NULL,
                    cooldown_until = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, key_id),
            )

    def update_provider_last_index(self, provider_id: str, index: int) -> None:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE ai_providers SET last_key_index = ?, updated_at = ? WHERE id = ?",
                (index, now, provider_id),
            )

    # ------------------------------------------------------------- models

    def list_models(self, provider_id: str | None = None) -> list[AIModel]:
        with self._lock, self._connect() as conn:
            if provider_id is None:
                rows = conn.execute(
                    "SELECT * FROM ai_models ORDER BY display_name COLLATE NOCASE"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ai_models WHERE provider_id = ? "
                    "ORDER BY display_name COLLATE NOCASE",
                    (provider_id,),
                ).fetchall()
        return [_row_model(r) for r in rows]

    def upsert_model(self, m: AIModel) -> AIModel:
        if not m.id:
            m.id = str(ulid.new())
        now = _now()
        if not m.created_at:
            m.created_at = now
        m.updated_at = now
        row = {
            "id": m.id,
            "provider_id": m.provider_id,
            "model_id": m.model_id,
            "display_name": m.display_name or m.model_id,
            "source": m.source,
            "enabled": _bool(m.enabled),
            "raw": _json(m.raw),
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_MODEL_SQL, row)
        return m

    def delete_model(self, model_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM ai_models WHERE id = ?", (model_id,))
            return cur.rowcount > 0

    def replace_discovered_models(
        self, provider_id: str, models: list[AIModel]
    ) -> list[AIModel]:
        """Replace every `source='discovered'` model for this provider.

        Manually-added models (`source='manual'`) are preserved as-is.
        Used by the discover-models endpoint which fetches the upstream
        model list and updates the catalogue in one shot.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM ai_models WHERE provider_id = ? AND source = 'discovered'",
                (provider_id,),
            )
            now = _now()
            saved: list[AIModel] = []
            for m in models:
                m.id = m.id or str(ulid.new())
                m.created_at = m.created_at or now
                m.updated_at = now
                m.source = "discovered"
                conn.execute(
                    _UPSERT_MODEL_SQL,
                    {
                        "id": m.id,
                        "provider_id": provider_id,
                        "model_id": m.model_id,
                        "display_name": m.display_name or m.model_id,
                        "source": "discovered",
                        "enabled": _bool(m.enabled),
                        "raw": _json(m.raw),
                        "created_at": m.created_at,
                        "updated_at": m.updated_at,
                    },
                )
                saved.append(m)
        return saved

    # ------------------------------------------------------------- routes

    def list_routes(self) -> list[AIRoute]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_routes ORDER BY task_id"
            ).fetchall()
        return [_row_route(r) for r in rows]

    def get_route(self, task_id: str) -> AIRoute | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_routes WHERE task_id = ?", (task_id,)
            ).fetchone()
        return _row_route(row) if row else None

    def upsert_route(self, r: AIRoute) -> AIRoute:
        now = _now()
        if not r.created_at:
            r.created_at = now
        r.updated_at = now
        row = {
            "task_id": r.task_id,
            "provider_id": r.provider_id,
            "model_id": r.model_id,
            "system_prompt": r.system_prompt,
            "stream": _opt_bool(r.stream),
            "temperature": r.temperature,
            "top_p": r.top_p,
            "frequency_penalty": r.frequency_penalty,
            "presence_penalty": r.presence_penalty,
            "max_output_tokens": r.max_output_tokens,
            "seed": r.seed,
            "reasoning_effort": r.reasoning_effort,
            "thinking_budget_tokens": r.thinking_budget_tokens,
            "include_reasoning": _opt_bool(r.include_reasoning),
            "stop_sequences": _json(r.stop_sequences),
            "extra_body_json": r.extra_body_json,
            "enabled": _bool(r.enabled),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_ROUTE_SQL, row)
        return r


# --------------------------------------------------------------------------- #
# SQL fragments
# --------------------------------------------------------------------------- #

_UPSERT_PROVIDER_SQL = """
INSERT INTO ai_providers (id, name, kind, base_url, organization, project,
                          headers, enabled, selection_mode, last_key_index,
                          created_at, updated_at)
VALUES (:id, :name, :kind, :base_url, :organization, :project,
        :headers, :enabled, :selection_mode, :last_key_index,
        :created_at, :updated_at)
ON CONFLICT(id) DO UPDATE SET
    name           = excluded.name,
    kind           = excluded.kind,
    base_url       = excluded.base_url,
    organization   = excluded.organization,
    project        = excluded.project,
    headers        = excluded.headers,
    enabled        = excluded.enabled,
    selection_mode = excluded.selection_mode,
    last_key_index = excluded.last_key_index,
    updated_at     = excluded.updated_at
"""

_UPSERT_KEY_SQL = """
INSERT INTO ai_provider_keys (id, provider_id, api_key, request_count,
                              success_count, failure_count, consecutive_failures,
                              last_used_at, last_succeeded_at, last_failed_at,
                              last_error, cooldown_until, created_at, updated_at)
VALUES (:id, :provider_id, :api_key, :request_count, :success_count,
        :failure_count, :consecutive_failures, :last_used_at, :last_succeeded_at,
        :last_failed_at, :last_error, :cooldown_until, :created_at, :updated_at)
ON CONFLICT(id) DO UPDATE SET
    provider_id          = excluded.provider_id,
    api_key              = excluded.api_key,
    request_count        = excluded.request_count,
    success_count        = excluded.success_count,
    failure_count        = excluded.failure_count,
    consecutive_failures = excluded.consecutive_failures,
    last_used_at         = excluded.last_used_at,
    last_succeeded_at    = excluded.last_succeeded_at,
    last_failed_at       = excluded.last_failed_at,
    last_error           = excluded.last_error,
    cooldown_until       = excluded.cooldown_until,
    updated_at           = excluded.updated_at
"""

_UPSERT_MODEL_SQL = """
INSERT INTO ai_models (id, provider_id, model_id, display_name, source, enabled,
                       raw, created_at, updated_at)
VALUES (:id, :provider_id, :model_id, :display_name, :source, :enabled,
        :raw, :created_at, :updated_at)
ON CONFLICT(id) DO UPDATE SET
    provider_id  = excluded.provider_id,
    model_id     = excluded.model_id,
    display_name = excluded.display_name,
    source       = excluded.source,
    enabled      = excluded.enabled,
    raw          = excluded.raw,
    updated_at   = excluded.updated_at
"""

_UPSERT_ROUTE_SQL = """
INSERT INTO ai_routes (task_id, provider_id, model_id, system_prompt, stream,
                       temperature, top_p, frequency_penalty, presence_penalty,
                       max_output_tokens, seed, reasoning_effort,
                       thinking_budget_tokens, include_reasoning,
                       stop_sequences, extra_body_json, enabled,
                       created_at, updated_at)
VALUES (:task_id, :provider_id, :model_id, :system_prompt, :stream,
        :temperature, :top_p, :frequency_penalty, :presence_penalty,
        :max_output_tokens, :seed, :reasoning_effort,
        :thinking_budget_tokens, :include_reasoning,
        :stop_sequences, :extra_body_json, :enabled,
        :created_at, :updated_at)
ON CONFLICT(task_id) DO UPDATE SET
    provider_id            = excluded.provider_id,
    model_id               = excluded.model_id,
    system_prompt          = excluded.system_prompt,
    stream                 = excluded.stream,
    temperature            = excluded.temperature,
    top_p                  = excluded.top_p,
    frequency_penalty      = excluded.frequency_penalty,
    presence_penalty       = excluded.presence_penalty,
    max_output_tokens      = excluded.max_output_tokens,
    seed                   = excluded.seed,
    reasoning_effort       = excluded.reasoning_effort,
    thinking_budget_tokens = excluded.thinking_budget_tokens,
    include_reasoning      = excluded.include_reasoning,
    stop_sequences         = excluded.stop_sequences,
    extra_body_json        = excluded.extra_body_json,
    enabled                = excluded.enabled,
    updated_at             = excluded.updated_at
"""


def default_ai_store_path() -> Path:
    """``<cwd>/runs/ai.sqlite``."""
    return Path.cwd() / "runs" / "ai.sqlite"


__all__ = [
    "AIModel",
    "AIProvider",
    "AIProviderKey",
    "AIProviderKeyRuntime",
    "AIRoute",
    "AIStore",
    "default_ai_store_path",
]
