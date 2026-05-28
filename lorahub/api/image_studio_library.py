"""Cross-dataset library: tag dictionary, trigger-word index, prompt templates.

Lives in the same SQLite file as ``ImageStudioStore`` but in dedicated tables
so the dataset-bound annotations / phash / pending-ops / embeddings stay
disjoint. The library is *global* — entries here outlive any single dataset
and are reusable from the image studio + future training jobs.

Schema:

* ``library_tags``           — curated tag dictionary entries (tag + category +
                                aliases + colour + notes). Keyed by tag string.
* ``library_trigger_words``  — trigger word ↔ character/concept index, plus the
                                list of datasets the trigger has been used in.
* ``library_prompt_templates`` — named prompt body + variable list, used by the
                                AI sub-system + caption tooling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ulid


_SCHEMA_TAGS = """\
CREATE TABLE IF NOT EXISTS library_tags (
    tag         TEXT PRIMARY KEY,
    category    TEXT NOT NULL DEFAULT 'other',
    aliases     TEXT,
    color       TEXT,
    notes       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_tags_category
    ON library_tags(category);
"""

_SCHEMA_TRIGGERS = """\
CREATE TABLE IF NOT EXISTS library_trigger_words (
    trigger_word    TEXT PRIMARY KEY,
    character_name  TEXT,
    concept         TEXT,
    datasets        TEXT,
    prompt_hint     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_triggers_character
    ON library_trigger_words(character_name);
"""

_SCHEMA_PROMPTS = """\
CREATE TABLE IF NOT EXISTS library_prompt_templates (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    body        TEXT NOT NULL,
    vars        TEXT,
    is_default  INTEGER NOT NULL DEFAULT 0,
    notes       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_prompts_category
    ON library_prompt_templates(category);
CREATE UNIQUE INDEX IF NOT EXISTS idx_library_prompts_name
    ON library_prompt_templates(name);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class TagEntry:
    tag: str
    category: str = "other"
    aliases: list[str] = field(default_factory=list)
    color: str | None = None
    notes: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class TriggerWordEntry:
    trigger_word: str
    character_name: str | None = None
    concept: str | None = None
    datasets: list[str] = field(default_factory=list)
    prompt_hint: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class PromptTemplate:
    id: str
    name: str
    category: str = "general"
    body: str = ""
    vars: list[str] = field(default_factory=list)
    is_default: bool = False
    notes: str | None = None
    created_at: str = ""
    updated_at: str = ""


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class ImageStudioLibrary:
    """CRUD for the cross-dataset library tables.

    Shares the SQLite file with ``ImageStudioStore`` but owns its own
    connection and lock. ``initialise()`` is idempotent — safe to run on every
    process start.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_TAGS)
            conn.executescript(_SCHEMA_TRIGGERS)
            conn.executescript(_SCHEMA_PROMPTS)
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

    # -- Tags -------------------------------------------------------------- #

    def upsert_tag(self, entry: TagEntry) -> TagEntry:
        with self._lock:
            now = _now_iso()
            entry.created_at = entry.created_at or now
            entry.updated_at = now
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO library_tags
                        (tag, category, aliases, color, notes,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(tag) DO UPDATE SET
                        category=excluded.category,
                        aliases=excluded.aliases,
                        color=excluded.color,
                        notes=excluded.notes,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entry.tag,
                        entry.category,
                        json.dumps(entry.aliases, ensure_ascii=False),
                        entry.color,
                        entry.notes,
                        entry.created_at,
                        entry.updated_at,
                    ),
                )
            return entry

    def list_tags(
        self,
        *,
        category: str | None = None,
        search: str | None = None,
    ) -> list[TagEntry]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if category is not None:
                clauses.append("category=?")
                params.append(category)
            if search:
                clauses.append("(tag LIKE ? OR aliases LIKE ? OR notes LIKE ?)")
                like = f"%{search}%"
                params.extend([like, like, like])
            where = " AND ".join(clauses) if clauses else "1=1"
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM library_tags WHERE {where} "  # noqa: S608
                    "ORDER BY category, tag",
                    params,
                ).fetchall()
            return [self._row_to_tag(r) for r in rows]

    def get_tag(self, tag: str) -> TagEntry | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM library_tags WHERE tag=?", (tag,)
                ).fetchone()
            return self._row_to_tag(row) if row else None

    def delete_tag(self, tag: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM library_tags WHERE tag=?", (tag,)
                )
            return cur.rowcount > 0

    def _row_to_tag(self, row: sqlite3.Row) -> TagEntry:
        aliases_raw = row["aliases"]
        aliases: list[str] = []
        if aliases_raw:
            try:
                aliases = json.loads(aliases_raw)
            except json.JSONDecodeError:
                aliases = []
        return TagEntry(
            tag=row["tag"],
            category=row["category"],
            aliases=aliases,
            color=row["color"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Trigger words ----------------------------------------------------- #

    def upsert_trigger(self, entry: TriggerWordEntry) -> TriggerWordEntry:
        with self._lock:
            now = _now_iso()
            entry.created_at = entry.created_at or now
            entry.updated_at = now
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO library_trigger_words
                        (trigger_word, character_name, concept,
                         datasets, prompt_hint,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(trigger_word) DO UPDATE SET
                        character_name=excluded.character_name,
                        concept=excluded.concept,
                        datasets=excluded.datasets,
                        prompt_hint=excluded.prompt_hint,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entry.trigger_word,
                        entry.character_name,
                        entry.concept,
                        json.dumps(entry.datasets, ensure_ascii=False),
                        entry.prompt_hint,
                        entry.created_at,
                        entry.updated_at,
                    ),
                )
            return entry

    def list_triggers(
        self,
        *,
        character_name: str | None = None,
        search: str | None = None,
    ) -> list[TriggerWordEntry]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if character_name is not None:
                clauses.append("character_name=?")
                params.append(character_name)
            if search:
                clauses.append(
                    "(trigger_word LIKE ? OR character_name LIKE ? "
                    "OR concept LIKE ?)"
                )
                like = f"%{search}%"
                params.extend([like, like, like])
            where = " AND ".join(clauses) if clauses else "1=1"
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM library_trigger_words WHERE {where} "  # noqa: S608
                    "ORDER BY trigger_word",
                    params,
                ).fetchall()
            return [self._row_to_trigger(r) for r in rows]

    def get_trigger(self, trigger_word: str) -> TriggerWordEntry | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM library_trigger_words WHERE trigger_word=?",
                    (trigger_word,),
                ).fetchone()
            return self._row_to_trigger(row) if row else None

    def delete_trigger(self, trigger_word: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM library_trigger_words WHERE trigger_word=?",
                    (trigger_word,),
                )
            return cur.rowcount > 0

    def _row_to_trigger(self, row: sqlite3.Row) -> TriggerWordEntry:
        datasets_raw = row["datasets"]
        datasets: list[str] = []
        if datasets_raw:
            try:
                datasets = json.loads(datasets_raw)
            except json.JSONDecodeError:
                datasets = []
        return TriggerWordEntry(
            trigger_word=row["trigger_word"],
            character_name=row["character_name"],
            concept=row["concept"],
            datasets=datasets,
            prompt_hint=row["prompt_hint"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Prompt templates -------------------------------------------------- #

    def upsert_prompt(self, tpl: PromptTemplate) -> PromptTemplate:
        with self._lock:
            now = _now_iso()
            tpl.created_at = tpl.created_at or now
            tpl.updated_at = now
            if not tpl.id:
                tpl.id = str(ulid.new())
            with self._connect() as conn:
                # is_default is unique implicitly per-category by convention;
                # if this row is set default, demote any other row in the same
                # category. Caller can intentionally have multiple defaults
                # across different categories.
                if tpl.is_default:
                    conn.execute(
                        "UPDATE library_prompt_templates "
                        "SET is_default=0, updated_at=? "
                        "WHERE category=? AND id<>?",
                        (now, tpl.category, tpl.id),
                    )
                conn.execute(
                    """INSERT INTO library_prompt_templates
                        (id, name, category, body, vars, is_default, notes,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        category=excluded.category,
                        body=excluded.body,
                        vars=excluded.vars,
                        is_default=excluded.is_default,
                        notes=excluded.notes,
                        updated_at=excluded.updated_at
                    """,
                    (
                        tpl.id,
                        tpl.name,
                        tpl.category,
                        tpl.body,
                        json.dumps(tpl.vars, ensure_ascii=False),
                        int(tpl.is_default),
                        tpl.notes,
                        tpl.created_at,
                        tpl.updated_at,
                    ),
                )
            return tpl

    def list_prompts(
        self,
        *,
        category: str | None = None,
    ) -> list[PromptTemplate]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if category is not None:
                clauses.append("category=?")
                params.append(category)
            where = " AND ".join(clauses) if clauses else "1=1"
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM library_prompt_templates WHERE {where} "  # noqa: S608
                    "ORDER BY category, name",
                    params,
                ).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def get_prompt(self, prompt_id: str) -> PromptTemplate | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM library_prompt_templates WHERE id=?",
                    (prompt_id,),
                ).fetchone()
            return self._row_to_prompt(row) if row else None

    def get_prompt_by_name(self, name: str) -> PromptTemplate | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM library_prompt_templates WHERE name=?",
                    (name,),
                ).fetchone()
            return self._row_to_prompt(row) if row else None

    def delete_prompt(self, prompt_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM library_prompt_templates WHERE id=?",
                    (prompt_id,),
                )
            return cur.rowcount > 0

    def _row_to_prompt(self, row: sqlite3.Row) -> PromptTemplate:
        vars_raw = row["vars"]
        vars_list: list[str] = []
        if vars_raw:
            try:
                vars_list = json.loads(vars_raw)
            except json.JSONDecodeError:
                vars_list = []
        return PromptTemplate(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            body=row["body"],
            vars=vars_list,
            is_default=bool(row["is_default"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


__all__ = [
    "ImageStudioLibrary",
    "PromptTemplate",
    "TagEntry",
    "TriggerWordEntry",
]
