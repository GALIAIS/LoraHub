"""SQLite-backed persistence for Image Studio annotations and ops.

Separate DB file (runs/image_studio.sqlite) so it can't corrupt or lock
the jobs/sweeps/AI stores. Schema covers:

  * image_annotations — per-image AI + user metadata
  * image_phash — perceptual hashes for L1 dedupe
  * image_pending_ops — queued edits before Apply
  * image_embeddings — cached AI embedding vectors for L2 similarity
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ulid

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS image_annotations (
    image_path           TEXT PRIMARY KEY,
    sha256               TEXT NOT NULL,
    width                INTEGER,
    height               INTEGER,
    bytes                INTEGER,
    ai_caption           TEXT,
    ai_caption_provider  TEXT,
    ai_caption_at        TEXT,
    ai_quality_score     REAL,
    ai_quality_label     TEXT,
    ai_quality_reason    TEXT,
    ai_quality_at        TEXT,
    ai_composition       TEXT,
    ai_composition_at    TEXT,
    ai_trigger_words     TEXT,
    ai_trigger_words_at  TEXT,
    user_quality_label   TEXT,
    user_notes           TEXT,
    soft_deleted         INTEGER NOT NULL DEFAULT 0,
    favorite             INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_image_sha256
    ON image_annotations(sha256);
CREATE INDEX IF NOT EXISTS idx_image_quality
    ON image_annotations(ai_quality_label);
CREATE INDEX IF NOT EXISTS idx_image_user_quality
    ON image_annotations(user_quality_label);
"""

_SCHEMA_PHASH = """\
CREATE TABLE IF NOT EXISTS image_phash (
    image_path  TEXT NOT NULL,
    algo        TEXT NOT NULL,
    hash        TEXT NOT NULL,
    PRIMARY KEY (image_path, algo)
);
CREATE INDEX IF NOT EXISTS idx_phash_value ON image_phash(algo, hash);
"""

_SCHEMA_PENDING_OPS = """\
CREATE TABLE IF NOT EXISTS image_pending_ops (
    id         TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    op         TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_ops_path
    ON image_pending_ops(image_path);
"""

_SCHEMA_EMBEDDINGS = """\
CREATE TABLE IF NOT EXISTS image_embeddings (
    image_path TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (image_path, model_id)
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class ImageAnnotation:
    image_path: str
    sha256: str
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    ai_caption: str | None = None
    ai_caption_provider: str | None = None
    ai_caption_at: str | None = None
    ai_quality_score: float | None = None
    ai_quality_label: str | None = None
    ai_quality_reason: str | None = None
    ai_quality_at: str | None = None
    ai_composition: str | None = None
    ai_composition_at: str | None = None
    ai_trigger_words: list[str] | None = None
    ai_trigger_words_at: str | None = None
    user_quality_label: str | None = None
    user_notes: str | None = None
    soft_deleted: bool = False
    favorite: bool = False
    updated_at: str = ""


@dataclass
class ImagePhash:
    image_path: str
    algo: str
    hash: str


@dataclass
class PendingOp:
    id: str
    image_path: str
    op: str
    payload: dict[str, Any]
    created_at: str = ""


@dataclass
class ImageEmbedding:
    image_path: str
    model_id: str
    dim: int
    vector: list[float] = field(default_factory=list)
    created_at: str = ""


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def default_image_studio_store_path() -> Path:
    """``<project_root>/runs/image_studio.sqlite``."""
    from lorahub.api.paths import runs_dir  # noqa: PLC0415

    return runs_dir() / "image_studio.sqlite"


class ImageStudioStore:
    """CRUD wrapper around the image_studio SQLite database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_SCHEMA_PHASH)
            conn.executescript(_SCHEMA_PENDING_OPS)
            conn.executescript(_SCHEMA_EMBEDDINGS)
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

    # -- Annotations -------------------------------------------------------- #

    def upsert_annotation(self, ann: ImageAnnotation) -> ImageAnnotation:
        with self._lock:
            ann.updated_at = ann.updated_at or _now_iso()
            trigger_json = (
                json.dumps(ann.ai_trigger_words, ensure_ascii=False)
                if ann.ai_trigger_words is not None
                else None
            )
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO image_annotations (
                        image_path, sha256, width, height, bytes,
                        ai_caption, ai_caption_provider, ai_caption_at,
                        ai_quality_score, ai_quality_label, ai_quality_reason,
                        ai_quality_at, ai_composition, ai_composition_at,
                        ai_trigger_words, ai_trigger_words_at,
                        user_quality_label, user_notes,
                        soft_deleted, favorite, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(image_path) DO UPDATE SET
                        sha256=excluded.sha256,
                        width=excluded.width, height=excluded.height,
                        bytes=excluded.bytes,
                        ai_caption=excluded.ai_caption,
                        ai_caption_provider=excluded.ai_caption_provider,
                        ai_caption_at=excluded.ai_caption_at,
                        ai_quality_score=excluded.ai_quality_score,
                        ai_quality_label=excluded.ai_quality_label,
                        ai_quality_reason=excluded.ai_quality_reason,
                        ai_quality_at=excluded.ai_quality_at,
                        ai_composition=excluded.ai_composition,
                        ai_composition_at=excluded.ai_composition_at,
                        ai_trigger_words=excluded.ai_trigger_words,
                        ai_trigger_words_at=excluded.ai_trigger_words_at,
                        user_quality_label=excluded.user_quality_label,
                        user_notes=excluded.user_notes,
                        soft_deleted=excluded.soft_deleted,
                        favorite=excluded.favorite,
                        updated_at=excluded.updated_at
                    """,
                    (
                        ann.image_path, ann.sha256, ann.width, ann.height,
                        ann.bytes, ann.ai_caption, ann.ai_caption_provider,
                        ann.ai_caption_at, ann.ai_quality_score,
                        ann.ai_quality_label, ann.ai_quality_reason,
                        ann.ai_quality_at, ann.ai_composition,
                        ann.ai_composition_at, trigger_json,
                        ann.ai_trigger_words_at, ann.user_quality_label,
                        ann.user_notes, int(ann.soft_deleted),
                        int(ann.favorite), ann.updated_at,
                    ),
                )
            return ann

    def get_annotation(self, image_path: str) -> ImageAnnotation | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM image_annotations WHERE image_path=?",
                    (image_path,),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_annotation(row)

    def list_annotations(
        self,
        paths: list[str] | None = None,
        *,
        quality_label: str | None = None,
        user_quality_label: str | None = None,
        soft_deleted: bool | None = None,
        favorite: bool | None = None,
    ) -> list[ImageAnnotation]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if paths is not None:
                placeholders = ",".join("?" * len(paths))
                clauses.append(f"image_path IN ({placeholders})")
                params.extend(paths)
            if quality_label is not None:
                clauses.append("ai_quality_label=?")
                params.append(quality_label)
            if user_quality_label is not None:
                clauses.append("user_quality_label=?")
                params.append(user_quality_label)
            if soft_deleted is not None:
                clauses.append("soft_deleted=?")
                params.append(int(soft_deleted))
            if favorite is not None:
                clauses.append("favorite=?")
                params.append(int(favorite))
            where = " AND ".join(clauses) if clauses else "1=1"
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM image_annotations WHERE {where}",  # noqa: S608
                    params,
                ).fetchall()
            return [self._row_to_annotation(r) for r in rows]

    def delete_annotation(self, image_path: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM image_annotations WHERE image_path=?",
                    (image_path,),
                )
            return cur.rowcount > 0

    def _row_to_annotation(self, row: sqlite3.Row) -> ImageAnnotation:
        trigger_raw = row["ai_trigger_words"]
        trigger: list[str] | None = None
        if trigger_raw:
            try:
                trigger = json.loads(trigger_raw)
            except json.JSONDecodeError:
                trigger = None
        return ImageAnnotation(
            image_path=row["image_path"],
            sha256=row["sha256"],
            width=row["width"],
            height=row["height"],
            bytes=row["bytes"],
            ai_caption=row["ai_caption"],
            ai_caption_provider=row["ai_caption_provider"],
            ai_caption_at=row["ai_caption_at"],
            ai_quality_score=row["ai_quality_score"],
            ai_quality_label=row["ai_quality_label"],
            ai_quality_reason=row["ai_quality_reason"],
            ai_quality_at=row["ai_quality_at"],
            ai_composition=row["ai_composition"],
            ai_composition_at=row["ai_composition_at"],
            ai_trigger_words=trigger,
            ai_trigger_words_at=row["ai_trigger_words_at"],
            user_quality_label=row["user_quality_label"],
            user_notes=row["user_notes"],
            soft_deleted=bool(row["soft_deleted"]),
            favorite=bool(row["favorite"]),
            updated_at=row["updated_at"],
        )

    # -- Phash -------------------------------------------------------------- #

    def upsert_phash(self, ph: ImagePhash) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO image_phash (image_path, algo, hash)
                    VALUES (?,?,?)
                    ON CONFLICT(image_path, algo) DO UPDATE SET hash=excluded.hash
                    """,
                    (ph.image_path, ph.algo, ph.hash),
                )

    def get_phashes(self, image_path: str) -> list[ImagePhash]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM image_phash WHERE image_path=?",
                    (image_path,),
                ).fetchall()
            return [ImagePhash(r["image_path"], r["algo"], r["hash"]) for r in rows]

    def list_phashes(self, algo: str) -> list[ImagePhash]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM image_phash WHERE algo=?", (algo,)
                ).fetchall()
            return [ImagePhash(r["image_path"], r["algo"], r["hash"]) for r in rows]

    def delete_phashes(self, image_path: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM image_phash WHERE image_path=?", (image_path,)
                )

    # -- Pending ops -------------------------------------------------------- #

    def add_pending_op(self, op: PendingOp) -> PendingOp:
        with self._lock:
            if not op.id:
                op.id = str(ulid.new())
            op.created_at = op.created_at or _now_iso()
            payload_json = json.dumps(op.payload, ensure_ascii=False)
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO image_pending_ops (id, image_path, op, payload, created_at)
                    VALUES (?,?,?,?,?)""",
                    (op.id, op.image_path, op.op, payload_json, op.created_at),
                )
            return op

    def list_pending_ops(self, image_path: str | None = None) -> list[PendingOp]:
        with self._lock:
            with self._connect() as conn:
                if image_path:
                    rows = conn.execute(
                        "SELECT * FROM image_pending_ops WHERE image_path=? ORDER BY created_at",
                        (image_path,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM image_pending_ops ORDER BY created_at"
                    ).fetchall()
            return [self._row_to_op(r) for r in rows]

    def delete_pending_op(self, op_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM image_pending_ops WHERE id=?", (op_id,)
                )
            return cur.rowcount > 0

    def clear_pending_ops(self, image_path: str) -> int:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM image_pending_ops WHERE image_path=?",
                    (image_path,),
                )
            return cur.rowcount

    def _row_to_op(self, row: sqlite3.Row) -> PendingOp:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        return PendingOp(
            id=row["id"],
            image_path=row["image_path"],
            op=row["op"],
            payload=payload,
            created_at=row["created_at"],
        )

    # -- Embeddings --------------------------------------------------------- #

    def upsert_embedding(self, emb: ImageEmbedding) -> None:
        with self._lock:
            emb.created_at = emb.created_at or _now_iso()
            blob = struct.pack(f"<{len(emb.vector)}f", *emb.vector)
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO image_embeddings
                        (image_path, model_id, dim, vector, created_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(image_path, model_id) DO UPDATE SET
                        dim=excluded.dim, vector=excluded.vector,
                        created_at=excluded.created_at
                    """,
                    (emb.image_path, emb.model_id, emb.dim, blob, emb.created_at),
                )

    def get_embedding(
        self, image_path: str, model_id: str
    ) -> ImageEmbedding | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM image_embeddings WHERE image_path=? AND model_id=?",
                    (image_path, model_id),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_embedding(row)

    def list_embeddings(self, model_id: str) -> list[ImageEmbedding]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM image_embeddings WHERE model_id=?",
                    (model_id,),
                ).fetchall()
            return [self._row_to_embedding(r) for r in rows]

    def delete_embeddings(self, image_path: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM image_embeddings WHERE image_path=?",
                    (image_path,),
                )

    def _row_to_embedding(self, row: sqlite3.Row) -> ImageEmbedding:
        blob: bytes = row["vector"]
        dim: int = row["dim"]
        vector = list(struct.unpack(f"<{dim}f", blob))
        return ImageEmbedding(
            image_path=row["image_path"],
            model_id=row["model_id"],
            dim=dim,
            vector=vector,
            created_at=row["created_at"],
        )


__all__ = [
    "ImageAnnotation",
    "ImageEmbedding",
    "ImagePhash",
    "ImageStudioStore",
    "PendingOp",
    "default_image_studio_store_path",
]
