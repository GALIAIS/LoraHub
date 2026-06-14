# Background Task Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable background task session store and migrate generic model downloads plus Anima model downloads onto it without changing their public response contracts.

**Architecture:** Introduce `lorahub/api/task_sessions.py` as a small SQLite-backed store for session summaries and bounded event logs. Keep router execution/threading mostly unchanged, but write state transitions and events through the store. Existing route-specific response shapes are adapted from the common session record so the frontend keeps working.

**Tech Stack:** Python 3.11, sqlite3, dataclasses, FastAPI routers, pytest, existing React Query frontend APIs.

---

## File Structure

- Create `lorahub/api/task_sessions.py`: generic store, dataclasses, status/event helpers, startup stale-running recovery.
- Modify `lorahub/api/app.py`: initialize a process-wide task session store during lifespan and expose a test seam `_task_session_store`.
- Modify `lorahub/api/routers/models.py`: persist model download sessions/events/results in the store; keep `/models/download`, `/models/download/latest`, and `/models/download/{id}` response shapes.
- Modify `lorahub/api/routers/backends.py`: persist Anima model download sessions/events/results in the store; keep current status endpoint shape.
- Create `tests/test_task_sessions.py`: store behavior unit tests.
- Modify `tests/test_api.py`: API persistence/latest tests for model and Anima downloads.
- No frontend changes are required in Batch 1 because the current endpoints remain compatible.

---

### Task 1: Add Generic Task Session Store

**Files:**
- Create: `lorahub/api/task_sessions.py`
- Test: `tests/test_task_sessions.py`

- [ ] **Step 1: Write failing store tests**

Create `tests/test_task_sessions.py` with these tests:

```python
from __future__ import annotations

from pathlib import Path

from lorahub.api.task_sessions import TaskEvent, TaskSessionStore


def test_task_session_store_create_update_and_latest(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    session = store.create(kind="model_download", title="owner/name", metadata={"repo_id": "owner/name"})

    store.append_event(session.id, TaskEvent(level="info", message="queued", percent=0))
    store.update(session.id, status="running", percent=12.5)
    store.append_event(session.id, TaskEvent(level="info", message="downloading", percent=12.5))
    store.update(session.id, status="succeeded", percent=100, result={"files": 1})

    loaded = store.get(session.id)
    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.percent == 100
    assert loaded.result == {"files": 1}
    assert [event.message for event in loaded.events] == ["queued", "downloading"]

    latest = store.latest("model_download")
    assert latest is not None
    assert latest.id == session.id


def test_task_session_store_marks_stale_running_interrupted(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    first = store.create(kind="model_download", title="first", metadata={})
    second = store.create(kind="anima_model_download", title="second", metadata={})
    store.update(first.id, status="running", percent=42)
    store.update(second.id, status="queued", percent=0)

    reopened = TaskSessionStore(tmp_path / "tasks.sqlite3")
    count = reopened.mark_stale_interrupted()

    assert count == 2
    assert reopened.get(first.id).status == "interrupted"
    assert reopened.get(second.id).status == "interrupted"
    assert reopened.get(first.id).events[-1].message.startswith("task interrupted")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_task_sessions.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'lorahub.api.task_sessions'`.

- [ ] **Step 3: Implement `lorahub/api/task_sessions.py`**

Create the file with:

```python
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["queued", "running", "succeeded", "failed", "canceled", "interrupted"]


@dataclass(frozen=True, slots=True)
class TaskEvent:
    level: str
    message: str
    percent: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "percent": self.percent,
            "payload": dict(self.payload),
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True)
class TaskSession:
    id: str
    kind: str
    title: str
    status: TaskStatus
    percent: float
    metadata: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    started_at: float
    updated_at: float
    finished_at: float | None
    events: list[TaskEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "percent": self.percent,
            "metadata": dict(self.metadata),
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "events": [event.to_dict() for event in self.events],
        }


class TaskSessionStore:
    def __init__(self, path: Path, *, max_events: int = 200) -> None:
        self.path = path.resolve()
        self.max_events = max(1, int(max_events))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_sessions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    percent REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    percent REAL,
                    payload_json TEXT NOT NULL,
                    ts REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES task_sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_sessions_kind_updated ON task_sessions(kind, updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_events_session_id_id ON task_events(session_id, id)"
            )

    def create(self, *, kind: str, title: str, metadata: dict[str, Any]) -> TaskSession:
        now = time.time()
        sid = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_sessions
                (id, kind, title, status, percent, metadata_json, result_json, error, started_at, updated_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, kind, title, "queued", 0.0, json.dumps(metadata), None, None, now, now, None),
            )
        loaded = self.get(sid)
        assert loaded is not None
        return loaded

    def update(
        self,
        session_id: str,
        *,
        status: TaskStatus | None = None,
        percent: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        loaded = self.get(session_id)
        if loaded is None:
            return
        next_status = status or loaded.status
        next_percent = loaded.percent if percent is None else max(0.0, min(100.0, float(percent)))
        now = time.time()
        finished_at = now if finished else loaded.finished_at
        result_json = json.dumps(result) if result is not None else (
            json.dumps(loaded.result) if loaded.result is not None else None
        )
        next_error = error if error is not None else loaded.error
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE task_sessions
                SET status=?, percent=?, result_json=?, error=?, updated_at=?, finished_at=?
                WHERE id=?
                """,
                (next_status, next_percent, result_json, next_error, now, finished_at, session_id),
            )

    def append_event(self, session_id: str, event: TaskEvent) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_events (session_id, level, message, percent, payload_json, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, event.level, event.message, event.percent, json.dumps(event.payload), event.ts),
            )
            conn.execute(
                "UPDATE task_sessions SET updated_at=?, percent=COALESCE(?, percent) WHERE id=?",
                (event.ts, event.percent, session_id),
            )
            rows = conn.execute(
                "SELECT id FROM task_events WHERE session_id=? ORDER BY id DESC LIMIT -1 OFFSET ?",
                (session_id, self.max_events),
            ).fetchall()
            if rows:
                ids = [str(row["id"]) for row in rows]
                conn.execute(f"DELETE FROM task_events WHERE id IN ({','.join('?' for _ in ids)})", ids)

    def get(self, session_id: str) -> TaskSession | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM task_sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_session(conn, row)

    def latest(self, kind: str) -> TaskSession | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM task_sessions WHERE kind=? ORDER BY updated_at DESC LIMIT 1",
                (kind,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(conn, row)

    def list_recent(self, *, kind: str | None = None, limit: int = 20) -> list[TaskSession]:
        limit = max(1, min(100, int(limit)))
        with self._lock, self._connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM task_sessions WHERE kind=? ORDER BY updated_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM task_sessions ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_session(conn, row) for row in rows]

    def mark_stale_interrupted(self) -> int:
        stale = self.list_recent(limit=100)
        count = 0
        for session in stale:
            if session.status not in ("queued", "running"):
                continue
            self.update(session.id, status="interrupted", error="task interrupted by server restart", finished=True)
            self.append_event(
                session.id,
                TaskEvent(level="warn", message="task interrupted by server restart", percent=session.percent),
            )
            count += 1
        return count

    def _row_to_session(self, conn: sqlite3.Connection, row: sqlite3.Row) -> TaskSession:
        event_rows = conn.execute(
            "SELECT * FROM task_events WHERE session_id=? ORDER BY id ASC",
            (row["id"],),
        ).fetchall()
        events = [
            TaskEvent(
                level=event_row["level"],
                message=event_row["message"],
                percent=event_row["percent"],
                payload=json.loads(event_row["payload_json"] or "{}"),
                ts=event_row["ts"],
            )
            for event_row in event_rows
        ]
        return TaskSession(
            id=row["id"],
            kind=row["kind"],
            title=row["title"],
            status=row["status"],
            percent=float(row["percent"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            finished_at=float(row["finished_at"]) if row["finished_at"] is not None else None,
            events=events,
        )
```

- [ ] **Step 4: Run store tests**

Run:

```bash
python -m pytest tests/test_task_sessions.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add lorahub/api/task_sessions.py tests/test_task_sessions.py
git commit -m "feat(tasks): add persistent session store"
```

---

### Task 2: Initialize Store in API App

**Files:**
- Modify: `lorahub/api/app.py`
- Test: `tests/test_task_sessions.py`

- [ ] **Step 1: Add default path helper test**

Append to `tests/test_task_sessions.py`:

```python
def test_default_task_store_path_is_named_tasks_db() -> None:
    from lorahub.api.task_sessions import default_task_store_path

    assert default_task_store_path().name == "tasks.sqlite3"
```

- [ ] **Step 2: Add default path and app singleton**

In `lorahub/api/task_sessions.py`, add:

```python
from platformdirs import user_state_path


def default_task_store_path() -> Path:
    path = user_state_path("lorahub", "lorahub") / "tasks.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
```

In `lorahub/api/app.py`, import `TaskSessionStore` and `default_task_store_path`, add module global:

```python
_task_session_store: TaskSessionStore | None = None
```

Inside lifespan before router work that might need sessions:

```python
global _task_session_store
if _task_session_store is None:
    _task_session_store = TaskSessionStore(default_task_store_path())
    interrupted = _task_session_store.mark_stale_interrupted()
    if interrupted:
        log.info("marked %d background task session(s) interrupted", interrupted)
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest tests/test_task_sessions.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit Task 2**

```bash
git add lorahub/api/app.py lorahub/api/task_sessions.py tests/test_task_sessions.py
git commit -m "feat(tasks): initialise session store"
```

---

### Task 3: Migrate Generic Model Downloads

**Files:**
- Modify: `lorahub/api/routers/models.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write API persistence test**

Add a test near existing model download tests in `tests/test_api.py`:

```python
def test_models_download_latest_survives_memory_clear(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from lorahub.api.routers import models as models_router
    from lorahub.core.models.downloader import DownloadProgress, DownloadResult

    def fake_download(req: Any, progress: Any = None) -> DownloadResult:
        if progress:
            progress(DownloadProgress(message="persisted progress", percent=33, files_done=1, files_total=3))
        target = req.target_dir or tmp_path / "model"
        target.mkdir(parents=True, exist_ok=True)
        return DownloadResult(target=target, files=1, total_bytes=10)

    monkeypatch.setattr(models_router, "download", fake_download)
    response = client.post(
        "/api/models/download",
        json={"repo_id": "owner/name", "target_dir": str(tmp_path / "downloaded")},
    )
    assert response.status_code == 202
    session_id = response.json()["session_id"]

    deadline = time.time() + 3
    latest = {}
    while time.time() < deadline:
        latest = client.get("/api/models/download/latest").json()
        if latest.get("status") == "succeeded":
            break
        time.sleep(0.01)

    models_router._sessions.clear()
    models_router._latest_session_id = None

    recovered = client.get("/api/models/download/latest").json()
    assert recovered["session_id"] == session_id
    assert recovered["status"] == "succeeded"
    assert recovered["events"][-2]["message"] == "persisted progress"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_api.py::test_models_download_latest_survives_memory_clear -q
```

Expected: fails because latest endpoint falls back to idle after dictionaries are cleared.

- [ ] **Step 3: Update router to write store**

In `lorahub/api/routers/models.py`, add helpers:

```python
_KIND_MODEL_DOWNLOAD = "model_download"


def _task_store():
    store = app_module._task_session_store
    if store is None:
        from lorahub.api.task_sessions import TaskSessionStore, default_task_store_path
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store
```

When creating `_DownloadSession`, also create a task session:

```python
task = _task_store().create(
    kind=_KIND_MODEL_DOWNLOAD,
    title=f"{req.source}:{req.repo_id}",
    metadata={
        "source": req.source,
        "repo_id": req.repo_id,
        "revision": req.revision,
        "target_dir": str(target) if target else None,
        "threads": req.threads,
        "paths": list(req.paths),
    },
)
session = _DownloadSession(session_id=task.id, ...)
```

In `_DownloadSession.add_progress`, after updating memory, append a `TaskEvent` and update percent. Use the store best-effort:

```python
from lorahub.api.task_sessions import TaskEvent
...
try:
    _task_store().append_event(self.session_id, TaskEvent(level="info", message=event.message, percent=event.percent, payload=asdict(event)))
except Exception:
    pass
```

When run starts, call:

```python
_task_store().update(session.session_id, status="running", percent=session.percent)
```

On success:

```python
payload = _result_payload(req, result)
_task_store().update(session.session_id, status="succeeded", percent=100, result=payload, finished=True)
```

On failure:

```python
_task_store().update(session.session_id, status="failed", error=str(exc), finished=True)
```

Update `/models/download/latest` and `/{session_id}` to read store when memory lacks the session. Add a converter from task session to the existing response shape using task metadata.

- [ ] **Step 4: Run model API tests**

Run:

```bash
python -m pytest tests/test_api.py::test_models_download_starts_session_and_reports_progress tests/test_api.py::test_models_download_latest_survives_memory_clear tests/test_api.py::test_models_download_latest_idle -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add lorahub/api/routers/models.py tests/test_api.py
git commit -m "feat(models): persist download status"
```

---

### Task 4: Migrate Anima Model Downloads

**Files:**
- Modify: `lorahub/api/routers/backends.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write Anima recovery test**

Add to `tests/test_api.py`:

```python
def test_anima_model_download_status_survives_memory_clear(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from lorahub.api.routers import backends as backends_router
    from lorahub.core.backends.anima_lora.models import DownloadEvent

    def fake_download_anima_models(**kwargs: Any) -> None:
        progress = kwargs.get("progress")
        if progress:
            progress(DownloadEvent("persisted anima progress", 55, 1, 3))

    monkeypatch.setattr(backends_router, "_download_anima_models", fake_download_anima_models)
    response = client.post("/api/backends/anima_lora/download-models")
    assert response.status_code == 202
    session_id = response.json()["session_id"]

    deadline = time.time() + 3
    while time.time() < deadline:
        status = client.get("/api/backends/anima_lora/download-models/status").json()
        if status.get("status") == "succeeded":
            break
        time.sleep(0.01)

    backends_router._anima_sessions.clear()
    backends_router._anima_active_session = None

    recovered = client.get("/api/backends/anima_lora/download-models/status").json()
    assert recovered["session_id"] == session_id
    assert recovered["status"] == "succeeded"
    assert recovered["events"][-1]["message"] == "persisted anima progress"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_api.py::test_anima_model_download_status_survives_memory_clear -q
```

Expected: fails because Anima status reads only in-memory dictionaries.

- [ ] **Step 3: Update Anima router to write store**

In `lorahub/api/routers/backends.py`, add:

```python
_KIND_ANIMA_MODEL_DOWNLOAD = "anima_model_download"
```

Reuse the same `_task_store()` helper shape from models router or import a shared helper if extracted. When starting Anima download, create a task session first and use its id as `_AnimaModelSession.session_id`.

In `_AnimaModelSession.add_event`, append a `TaskEvent(level="info", message=event.message, percent=event.percent, payload=asdict(event))` and update percent.

On start/success/failure, call `_task_store().update(...)` with `running`, `succeeded`, `failed`.

Update `anima_model_download_status()` to recover latest `_KIND_ANIMA_MODEL_DOWNLOAD` from the store when memory has no session.

- [ ] **Step 4: Run Anima tests**

Run:

```bash
python -m pytest tests/test_api.py::test_anima_model_download_defaults_to_modelscope tests/test_api.py::test_anima_model_download_status_survives_memory_clear -q
```

Expected: both pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add lorahub/api/routers/backends.py tests/test_api.py
git commit -m "feat(anima): persist model download status"
```

---

### Task 5: Expose Generic Task Read API

**Files:**
- Create: `lorahub/api/routers/tasks.py`
- Modify: `lorahub/api/routers/__init__.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write route tests**

Add to `tests/test_api.py`:

```python
def test_tasks_latest_and_list_routes(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.api import app as app_module
    from lorahub.api.task_sessions import TaskEvent, TaskSessionStore

    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    monkeypatch.setattr(app_module, "_task_session_store", store)
    session = store.create(kind="model_download", title="owner/name", metadata={"repo_id": "owner/name"})
    store.append_event(session.id, TaskEvent(level="info", message="hello", percent=1))

    latest = client.get("/api/tasks/latest?kind=model_download")
    assert latest.status_code == 200
    assert latest.json()["id"] == session.id

    listing = client.get("/api/tasks?kind=model_download")
    assert listing.status_code == 200
    assert listing.json()["tasks"][0]["id"] == session.id
```

- [ ] **Step 2: Implement router**

Create `lorahub/api/routers/tasks.py`:

```python
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from lorahub.api import app as app_module
from lorahub.api.task_sessions import TaskSessionStore, default_task_store_path

router = APIRouter(prefix="/api")


def _store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


@router.get("/tasks")
def list_tasks(kind: str | None = None, limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    return {"tasks": [session.to_dict() for session in _store().list_recent(kind=kind, limit=limit)]}


@router.get("/tasks/latest")
def latest_task(kind: str) -> dict[str, Any]:
    session = _store().latest(kind)
    if session is None:
        raise HTTPException(status_code=404, detail="task session not found")
    return session.to_dict()


@router.get("/tasks/{session_id}")
def get_task(session_id: str) -> dict[str, Any]:
    session = _store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="task session not found")
    return session.to_dict()
```

Add it to `lorahub/api/routers/__init__.py` in the router list.

- [ ] **Step 3: Run route tests**

Run:

```bash
python -m pytest tests/test_api.py::test_tasks_latest_and_list_routes -q
```

Expected: pass.

- [ ] **Step 4: Commit Task 5**

```bash
git add lorahub/api/routers/tasks.py lorahub/api/routers/__init__.py tests/test_api.py
git commit -m "feat(tasks): expose task session API"
```

---

### Task 6: Final Verification

**Files:**
- No code changes unless failures are found.

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
python -m pytest tests/test_task_sessions.py tests/test_api.py -k "task or models_download or anima_model_download" -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax compile**

Run:

```bash
python -m py_compile lorahub/api/task_sessions.py lorahub/api/routers/models.py lorahub/api/routers/backends.py lorahub/api/routers/tasks.py lorahub/api/app.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run frontend build only if frontend files changed**

No frontend files should change in this batch. If they did, run:

```bash
npm run build
```

from `web/`. Expected: TypeScript and Vite build pass.

- [ ] **Step 4: Review git status**

Run:

```bash
git status --short --branch
```

Expected: branch ahead by the task commits and no unstaged changes.

- [ ] **Step 5: Push**

Run:

```bash
git push origin dev
```

Expected: `dev -> dev`.
