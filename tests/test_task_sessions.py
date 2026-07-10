from __future__ import annotations

from pathlib import Path

from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSessionStore,
    prune_terminal_session_cache,
)


def test_task_session_store_create_update_and_latest(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    session = store.create(
        kind="model_download",
        title="owner/name",
        metadata={"repo_id": "owner/name"},
    )

    store.append_event(session.id, TaskEvent(level="info", message="queued", percent=0))
    store.update(session.id, status="running", percent=12.5)
    store.append_event(
        session.id,
        TaskEvent(level="info", message="downloading", percent=12.5),
    )
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
    assert reopened.get(first.id).status == "interrupted"  # type: ignore[union-attr]
    assert reopened.get(second.id).status == "interrupted"  # type: ignore[union-attr]
    assert reopened.get(first.id).events[-1].message.startswith(  # type: ignore[union-attr]
        "task interrupted",
    )


def test_task_session_store_lists_only_active_sessions(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    queued = store.create(kind="queued", title="queued", metadata={})
    running = store.create(kind="running", title="running", metadata={})
    stopping = store.create(kind="stopping", title="stopping", metadata={})
    finished = store.create(kind="finished", title="finished", metadata={})
    store.update(running.id, status="running")
    store.update(stopping.id, status="stop_requested")
    store.update(finished.id, status="succeeded", finished=True)

    assert {session.id for session in store.list_active()} == {
        queued.id,
        running.id,
        stopping.id,
    }


def test_stop_request_cannot_be_reopened_or_overwrite_terminal_state(
    tmp_path: Path,
) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    task = store.create(kind="download", title="download", metadata={})
    store.update(task.id, status="running", percent=12)

    assert store.request_stop(task.id, percent=15) is True
    store.update(task.id, status="running", percent=10)
    stopping = store.get(task.id)
    assert stopping is not None
    assert stopping.status == "stop_requested"
    assert stopping.percent == 15

    store.update(task.id, status="canceled", finished=True)
    store.update(task.id, status="succeeded", percent=100, finished=True)
    terminal = store.get(task.id)
    assert terminal is not None
    assert terminal.status == "canceled"
    assert terminal.percent == 15
    assert store.request_stop(task.id) is False


def test_stop_request_wins_race_with_worker_success(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    task = store.create(kind="download", title="download", metadata={})
    store.update(task.id, status="running", percent=80)

    assert store.request_stop(task.id) is True
    store.update(
        task.id,
        status="succeeded",
        percent=100,
        result={"files": 1},
        finished=True,
    )

    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.status == "canceled"
    assert loaded.percent == 100
    assert loaded.finished_at is not None


def test_task_events_cannot_regress_persisted_progress(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    task = store.create(kind="download", title="download", metadata={})
    store.update(task.id, status="running", percent=60)

    store.append_event(
        task.id,
        TaskEvent(level="info", message="late event", percent=25),
    )

    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.percent == 60


def test_task_session_store_marks_all_stale_running_interrupted(tmp_path: Path) -> None:
    store = TaskSessionStore(tmp_path / "tasks.sqlite3")
    ids: list[str] = []
    for index in range(105):
        session = store.create(kind="bulk", title=f"task-{index}", metadata={})
        store.update(session.id, status="running", percent=index % 100)
        ids.append(session.id)

    reopened = TaskSessionStore(tmp_path / "tasks.sqlite3")
    count = reopened.mark_stale_interrupted()

    assert count == 105
    assert all(reopened.get(session_id).status == "interrupted" for session_id in ids)  # type: ignore[union-attr]


def test_default_task_store_path_is_named_tasks_db() -> None:
    from lorahub.api.task_sessions import default_task_store_path

    assert default_task_store_path().name == "tasks.sqlite3"


def test_prune_terminal_session_cache_keeps_active_and_recent() -> None:
    from types import SimpleNamespace

    sessions = {
        "old": SimpleNamespace(status="succeeded", finished_at=1.0),
        "recent": SimpleNamespace(status="failed", finished_at=3.0),
        "middle": SimpleNamespace(status="canceled", finished_at=2.0),
        "active": SimpleNamespace(status="running", started_at=0.0),
    }

    prune_terminal_session_cache(sessions, keep=2)

    assert set(sessions) == {"recent", "middle", "active"}
