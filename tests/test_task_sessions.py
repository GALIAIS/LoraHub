from __future__ import annotations

from pathlib import Path

from lorahub.api.task_sessions import TaskEvent, TaskSessionStore


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
