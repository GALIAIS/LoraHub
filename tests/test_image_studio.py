"""Tests for ImageStudioStore + router endpoints."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module
from lorahub.api.image_studio_store import (
    ImageAnnotation,
    ImageEmbedding,
    ImagePhash,
    ImageStudioStore,
    PendingOp,
)
from lorahub.api.image_studio_library import (
    ImageStudioLibrary,
    PromptTemplate,
    TagEntry,
    TriggerWordEntry,
)
from lorahub.api.task_sessions import TaskSessionStore


# --------------------------------------------------------------------------- #
# Store unit tests
# --------------------------------------------------------------------------- #


def test_annotation_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    ann = ImageAnnotation(
        image_path="/data/img001.png",
        sha256="abc123",
        width=1024,
        height=768,
        bytes=500_000,
        ai_quality_label="good",
        ai_trigger_words=["1girl", "blue_hair"],
        favorite=True,
    )
    s.upsert_annotation(ann)
    fetched = s.get_annotation("/data/img001.png")
    assert fetched is not None
    assert fetched.sha256 == "abc123"
    assert fetched.width == 1024
    assert fetched.ai_trigger_words == ["1girl", "blue_hair"]
    assert fetched.favorite is True
    assert fetched.updated_at != ""


def test_annotation_upsert_overwrites(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(
        image_path="/x.png", sha256="aaa", user_notes="first"
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/x.png", sha256="aaa", user_notes="second"
    ))
    fetched = s.get_annotation("/x.png")
    assert fetched is not None
    assert fetched.user_notes == "second"


def test_annotation_list_filters(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(
        image_path="/a.png", sha256="a", ai_quality_label="good", favorite=True
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/b.png", sha256="b", ai_quality_label="bad"
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/c.png", sha256="c", ai_quality_label="good", soft_deleted=True
    ))
    assert len(s.list_annotations(quality_label="good")) == 2
    assert len(s.list_annotations(favorite=True)) == 1
    assert len(s.list_annotations(soft_deleted=True)) == 1


def test_annotation_delete(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(image_path="/x.png", sha256="x"))
    assert s.delete_annotation("/x.png") is True
    assert s.get_annotation("/x.png") is None
    assert s.delete_annotation("/x.png") is False


def test_phash_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/img.png", "phash64", "abcdef01"))
    s.upsert_phash(ImagePhash("/img.png", "dhash64", "12345678"))
    hashes = s.get_phashes("/img.png")
    assert len(hashes) == 2
    algos = {h.algo for h in hashes}
    assert algos == {"phash64", "dhash64"}


def test_phash_upsert_overwrites(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/img.png", "phash64", "old"))
    s.upsert_phash(ImagePhash("/img.png", "phash64", "new"))
    hashes = s.get_phashes("/img.png")
    assert len(hashes) == 1
    assert hashes[0].hash == "new"


def test_phash_list_by_algo(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/a.png", "phash64", "aaa"))
    s.upsert_phash(ImagePhash("/b.png", "phash64", "bbb"))
    s.upsert_phash(ImagePhash("/c.png", "dhash64", "ccc"))
    assert len(s.list_phashes("phash64")) == 2
    assert len(s.list_phashes("dhash64")) == 1


def test_pending_ops_lifecycle(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    op = s.add_pending_op(PendingOp(
        id="", image_path="/img.png", op="rotate", payload={"degrees": 90}
    ))
    assert op.id != ""
    assert op.created_at != ""
    ops = s.list_pending_ops("/img.png")
    assert len(ops) == 1
    assert ops[0].payload == {"degrees": 90}
    assert s.delete_pending_op(op.id) is True
    assert s.list_pending_ops("/img.png") == []


def test_pending_ops_clear(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.add_pending_op(PendingOp(id="", image_path="/a.png", op="rotate", payload={}))
    s.add_pending_op(PendingOp(id="", image_path="/a.png", op="flip", payload={}))
    s.add_pending_op(PendingOp(id="", image_path="/b.png", op="rotate", payload={}))
    assert s.clear_pending_ops("/a.png") == 2
    assert s.list_pending_ops("/a.png") == []
    assert len(s.list_pending_ops("/b.png")) == 1


def test_embedding_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    vec = [0.1, 0.2, 0.3, 0.4]
    emb = ImageEmbedding(
        image_path="/img.png", model_id="clip-v1", dim=4, vector=vec
    )
    s.upsert_embedding(emb)
    fetched = s.get_embedding("/img.png", "clip-v1")
    assert fetched is not None
    assert fetched.dim == 4
    assert len(fetched.vector) == 4
    assert abs(fetched.vector[0] - 0.1) < 1e-6
    assert abs(fetched.vector[3] - 0.4) < 1e-6


def test_embedding_list_by_model(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_embedding(ImageEmbedding("/a.png", "m1", 2, [1.0, 2.0]))
    s.upsert_embedding(ImageEmbedding("/b.png", "m1", 2, [3.0, 4.0]))
    s.upsert_embedding(ImageEmbedding("/c.png", "m2", 2, [5.0, 6.0]))
    assert len(s.list_embeddings("m1")) == 2
    assert len(s.list_embeddings("m2")) == 1


def test_embedding_delete(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_embedding(ImageEmbedding("/x.png", "m1", 2, [1.0, 2.0]))
    s.delete_embeddings("/x.png")
    assert s.get_embedding("/x.png", "m1") is None


# --------------------------------------------------------------------------- #
# Router HTTP tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.chdir(tmp_path)
    # FastAPI runs sync handlers on a worker thread whose ``os.getcwd()``
    # can lag behind the main thread's ``chdir`` on some platforms (e.g.
    # WSL with the cwd on a ``/mnt/...`` drvfs mount), so the dataset
    # path allow-list — which derives one root from cwd — would reject
    # files under ``tmp_path``. Pin the allow-list with an explicit env
    # root so the fixture works regardless of thread-cwd behaviour.
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    store = ImageStudioStore(tmp_path / "is.sqlite")
    monkeypatch.setattr(app_module, "_image_studio_store", store)
    library = ImageStudioLibrary(tmp_path / "is.sqlite")
    monkeypatch.setattr(app_module, "_image_studio_library", library)
    monkeypatch.setattr(app_module, "_task_session_store", TaskSessionStore(tmp_path / "tasks.sqlite3"))
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a sample dataset directory with images."""
    d = tmp_path / "dataset"
    d.mkdir()
    from PIL import Image  # noqa: PLC0415
    for name in ("a.png", "b.png", "c.png"):
        img = Image.new("RGB", (64, 64), color="red")
        img.save(d / name)
    (d / "a.txt").write_text("caption for a", encoding="utf-8")
    return d


def test_list_images(client: TestClient, sample_dir: Path) -> None:
    r = client.get("/api/image-studio/list", params={"path": str(sample_dir)})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    names = {item["name"] for item in body["items"]}
    assert names == {"a.png", "b.png", "c.png"}


def test_list_pagination(client: TestClient, sample_dir: Path) -> None:
    r = client.get("/api/image-studio/list", params={
        "path": str(sample_dir), "limit": 2, "page": 1
    })
    assert r.status_code == 200
    assert len(r.json()["items"]) == 2
    r2 = client.get("/api/image-studio/list", params={
        "path": str(sample_dir), "limit": 2, "page": 2
    })
    assert len(r2.json()["items"]) == 1


def test_dataset_name_endpoints_reject_dot_paths(
    client: TestClient, tmp_path: Path
) -> None:
    dataset = tmp_path / "safe"
    dataset.mkdir()
    (tmp_path / "outside.txt").write_text("keep", encoding="utf-8")

    r = client.delete("/api/image-studio/datasets/..%5C")
    assert r.status_code in (400, 404), r.text

    assert dataset.is_dir()
    assert (tmp_path / "outside.txt").is_file()


def test_dataset_upload_normalizes_plain_file_names(
    client: TestClient, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    r = client.post(
        "/api/image-studio/datasets/dataset/upload",
        files={"files": ("..\\escape.png", b"not-image", "image/png")},
    )

    assert r.status_code == 200, r.text
    assert not (tmp_path / "escape.png").exists()
    assert (dataset / "escape.png").read_bytes() == b"not-image"


def test_get_image(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    r = client.get("/api/image-studio/image", params={"path": img_path})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "a.png"
    assert body["captionExists"] is True
    assert body["caption"] == "caption for a"
    assert body["phash"] == {}
    assert body["pendingOps"] == []


def test_smart_caption_writes_task_session(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from lorahub.api.ai_store import AIRoute

    class FakeAIStore:
        def get_route(self, _task_id: str) -> AIRoute:
            return AIRoute(
                task_id="tagging.assist",
                provider_id="fake-provider",
                model_id="fake-model",
            )

    monkeypatch.setattr(app_module, "_ai_store", FakeAIStore())

    response = client.post(
        "/api/image-studio/ai/smart-caption",
        json={
            "path": str(sample_dir),
            "skipExisting": True,
            "useWd14": False,
            "captionMode": "style",
            "triggerWord": "teststyle",
            "concurrency": 1,
            "taggerConcurrency": 1,
        },
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]
    assert response.json()["total"] == 2
    assert response.json()["skipped"] == 1
    assert response.json()["total"] == 2
    assert response.json()["skipped"] == 1

    deadline = time.time() + 5
    latest: dict | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/ai/smart-caption/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_smart_caption",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["path"] == str(sample_dir)
    assert latest["result"]["processed"] == 2
    assert latest["events"][-1]["message"].startswith("finished")


def test_smart_caption_status_reads_persisted_result_after_memory_clear(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from lorahub.api.ai_store import AIRoute
    from lorahub.api.routers.image_studio import ai as ai_router

    class FakeAIStore:
        def get_route(self, _task_id: str) -> AIRoute:
            return AIRoute(
                task_id="tagging.assist",
                provider_id="fake-provider",
                model_id="fake-model",
            )

    monkeypatch.setattr(app_module, "_ai_store", FakeAIStore())

    response = client.post(
        "/api/image-studio/ai/smart-caption",
        json={
            "path": str(sample_dir),
            "skipExisting": True,
            "useWd14": False,
            "concurrency": 1,
            "taggerConcurrency": 1,
        },
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]
    assert response.json()["total"] == 2
    assert response.json()["skipped"] == 1

    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/ai/smart-caption/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            break
        time.sleep(0.02)

    ai_router._smart_caption_sessions.clear()
    recovered = client.get(
        f"/api/image-studio/ai/smart-caption/status/{session_id}",
    )
    assert recovered.status_code == 200, recovered.text
    body = recovered.json()
    assert body["session_id"] == session_id
    assert body["status"] == "succeeded"
    assert body["processed"] == 2


@pytest.mark.parametrize(
    ("kind", "url"),
    [
        ("image_studio_smart_caption", "/api/image-studio/ai/smart-caption/status/{sid}"),
        ("image_studio_quality", "/api/image-studio/ai/quality/status/{sid}"),
        ("image_studio_caption", "/api/image-studio/ai/caption/status/{sid}"),
        ("image_studio_trigger_words", "/api/image-studio/ai/trigger-words/status/{sid}"),
        ("image_studio_auto_rotate", "/api/image-studio/curate/auto-rotate/status/{sid}"),
        ("image_studio_batch_resize", "/api/image-studio/curate/batch-resize/status/{sid}"),
    ],
)
def test_image_studio_status_recovers_interrupted_task_without_result(
    client: TestClient, sample_dir: Path, kind: str, url: str
) -> None:
    task = app_module._task_session_store.create(
        kind=kind,
        title="interrupted task",
        metadata={"path": str(sample_dir), "dataset_path": str(sample_dir), "total": 3},
    )
    app_module._task_session_store.update(
        task.id,
        status="running",
        percent=42,
    )
    app_module._task_session_store.mark_stale_interrupted()

    recovered = client.get(url.format(sid=task.id))

    assert recovered.status_code == 200, recovered.text
    body = recovered.json()
    assert body["session_id"] == task.id
    assert body["status"] == "interrupted"
    assert body["percent"] == 42
    assert body["error"] == "task interrupted by server restart"
    assert body["events"][-1]["message"] == "task interrupted by server restart"


def test_ai_quality_writes_task_session_and_recovers_status(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from dataclasses import dataclass

    from lorahub.api.ai_store import AIRoute
    from lorahub.api.routers.image_studio import ai as ai_router
    from lorahub.core.ai import client as ai_client

    class FakeAIStore:
        def get_route(self, _task_id: str) -> AIRoute:
            return AIRoute(
                task_id="quality.score",
                provider_id="fake-provider",
                model_id="fake-model",
            )

    @dataclass
    class FakeAIResult:
        content: str
        provider_name: str = "fake"
        model_id: str = "fake-model"
        raw: dict[str, Any] | None = None

    monkeypatch.setattr(app_module, "_ai_store", FakeAIStore())
    monkeypatch.setattr(
        ai_client,
        "invoke",
        lambda *_args, **_kwargs: FakeAIResult(
            '{"score": 80, "label": "good", "reason": "sharp"}',
        ),
    )

    response = client.post(
        "/api/image-studio/ai/quality/start",
        json={"path": str(sample_dir), "recursive": False, "skipScored": True},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/ai/quality/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_quality",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["processed"] == 3
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["path"] == str(sample_dir)
    assert latest["result"]["processed"] == 3
    assert latest["events"][-1]["message"].startswith("finished")

    ai_router._quality_sessions.clear()
    recovered = client.get(f"/api/image-studio/ai/quality/status/{session_id}")
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["processed"] == 3


def test_trigger_words_writes_task_session_and_recovers_status(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from dataclasses import dataclass

    from lorahub.api.ai_store import AIRoute
    from lorahub.api.routers.image_studio import ai as ai_router
    from lorahub.core.ai import client as ai_client

    class FakeAIStore:
        def get_route(self, _task_id: str) -> AIRoute:
            return AIRoute(
                task_id="trigger.words",
                provider_id="fake-provider",
                model_id="fake-model",
            )

    @dataclass
    class FakeAIResult:
        content: str
        provider_name: str = "fake"
        model_id: str = "fake-model"
        raw: dict[str, Any] | None = None

    monkeypatch.setattr(app_module, "_ai_store", FakeAIStore())
    monkeypatch.setattr(
        ai_client,
        "invoke",
        lambda *_args, **_kwargs: FakeAIResult(
            '{"triggers": ["crimson cloak", "star wand"]}',
        ),
    )

    response = client.post(
        "/api/image-studio/ai/trigger-words/start",
        json={"path": str(sample_dir), "recursive": False, "skipAnalyzed": True},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/ai/trigger-words/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_trigger_words",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["processed"] == 3
    assert status["dataset_top"][0] == {"trigger": "crimson cloak", "count": 3}
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["path"] == str(sample_dir)
    assert latest["result"]["processed"] == 3
    assert latest["events"][-1]["message"].startswith("finished")

    ai_router._trigger_words_sessions.clear()
    recovered = client.get(
        f"/api/image-studio/ai/trigger-words/status/{session_id}",
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["processed"] == 3


def test_ai_caption_writes_task_session_and_recovers_status(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from dataclasses import dataclass

    from lorahub.api.ai_store import AIRoute
    from lorahub.api.routers.image_studio import ai as ai_router
    from lorahub.core.ai import client as ai_client

    class FakeAIStore:
        def get_route(self, _task_id: str) -> AIRoute:
            return AIRoute(
                task_id="tagging.assist",
                provider_id="fake-provider",
                model_id="fake-model",
            )

    @dataclass
    class FakeAIResult:
        content: str
        provider_name: str = "fake"
        model_id: str = "fake-model"
        raw: dict[str, Any] | None = None

    monkeypatch.setattr(app_module, "_ai_store", FakeAIStore())
    monkeypatch.setattr(
        ai_client,
        "invoke",
        lambda *_args, **_kwargs: FakeAIResult("caption text"),
    )

    response = client.post(
        "/api/image-studio/ai/caption/start",
        json={"path": str(sample_dir), "recursive": False, "skipAnnotated": True},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/ai/caption/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_caption",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["processed"] == 2
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["path"] == str(sample_dir)
    assert latest["result"]["processed"] == 2
    assert latest["events"][-1]["message"].startswith("finished")

    ai_router._caption_sessions.clear()
    recovered = client.get(f"/api/image-studio/ai/caption/status/{session_id}")
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["processed"] == 2


def test_image_studio_tagging_writes_task_session(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from lorahub.api.routers.image_studio import tagging as tagging_router

    class FakeTagger:
        active_provider = "CPUExecutionProvider"

        def load(self) -> None:
            pass

        def tag_directory(self, directory: Path, **kwargs: Any) -> list[Any]:
            from lorahub.core.tagging.wd14 import _iter_images  # noqa: PLC0415

            results: list[Any] = []
            for image in _iter_images(directory, recursive=kwargs["recursive"]):
                image.with_suffix(".txt").write_text("studio tag", encoding="utf-8")
                kwargs["on_progress"](image, object())
                results.append(object())
            return results

    monkeypatch.setattr(tagging_router, "_build_is_tagger", lambda _req: FakeTagger())

    response = client.post(
        "/api/image-studio/tagging/start",
        json={"path": str(sample_dir), "device": "cpu", "overwrite": True},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(f"/api/image-studio/tagging/{session_id}").json()
        if status["status"] in {"succeeded", "failed"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_tagging",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["metadata"]["path"] == str(sample_dir)
    assert latest["metadata"]["tagger"] == "wd14"
    assert latest["status"] == "succeeded"
    assert latest["result"]["written"] == 3
    assert latest["events"][-1]["message"].startswith("done")


def test_image_studio_tagging_status_reads_persisted_result_after_memory_clear(
    client: TestClient,
    sample_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from lorahub.api.routers.image_studio import tagging as tagging_router

    class FakeTagger:
        active_provider = "CPUExecutionProvider"

        def load(self) -> None:
            pass

        def tag_directory(self, directory: Path, **kwargs: Any) -> list[Any]:
            from lorahub.core.tagging.wd14 import _iter_images  # noqa: PLC0415

            results: list[Any] = []
            for image in _iter_images(directory, recursive=kwargs["recursive"]):
                image.with_suffix(".txt").write_text("studio tag", encoding="utf-8")
                kwargs["on_progress"](image, object())
                results.append(object())
            return results

    monkeypatch.setattr(tagging_router, "_build_is_tagger", lambda _req: FakeTagger())

    response = client.post(
        "/api/image-studio/tagging/start",
        json={"path": str(sample_dir), "device": "cpu", "overwrite": True},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get(f"/api/image-studio/tagging/{session_id}").json()
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)

    tagging_router._is_tagging_sessions.clear()
    recovered = client.get(f"/api/image-studio/tagging/{session_id}")
    assert recovered.status_code == 200, recovered.text
    body = recovered.json()
    assert body["session_id"] == session_id
    assert body["status"] == "succeeded"
    assert body["written"] == 3


def test_image_studio_tagging_status_recovers_interrupted_task(
    client: TestClient, sample_dir: Path
) -> None:
    from lorahub.api import app as app_module
    from lorahub.api.routers.image_studio import tagging as tagging_router

    task = app_module._task_session_store.create(
        kind="image_studio_tagging",
        title="wd14:dataset",
        metadata={
            "path": str(sample_dir),
            "tagger": "wd14",
            "model_id": "wd14-vit-v2",
            "device": "cpu",
            "overwrite": True,
            "recursive": False,
        },
    )
    app_module._task_session_store.update(task.id, status="running", percent=52)
    app_module._task_session_store.mark_stale_interrupted()
    tagging_router._is_tagging_sessions.clear()

    recovered = client.get(f"/api/image-studio/tagging/{task.id}")

    assert recovered.status_code == 200, recovered.text
    body = recovered.json()
    assert body["session_id"] == task.id
    assert body["status"] == "interrupted"
    assert body["percent"] == 52
    assert body["error"] == "task interrupted by server restart"


def test_annotation_crud_via_api(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    r = client.put("/api/image-studio/annotations", json={
        "path": img_path, "userQualityLabel": "good", "favorite": True
    })
    assert r.status_code == 200
    assert r.json()["annotation"]["favorite"] is True
    assert r.json()["annotation"]["userQualityLabel"] == "good"

    r2 = client.delete("/api/image-studio/annotations", params={"path": img_path})
    assert r2.status_code == 200


def test_pending_ops_via_api(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "b.png")
    r = client.post("/api/image-studio/ops", json={
        "path": img_path, "op": "rotate", "payload": {"degrees": 90}
    })
    assert r.status_code == 200
    op_id = r.json()["id"]
    assert op_id

    r2 = client.get("/api/image-studio/ops", params={"path": img_path})
    assert len(r2.json()["ops"]) == 1

    r3 = client.delete(f"/api/image-studio/ops/{op_id}")
    assert r3.status_code == 200


def test_apply_ops_rotate(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "b.png")
    client.post("/api/image-studio/ops", json={
        "path": img_path, "op": "rotate", "payload": {"degrees": 90}
    })
    r = client.post("/api/image-studio/ops/apply", json={"path": img_path})
    assert r.status_code == 200
    assert len(r.json()["applied"]) == 1
    assert r.json()["errors"] == []


# --------------------------------------------------------------------------- #
# Phash unit tests
# --------------------------------------------------------------------------- #


def test_phash64_deterministic(tmp_path: Path) -> None:
    from lorahub.core.phash import phash64

    from PIL import Image
    img = Image.new("RGB", (64, 64), color="blue")
    p = tmp_path / "blue.png"
    img.save(p)
    h1 = phash64(p)
    h2 = phash64(p)
    assert h1 == h2
    assert len(h1) == 16  # 64 bits = 16 hex chars


def test_dhash64_deterministic(tmp_path: Path) -> None:
    from lorahub.core.phash import dhash64

    from PIL import Image
    img = Image.new("RGB", (64, 64), color="green")
    p = tmp_path / "green.png"
    img.save(p)
    h1 = dhash64(p)
    h2 = dhash64(p)
    assert h1 == h2
    assert len(h1) == 16


def test_hamming_distance() -> None:
    from lorahub.core.phash import hamming_distance
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    assert hamming_distance("ffffffffffffffff", "0000000000000000") == 64


def test_similar_images_low_distance(tmp_path: Path) -> None:
    from lorahub.core.phash import hamming_distance, phash64

    from PIL import Image
    img1 = Image.new("RGB", (128, 128), color=(100, 150, 200))
    img2 = img1.copy()
    # Slightly modify one pixel
    img2.putpixel((0, 0), (101, 150, 200))
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    img1.save(p1)
    img2.save(p2)
    h1 = phash64(p1)
    h2 = phash64(p2)
    assert hamming_distance(h1, h2) <= 5


# --------------------------------------------------------------------------- #
# Dedupe router tests
# --------------------------------------------------------------------------- #


def test_dedupe_scan_and_clusters(client: TestClient, sample_dir: Path) -> None:
    # Create two identical images (should cluster together)
    from PIL import Image
    img = Image.new("RGB", (64, 64), color="red")
    img.save(sample_dir / "dup1.png")
    img.save(sample_dir / "dup2.png")

    r = client.post("/api/image-studio/dedupe/scan", json={
        "path": str(sample_dir), "algo": "phash64", "threshold": 10
    })
    assert r.status_code == 200
    assert r.json()["computed"] >= 2

    r2 = client.get("/api/image-studio/dedupe/clusters", params={
        "path": str(sample_dir), "kind": "phash", "threshold": 10
    })
    assert r2.status_code == 200
    clusters = r2.json()["clusters"]
    # At least one cluster with the two identical images
    assert len(clusters) >= 1
    dup_cluster = next(
        (c for c in clusters if any("dup1" in m["path"] for m in c["members"])),
        None,
    )
    assert dup_cluster is not None
    assert len(dup_cluster["members"]) >= 2


def test_batch_delete(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "c.png")
    r = client.post("/api/image-studio/dedupe/batch-delete", json={
        "paths": [img_path]
    })
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 1
    assert not (sample_dir / "c.png").exists()


def test_batch_delete_blocks_favorites(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    # Mark as favorite
    client.put("/api/image-studio/annotations", json={
        "path": img_path, "favorite": True
    })
    r = client.post("/api/image-studio/dedupe/batch-delete", json={
        "paths": [img_path]
    })
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0
    assert len(r.json()["errors"]) == 1
    assert "favourite" in r.json()["errors"][0]["error"]


def test_batch_resize_writes_task_session_and_recovers_status(
    client: TestClient,
    sample_dir: Path,
) -> None:
    import time

    from lorahub.api.routers.image_studio import curate as curate_router

    response = client.post(
        "/api/image-studio/curate/batch-resize/start",
        json={
            "dataset_path": str(sample_dir),
            "target_short_edge": 128,
            "filter": "bilinear",
            "upscale": True,
            "recursive": False,
        },
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]
    assert response.json()["total"] == 3

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/curate/batch-resize/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_batch_resize",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["processed"] == 3
    assert status["resampled_count"] == 3
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["dataset_path"] == str(sample_dir)
    assert latest["result"]["resampled_count"] == 3
    assert latest["events"][-1]["message"].startswith("finished")

    curate_router._batch_resize_sessions.clear()
    recovered = client.get(
        f"/api/image-studio/curate/batch-resize/status/{session_id}",
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["resampled_count"] == 3


def test_auto_rotate_writes_task_session_and_recovers_status(
    client: TestClient,
    sample_dir: Path,
) -> None:
    import time

    from lorahub.api.routers.image_studio import curate as curate_router

    response = client.post(
        "/api/image-studio/curate/auto-rotate/start",
        json={
            "dataset_path": str(sample_dir),
            "recursive": False,
        },
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]
    assert response.json()["total"] == 3

    deadline = time.time() + 5
    status: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        status = client.get(
            f"/api/image-studio/curate/auto-rotate/status/{session_id}",
        ).json()
        if status["status"] in {"succeeded", "failed", "canceled"}:
            latest_response = client.get(
                "/api/tasks/latest?kind=image_studio_auto_rotate",
            )
            if latest_response.status_code == 200:
                latest = latest_response.json()
            break
        time.sleep(0.02)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["processed"] == 3
    assert status["rotated_count"] == 0
    assert status["skipped_count"] == 3
    assert latest is not None
    assert latest["id"] == session_id
    assert latest["status"] == "succeeded"
    assert latest["metadata"]["dataset_path"] == str(sample_dir)
    assert latest["result"]["skipped_count"] == 3
    assert latest["events"][-1]["message"].startswith("finished")

    curate_router._auto_rotate_sessions.clear()
    recovered = client.get(
        f"/api/image-studio/curate/auto-rotate/status/{session_id}",
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["skipped_count"] == 3


# --------------------------------------------------------------------------- #
# Library — store unit tests
# --------------------------------------------------------------------------- #


def test_library_tag_round_trip(tmp_path: Path) -> None:
    lib = ImageStudioLibrary(tmp_path / "is.sqlite")
    saved = lib.upsert_tag(
        TagEntry(
            tag="blue hair",
            category="character",
            aliases=["azure hair"],
            color="#3b82f6",
            notes="lead character",
        )
    )
    assert saved.created_at
    assert saved.updated_at
    fetched = lib.get_tag("blue hair")
    assert fetched is not None
    assert fetched.aliases == ["azure hair"]
    assert fetched.notes == "lead character"


def test_library_tag_search(tmp_path: Path) -> None:
    lib = ImageStudioLibrary(tmp_path / "is.sqlite")
    lib.upsert_tag(TagEntry(tag="blue hair", category="character"))
    lib.upsert_tag(TagEntry(tag="red hair", category="character"))
    lib.upsert_tag(TagEntry(tag="masterpiece", category="quality"))
    assert {t.tag for t in lib.list_tags(category="character")} == {
        "blue hair",
        "red hair",
    }
    assert {t.tag for t in lib.list_tags(search="blue")} == {"blue hair"}


def test_library_trigger_round_trip(tmp_path: Path) -> None:
    lib = ImageStudioLibrary(tmp_path / "is.sqlite")
    lib.upsert_trigger(
        TriggerWordEntry(
            trigger_word="aelina",
            character_name="Aelina",
            datasets=["proj-a", "proj-b"],
        )
    )
    fetched = lib.get_trigger("aelina")
    assert fetched is not None
    assert fetched.datasets == ["proj-a", "proj-b"]


def test_library_prompt_default_demotes_others(tmp_path: Path) -> None:
    lib = ImageStudioLibrary(tmp_path / "is.sqlite")
    a = lib.upsert_prompt(
        PromptTemplate(id="", name="Anima A", category="caption", body="x", is_default=True)
    )
    b = lib.upsert_prompt(
        PromptTemplate(id="", name="Anima B", category="caption", body="y", is_default=True)
    )
    refreshed_a = lib.get_prompt(a.id)
    refreshed_b = lib.get_prompt(b.id)
    assert refreshed_a is not None and refreshed_a.is_default is False
    assert refreshed_b is not None and refreshed_b.is_default is True


# --------------------------------------------------------------------------- #
# Library — API endpoint tests
# --------------------------------------------------------------------------- #


def test_library_tags_api_crud(client: TestClient) -> None:
    r = client.put(
        "/api/image-studio/library/tags/blue%20hair",
        json={
            "tag": "blue hair",
            "category": "character",
            "aliases": ["azure hair"],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["tag"] == "blue hair"

    listing = client.get("/api/image-studio/library/tags?category=character").json()
    assert any(t["tag"] == "blue hair" for t in listing["tags"])

    r = client.delete("/api/image-studio/library/tags/blue%20hair")
    assert r.status_code == 200
    assert r.json()["deleted"] is True


def test_library_tags_path_body_mismatch_rejected(client: TestClient) -> None:
    r = client.put(
        "/api/image-studio/library/tags/foo",
        json={"tag": "bar"},
    )
    assert r.status_code == 400


def test_library_triggers_api_crud(client: TestClient) -> None:
    r = client.put(
        "/api/image-studio/library/triggers/aelina",
        json={
            "triggerWord": "aelina",
            "characterName": "Aelina",
            "datasets": ["d1"],
        },
    )
    assert r.status_code == 200
    assert r.json()["triggerWord"] == "aelina"

    listing = client.get(
        "/api/image-studio/library/triggers?characterName=Aelina"
    ).json()
    assert any(t["triggerWord"] == "aelina" for t in listing["triggers"])

    r = client.delete("/api/image-studio/library/triggers/aelina")
    assert r.status_code == 200


def test_library_prompts_api_crud(client: TestClient) -> None:
    r = client.post(
        "/api/image-studio/library/prompts",
        json={"name": "Anima Caption", "category": "caption", "body": "hello"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert pid

    # Duplicate name should 409.
    r2 = client.post(
        "/api/image-studio/library/prompts",
        json={"name": "Anima Caption", "category": "caption", "body": "x"},
    )
    assert r2.status_code == 409

    # Update via PUT should change body and bump updated_at.
    r3 = client.put(
        f"/api/image-studio/library/prompts/{pid}",
        json={"name": "Anima Caption", "category": "caption", "body": "world"},
    )
    assert r3.status_code == 200
    assert r3.json()["body"] == "world"

    listing = client.get(
        "/api/image-studio/library/prompts?category=caption"
    ).json()
    assert any(p["id"] == pid for p in listing["prompts"])

    r4 = client.delete(f"/api/image-studio/library/prompts/{pid}")
    assert r4.status_code == 200


def test_library_prompts_post_rejects_id(client: TestClient) -> None:
    r = client.post(
        "/api/image-studio/library/prompts",
        json={"id": "should-not-allow", "name": "x", "body": "y"},
    )
    assert r.status_code == 400
