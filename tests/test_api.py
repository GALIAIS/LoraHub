"""Tests for the LoraHub HTTP API."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lorahub.api import state


def _make_stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        print("epoch 1/1", flush=True)
        print("saving checkpoint: out.safetensors", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    (root / "train_network.py").write_text(stub, encoding="utf-8")
    (root / "sdxl_train_network.py").write_text(stub, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    yield fresh


@pytest.fixture
def client() -> TestClient:
    from lorahub.api.app import app

    return TestClient(app)


def _recipe_payload(tmp_path: Path) -> dict[str, Any]:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {
            "sd_scripts_path": str(sd),
            "python_executable": __import__("sys").executable,
        },
    }


def test_health_returns_version(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()


def test_recipe_schema_is_valid_json_schema(client: TestClient) -> None:
    r = client.get("/api/recipes/schema")
    assert r.status_code == 200
    schema = r.json()
    assert schema["title"] == "RecipeConfig"
    assert "base_model" in schema["$defs"] or "base_model" in str(schema)


def test_list_jobs_starts_empty(client: TestClient) -> None:
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json() == {"jobs": []}


def test_get_unknown_job_returns_404(client: TestClient) -> None:
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_create_and_complete_job(client: TestClient, tmp_path: Path) -> None:
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    r = client.post("/api/jobs", json=payload)
    assert r.status_code == 202, r.text
    summary = r.json()
    assert summary["state"] in ("queued", "running", "succeeded", "failed")

    job_id = summary["id"]

    # Wait up to ~30s for the stub kohya to finish
    import time

    deadline = time.time() + 30
    while time.time() < deadline:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)

    final = client.get(f"/api/jobs/{job_id}").json()
    assert final["state"] == "succeeded", final
    assert final["returncode"] == 0


def test_recent_events_returned_after_completion(
    client: TestClient, tmp_path: Path
) -> None:
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    job_id = client.post("/api/jobs", json=payload).json()["id"]

    import time

    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)

    events = client.get(f"/api/jobs/{job_id}/events").json()["events"]
    assert any(e["type"] == "epoch_end" for e in events)
    assert any(e["type"] == "checkpoint_saved" for e in events)
    assert events[-1]["type"] == "done"


def test_invalid_recipe_returns_422(client: TestClient) -> None:
    r = client.post("/api/jobs", json={"recipe": {"missing": "everything"}})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Recipe template browsing
# --------------------------------------------------------------------------- #


@pytest.fixture
def recipes_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the API at an isolated recipes directory."""
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    monkeypatch.setenv("LORAHUB_RECIPES_DIR", str(rdir))
    return rdir


def _write_valid_recipe(rdir: Path, name: str = "demo") -> Path:
    ckpt = rdir.parent / "model.safetensors"
    ckpt.write_bytes(b"")
    data = rdir.parent / "data"
    data.mkdir(exist_ok=True)
    body = textwrap.dedent(
        f"""
        base_model:
          arch: sdxl
          checkpoint: {ckpt!s}
        dataset:
          source: {data!s}
        schedule:
          epochs: 2
          batch_size: 1
        sampling:
          enabled: false
        """
    ).strip() + "\n"
    p = rdir / f"{name}.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_list_recipes_returns_valid_and_invalid(
    client: TestClient, recipes_dir: Path
) -> None:
    _write_valid_recipe(recipes_dir, "good")
    (recipes_dir / "broken.yaml").write_text("base_model: {}\n", encoding="utf-8")
    (recipes_dir / "ignore-me.txt").write_text("not yaml", encoding="utf-8")

    r = client.get("/api/recipes")
    assert r.status_code == 200
    body = r.json()
    names = {it["name"] for it in body["recipes"]}
    assert names == {"good", "broken"}

    good = next(it for it in body["recipes"] if it["name"] == "good")
    assert good["valid"] is True
    assert good["arch"] == "sdxl"
    assert "epoch" in good["summary"]

    broken = next(it for it in body["recipes"] if it["name"] == "broken")
    assert broken["valid"] is False
    assert broken["error"]


def test_get_recipe_returns_content_and_parsed(
    client: TestClient, recipes_dir: Path
) -> None:
    _write_valid_recipe(recipes_dir, "good")
    r = client.get("/api/recipes/good")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "good"
    assert "base_model:" in body["content"]
    assert body["parsed"]["base_model"]["arch"] == "sdxl"
    assert body["error"] is None


def test_get_recipe_missing_returns_404(
    client: TestClient, recipes_dir: Path
) -> None:
    r = client.get("/api/recipes/nope")
    assert r.status_code == 404


def test_get_recipe_blocks_path_traversal(
    client: TestClient, recipes_dir: Path
) -> None:
    r = client.get("/api/recipes/..%2Fpasswd")
    # FastAPI normalizes %2F into /, our handler rejects bare names with slashes
    assert r.status_code in (400, 404)


def test_recipe_schema_still_resolves_under_recipes_prefix(
    client: TestClient, recipes_dir: Path
) -> None:
    # /recipes/schema must keep working alongside /recipes/{name}
    r = client.get("/api/recipes/schema")
    assert r.status_code == 200
    assert r.json()["title"] == "RecipeConfig"
