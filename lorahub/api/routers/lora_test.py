"""LoRA testbench endpoints for post-training image generation."""

from __future__ import annotations

import json
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api import state
from lorahub.api.jobs_helpers import _list_workspace_files, _resolve_workspace_file
from lorahub.api.paths import runs_dir
from lorahub.api.task_sessions import TaskEvent, TaskSessionStore, default_task_store_path
from lorahub.core.backends.anima_lora import bootstrap as anima_bootstrap
from lorahub.core.config.schema import TrainingConfig

router = APIRouter(prefix="/api")

_KIND = "lora-test"
_RESULT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


class GenerateRequest(BaseModel):
    job_id: str
    checkpoint_path: str
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)
    width: int = Field(default=912, ge=256, le=2048)
    height: int = Field(default=1632, ge=256, le=2048)
    seed: int = -1
    batch_count: int = Field(default=4, ge=1, le=32)
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=4.5, ge=0.0, le=30.0)
    sampler: str = Field(default="euler", max_length=32)
    lora_weight: float = Field(default=1.0, ge=-2.0, le=2.0)
    output_format: Literal["png"] = "png"

    @property
    def validated_sampler(self) -> str:
        if self.sampler not in {"euler", "er_sde", "lcm"}:
            raise ValueError("sampler must be one of euler, er_sde, lcm")
        return self.sampler


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    job: state.JobRecord
    cfg: TrainingConfig
    checkpoint: Path
    checkpoint_rel: str


def _store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


def _cfg_output_name(cfg: dict[str, Any]) -> str | None:
    output = cfg.get("output") if isinstance(cfg, dict) else None
    if isinstance(output, dict) and isinstance(output.get("name"), str):
        return output["name"]
    return None


def _cfg_backend(cfg: dict[str, Any]) -> str | None:
    backend = cfg.get("backend") if isinstance(cfg, dict) else None
    if isinstance(backend, dict) and isinstance(backend.get("type"), str):
        return backend["type"]
    return None


def _cfg_base_model(cfg: dict[str, Any]) -> dict[str, Any]:
    base = cfg.get("baseModel") or cfg.get("base_model")
    return base if isinstance(base, dict) else {}


def _resolve_model(job_id: str, checkpoint_path: str) -> _ResolvedModel:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.workspace.is_dir():
        raise HTTPException(status_code=404, detail="workspace missing on disk")
    try:
        cfg = TrainingConfig.model_validate(job.config_snapshot or {})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"job config is invalid: {exc}") from exc
    try:
        checkpoint = _resolve_workspace_file(job.workspace, checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not checkpoint.is_file():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    if checkpoint.suffix.lower() not in {".safetensors", ".sft"}:
        raise HTTPException(status_code=422, detail="checkpoint must be .safetensors or .sft")
    rel = checkpoint.relative_to(job.workspace.resolve()).as_posix()
    allowed = {
        str(entry.get("path"))
        for entry in _list_workspace_files(job.workspace).get("checkpoints", [])
        if entry.get("path")
    }
    if rel not in allowed:
        raise HTTPException(status_code=422, detail="checkpoint is not a LoRA artifact")
    return _ResolvedModel(job=job, cfg=cfg, checkpoint=checkpoint, checkpoint_rel=rel)


@router.get("/lora-test/models")
def list_lora_test_models() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in state.registry.list():
        if not job.workspace.is_dir():
            continue
        buckets = _list_workspace_files(job.workspace)
        checkpoints = buckets.get("checkpoints", [])
        if not checkpoints:
            continue
        cfg = job.config_snapshot or {}
        if not isinstance(cfg, dict):
            cfg = {}
        rows.append(
            {
                "job_id": job.id,
                "output_name": _cfg_output_name(cfg),
                "workspace": str(job.workspace),
                "state": job.state.value,
                "backend": _cfg_backend(cfg),
                "base_model": _cfg_base_model(cfg),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "checkpoints": checkpoints,
            }
        )
    rows.sort(key=lambda r: r.get("finished_at") or r.get("created_at") or "", reverse=True)
    return {"jobs": rows}


@router.post("/lora-test/generate")
def start_generation(req: GenerateRequest) -> dict[str, Any]:
    try:
        req.validated_sampler
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    resolved = _resolve_model(req.job_id, req.checkpoint_path)
    if resolved.cfg.backend.type != "anima_lora":
        raise HTTPException(status_code=422, detail="only anima_lora LoRA generation is supported")

    session = _store().create(
        kind=_KIND,
        title=f"LoRA test {resolved.checkpoint.name}",
        metadata={
            "job_id": resolved.job.id,
            "checkpoint_path": resolved.checkpoint_rel,
            "checkpoint_name": resolved.checkpoint.name,
            "backend": resolved.cfg.backend.type,
            "request": req.model_dump(),
        },
    )
    cancel_evt = threading.Event()
    with _cancel_lock:
        _cancel_events[session.id] = cancel_evt
    thread = threading.Thread(
        target=_run_generation_session,
        args=(session.id, req, cancel_evt),
        daemon=True,
        name=f"lora-test-{session.id[-6:]}",
    )
    thread.start()
    return {"session_id": session.id}


@router.get("/lora-test/sessions/{session_id}")
def get_generation_session(session_id: str) -> dict[str, Any]:
    session = _store().get(session_id)
    if session is None or session.kind != _KIND:
        raise HTTPException(status_code=404, detail="lora test session not found")
    return session.to_dict()


@router.post("/lora-test/sessions/{session_id}/cancel")
def cancel_generation_session(session_id: str) -> dict[str, Any]:
    session = _store().get(session_id)
    if session is None or session.kind != _KIND:
        raise HTTPException(status_code=404, detail="lora test session not found")
    with _cancel_lock:
        evt = _cancel_events.get(session_id)
    if evt is not None:
        evt.set()
    if session.status in {"queued", "running"}:
        _store().update(session_id, status="canceled", error="canceled by user", finished=True)
        _store().append_event(
            session_id,
            TaskEvent(level="warn", message="canceled by user", percent=session.percent),
        )
    return {"canceled": True}


@router.get("/lora-test/results/{session_id}/file")
def get_result_file(session_id: str, path: str) -> FileResponse:
    root = _session_output_dir(session_id)
    try:
        target = (root / path).resolve()
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes result directory") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    disposition = "inline" if target.suffix.lower() in _RESULT_IMAGE_SUFFIXES else "attachment"
    media_type = "image/png" if target.suffix.lower() == ".png" else "application/octet-stream"
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        content_disposition_type=disposition,
    )


def _session_output_dir(session_id: str) -> Path:
    return (runs_dir() / "lora-test" / session_id).resolve()


def _run_generation_session(
    session_id: str,
    req: GenerateRequest,
    cancel_evt: threading.Event,
) -> None:
    store = _store()
    results: list[dict[str, Any]] = []
    try:
        resolved = _resolve_model(req.job_id, req.checkpoint_path)
        out_dir = _session_output_dir(session_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        store.update(session_id, status="running", percent=1)
        store.append_event(session_id, TaskEvent(level="info", message="loading model", percent=1))
        for index in range(req.batch_count):
            if cancel_evt.is_set():
                store.update(
                    session_id,
                    status="canceled",
                    percent=100,
                    result={"images": results, "output_dir": str(out_dir)},
                    error="canceled by user",
                    finished=True,
                )
                return
            seed = random.randrange(0, 2**31 - 1) if req.seed < 0 else req.seed + index
            image_path = out_dir / f"{index + 1:03d}_{seed}.{req.output_format}"
            percent = 5 + (index / max(req.batch_count, 1)) * 90
            store.append_event(
                session_id,
                TaskEvent(
                    level="info",
                    message=f"generating {index + 1}/{req.batch_count}",
                    percent=percent,
                    payload={"seed": seed},
                ),
            )
            _run_anima_inference(resolved, req, image_path, seed, cancel_evt)
            sidecar = image_path.with_suffix(".json")
            meta = {
                "path": image_path.name,
                "seed": seed,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "cfg": req.cfg,
                "sampler": req.sampler,
                "lora_weight": req.lora_weight,
                "checkpoint_path": resolved.checkpoint_rel,
                "job_id": resolved.job.id,
            }
            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(meta)
        store.update(
            session_id,
            status="succeeded",
            percent=100,
            result={"images": results, "output_dir": str(out_dir)},
            finished=True,
        )
        store.append_event(session_id, TaskEvent(level="info", message="generation complete", percent=100))
    except Exception as exc:  # noqa: BLE001
        if cancel_evt.is_set():
            store.update(
                session_id,
                status="canceled",
                error="canceled by user",
                result={"images": results},
                finished=True,
            )
            return
        store.update(
            session_id,
            status="failed",
            error=str(exc),
            result={"images": results},
            finished=True,
        )
        store.append_event(session_id, TaskEvent(level="error", message=str(exc)))
    finally:
        with _cancel_lock:
            _cancel_events.pop(session_id, None)


def _run_anima_inference(
    resolved: _ResolvedModel,
    req: GenerateRequest,
    out_path: Path,
    seed: int,
    cancel_evt: threading.Event,
) -> None:
    cfg = resolved.cfg
    bm = cfg.base_model
    paths = bm.arch_paths
    if bm.arch != "anima":
        raise RuntimeError(f"anima_lora generation requires base_model.arch='anima', got {bm.arch!r}")
    if paths.ae is None or paths.qwen3 is None:
        raise RuntimeError("anima_lora generation requires baseModel.archPaths.ae and qwen3")
    env = anima_bootstrap.resolve(
        config_path=cfg.backend.repo_path,
        config_python=cfg.backend.python_executable,
    )
    argv = [
        str(env.python_executable),
        str(env.script("inference.py")),
        "--dit",
        str(bm.checkpoint),
        "--vae",
        str(paths.ae),
        "--text_encoder",
        str(paths.qwen3),
        "--lora_weight",
        str(resolved.checkpoint),
        "--lora_multiplier",
        repr(float(req.lora_weight)),
        "--prompt",
        req.prompt,
        "--image_size",
        str(req.height),
        str(req.width),
        "--infer_steps",
        str(req.steps),
        "--guidance_scale",
        repr(float(req.cfg)),
        "--sampler",
        req.validated_sampler,
        "--save_path",
        str(out_path),
        "--seed",
        str(seed),
    ]
    if req.negative_prompt.strip():
        argv += ["--negative_prompt", req.negative_prompt.strip()]
    proc = subprocess.Popen(
        argv,
        cwd=env.repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while proc.poll() is None:
        if cancel_evt.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError("canceled by user")
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        tail = "\n".join((stderr or stdout or "").splitlines()[-20:])
        raise RuntimeError(f"anima_lora inference failed ({proc.returncode}): {tail}")
    if not out_path.is_file():
        raise RuntimeError("anima_lora inference finished but did not write an image")
