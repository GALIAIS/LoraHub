"""LoRA testbench endpoints for post-training image generation."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
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
from lorahub.api.dataset_files import is_link_like
from lorahub.api.jobs_helpers import _list_workspace_files, _resolve_workspace_file
from lorahub.api.paths import runs_dir
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSessionStore,
    default_task_store_path,
    persist_stop_request,
)
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
    width: int = Field(default=896, ge=256, le=2048)
    height: int = Field(default=1632, ge=256, le=2048)
    seed: int = -1
    batch_count: int = Field(default=4, ge=1, le=32)
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=4.5, ge=0.0, le=30.0)
    sampler: str = Field(default="euler", max_length=32)
    lora_weight: float = Field(default=1.0, ge=-2.0, le=2.0)
    loras: list["LoraInput"] = Field(default_factory=list, max_length=8)
    x_axis: "AxisInput | None" = None
    y_axis: "AxisInput | None" = None
    output_format: Literal["png"] = "png"

    @property
    def validated_sampler(self) -> str:
        if self.sampler not in {"euler", "er_sde", "lcm"}:
            raise ValueError("sampler must be one of euler, er_sde, lcm")
        return self.sampler


class LoraInput(BaseModel):
    job_id: str | None = None
    checkpoint_path: str
    weight: float = Field(default=1.0, ge=-2.0, le=2.0)


class AxisInput(BaseModel):
    field: Literal[
        "variant",
        "prompt",
        "negative_prompt",
        "seed",
        "lora_weight",
        "cfg",
        "steps",
        "sampler",
        "size",
        "checkpoint",
    ]
    values: list[str] = Field(min_length=1, max_length=16)


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    job: state.JobRecord
    cfg: TrainingConfig
    checkpoint: Path
    checkpoint_rel: str


@dataclass(frozen=True, slots=True)
class _GenerationCase:
    index: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler: str
    loras: list[_ResolvedModel]
    multipliers: list[float]
    x_label: str | None = None
    y_label: str | None = None


GenerateRequest.model_rebuild()


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
        _validate_anima_size(req.width, req.height)
        for axis in (req.x_axis, req.y_axis):
            if axis is not None and axis.field == "size":
                for raw in axis.values:
                    _parse_size_axis_value(raw)
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
    store = _store()
    session = store.get(session_id)
    if session is None or session.kind != _KIND:
        raise HTTPException(status_code=404, detail="lora test session not found")
    if session.status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"generation is {session.status}")
    if not persist_stop_request(store, session_id, percent=session.percent):
        current = store.get(session_id)
        status = current.status if current is not None else "unavailable"
        raise HTTPException(status_code=409, detail=f"generation is {status}")
    with _cancel_lock:
        evt = _cancel_events.get(session_id)
    if evt is not None:
        evt.set()
    store.append_event(
        session_id,
        TaskEvent(
            level="warn",
            message="cancellation requested",
            percent=session.percent,
        ),
    )
    return {"canceled": True}


@router.get("/lora-test/results/{session_id}/file")
def get_result_file(session_id: str, path: str) -> FileResponse:
    session = _store().get(session_id)
    if session is None or session.kind != _KIND:
        raise HTTPException(status_code=404, detail="generation session not found")
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
    if len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
        raise ValueError("invalid generation session id")
    runs = runs_dir().resolve()
    parent = runs / "lora-test"
    if is_link_like(parent):
        raise ValueError("LoRA test output directory cannot be a link")
    parent.mkdir(exist_ok=True)
    target = parent / session_id
    if is_link_like(target):
        raise ValueError("generation output directory cannot be a link")
    resolved = target.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError("generation output escapes the runs directory") from exc
    return resolved


def _run_generation_session(
    session_id: str,
    req: GenerateRequest,
    cancel_evt: threading.Event,
) -> None:
    store = _store()
    results: list[dict[str, Any]] = []
    try:
        if cancel_evt.is_set():
            store.update(
                session_id,
                status="canceled",
                error="canceled by user",
                result={"images": results},
                finished=True,
            )
            return
        resolved = _resolve_model(req.job_id, req.checkpoint_path)
        loras, weights = _resolve_loras(req, resolved)
        cases = _build_cases(req, resolved, loras, weights)
        if cancel_evt.is_set():
            store.update(
                session_id,
                status="canceled",
                error="canceled by user",
                result={"images": results},
                finished=True,
            )
            return
        out_dir = _session_output_dir(session_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        store.update(session_id, status="running", percent=1)
        store.append_event(session_id, TaskEvent(level="info", message="loading model", percent=1))
        for case in cases:
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
            image_path = out_dir / f"{case.index + 1:03d}_{case.seed}.{req.output_format}"
            percent = 5 + (case.index / max(len(cases), 1)) * 90
            store.append_event(
                session_id,
                TaskEvent(
                    level="info",
                    message=f"generating {case.index + 1}/{len(cases)}",
                    percent=percent,
                    payload={"seed": case.seed, "x": case.x_label, "y": case.y_label},
                ),
            )
            _run_anima_inference(resolved, req, case, image_path, cancel_evt)
            sidecar = image_path.with_suffix(".json")
            meta = {
                "path": image_path.name,
                "seed": case.seed,
                "prompt": case.prompt,
                "negative_prompt": case.negative_prompt,
                "width": case.width,
                "height": case.height,
                "steps": case.steps,
                "cfg": case.cfg,
                "sampler": case.sampler,
                "lora_weight": case.multipliers[0] if case.multipliers else 0.0,
                "loras": [
                    {
                        "job_id": lora.job.id,
                        "checkpoint_path": lora.checkpoint_rel,
                        "checkpoint_name": lora.checkpoint.name,
                        "weight": case.multipliers[i],
                    }
                    for i, lora in enumerate(case.loras)
                ],
                "checkpoint_path": case.loras[0].checkpoint_rel if case.loras else "",
                "job_id": resolved.job.id,
                "x_label": case.x_label,
                "y_label": case.y_label,
            }
            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(meta)
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
        grid_path = _maybe_write_xy_grid(out_dir, results, req)
        store.update(
            session_id,
            status="succeeded",
            percent=100,
            result={
                "images": results,
                "grid": grid_path.name if grid_path is not None else None,
                "output_dir": str(out_dir),
            },
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


def _resolve_loras(
    req: GenerateRequest,
    primary: _ResolvedModel,
) -> tuple[list[_ResolvedModel], list[float]]:
    if not req.loras:
        return [primary], [req.lora_weight]
    resolved: list[_ResolvedModel] = []
    weights: list[float] = []
    seen: set[str] = set()
    for item in req.loras:
        model = _resolve_model(item.job_id or req.job_id, item.checkpoint_path)
        key = str(model.checkpoint.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(model)
        weights.append(item.weight)
    return resolved, weights


def _build_cases(
    req: GenerateRequest,
    primary: _ResolvedModel,
    loras: list[_ResolvedModel],
    lora_weights: list[float],
) -> list[_GenerationCase]:
    base_weights = list(lora_weights)
    x_values = _axis_values(req.x_axis) if req.x_axis else [(None, None)]
    y_values = _axis_values(req.y_axis) if req.y_axis else [(None, None)]
    cases: list[_GenerationCase] = []
    grid_mode = req.x_axis is not None or req.y_axis is not None
    total = len(x_values) * len(y_values) if grid_mode else req.batch_count
    for i in range(total):
        x_label, x_override = x_values[i % len(x_values)] if grid_mode else (None, {})
        y_label, y_override = y_values[i // len(x_values)] if grid_mode else (None, {})
        override = {**(x_override or {}), **(y_override or {})}
        case_loras = loras
        weights = list(base_weights)
        prompt = req.prompt
        negative_prompt = req.negative_prompt
        seed_override: int | None = None
        width = req.width
        height = req.height
        variant = override.get("variant")
        if variant not in (None, "base", "lora"):
            raise ValueError("variant axis values must be base or lora")
        if "checkpoint" in override and variant != "base":
            case_loras = [_resolve_model(req.job_id, str(override["checkpoint"]))]
            weights = [req.lora_weight]
        if "lora_weight" in override and weights:
            weights[0] = float(override["lora_weight"])
        if "size" in override:
            width, height = override["size"]
        if "prompt" in override:
            prompt = str(override["prompt"])
        if "negative_prompt" in override:
            negative_prompt = str(override["negative_prompt"])
        if "seed" in override:
            seed_override = int(override["seed"])
        if variant == "base":
            case_loras = []
            weights = []
        seed = (
            seed_override
            if seed_override is not None
            else random.randrange(0, 2**31 - 1) if req.seed < 0 else req.seed + i
        )
        cases.append(
            _GenerationCase(
                index=i,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                steps=int(override.get("steps", req.steps)),
                cfg=float(override.get("cfg", req.cfg)),
                sampler=str(override.get("sampler", req.sampler)),
                loras=case_loras,
                multipliers=weights,
                x_label=x_label,
                y_label=y_label,
            )
        )
    return cases


def _axis_values(axis: AxisInput) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for raw in axis.values:
        value = raw.strip()
        if not value:
            continue
        if axis.field in {"lora_weight", "cfg"}:
            parsed: Any = float(value)
        elif axis.field in {"steps", "seed"}:
            parsed = int(value)
        elif axis.field == "sampler":
            if value not in {"euler", "er_sde", "lcm"}:
                raise ValueError("sampler axis values must be euler, er_sde or lcm")
            parsed = value
        elif axis.field == "size":
            parsed = _parse_size_axis_value(value)
        elif axis.field == "variant":
            parsed = value.lower()
        elif axis.field == "negative_prompt" and value.lower() in {
            "empty",
            "none",
            "__empty__",
            "空",
            "无",
        }:
            parsed = ""
            value = "empty"
        else:
            parsed = value
        out.append((f"{axis.field}={value}", {axis.field: parsed}))
    if not out:
        raise ValueError("axis values cannot be empty")
    return out


def _parse_size_axis_value(value: str) -> tuple[int, int]:
    raw = value.lower().replace(" ", "")
    for sep in ("x", "*", "×"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            break
    else:
        raise ValueError("size axis values must look like 912x1632")
    width = int(left)
    height = int(right)
    _validate_anima_size(width, height)
    return width, height


def _validate_anima_size(width: int, height: int) -> None:
    if width < 256 or width > 2048 or height < 256 or height > 2048:
        raise ValueError("size axis width and height must be between 256 and 2048")
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError("anima_lora generation width and height must be divisible by 32")


def _maybe_write_xy_grid(
    out_dir: Path,
    results: list[dict[str, Any]],
    req: GenerateRequest,
) -> Path | None:
    if req.x_axis is None and req.y_axis is None:
        return None
    from PIL import Image, ImageDraw

    paths = [out_dir / str(item["path"]) for item in results]
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return None
    cols = len([v for v in req.x_axis.values if v.strip()]) if req.x_axis else 1
    rows = len([v for v in req.y_axis.values if v.strip()]) if req.y_axis else 1
    thumb_w, thumb_h = images[0].size
    label_h = 28
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        x = idx % cols
        y = idx // cols
        left = x * thumb_w
        top = y * (thumb_h + label_h)
        canvas.paste(image, (left, top + label_h))
        label = " / ".join(
            part for part in (results[idx].get("x_label"), results[idx].get("y_label")) if part
        )
        draw.text((left + 8, top + 7), label or str(idx + 1), fill=(20, 20, 20))
    target = out_dir / "xy_grid.png"
    canvas.save(target)
    return target


def _run_anima_inference(
    resolved: _ResolvedModel,
    req: GenerateRequest,
    case: _GenerationCase,
    out_path: Path,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inference_dir = Path(
        tempfile.mkdtemp(dir=out_path.parent, prefix=".lorahub-lora-test-")
    )
    log_path = out_path.with_suffix(".log")
    try:
        argv = [
            str(env.python_executable),
            str(env.script("inference.py")),
            "--dit",
            str(bm.checkpoint),
            "--vae",
            str(paths.ae),
            "--text_encoder",
            str(paths.qwen3),
            "--prompt",
            case.prompt,
            "--image_size",
            str(case.height),
            str(case.width),
            "--infer_steps",
            str(case.steps),
            "--guidance_scale",
            repr(float(case.cfg)),
            "--sampler",
            case.sampler,
            "--save_path",
            str(inference_dir),
            "--seed",
            str(case.seed),
        ]
        if case.loras:
            argv += [
                "--lora_weight",
                *[str(lora.checkpoint) for lora in case.loras],
                "--lora_multiplier",
                *[repr(float(weight)) for weight in case.multipliers],
            ]
        if case.negative_prompt.strip():
            argv += ["--negative_prompt", case.negative_prompt.strip()]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                argv,
                cwd=env.repo_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0,
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
        if proc.returncode != 0:
            tail = _tail_text(log_path, lines=20)
            raise RuntimeError(f"anima_lora inference failed ({proc.returncode}): {tail}")
        generated = sorted(
            inference_dir.glob("*.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if generated:
            generated[0].replace(out_path)
        if not out_path.is_file():
            tail = _tail_text(log_path, lines=20)
            raise RuntimeError(
                f"anima_lora inference finished but did not write an image: {tail}"
            )
    finally:
        shutil.rmtree(inference_dir, ignore_errors=True)


def _tail_text(path: Path, *, lines: int) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
