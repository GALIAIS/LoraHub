"""Seed realistic local demo jobs for UI/mobile development.

The script writes ignored runtime data under ``runs/`` and upserts the
corresponding rows into ``runs/jobs.sqlite``. It is intentionally safe to run
multiple times: demo job IDs are stable and their workspaces are replaced.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from lorahub.api.paths import runs_dir
from lorahub.api.state import JobRecord, JobState
from lorahub.api.store import JobStore, default_store_path
from lorahub.core.events import EventType, TrainingEvent


DEMO_PREFIX = "demo-mobile-"


def _config(name: str, dataset: Path, output: Path, *, backend: str) -> dict[str, Any]:
    return {
        "name": name,
        "baseModel": {
            "arch": "sdxl",
            "checkpoint": str(runs_dir() / "demo-assets" / "base-model.safetensors"),
        },
        "dataset": {"source": str(dataset), "captionExtension": ".txt"},
        "schedule": {
            "epochs": 6,
            "batchSize": 2,
            "gradientAccumulationSteps": 2,
            "learningRate": 0.00012,
            "maxTrainSteps": 240,
        },
        "network": {
            "rank": 16,
            "alpha": 8,
            "networkDropout": 0.05,
            "algo": "tlora" if backend == "anima_lora" else "lora",
        },
        "optimizer": {"type": "AdamW8bit"},
        "sampling": {
            "enabled": True,
            "triggerWord": "demochar",
            "prompts": [
                "demochar portrait, studio light",
                "demochar full body, cinematic street",
            ],
        },
        "output": {"name": name, "outputDir": str(output)},
        "monitoring": {"enableWandb": backend == "diffusion-pipe"},
        "backend": {"type": backend},
    }


def _write_png(path: Path, *, label: str, hue: int, size: tuple[int, int] = (768, 768)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = Image.new("RGB", size, (22, 25, 31))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = (hue + x // 8 + y // 16) % 255
            g = (80 + x // 14 + hue // 4) % 255
            b = (150 + y // 10 + hue // 2) % 255
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle((36, h - 172, w - 36, h - 36), radius=18, fill=(0, 0, 0, 130))
    font = ImageFont.load_default()
    draw.text((58, h - 140), label, fill=(255, 255, 255, 235), font=font)
    draw.text((58, h - 110), "LoRaHub mobile demo sample", fill=(255, 255, 255, 180), font=font)
    img.save(path)


def _write_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 9):
        sample = root / f"demochar_{i:02d}.png"
        _write_png(sample, label=f"dataset image {i:02d}", hue=18 * i, size=(512, 512))
        sample.with_suffix(".txt").write_text(
            f"demochar, training sample {i:02d}, clean caption\n",
            encoding="utf-8",
        )


def _event(
    event_type: EventType,
    payload: dict[str, Any],
    *,
    ts: float,
    job_id: str,
) -> TrainingEvent:
    return TrainingEvent(type=event_type, payload=payload, timestamp=ts, job_id=job_id)


def _write_events(
    workspace: Path,
    *,
    job_id: str,
    total_steps: int,
    final_step: int,
    base_ts: float,
    overfit: bool = False,
    failed: bool = False,
) -> None:
    events: list[TrainingEvent] = [
        _event(
            EventType.log,
            {"level": "info", "message": "resolved config and dataset manifest"},
            ts=base_ts,
            job_id=job_id,
        ),
        _event(
            EventType.cache_progress,
            {"phase": "latents", "done": 8, "total": 8},
            ts=base_ts + 2,
            job_id=job_id,
        ),
    ]

    checkpoints_dir = workspace / "checkpoints"
    samples_dir = workspace / "samples"
    step_values = list(range(10, final_step + 1, 10))
    if final_step not in step_values:
        step_values.append(final_step)
    for step in step_values:
        epoch = max(1, math.ceil(step / (total_steps / 6)))
        decay = step / total_steps
        train_loss = max(0.065, 0.92 * math.exp(-2.9 * decay) + 0.045 * math.sin(step / 13))
        if failed and step > final_step - 20:
            train_loss += 0.18
        ts = base_ts + step * 5
        events.append(
            _event(
                EventType.step,
                {
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "loss": round(train_loss, 5),
                    "lr": round(0.00012 * max(0.12, 1 - decay), 8),
                    "iter_time_s": round(1.55 + 0.2 * math.sin(step / 17), 3),
                    "samples_per_sec": round(1.25 + 0.15 * math.cos(step / 19), 3),
                },
                ts=ts,
                job_id=job_id,
            )
        )
        if step % 30 == 0:
            events.append(
                _event(
                    EventType.gpu_sample,
                    {
                        "gpu_index": 0,
                        "util_percent": min(99, 68 + step % 29),
                        "vram_used_mib": 13200 + step * 9,
                        "vram_total_mib": 24564,
                        "temperature_c": 61 + step % 12,
                    },
                    ts=ts + 0.5,
                    job_id=job_id,
                )
            )
        if step % 40 == 0:
            val_loss = 0.38 + (0.08 * (step / total_steps) if overfit else -0.16 * (step / total_steps))
            events.append(
                _event(
                    EventType.validation,
                    {"epoch": epoch, "step": step, "val_loss": round(val_loss, 5)},
                    ts=ts + 1,
                    job_id=job_id,
                )
            )
        if step % 60 == 0:
            events.append(
                _event(
                    EventType.epoch_end,
                    {"epoch": epoch, "total_epochs": 6},
                    ts=ts + 2,
                    job_id=job_id,
                )
            )
            ckpt = checkpoints_dir / f"demo-lora-step{step:06d}.safetensors"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes((f"demo checkpoint {step}\n").encode("utf-8") + b"\0" * 256)
            rel_ckpt = ckpt.relative_to(workspace).as_posix()
            events.append(
                _event(
                    EventType.checkpoint_saved,
                    {"path": rel_ckpt, "step": step},
                    ts=ts + 3,
                    job_id=job_id,
                )
            )
            sample = samples_dir / f"demochar_step{step:06d}_seed42.png"
            _write_png(sample, label=f"checkpoint step {step}", hue=step % 255)
            rel_sample = sample.relative_to(workspace).as_posix()
            events.append(
                _event(
                    EventType.sample_ready,
                    {"path": rel_sample, "step": step, "prompt": "demochar portrait"},
                    ts=ts + 4,
                    job_id=job_id,
                )
            )
            events.append(
                _event(
                    EventType.lora_spectrum,
                    {
                        "checkpoint": rel_ckpt,
                        "step": step,
                        "layers": 128,
                        "effective_rank": round(7.2 + step / 80, 3),
                        "top1_energy": round(0.41 - min(step / 1600, 0.12), 3),
                        "fro_norm": round(1.35 + step / 300, 3),
                    },
                    ts=ts + 5,
                    job_id=job_id,
                )
            )

    grid = workspace / "samples" / "grids" / "comparison_grid.png"
    _write_png(grid, label="derived grid ignored by regular sample metrics", hue=210, size=(1024, 768))
    events.append(
        _event(
            EventType.sample_ready,
            {
                "path": grid.relative_to(workspace).as_posix(),
                "step": final_step,
                "kind": "grid",
            },
            ts=base_ts + final_step * 5 + 7,
            job_id=job_id,
        )
    )

    if failed:
        events.append(
            _event(
                EventType.diagnostic_warning,
                {
                    "category": "cuda_oom",
                    "severity": "error",
                    "message": "CUDA out of memory while allocating attention block",
                    "remediation": "Lower batch size or enable gradient checkpointing.",
                    "evidence": "RuntimeError: CUDA out of memory",
                    "source": "stderr",
                },
                ts=base_ts + final_step * 5 + 9,
                job_id=job_id,
            )
        )
        events.append(
            _event(
                EventType.error,
                {"message": "training failed with CUDA OOM", "returncode": 1},
                ts=base_ts + final_step * 5 + 10,
                job_id=job_id,
            )
        )
    else:
        events.append(
            _event(
                EventType.done,
                {"returncode": 0},
                ts=base_ts + final_step * 5 + 10,
                job_id=job_id,
            )
        )

    (workspace / "events.jsonl").write_text(
        "\n".join(event.to_json() for event in events) + "\n",
        encoding="utf-8",
    )


def _job_record(
    *,
    job_id: str,
    workspace: Path,
    config: dict[str, Any],
    state: JobState,
    created_at: datetime,
    started_at: datetime | None,
    finished_at: datetime | None,
    returncode: int | None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JobRecord:
    return JobRecord(
        id=job_id,
        state=state,
        workspace=workspace,
        config_snapshot=config,
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        returncode=returncode,
        error=error,
        pid=None,
        metadata=metadata,
    )


def seed_demo_jobs(*, clear: bool = False) -> list[str]:
    runs = runs_dir()
    assets = runs / "demo-assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "base-model.safetensors").write_bytes(b"demo base model placeholder\n")

    store = JobStore(default_store_path())

    if clear:
        for path in runs.glob(f"{DEMO_PREFIX}*"):
            if path.is_dir():
                shutil.rmtree(path)
        for record in store.list():
            if record.id.startswith(DEMO_PREFIX):
                store.delete(record.id)

    dataset = runs / "demo-mobile-dataset"
    _write_dataset(dataset)

    now = datetime.now(UTC).replace(microsecond=0)
    specs = [
        {
            "suffix": "tlora-succeeded",
            "name": "demo-tlora-character",
            "state": JobState.succeeded,
            "backend": "anima_lora",
            "age": timedelta(hours=7),
            "steps": 240,
            "final": 240,
            "overfit": False,
            "returncode": 0,
            "error": None,
        },
        {
            "suffix": "sdxl-overfit",
            "name": "demo-overfit-style",
            "state": JobState.succeeded,
            "backend": "kohya",
            "age": timedelta(hours=18),
            "steps": 220,
            "final": 220,
            "overfit": True,
            "returncode": 0,
            "error": None,
        },
        {
            "suffix": "interrupted",
            "name": "demo-interrupted-preview",
            "state": JobState.interrupted,
            "backend": "diffusion-pipe",
            "age": timedelta(minutes=52),
            "steps": 260,
            "final": 150,
            "overfit": False,
            "returncode": None,
            "error": "interrupted by demo restart",
        },
        {
            "suffix": "failed-oom",
            "name": "demo-failed-oom",
            "state": JobState.failed,
            "backend": "anima_lora",
            "age": timedelta(days=1, hours=3),
            "steps": 180,
            "final": 90,
            "overfit": False,
            "returncode": 1,
            "error": "CUDA out of memory while allocating attention block",
        },
    ]

    ids: list[str] = []
    for index, spec in enumerate(specs):
        job_id = f"{DEMO_PREFIX}{spec['suffix']}"
        ids.append(job_id)
        workspace = runs / job_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        output = workspace / "output"
        output.mkdir(parents=True, exist_ok=True)
        config = _config(str(spec["name"]), dataset, output, backend=str(spec["backend"]))
        (workspace / "config.yaml").write_text(
            f"name: {spec['name']}\nbackend: {spec['backend']}\n",
            encoding="utf-8",
        )
        base_ts = time.time() - (index + 1) * 7200
        _write_events(
            workspace,
            job_id=job_id,
            total_steps=int(spec["steps"]),
            final_step=int(spec["final"]),
            base_ts=base_ts,
            overfit=bool(spec["overfit"]),
            failed=spec["state"] == JobState.failed,
        )
        created = now - spec["age"]
        started = created + timedelta(minutes=3)
        finished = (
            None
            if spec["state"] == JobState.running
            else started + timedelta(seconds=int(spec["final"]) * 5 + 45)
        )
        metadata = {
            "demo": True,
            "axis_values": {"rank": config["network"]["rank"], "lr": config["schedule"]["learningRate"]},
        }
        if spec["backend"] == "diffusion-pipe":
            metadata["wandb_run_url"] = "https://wandb.ai/demo/lorahub/runs/mobile-demo"
        store.upsert(
            _job_record(
                job_id=job_id,
                workspace=workspace,
                config=config,
                state=spec["state"],
                created_at=created,
                started_at=started,
                finished_at=finished,
                returncode=spec["returncode"],
                error=spec["error"],
                metadata=metadata,
            )
        )
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="replace existing demo workspaces")
    args = parser.parse_args()
    ids = seed_demo_jobs(clear=args.clear)
    print("Seeded demo jobs:")
    for job_id in ids:
        print(f"  {job_id}")
    print(f"Store: {default_store_path()}")


if __name__ == "__main__":
    main()
