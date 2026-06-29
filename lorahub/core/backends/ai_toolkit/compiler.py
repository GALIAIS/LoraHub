"""Compile LoraHub TrainingConfig into an ostris/ai-toolkit YAML job."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lorahub.core.config.schema import TrainingConfig

__all__ = ["CompilationError", "compile_config"]


class CompilationError(ValueError):
    """Raised when a config cannot be expressed as an ai-toolkit job."""


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    if cfg.backend.type != "ai_toolkit":
        raise CompilationError(f"ai_toolkit compiler got backend.type={cfg.backend.type!r}")
    if cfg.base_model.arch != "krea2":
        raise CompilationError("ai_toolkit currently supports arch='krea2' in LoraHub")

    workspace = workspace.resolve()
    config_path = workspace / "_lorahub_ai_toolkit.yaml"
    job = _build_job(cfg, workspace)
    _overlay_extra_args(job, cfg.backend.extra_args)
    return [str(config_path)], {
        config_path: yaml.safe_dump(job, sort_keys=False, allow_unicode=True)
    }


def _build_job(cfg: TrainingConfig, workspace: Path) -> dict[str, Any]:
    output_name = cfg.output.name or "lorahub_ai_toolkit"
    model_name = str(cfg.base_model.checkpoint) if str(cfg.base_model.checkpoint) else "krea/Krea-2-Raw"
    if model_name in {".", ""}:
        model_name = "krea/Krea-2-Raw"

    train_steps = cfg.schedule.max_steps or max(1, int(cfg.schedule.epochs) * 100)
    width, height = _resolution_pair(cfg.sampling.resolution or cfg.dataset.resolution)
    dataset_width, dataset_height = _resolution_pair(cfg.dataset.resolution)

    process: dict[str, Any] = {
        "type": "diffusion_trainer",
        "training_folder": str(workspace / "ai_toolkit_output"),
        "device": "cuda",
        "network": {
            "type": "lora",
            "linear": int(cfg.network.rank),
            "linear_alpha": int(cfg.network.alpha),
        },
        "save": {
            "dtype": _dtype(cfg.output.save_dtype),
            "save_every": int(cfg.output.save_every_n_steps or train_steps),
            "max_step_saves_to_keep": int(cfg.output.save_last_n_steps or 4),
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": str(cfg.dataset.source),
                "caption_ext": cfg.dataset.caption.ext.lstrip("."),
                "caption_dropout_rate": float(cfg.dataset.caption.drop_rate),
                "resolution": [int(dataset_width), int(dataset_height)],
                "num_repeats": int(cfg.dataset.num_repeats),
                "cache_latents_to_disk": bool(cfg.cache_latents_to_disk),
            }
        ],
        "train": {
            "batch_size": int(cfg.schedule.batch_size),
            "steps": int(train_steps),
            "gradient_accumulation": int(cfg.schedule.grad_accum),
            "train_unet": bool(cfg.network.target_unet),
            "train_text_encoder": bool(cfg.network.target_text_encoder),
            "gradient_checkpointing": bool(cfg.gradient_checkpointing),
            "disable_sampling": not bool(cfg.sampling.enabled),
            "noise_scheduler": "flowmatch",
            "optimizer": cfg.optimizer.type,
            "lr": float(cfg.optimizer.lr.unet),
            "dtype": _dtype(cfg.precision),
        },
        "model": {
            "name_or_path": model_name,
            "arch": "krea2",
            "quantize": True,
            "qtype": "qfloat8",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": False,
            "compile": bool(cfg.optimization.torch_compile),
        },
        "sample": {
            "sample_every": int(cfg.sampling.every_n_steps or train_steps),
            "width": int(width),
            "height": int(height),
            "seed": int(cfg.sampling.seed),
            "sample_steps": int(cfg.sampling.inference_steps),
            "guidance_scale": float(cfg.sampling.inference_cfg),
            "prompts": [p.prompt for p in cfg.sampling.prompts] or ["a high quality image"],
        },
        "logging": {"log_every": 1, "use_ui_logger": False},
    }
    if cfg.schedule.seed is not None:
        process["train"]["seed"] = int(cfg.schedule.seed)
    return {
        "job": "extension",
        "config": {"name": output_name, "process": [process]},
        "meta": {"name": output_name, "version": "1.0"},
    }


def _resolution_pair(value: list[int] | tuple[int, ...]) -> tuple[int, int]:
    if len(value) == 1:
        return int(value[0]), int(value[0])
    return int(value[0]), int(value[1])


def _dtype(value: str) -> str:
    if value in {"bf16", "bfloat16"}:
        return "bf16"
    if value in {"fp32", "float", "float32"}:
        return "float32"
    return "fp16"


def _overlay_extra_args(job: dict[str, Any], extra: dict[str, Any]) -> None:
    for raw_key, value in extra.items():
        if value is None or value is False:
            continue
        key = raw_key.lstrip("-")
        parts = [p for p in key.split(".") if p]
        if not parts:
            continue
        cursor: Any = job["config"]["process"][0]
        if parts[0] in {"job", "config", "meta"}:
            cursor = job
        for part in parts[:-1]:
            if not isinstance(cursor, dict):
                break
            cursor = cursor.setdefault(part, {})
        else:
            if isinstance(cursor, dict):
                cursor[parts[-1]] = _coerce(value)


def _coerce(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in lowered or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value
