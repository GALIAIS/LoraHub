"""Compile LoraHub TrainingConfig into an ostris/ai-toolkit YAML job."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from lorahub.core.config.backends.ai_toolkit import AiToolkitOptions
from lorahub.core.config.schema import TrainingConfig

__all__ = ["CompilationError", "compile_config"]


class CompilationError(ValueError):
    """Raised when a config cannot be expressed as an ai-toolkit job."""


_KREA2_NETWORK_TYPES = {"lora", "dora", "loha", "lokr", "lorm"}


def _require_training_dataset(cfg: TrainingConfig) -> None:
    """Reject incomplete form state before it turns into a YAML path string."""
    if cfg.dataset.subsets:
        missing = [
            str(index + 1)
            for index, subset in enumerate(cfg.dataset.subsets)
            if subset.path is None
        ]
        if missing:
            raise CompilationError(
                "ai_toolkit requires dataset.subsets[].path for every active subset "
                f"(missing: {', '.join(missing)})"
            )
        return
    if cfg.dataset.source is None:
        raise CompilationError(
            "ai_toolkit requires dataset.source when no dataset subsets are configured"
        )


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    if cfg.backend.type != "ai_toolkit":
        raise CompilationError(f"ai_toolkit compiler got backend.type={cfg.backend.type!r}")
    if cfg.base_model.arch != "krea2":
        raise CompilationError("ai_toolkit currently supports arch='krea2' in LoraHub")
    _require_training_dataset(cfg)

    workspace = workspace.resolve()
    config_path = workspace / "_lorahub_ai_toolkit.yaml"
    job = _build_job(cfg, workspace)
    _overlay_extra_args(job, cfg.backend.extra_args)
    _sanitize_job(job)
    return [str(config_path)], {
        config_path: yaml.safe_dump(job, sort_keys=False, allow_unicode=True)
    }


def _build_job(cfg: TrainingConfig, workspace: Path) -> dict[str, Any]:
    options = cfg.backend.ai_toolkit or AiToolkitOptions()
    output_name = cfg.output.name or "lorahub_ai_toolkit"
    model_name = str(cfg.base_model.checkpoint) if str(cfg.base_model.checkpoint) else "krea/Krea-2-Raw"
    if model_name in {".", ""}:
        model_name = "krea/Krea-2-Raw"

    width, height = _resolution_pair(cfg.sampling.resolution or cfg.dataset.resolution)

    sample = _sample_section(cfg, options, width=width, height=height)
    process: dict[str, Any] = {
        "type": "diffusion_trainer",
        "training_folder": str(workspace / "ai_toolkit_output"),
        "device": "cuda",
        "network": _network_section(cfg, options),
        "save": _save_section(cfg, options),
        "datasets": _dataset_sections(cfg, options),
        "train": _train_section(cfg, options),
        "model": _model_section(cfg, options, model_name=model_name),
        "sample": sample,
        "logging": {
            "log_every": int(options.logging.log_every),
            "verbose": bool(options.logging.verbose),
            "use_wandb": bool(options.logging.use_wandb),
            "use_ui_logger": False,
            "project_name": options.logging.project_name,
            "run_name": options.logging.run_name,
        },
    }
    if cfg.schedule.seed is not None:
        process["train"]["seed"] = int(cfg.schedule.seed)
    if options.dataset.trigger_word:
        process["trigger_word"] = options.dataset.trigger_word
    return {
        "job": "extension",
        "config": {"name": output_name, "process": [process]},
        "meta": {"name": output_name, "version": "1.0"},
    }


def _sample_section(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "sampler": "flowmatch",
        "sample_every": (
            int(cfg.sampling.every_n_steps)
            if cfg.sampling.every_n_steps is not None
            else None
        ),
        "sample_every_n_epochs": (
            int(cfg.sampling.every_n_epochs)
            if cfg.sampling.every_n_epochs is not None
            else None
        ),
        "width": int(width),
        "height": int(height),
        "neg": "",
        "seed": int(cfg.sampling.seed),
        "sample_steps": int(cfg.sampling.inference_steps),
        "guidance_scale": float(cfg.sampling.inference_cfg),
        "format": options.sample.format,
        "walk_seed": bool(options.sample.walk_seed),
        "network_multiplier": float(options.sample.network_multiplier),
    }
    prompt_rows = cfg.sampling.prompts
    if not prompt_rows:
        sample["prompts"] = ["a high quality image"]
        return sample

    samples: list[dict[str, Any]] = []
    for row in prompt_rows:
        item: dict[str, Any] = {"prompt": row.prompt}
        if row.negative:
            item["neg"] = row.negative
        if row.width is not None:
            item["width"] = int(row.width)
        if row.height is not None:
            item["height"] = int(row.height)
        if row.seed is not None:
            item["seed"] = int(row.seed)
        if row.cfg is not None:
            item["guidance_scale"] = float(row.cfg)
        if row.steps is not None:
            item["sample_steps"] = int(row.steps)
        samples.append(item)
    sample["prompts"] = [item["prompt"] for item in samples]
    sample["samples"] = samples
    return sample


def _resolution_pair(value: list[int] | tuple[int, ...]) -> tuple[int, int]:
    if len(value) == 1:
        return int(value[0]), int(value[0])
    return int(value[0]), int(value[1])


def _network_section(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
) -> dict[str, Any]:
    n = cfg.network
    if not n.target_unet:
        raise CompilationError("ai_toolkit Krea2 requires network.target_unet=true")
    if n.target_text_encoder:
        raise CompilationError(
            "ai_toolkit Krea2 does not support text-encoder LoRA training"
        )
    if n.type not in _KREA2_NETWORK_TYPES:
        allowed = ", ".join(sorted(_KREA2_NETWORK_TYPES))
        raise CompilationError(
            f"ai_toolkit krea2 supports network.type in {{{allowed}}}; got {n.type!r}"
        )
    network: dict[str, Any] = {
        "type": n.type,
        "linear": int(n.rank),
        "linear_alpha": int(n.alpha),
    }
    if n.type == "lorm":
        network["lorm"] = {
            "extract_mode": options.network.lorm_extract_mode,
            "extract_mode_param": (
                options.network.lorm_extract_mode_param
                if options.network.lorm_extract_mode_param is not None
                else int(n.rank)
            ),
            "parameter_threshold": options.network.lorm_parameter_threshold,
        }
    if n.network_dropout > 0:
        network["dropout"] = float(n.network_dropout)
    network_kwargs: dict[str, Any] = {}
    if n.rank_dropout > 0:
        network_kwargs["rank_dropout"] = float(n.rank_dropout)
    if n.module_dropout > 0:
        network_kwargs["module_dropout"] = float(n.module_dropout)
    if network_kwargs:
        network["network_kwargs"] = network_kwargs
    if n.init_from is not None:
        network["pretrained_lora_path"] = str(n.init_from)
    if n.type == "lokr":
        network["lokr_factor"] = int(options.network.lokr_factor)
        network["lokr_full_rank"] = bool(options.network.lokr_full_rank)
        network["old_lokr_format"] = bool(options.network.old_lokr_format)
    return network


def _dataset_sections(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
) -> list[dict[str, Any]]:
    dataset_options = options.dataset
    resolutions = dataset_options.resolutions or [
        _legacy_dataset_resolution(cfg.dataset.resolution)
    ]
    resolution: int | list[int]
    resolution = resolutions[0] if len(resolutions) == 1 else list(resolutions)

    shared: dict[str, Any] = {
        "caption_ext": cfg.dataset.caption.ext.lstrip("."),
        "caption_dropout_rate": float(cfg.dataset.caption.drop_rate),
        "resolution": resolution,
        "buckets": bool(dataset_options.buckets),
        "random_crop": bool(dataset_options.random_crop),
        "random_scale": bool(dataset_options.random_scale),
        "scale": float(dataset_options.scale),
        "flip_x": bool(dataset_options.flip_x),
        "flip_y": bool(dataset_options.flip_y),
        "shuffle_tokens": bool(dataset_options.shuffle_tokens),
        "token_dropout_rate": float(dataset_options.token_dropout_rate),
        "keep_tokens": int(dataset_options.keep_tokens),
        "cache_latents": bool(dataset_options.cache_latents),
        "cache_latents_to_disk": bool(cfg.cache_latents_to_disk),
        "cache_text_embeddings": bool(dataset_options.cache_text_embeddings),
        "load_image_when_caching_latents": bool(
            dataset_options.load_image_when_caching_latents
        ),
        "num_workers": int(dataset_options.num_workers),
    }
    if dataset_options.prefetch_factor is not None:
        shared["prefetch_factor"] = int(dataset_options.prefetch_factor)
    if dataset_options.default_caption:
        shared["default_caption"] = dataset_options.default_caption
    if dataset_options.trigger_word:
        shared["trigger_word"] = dataset_options.trigger_word

    datasets: list[dict[str, Any]] = []
    if cfg.dataset.subsets:
        for subset in cfg.dataset.subsets:
            item = {
                **shared,
                "folder_path": str(subset.path),
                "num_repeats": int(subset.num_repeats),
            }
            if subset.mask_path is not None:
                item["mask_path"] = str(subset.mask_path)
            datasets.append(item)
    else:
        datasets.append(
            {
                **shared,
                "folder_path": str(cfg.dataset.source),
                "num_repeats": int(cfg.dataset.num_repeats),
            }
        )

    if cfg.dataset.reg_source is not None:
        datasets.append(
            {
                **shared,
                "folder_path": str(cfg.dataset.reg_source),
                "num_repeats": 1,
                "is_reg": True,
            }
        )
    return datasets


def _legacy_dataset_resolution(value: list[int] | tuple[int, ...]) -> int:
    if len(value) == 1:
        return int(value[0])
    width, height = int(value[0]), int(value[1])
    if width == height:
        return width
    # ai-toolkit accepts a square pixel budget and preserves aspect ratio via buckets.
    return max(64, int(round(math.sqrt(width * height) / 16.0) * 16))


def _train_section(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
) -> dict[str, Any]:
    train_options = options.train
    skip_first_sample = bool(train_options.skip_first_sample)
    if (
        "at_first" in cfg.sampling.model_fields_set
        or "skip_first_sample" not in train_options.model_fields_set
    ):
        skip_first_sample = not bool(cfg.sampling.at_first)
    optimizer_name = cfg.optimizer.type.lower()
    optimizer_params: dict[str, Any] = {}
    if optimizer_name in {
        "adam",
        "adam8bit",
        "adamw",
        "adamw8bit",
        "prodigy",
        "prodigy8bit",
    }:
        optimizer_params["betas"] = [float(value) for value in cfg.optimizer.betas]
    elif optimizer_name in {"automagic", "automagic2", "automagic3"}:
        optimizer_params["beta2"] = float(cfg.optimizer.betas[1])
    optimizer_params["weight_decay"] = float(cfg.optimizer.weight_decay)
    optimizer_params.update(
        {key: _coerce(value) for key, value in cfg.optimizer.optimizer_args.items()}
    )

    scheduler_name: str
    if "lr_scheduler" in train_options.model_fields_set:
        scheduler_name = train_options.lr_scheduler
    elif "schedule" in cfg.optimizer.model_fields_set:
        scheduler_name = cfg.optimizer.schedule
    else:
        scheduler_name = "constant"
    scheduler_params = {
        key: _coerce(value) for key, value in cfg.optimizer.scheduler_args.items()
    }
    if scheduler_name == "constant_with_warmup":
        scheduler_params.setdefault("num_warmup_steps", int(cfg.optimizer.warmup_steps))
    if cfg.schedule.lr_decay_steps is not None:
        scheduler_params.setdefault("total_iters", int(cfg.schedule.lr_decay_steps))
    elif (
        scheduler_name == "cosine_with_restarts"
        and cfg.schedule.max_steps is not None
        and not _uses_epoch_schedule(cfg)
    ):
        scheduler_params.setdefault(
            "total_iters",
            max(1, int(cfg.schedule.max_steps) // int(cfg.optimizer.scheduler_num_cycles)),
        )
    if (
        cfg.optimizer.scheduler_min_lr_ratio is not None
        and scheduler_name in {"cosine", "cosine_with_restarts"}
    ):
        scheduler_params.setdefault(
            "eta_min",
            float(cfg.optimizer.lr.unet) * float(cfg.optimizer.scheduler_min_lr_ratio),
        )

    train: dict[str, Any] = {
        "batch_size": int(cfg.schedule.batch_size),
        "gradient_accumulation": int(cfg.schedule.grad_accum),
        "train_unet": bool(cfg.network.target_unet),
        "train_text_encoder": bool(cfg.network.target_text_encoder),
        "gradient_checkpointing": bool(cfg.gradient_checkpointing),
        "disable_sampling": not bool(cfg.sampling.enabled),
        "noise_scheduler": "flowmatch",
        "optimizer": cfg.optimizer.type,
        "optimizer_params": optimizer_params,
        "lr": float(cfg.optimizer.lr.unet),
        "lr_scheduler": scheduler_name,
        "lr_scheduler_params": scheduler_params,
        "max_grad_norm": float(cfg.optimizer.max_grad_norm),
        "dtype": _dtype(cfg.precision),
        "content_or_style": train_options.content_or_style,
        "timestep_type": train_options.timestep_type,
        "loss_type": train_options.loss_type,
        "min_denoising_steps": int(train_options.min_denoising_steps),
        "max_denoising_steps": int(train_options.max_denoising_steps),
        "min_snr_gamma": train_options.min_snr_gamma,
        "noise_offset": float(train_options.noise_offset),
        "prompt_dropout_prob": float(train_options.prompt_dropout_prob),
        "skip_first_sample": skip_first_sample,
        "force_first_sample": bool(train_options.force_first_sample),
        "unload_text_encoder": bool(train_options.unload_text_encoder),
        "cache_text_embeddings": bool(options.dataset.cache_text_embeddings),
        "ema_config": {
            "use_ema": bool(train_options.use_ema),
            "ema_decay": float(train_options.ema_decay),
            "use_feedback": bool(train_options.ema_use_feedback),
            "param_multiplier": float(train_options.ema_param_multiplier),
        },
        "max_loss": train_options.max_loss,
    }
    if _uses_epoch_schedule(cfg):
        # ai-toolkit derives the exact step count only after it has built the
        # real dataloader. This accounts for repeats, buckets, batch size, and
        # gradient accumulation instead of guessing from a fixed multiplier.
        train["epochs"] = int(cfg.schedule.epochs)
        if cfg.schedule.max_steps is not None:
            train["max_steps"] = int(cfg.schedule.max_steps)
    else:
        max_steps = cfg.schedule.max_steps
        if max_steps is None:
            raise CompilationError("ai-toolkit step schedule requires max_steps")
        train["steps"] = int(max_steps)
    return train


def _uses_epoch_schedule(cfg: TrainingConfig) -> bool:
    """Use epochs unless the config explicitly selects a step-only run."""
    return (
        cfg.schedule.max_steps is None
        or "epochs" in cfg.schedule.model_fields_set
    )


def _model_section(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
    *,
    model_name: str,
) -> dict[str, Any]:
    model_options = options.model
    compile_model = (
        bool(cfg.optimization.torch_compile)
        if model_options.compile is None
        else bool(model_options.compile)
    )
    model_kwargs: dict[str, Any] = {"max_text_length": model_options.max_text_length}
    for key, value in (
        ("checkpoint_filename", model_options.checkpoint_filename),
        ("vae_path", model_options.vae_path),
        ("text_encoder_path", model_options.text_encoder_path),
    ):
        if value:
            model_kwargs[key] = value

    model: dict[str, Any] = {
        "name_or_path": model_name,
        "arch": "krea2",
        "quantize": bool(model_options.quantize),
        "qtype": model_options.qtype,
        "quantize_te": bool(model_options.quantize_text_encoder),
        "qtype_te": model_options.qtype_text_encoder,
        "low_vram": bool(model_options.low_vram),
        "layer_offloading": bool(model_options.layer_offloading),
        "layer_offloading_transformer_percent": float(
            model_options.layer_offloading_transformer_percent
        ),
        "layer_offloading_text_encoder_percent": float(
            model_options.layer_offloading_text_encoder_percent
        ),
        "compile": compile_model,
        "block_compile": bool(model_options.block_compile),
        "compile_mode": model_options.compile_mode,
        "compile_fullgraph": bool(model_options.compile_fullgraph),
        "compile_dynamic": bool(model_options.compile_dynamic),
        "cache_size_limit": model_options.cache_size_limit,
        "model_kwargs": model_kwargs,
    }
    if model_options.assistant_lora_path:
        model["assistant_lora_path"] = model_options.assistant_lora_path
    return model


def _save_section(
    cfg: TrainingConfig,
    options: AiToolkitOptions,
) -> dict[str, Any]:
    save_options = options.save
    return {
        "dtype": _dtype(cfg.output.save_dtype),
        "save_every": (
            int(cfg.output.save_every_n_steps)
            if cfg.output.save_every_n_steps is not None
            else None
        ),
        "save_every_n_epochs": int(cfg.output.save_every_n_epochs),
        "max_step_saves_to_keep": int(cfg.output.save_last_n_steps or 4),
        "push_to_hub": bool(save_options.push_to_hub),
        "hf_repo_id": save_options.hf_repo_id,
        "hf_private": bool(save_options.hf_private),
    }


def _dtype(value: str) -> str:
    if value in {"bf16", "bfloat16"}:
        return "bf16"
    if value in {"fp32", "float", "float32"}:
        return "float32"
    return "fp16"


def _overlay_extra_args(job: dict[str, Any], extra: dict[str, Any]) -> None:
    for raw_key, value in extra.items():
        if value is None:
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


def _sanitize_job(job: dict[str, Any]) -> None:
    process = job["config"]["process"][0]
    sample = process.setdefault("sample", {})
    if not isinstance(sample, dict):
        process["sample"] = sample = {}

    neg = sample.get("neg", "")
    sample["neg"] = neg if isinstance(neg, str) else ""

    prompts = sample.get("prompts", [])
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list):
        prompts = []
    prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
    sample["prompts"] = prompts or ["a high quality image"]
    raw_samples = sample.get("samples", [])
    if isinstance(raw_samples, dict):
        raw_samples = [raw_samples]
    if isinstance(raw_samples, list):
        clean_samples = []
        for item in raw_samples:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            clean = dict(item)
            clean["prompt"] = prompt
            if "neg" in clean and not isinstance(clean["neg"], str):
                clean["neg"] = ""
            clean_samples.append(clean)
        if clean_samples:
            sample["samples"] = clean_samples
            sample["prompts"] = [item["prompt"] for item in clean_samples]
        else:
            sample.pop("samples", None)
    else:
        sample.pop("samples", None)

    network = process.get("network", {})
    if isinstance(network, dict):
        ntype = str(network.get("type", "lora")).lower()
        if ntype not in _KREA2_NETWORK_TYPES:
            allowed = ", ".join(sorted(_KREA2_NETWORK_TYPES))
            raise CompilationError(
                f"ai_toolkit krea2 supports network.type in {{{allowed}}}; got {ntype!r}"
            )
        network["type"] = ntype


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
