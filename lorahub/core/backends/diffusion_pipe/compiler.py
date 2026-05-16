"""Compile a semantic ``RecipeConfig`` into diffusion-pipe TOML configs.

diffusion-pipe is config-driven instead of CLI-flag-driven: ``train.py`` takes
a single ``--config`` pointing at a TOML file, and that TOML in turn points at
a dataset TOML. This module produces both files (purely as strings + paths)
so the launcher can write them to disk and shell out.

The compiler is deliberately a pure function: callers pass in the recipe and
a workspace directory, and get back ``(argv, files_to_write)`` where
``argv`` is what to append after ``train.py`` and ``files_to_write`` maps
absolute paths to file contents. No disk I/O here keeps the unit tests fast.

Reference shape: see ``diffusion-pipe/examples/main_example.toml`` and
``diffusion-pipe/examples/dataset.toml`` in the upstream repo.

Follow-ups (not yet wired through the schema):
  * Per-subset ``mask_path`` for masked training.
  * Multiple ``[[directory]]`` blocks (we currently emit one).
  * Video ``frame_buckets`` beyond ``[1]``.
  * ``wandb_api_key`` is intentionally not in the recipe; dp reads
    ``$WANDB_API_KEY`` directly so secrets stay out of the toml.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.config.schema import DiffusionPipeOptions, RecipeConfig

__all__ = ["CompilationError", "compile_recipe"]


class CompilationError(ValueError):
    """Raised when a recipe cannot be expressed in diffusion-pipe's vocabulary."""


# Map our base_model.arch literals to the `[model] type = "..."` value
# diffusion-pipe expects in its TOML config. Mirrors upstream's
# docs/supported_models.md. Note the dash-vs-underscore mismatches: dp
# spells `hunyuan_video` as `hunyuan-video` and `ltx_video` as `ltx-video`
# (every other entry happens to match the schema literal verbatim).
# Arches not in this map are kohya-only and rejected up front.
_DP_MODEL_TYPE_MAP: dict[str, str] = {
    "sdxl": "sdxl",
    "sd3": "sd3",
    "flux": "flux",
    "flux2": "flux2",
    "lumina": "lumina_2",
    "chroma": "chroma",
    "hidream": "hidream",
    "omnigen2": "omnigen2",
    "auraflow": "auraflow",
    "qwen_image": "qwen_image",
    "cosmos": "cosmos",
    "cosmos_predict2": "cosmos_predict2",
    "anima": "anima",
    "hunyuan_image": "hunyuan_image",
    "hunyuan_video": "hunyuan-video",  # upstream uses a hyphen here
    "hunyuan_video_15": "hunyuan_video_15",
    "ltx_video": "ltx-video",  # upstream uses a hyphen here
    "ltx2": "ltx2",
    "wan": "wan",  # covers Wan2.1 and Wan2.2
    "z_image": "z_image",
    "ernie_image": "ernie_image",
}

# Convenience set used by validators.
_SUPPORTED_ARCHS: frozenset[str] = frozenset(_DP_MODEL_TYPE_MAP)

# Arches whose [model] section takes a single-file `checkpoint_path =`. Every
# other supported arch uses `diffusers_path =` (folder of Diffusers shards).
# Mirrors the historical default for SDXL plus the dp examples for image
# diffusion models that ship as single safetensors (e.g. AuraFlow, Chroma).
_CHECKPOINT_PATH_ARCHES: frozenset[str] = frozenset({"sdxl"})

# Map our optimizer.type values onto names diffusion-pipe accepts directly.
# Anything not in this map is passed through verbatim and resolved by
# diffusion-pipe via pytorch_optimizer (which is its escape hatch).
_OPTIMIZER_MAP: dict[str, str] = {
    "adamw": "adamw",
    "adamw8bit": "adamw8bit",
    "adamw_optimi": "adamw_optimi",
    "adamw8bitkahan": "AdamW8bitKahan",
    "lion": "Lion",
    "lion8bit": "Lion8bit",
    "prodigy": "Prodigy",
    "automagic": "automagic",
}

# Recipe scheduler -> diffusion-pipe scheduler. dp only ships constant /
# linear / cosine; anything implying restarts collapses to plain cosine.
_SCHEDULER_MAP: dict[str, str] = {
    "constant": "constant",
    "linear": "linear",
    "cosine": "cosine",
    "cosine_with_restarts": "cosine",
}


def compile_recipe(
    recipe: RecipeConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Translate a recipe into ``(argv, files_to_write)`` for diffusion-pipe.

    ``argv`` is what to append after ``python train.py`` -- typically
    ``["--deepspeed", "--config", "<workspace>/diffusion_pipe.toml"]``.
    ``files_to_write`` is a mapping of absolute path to file contents the
    launcher must write before spawning the process.
    """
    arch = recipe.base_model.arch
    if arch not in _SUPPORTED_ARCHS:
        msg = (
            f"diffusion-pipe does not support arch {arch!r}; "
            f"supported: {sorted(_SUPPORTED_ARCHS)}. Use the kohya backend instead."
        )
        raise CompilationError(msg)

    workspace = workspace.resolve()
    config_path = workspace / "diffusion_pipe.toml"
    dataset_path = workspace / "dataset.toml"

    files: dict[Path, str] = {
        dataset_path: _build_dataset_toml(recipe),
        config_path: _build_main_toml(recipe, workspace, dataset_path),
    }
    argv = ["--deepspeed", "--config", str(config_path)]
    return argv, files


# --------------------------------------------------------------------------- #
# Main config TOML
# --------------------------------------------------------------------------- #


def _dp_options(recipe: RecipeConfig) -> DiffusionPipeOptions:
    """Return the dp options block, falling back to defaults when unset."""
    return recipe.backend.diffusion_pipe or DiffusionPipeOptions()


def _build_main_toml(recipe: RecipeConfig, workspace: Path, dataset_path: Path) -> str:
    out_dir = recipe.output.output_dir or (workspace / "output")
    opts = _dp_options(recipe)
    parts: list[str] = [
        "# Auto-generated by lorahub. Do not edit by hand.",
        f"output_dir = {_toml_str(str(out_dir))}",
        f"dataset = {_toml_str(str(dataset_path))}",
        "",
        f"epochs = {recipe.schedule.epochs}",
        f"micro_batch_size_per_gpu = {recipe.schedule.batch_size}",
        f"gradient_accumulation_steps = {recipe.schedule.grad_accum}",
        f"pipeline_stages = {opts.pipeline_stages}",
        f"gradient_clipping = {opts.gradient_clipping}",
        f"warmup_steps = {recipe.optimizer.warmup_steps}",
    ]
    if opts.blocks_to_swap > 0:
        parts.append(f"blocks_to_swap = {opts.blocks_to_swap}")
    if opts.compile:
        parts.append("compile = true")
    scheduler = _scheduler_for(recipe.optimizer.schedule)
    if scheduler != "constant":
        parts.append(f"lr_scheduler = {_toml_str(scheduler)}")
    if recipe.schedule.max_steps is not None:
        parts.append(f"max_steps = {recipe.schedule.max_steps}")

    parts += [
        "",
        f"save_every_n_epochs = {recipe.output.save_every_n_epochs}",
        f"activation_checkpointing = {_toml_bool(recipe.gradient_checkpointing)}",
        f"partition_method = {_toml_str(opts.partition_method)}",
        f"save_dtype = {_toml_str(_save_dtype(recipe.output.save_dtype))}",
        f"caching_batch_size = {opts.caching_batch_size}",
        f"steps_per_print = {opts.steps_per_print}",
        "",
    ]

    if opts.eval_every_n_epochs is not None:
        parts += [
            f"eval_every_n_epochs = {opts.eval_every_n_epochs}",
            f"eval_before_first_step = {_toml_bool(opts.eval_before_first_step)}",
            f"eval_micro_batch_size_per_gpu = {opts.eval_micro_batch_size_per_gpu}",
            "",
        ]

    parts += _model_section(recipe)
    parts += [""]
    parts += _adapter_section(recipe)
    parts += [""]
    parts += _optimizer_section(recipe)
    parts += [""]
    parts += _monitoring_section(opts)
    parts += [""]
    return "\n".join(parts)


def _monitoring_section(opts: DiffusionPipeOptions) -> list[str]:
    """Emit the ``[monitoring]`` block.

    ``wandb_api_key`` is intentionally absent: dp picks it up from
    ``$WANDB_API_KEY`` so secrets never touch the on-disk recipe.
    """
    lines: list[str] = ["[monitoring]", f"enable_wandb = {_toml_bool(opts.enable_wandb)}"]
    if opts.tracker_name is not None:
        lines.append(f"wandb_tracker_name = {_toml_str(opts.tracker_name)}")
    if opts.run_name is not None:
        lines.append(f"wandb_run_name = {_toml_str(opts.run_name)}")
    return lines


def _model_section(recipe: RecipeConfig) -> list[str]:
    arch = recipe.base_model.arch
    dp_type = _DP_MODEL_TYPE_MAP[arch]  # presence guaranteed by compile_recipe
    dtype = "bfloat16" if recipe.precision in ("bf16", "fp32") else "float16"
    lines: list[str] = ["[model]", f"type = {_toml_str(dp_type)}"]

    if arch in _CHECKPOINT_PATH_ARCHES:
        lines.append(f"checkpoint_path = {_toml_str(str(recipe.base_model.checkpoint))}")
    else:
        # Most arches expect a Diffusers folder. Per-arch path overrides
        # (transformer_path / vae_path / llm_path / clip_l_path / t5_path...
        # depending on the arch) come from `backend.diffusion_pipe.model_paths`
        # below and win over this default.
        lines.append(f"diffusers_path = {_toml_str(str(recipe.base_model.checkpoint))}")

    lines.append(f"dtype = {_toml_str(dtype)}")

    # Free-form arch-specific path bag. Keys are emitted verbatim so users
    # can drop in any path field upstream's [model] section accepts (e.g.
    # `transformer_path = "..."`, `vae_path = "..."`, `llm_path = "..."`).
    # Empty by default -- existing SDXL/Flux/SD3 recipes produce identical
    # TOML to before. Keys provided here override duplicate defaults
    # rendered above (last-write-wins matches the rest of the compiler).
    opts = _dp_options(recipe)
    if opts.model_paths:
        seen_keys = {line.split(" =", 1)[0].strip() for line in lines if " =" in line}
        for key, value in opts.model_paths.items():
            if key in seen_keys:
                lines = [
                    line for line in lines
                    if not line.startswith(f"{key} =")
                ]
            lines.append(f"{key} = {_toml_str(value)}")

    return lines


def _adapter_section(recipe: RecipeConfig) -> list[str]:
    n = recipe.network
    if n.type != "lora":
        # diffusion-pipe upstream only ships a `lora` adapter type today.
        # Surface that early instead of letting train.py crash with
        # `Adapter type {x} is not implemented`.
        msg = (
            f"diffusion-pipe only supports network.type='lora'; got {n.type!r}. "
            "Use the kohya backend for locon/loha/dora."
        )
        raise CompilationError(msg)

    # diffusion-pipe forces alpha = rank (see train.py:118-122). It will
    # reject the toml if `alpha` is set explicitly, so we omit the field.
    return [
        "[adapter]",
        "type = 'lora'",
        f"rank = {n.rank}",
    ]


def _optimizer_section(recipe: RecipeConfig) -> list[str]:
    o = recipe.optimizer
    opt_type = _OPTIMIZER_MAP.get(o.type.lower(), o.type)
    parts = [
        "[optimizer]",
        f"type = {_toml_str(opt_type)}",
        f"lr = {o.lr.unet}",
        f"betas = [{o.betas[0]}, {o.betas[1]}]",
        f"weight_decay = {o.weight_decay}",
        f"eps = {o.eps}",
    ]
    # Free-form optimizer_args -> toml lines. Keys win over the dedicated
    # fields when names collide (matches the kohya backend's behaviour).
    seen = {"type", "lr", "betas", "weight_decay", "eps"}
    for key, value in o.optimizer_args.items():
        if key in seen:
            # Replace the prior entry instead of appending a duplicate.
            parts = [p for p in parts if not p.startswith(f"{key} =")]
        parts.append(f"{key} = {_toml_str(value)}")
    return parts


def _scheduler_for(name: str) -> str:
    return _SCHEDULER_MAP.get(name, "constant")


def _save_dtype(name: str) -> str:
    if name == "fp16":
        return "float16"
    if name == "bf16":
        return "bfloat16"
    return "float32"


# --------------------------------------------------------------------------- #
# Dataset TOML
# --------------------------------------------------------------------------- #


def _build_dataset_toml(recipe: RecipeConfig) -> str:
    ds = recipe.dataset
    opts = _dp_options(recipe)
    if len(ds.resolution) == 1:
        resolutions = f"[{ds.resolution[0]}]"
    else:
        # diffusion-pipe accepts (width, height) pairs as nested arrays.
        resolutions = f"[[{ds.resolution[0]}, {ds.resolution[1]}]]"

    parts: list[str] = [
        "# Auto-generated by lorahub. Do not edit by hand.",
        f"resolutions = {resolutions}",
        f"enable_ar_bucket = {_toml_bool(ds.bucket.enabled)}",
    ]
    if ds.bucket.enabled:
        parts += [
            f"min_ar = {opts.min_ar}",
            f"max_ar = {opts.max_ar}",
            f"num_ar_buckets = {opts.num_ar_buckets}",
        ]
    # `frame_buckets = [1]` keeps the dataset image-only; it's safe for
    # every image arch the schema currently ships (sdxl, flux, sd3).
    parts += [
        "frame_buckets = [1]",
        f"cache_shuffle_num = {opts.cache_shuffle_num}",
        f"skip_empty_caption = {_toml_bool(opts.skip_empty_caption)}",
        "",
        # TODO: support multiple `[[directory]]` blocks plus per-subset
        # `mask_path`. Single-directory covers the 95% case today.
        "[[directory]]",
        f"path = {_toml_str(str(ds.source))}",
        f"num_repeats = {ds.num_repeats}",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# TOML literal helpers
# --------------------------------------------------------------------------- #


def _toml_str(value: str) -> str:
    """Emit a TOML basic string literal (escaped) for ``value``."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"
