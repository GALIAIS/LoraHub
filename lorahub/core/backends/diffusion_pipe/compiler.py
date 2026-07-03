"""Compile a semantic ``TrainingConfig`` into diffusion-pipe TOML configs.

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

Notes:
  * ``wandb_api_key`` is intentionally not in the recipe; dp reads
    ``$WANDB_API_KEY`` directly so secrets stay out of the toml.
  * Schema fields whose dp upstream does not consume (kohya-only loss /
    optimization / sampling knobs) are skipped silently by this compiler.
    See the module-level ``_KOHYA_ONLY_*`` debug log emissions if you need
    to audit which fields are being dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lorahub.core.config.schema import (
    ArchPathsConfig,
    DiffusionPipeOptions,
    TrainingConfig,
)

__all__ = ["CompilationError", "compile_config"]

_log = logging.getLogger(__name__)


class CompilationError(ValueError):
    """Raised when a config cannot be expressed in diffusion-pipe's vocabulary."""


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


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Translate a recipe into ``(argv, files_to_write)`` for diffusion-pipe.

    ``argv`` is what to append after ``python train.py`` -- typically
    ``["--deepspeed", "--config", "<workspace>/diffusion_pipe.toml"]``.
    ``files_to_write`` is a mapping of absolute path to file contents the
    launcher must write before spawning the process.
    """
    arch = cfg.base_model.arch
    if arch not in _SUPPORTED_ARCHS:
        msg = (
            f"diffusion-pipe does not support arch {arch!r}; "
            f"supported: {sorted(_SUPPORTED_ARCHS)}. Use the kohya backend instead."
        )
        raise CompilationError(msg)

    workspace = workspace.resolve()
    config_path = workspace / "diffusion_pipe.toml"
    dataset_path = workspace / "dataset.toml"

    _log_attention_choice(cfg)

    files: dict[Path, str] = {
        dataset_path: _build_dataset_toml(cfg),
        config_path: _build_main_toml(cfg, workspace, dataset_path),
    }
    argv = ["--deepspeed", "--config", str(config_path)]
    # Cross-job resume: dp's train.py takes --resume_from_checkpoint=<basename>
    # where basename is a child directory of output_dir (e.g. a timestamped run
    # dir). We accept a full path on cfg.resume.resume_from for symmetry with
    # the other backends and reduce it to the basename here. Caller is
    # responsible for keeping output.output_dir == resume_from.parent — the
    # clone-with-state API and _validate_resume_target enforce that.
    if cfg.resume.resume_from is not None:
        basename = Path(str(cfg.resume.resume_from)).name
        argv.append(f"--resume_from_checkpoint={basename}")
    return argv, files


def _log_attention_choice(cfg: TrainingConfig) -> None:
    """Emit a friendly log line for backend-specific attention handling.

    diffusion-pipe upstream auto-detects FlashAttention 3 by importing
    ``flash_attn_interface`` first and falling back to ``flash_attn`` (FA2)
    when the import fails. The recipe's attention.training therefore can't
    rewrite TOML or argv — it can only nudge the user toward the right
    install. We keep this advisory so a user choosing flash3/flash4 in the
    UI gets confirmation that their choice is meaningful, plus a nudge to
    install the wheel when needed.
    """
    backend = cfg.attention.training
    if backend == "flash":
        _log.info(
            "diffusion-pipe auto-uses flash_attn (FA2) when the package is "
            "importable; install via the 'install-flash-attn' button or "
            "manually with `pip install flash-attn --no-build-isolation`."
        )
    elif backend in ("flash3", "flash4"):
        _log.info(
            "diffusion-pipe auto-detects FlashAttention; ensure the "
            "%s wheel (%s) is installed in the dp venv to actually use it.",
            backend,
            "flash-attn-3" if backend == "flash3" else "flash-attn (4.x)",
        )
    elif backend in ("torch", "sdpa", "flex", "xformers"):
        _log.warning(
            "diffusion-pipe does not honour attention.training=%r; the "
            "trainer will use whichever flash_attn variant it imports.",
            backend,
        )


# --------------------------------------------------------------------------- #
# Main config TOML
# --------------------------------------------------------------------------- #


def _dp_options(cfg: TrainingConfig) -> DiffusionPipeOptions:
    """Return the dp options block, falling back to defaults when unset."""
    return cfg.backend.diffusion_pipe or DiffusionPipeOptions()


def _build_main_toml(cfg: TrainingConfig, workspace: Path, dataset_path: Path) -> str:
    out_dir = cfg.output.output_dir or (workspace / "output")
    opts = _dp_options(cfg)
    parts: list[str] = [
        "# Auto-generated by lorahub. Do not edit by hand.",
    ]
    if cfg.attention.training in ("flash3", "flash4"):
        parts.append(
            f"# attention.training = '{cfg.attention.training}' "
            "(diffusion-pipe auto-detects FlashAttention; install the "
            f"{'flash-attn-3' if cfg.attention.training == 'flash3' else 'flash-attn 4.x'} "
            "wheel into the dp venv)"
        )
    parts += [
        f"output_dir = {_toml_str(str(out_dir))}",
        f"dataset = {_toml_str(str(dataset_path))}",
        "",
        f"epochs = {cfg.schedule.epochs}",
        f"micro_batch_size_per_gpu = {cfg.schedule.batch_size}",
        f"gradient_accumulation_steps = {cfg.schedule.grad_accum}",
        f"pipeline_stages = {opts.pipeline_stages}",
        f"gradient_clipping = {opts.gradient_clipping}",
        f"warmup_steps = {cfg.optimizer.warmup_steps}",
    ]
    # Top-level `cfg.optimization.blocks_to_swap` is the cross-backend
    # source of truth. We keep `backend.diffusion_pipe.blocks_to_swap` for
    # backwards compatibility: when the top-level value is unset (default
    # 0) we fall back to the dp-specific knob so older recipes still emit
    # the same TOML they did before.
    blocks_to_swap = (
        cfg.optimization.blocks_to_swap
        if cfg.optimization.blocks_to_swap > 0
        else opts.blocks_to_swap
    )
    if blocks_to_swap > 0:
        parts.append(f"blocks_to_swap = {blocks_to_swap}")
    if opts.compile:
        parts.append("compile = true")
    if opts.reentrant_activation_checkpointing:
        parts.append("reentrant_activation_checkpointing = true")
    if opts.disable_block_swap_for_eval:
        parts.append("disable_block_swap_for_eval = true")
    # Mixed image/video training batch overrides.
    if opts.image_micro_batch_size_per_gpu is not None:
        parts.append(
            f"image_micro_batch_size_per_gpu = {opts.image_micro_batch_size_per_gpu}"
        )
    if opts.image_eval_micro_batch_size_per_gpu is not None:
        parts.append(
            "image_eval_micro_batch_size_per_gpu = "
            f"{opts.image_eval_micro_batch_size_per_gpu}"
        )
    # Force-flat LR overrides the scheduler (resume tweak).
    if opts.force_constant_lr is not None:
        parts.append(f"force_constant_lr = {opts.force_constant_lr}")
    # CFG-style training: drop captions for some fraction of steps.
    if opts.uncond_fraction > 0:
        parts.append(f"uncond_fraction = {opts.uncond_fraction}")
    # Tensorboard X-axis switch.
    if opts.x_axis_examples:
        parts.append("x_axis_examples = true")
    # Console log cadence (default 1, only emit when overridden).
    if opts.logging_steps != 1:
        parts.append(f"logging_steps = {opts.logging_steps}")
    # Video clip extraction strategy. Only emit when non-default to keep
    # existing image-only TOML byte-identical.
    if opts.video_clip_mode != "single_beginning":
        parts.append(f"video_clip_mode = {_toml_str(opts.video_clip_mode)}")
    # Caching parallelism (dp top-level).
    if cfg.dataloader.map_num_proc is not None:
        parts.append(f"map_num_proc = {cfg.dataloader.map_num_proc}")
    # Pseudo Huber loss constant; the only LossConfig field dp consumes.
    if cfg.loss.pseudo_huber_c is not None:
        parts.append(f"pseudo_huber_c = {cfg.loss.pseudo_huber_c}")
    scheduler = _scheduler_for(cfg.optimizer.schedule)
    if scheduler != "constant":
        parts.append(f"lr_scheduler = {_toml_str(scheduler)}")
    if cfg.schedule.max_steps is not None:
        parts.append(f"max_steps = {cfg.schedule.max_steps}")

    # Save cadence. dp's saver tests for the *presence* of each key, not
    # the value, so we only emit `save_every_n_epochs` when the user
    # actually wants epoch-level saves. Step-level cadence (when set)
    # supersedes it; emitting both would produce double checkpoints
    # whenever a step-save and an epoch-save fall on the same iteration.
    save_block: list[str] = []
    if cfg.output.save_every_n_steps is None:
        save_block.append(
            f"save_every_n_epochs = {cfg.output.save_every_n_epochs}"
        )
    parts += [
        "",
        *save_block,
        f"activation_checkpointing = {_toml_bool(cfg.gradient_checkpointing)}",
        f"partition_method = {_toml_str(opts.partition_method)}",
        f"save_dtype = {_toml_str(_save_dtype(cfg.output.save_dtype))}",
        f"caching_batch_size = {opts.caching_batch_size}",
        f"steps_per_print = {opts.steps_per_print}",
        "",
    ]

    # Manual partition split (used with partition_method='manual').
    if opts.partition_split is not None:
        parts.append(
            f"partition_split = [{', '.join(str(n) for n in opts.partition_split)}]"
        )

    # Output cadence at step / examples granularity (kohya parity).
    if cfg.output.save_every_n_steps is not None:
        parts.append(f"save_every_n_steps = {cfg.output.save_every_n_steps}")
    if cfg.output.save_every_n_examples is not None:
        parts.append(f"save_every_n_examples = {cfg.output.save_every_n_examples}")

    # DeepSpeed checkpoint cadence (separate from `save_*`). dp's default
    # is `checkpoint_every_n_minutes = 120`; emit only when the recipe
    # overrides at least one of the two so default TOML stays byte-identical.
    if opts.checkpoint_every_n_epochs is not None:
        parts.append(f"checkpoint_every_n_epochs = {opts.checkpoint_every_n_epochs}")
    if opts.checkpoint_every_n_minutes is not None:
        parts.append(
            f"checkpoint_every_n_minutes = {opts.checkpoint_every_n_minutes}"
        )

    if (
        opts.eval_every_n_epochs is not None
        or opts.eval_every_n_steps is not None
        or opts.eval_every_n_examples is not None
    ):
        parts.append("")
        if opts.eval_every_n_epochs is not None:
            parts.append(f"eval_every_n_epochs = {opts.eval_every_n_epochs}")
        if opts.eval_every_n_steps is not None:
            parts.append(f"eval_every_n_steps = {opts.eval_every_n_steps}")
        if opts.eval_every_n_examples is not None:
            parts.append(f"eval_every_n_examples = {opts.eval_every_n_examples}")
        parts += [
            f"eval_before_first_step = {_toml_bool(opts.eval_before_first_step)}",
            f"eval_micro_batch_size_per_gpu = {opts.eval_micro_batch_size_per_gpu}",
            f"eval_gradient_accumulation_steps = {opts.eval_gradient_accumulation_steps}",
            "",
        ]

    # Independent eval datasets (top-level inline-table array).
    if opts.eval_datasets:
        entries = []
        for entry in opts.eval_datasets:
            name = entry.get("name", "")
            cfg_path = entry.get("config_path", entry.get("config", ""))
            entries.append(
                f"{{ name = {_toml_str(name)}, config = {_toml_str(cfg_path)} }}"
            )
        parts.append(f"eval_datasets = [{', '.join(entries)}]")
        parts.append("")

    parts += _model_section(cfg)
    parts += [""]
    parts += _adapter_section(cfg)
    parts += [""]
    parts += _optimizer_section(cfg)
    parts += [""]
    parts += _monitoring_section(cfg, opts)
    parts += [""]
    parts += _extra_args_section(cfg)
    _log_dropped_kohya_only_fields(cfg)
    return "\n".join(parts)


def _extra_args_section(cfg: TrainingConfig) -> list[str]:
    """Emit ``cfg.backend.extra_args`` as top-level TOML scalars.

    Escape hatch for any dp TOML key that isn't (yet) typed on the
    schema. Strings ``"true"`` / ``"false"`` (case-insensitive) coerce
    to real bools so the form-driven editor and direct YAML edits feed
    dp the same shape. ``None`` and ``False`` skip the entry entirely.

    Users are on their own for typos: dp's TOML parser will error out
    at startup if a key isn't recognized.
    """
    extra = cfg.backend.extra_args if cfg.backend else None
    if not extra:
        return []
    lines: list[str] = []
    for raw_key, value in extra.items():
        key = raw_key.lstrip("-")
        normalized: Any = value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                normalized = True
            elif lowered == "false":
                normalized = False
        if normalized is None or normalized is False:
            continue
        if normalized is True:
            lines.append(f"{key} = true")
        elif isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
            lines.append(f"{key} = {normalized}")
        else:
            lines.append(f"{key} = {_toml_str(str(normalized))}")
    if lines:
        lines.insert(0, "# backend.extra_args (escape hatch)")
        lines.append("")
    return lines


def _monitoring_section(
    cfg: TrainingConfig, opts: DiffusionPipeOptions
) -> list[str]:
    """Emit the ``[monitoring]`` block.

    Reads from the top-level ``cfg.monitoring`` (the public, all-backend
    surface). Legacy ``opts.enable_wandb`` / ``opts.tracker_name`` /
    ``opts.run_name`` are honored as a fallback for configs saved before
    ``MonitoringConfig`` was promoted out of ``DiffusionPipeOptions``.

    Only the three diffusion-pipe-recognized TOML keys are emitted
    (``enable_wandb`` / ``wandb_tracker_name`` / ``wandb_run_name``);
    the broader wandb identity (entity, tags, notes, run_id, group,
    job_type, mode, resume, base_url) is delivered through ``WANDB_*``
    environment variables injected by ``lorahub.api.wandb_env``.

    ``wandb_api_key`` is intentionally absent: dp picks it up from
    ``$WANDB_API_KEY`` so secrets never touch the on-disk recipe.
    """
    monitoring = cfg.monitoring
    if monitoring.enable_wandb or monitoring.project or monitoring.run_name:
        enable = monitoring.enable_wandb
        project = monitoring.project
        run_name = monitoring.run_name
    else:
        enable = opts.enable_wandb
        project = opts.tracker_name
        run_name = opts.run_name

    lines: list[str] = ["[monitoring]", f"enable_wandb = {_toml_bool(enable)}"]
    if project is not None:
        lines.append(f"wandb_tracker_name = {_toml_str(project)}")
    if run_name is not None:
        lines.append(f"wandb_run_name = {_toml_str(run_name)}")
    return lines


def _model_section(cfg: TrainingConfig) -> list[str]:
    arch = cfg.base_model.arch
    dp_type = _DP_MODEL_TYPE_MAP[arch]  # presence guaranteed by compile_config
    dtype = "bfloat16" if cfg.precision in ("bf16", "fp32") else "float16"
    lines: list[str] = ["[model]", f"type = {_toml_str(dp_type)}"]

    if arch in _CHECKPOINT_PATH_ARCHES:
        lines.append(f"checkpoint_path = {_toml_str(str(cfg.base_model.checkpoint))}")
    else:
        # Most arches expect a Diffusers folder. Per-arch path overrides
        # (transformer_path / vae_path / llm_path / clip_l_path / t5_path...
        # depending on the arch) come from `backend.diffusion_pipe.model_paths`
        # below and win over this default.
        lines.append(f"diffusers_path = {_toml_str(str(cfg.base_model.checkpoint))}")

    lines.append(f"dtype = {_toml_str(dtype)}")

    opts = _dp_options(cfg)
    # Optional [model]-level dtype overrides + sampler. dp accepts these as
    # plain strings and resolves them via `DTYPE_MAP` (utils/common.py).
    if opts.transformer_dtype is not None:
        lines.append(f"transformer_dtype = {_toml_str(opts.transformer_dtype)}")
    if opts.diffusion_model_dtype is not None:
        lines.append(
            f"diffusion_model_dtype = {_toml_str(opts.diffusion_model_dtype)}"
        )
    if opts.timestep_sample_method is not None:
        lines.append(
            f"timestep_sample_method = {_toml_str(opts.timestep_sample_method)}"
        )

    # Typed arch-specific paths (transformer_path / vae_path / llm_path / ...)
    # come from `cfg.base_model.arch_paths`. Bare `base_model.vae` falls back
    # to `vae_path` if `arch_paths.vae` isn't set, so existing recipes that
    # used the legacy `base_model.vae` field keep working.
    arch_path_lines = _arch_paths_lines(cfg.base_model.arch_paths, cfg.base_model.vae)
    lines = _merge_kv_lines(lines, arch_path_lines)

    # Free-form arch-specific path bag. Keys are emitted verbatim so users
    # can drop in any path field upstream's [model] section accepts.
    # `model_paths` is the explicit override channel: when a key collides
    # with anything we already emitted (default `diffusers_path`, an arch
    # path entry, the model dtype, ...) the user-provided value wins.
    if opts.model_paths:
        legacy_lines = [
            f"{key} = {_toml_str(value)}" for key, value in opts.model_paths.items()
        ]
        lines = _merge_kv_lines(lines, legacy_lines)

    return lines


def _arch_paths_lines(
    paths: ArchPathsConfig,
    legacy_vae: Path | None,
) -> list[str]:
    """Render typed arch-specific [model] keys from ``ArchPathsConfig``.

    Only fields the user actually set surface in the TOML. ``ArchPathsConfig``
    has no `vae` field today (the canonical channel is the long-standing
    ``base_model.vae`` path); we still emit `vae_path` here when the legacy
    field is set so dp gets a consistent set of keys.
    """
    out: list[str] = []

    # `base_model.vae` is the canonical VAE path channel. Render as `vae_path`
    # whenever it is set so dp can pick it up alongside the typed paths.
    if legacy_vae is not None:
        out.append(f"vae_path = {_toml_str(str(legacy_vae))}")

    _path_fields: list[tuple[str, Path | None]] = [
        ("transformer_path", paths.transformer),
        ("text_encoder_path", paths.text_encoder),
        ("llm_path", paths.llm),
        ("byt5_path", paths.byt5),
        ("clip_l_path", paths.clip_l),
        ("clip_g_path", paths.clip_g),
        ("t5xxl_path", paths.t5xxl),
        ("ae_path", paths.ae),
        ("qwen3_path", paths.qwen3),
        ("t5_tokenizer_path", paths.t5_tokenizer),
        ("llm_adapter_path", paths.llm_adapter),
    ]
    for key, value in _path_fields:
        if value is not None:
            out.append(f"{key} = {_toml_str(str(value))}")

    # Token length caps (None means "let dp default").
    if paths.t5xxl_max_token_length is not None:
        out.append(f"t5xxl_max_token_length = {paths.t5xxl_max_token_length}")
    if paths.qwen3_max_token_length is not None:
        out.append(f"qwen3_max_token_length = {paths.qwen3_max_token_length}")
    if paths.t5_max_token_length is not None:
        out.append(f"t5_max_token_length = {paths.t5_max_token_length}")

    # Attention masking + per-encoder dropout (FLUX/SD3).
    if paths.apply_t5_attn_mask:
        out.append("apply_t5_attn_mask = true")
    if paths.apply_lg_attn_mask:
        out.append("apply_lg_attn_mask = true")
    if paths.t5_dropout_rate > 0:
        out.append(f"t5_dropout_rate = {paths.t5_dropout_rate}")
    if paths.clip_l_dropout_rate > 0:
        out.append(f"clip_l_dropout_rate = {paths.clip_l_dropout_rate}")
    if paths.clip_g_dropout_rate > 0:
        out.append(f"clip_g_dropout_rate = {paths.clip_g_dropout_rate}")

    # FLUX dev distilled guidance baked into the LoRA.
    if paths.guidance_scale is not None:
        out.append(f"guidance_scale = {paths.guidance_scale}")

    # VAE memory tweaks.
    if paths.vae_chunk_size is not None:
        out.append(f"vae_chunk_size = {paths.vae_chunk_size}")
    if paths.text_encoder_cpu:
        out.append("text_encoder_cpu = true")

    return out


def _merge_kv_lines(existing: list[str], extra: list[str]) -> list[str]:
    """Merge ``extra`` `key = value` lines into ``existing``, last-write-wins.

    Used to layer arch-specific path overrides on top of the default
    `diffusers_path` line, and then to layer the user's free-form
    `model_paths` dict on top of both. Each later assignment removes any
    earlier line whose key matches.
    """
    merged = list(existing)
    for line in extra:
        if " =" not in line:
            merged.append(line)
            continue
        key = line.split(" =", 1)[0].strip()
        merged = [m for m in merged if not m.startswith(f"{key} =")]
        merged.append(line)
    return merged


def _adapter_section(cfg: TrainingConfig) -> list[str]:
    n = cfg.network
    if n.type != "lora":
        # diffusion-pipe upstream only ships a `lora` adapter type today.
        # Surface that early instead of letting train.py crash with
        # `Adapter type {x} is not implemented`.
        msg = (
            f"diffusion-pipe only supports network.type='lora'; got {n.type!r}. "
            "Use the kohya backend for locon/loha/lokr/dora."
        )
        raise CompilationError(msg)

    # diffusion-pipe forces alpha = rank (see train.py:118-122). It will
    # reject the toml if `alpha` is set explicitly, so we omit the field.
    lines: list[str] = [
        "[adapter]",
        "type = 'lora'",
        f"rank = {n.rank}",
    ]
    # LoRA dtype on dp uses long form names (`bfloat16` / `float16` / `float32`).
    if n.dtype is not None:
        lines.append(f"dtype = {_toml_str(_save_dtype(n.dtype))}")
    if n.init_from is not None:
        lines.append(f"init_from_existing = {_toml_str(str(n.init_from))}")
    # `fuse_adapters` is an experimental Flux/Chroma feature: load extra
    # LoRAs into the base weights before training (see
    # ``diffusion-pipe/models/flux.py``). dp accepts an inline-table array
    # of `{path = "...", weight = N}` entries; the recipe also accepts
    # `multiplier` as an alias for `weight` since that's how kohya names it.
    if n.fuse_adapters:
        entries: list[str] = []
        for entry in n.fuse_adapters:
            path = entry.get("path", "")
            weight = entry.get("weight", entry.get("multiplier", 1.0))
            entries.append(
                f"{{ path = {_toml_str(str(path))}, weight = {weight} }}"
            )
        lines.append(f"fuse_adapters = [{', '.join(entries)}]")
    return lines


def _optimizer_section(cfg: TrainingConfig) -> list[str]:
    o = cfg.optimizer
    opt_type = _OPTIMIZER_MAP.get(o.type.lower(), o.type)
    parts = [
        "[optimizer]",
        f"type = {_toml_str(opt_type)}",
        f"lr = {o.lr.unet}",
        f"betas = [{o.betas[0]}, {o.betas[1]}]",
        f"weight_decay = {o.weight_decay}",
        f"eps = {o.eps}",
    ]
    # `cfg.optimization.full_bf16` -> dp `optim_dtype = "bf16"`.
    #
    # dp/train.py forwards every key in the [optimizer] block straight to
    # the optimizer constructor (klass(param_groups, **kwargs)). Most
    # AdamW variants accept `optim_dtype` and use it to allocate the
    # state in bf16, but the bitsandbytes 8-bit and 4-bit families keep
    # state in fp32 by design and reject the kwarg with TypeError. So we
    # only emit `optim_dtype` for optimizers we know accept it; for the
    # rest we silently drop full_bf16 because the optimizer footprint is
    # already tiny.
    if cfg.optimization.full_bf16 and not _is_quantized_optimizer(opt_type):
        parts.append(f"optim_dtype = {_toml_str('bf16')}")
    # dp gradient_release: chunk-wise grad release for memory savings (see
    # `diffusion-pipe/train.py:407`). dp forces gradient_clipping=0 when
    # this is on; we just emit the flag and let dp wire the rest.
    if o.gradient_release:
        parts.append("gradient_release = true")
    seen = {"type", "lr", "betas", "weight_decay", "eps", "optim_dtype", "gradient_release"}
    # Free-form optimizer_args -> toml lines. Keys win over the dedicated
    # fields when names collide (matches the kohya backend's behaviour).
    for key, value in o.optimizer_args.items():
        if key in seen:
            # Replace the prior entry instead of appending a duplicate.
            parts = [p for p in parts if not p.startswith(f"{key} =")]
        parts.append(f"{key} = {_toml_str(value)}")
    return parts


# Optimizer types whose constructor refuses `optim_dtype`. Lower-cased for
# case-insensitive matching against either the user's `optimizer.type`
# string or its mapped diffusion-pipe equivalent.
_QUANTIZED_OPTIMIZERS = {
    "adamw8bit",
    "adamw4bit",
    "adamw_8bit",
    "adamw_4bit",
    "lion8bit",
    "lion_8bit",
    "paged_adamw_8bit",
    "paged_adamw8bit",
    "paged_lion_8bit",
    "paged_lion8bit",
    "adamw8bitkahan",
}


def _is_quantized_optimizer(name: str) -> bool:
    return name.lower() in _QUANTIZED_OPTIMIZERS


def _scheduler_for(name: str) -> str:
    return _SCHEDULER_MAP.get(name, "constant")


def _save_dtype(name: str) -> str:
    """Translate our `fp16` / `bf16` / `fp32` / `float` aliases into dp names.

    dp accepts the long-form names (`float16` / `bfloat16` / `float32`) via
    `utils.common.DTYPE_MAP`. The legacy `OutputConfig.save_dtype` literal
    "float" predates fp32 and continues to mean float32; we keep accepting it
    for backward compatibility.
    """
    if name in ("fp16", "float16"):
        return "float16"
    if name in ("bf16", "bfloat16"):
        return "bfloat16"
    return "float32"


# --------------------------------------------------------------------------- #
# Dataset TOML
# --------------------------------------------------------------------------- #


def _build_dataset_toml(cfg: TrainingConfig) -> str:
    ds = cfg.dataset
    opts = _dp_options(cfg)
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
    # An explicit `ar_buckets` list (under `bucket.ar_buckets`) overrides
    # the parametric min/max/num triple. dp's `utils/dataset.py` reads
    # `directory_config.get('ar_buckets', dataset_config.get('ar_buckets'))`
    # so emitting it at the top level applies to every directory unless
    # a subset overrides it.
    if ds.bucket.ar_buckets:
        ar_str = ", ".join(str(v) for v in ds.bucket.ar_buckets)
        parts.append(f"ar_buckets = [{ar_str}]")
    elif ds.bucket.enabled:
        parts += [
            f"min_ar = {opts.min_ar}",
            f"max_ar = {opts.max_ar}",
            f"num_ar_buckets = {opts.num_ar_buckets}",
        ]
    # `frame_buckets` defaults to [1] (image-only). Only emit a different
    # value so existing image-only recipes keep producing identical TOML.
    if ds.frame_buckets == [1]:
        parts.append("frame_buckets = [1]")
    else:
        fb_str = ", ".join(str(v) for v in ds.frame_buckets)
        parts.append(f"frame_buckets = [{fb_str}]")

    parts += [
        f"cache_shuffle_num = {opts.cache_shuffle_num}",
        f"skip_empty_caption = {_toml_bool(opts.skip_empty_caption)}",
    ]
    # Per-tag-shuffle delimiter (dp key is `cache_shuffle_delimiter`, not
    # `shuffle_delimiter`) and the legacy whole-caption shuffle toggle.
    # Default `cache_shuffle_delimiter = ", "` is dp's built-in fallback,
    # so we only emit when the recipe overrides it.
    if cfg.dataset.caption.shuffle_delimiter is not None:
        parts.append(
            f"cache_shuffle_delimiter = {_toml_str(cfg.dataset.caption.shuffle_delimiter)}"
        )
    if cfg.dataset.caption.shuffle_tags:
        parts.append("shuffle_tags = true")

    parts.append("")

    # When `subsets` is non-empty it OVERRIDES the single-directory
    # `source` + `num_repeats` pair: dp emits one [[directory]] per entry,
    # each with its own optional mask / ar_buckets / caption_prefix.
    if ds.subsets:
        for sub in ds.subsets:
            parts.append("[[directory]]")
            parts.append(f"path = {_toml_str(str(sub.path))}")
            parts.append(f"num_repeats = {sub.num_repeats}")
            if sub.mask_path is not None:
                parts.append(f"mask_path = {_toml_str(str(sub.mask_path))}")
            if sub.ar_buckets:
                ar_str = ", ".join(str(v) for v in sub.ar_buckets)
                parts.append(f"ar_buckets = [{ar_str}]")
            if sub.caption_prefix is not None:
                parts.append(f"caption_prefix = {_toml_str(sub.caption_prefix)}")
            parts.append("")
    else:
        parts += [
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


# --------------------------------------------------------------------------- #
# Kohya-only fields: log at debug level so dropped values are auditable.
# --------------------------------------------------------------------------- #


def _log_dropped_kohya_only_fields(cfg: TrainingConfig) -> None:
    """Log fields the recipe sets but the dp backend doesn't consume.

    These are kohya argv with no dp equivalent (advanced loss / noise
    shaping, augmentation, fp8/lowram knobs, scheduler module overrides,
    metadata stamping, validation cadence, ...). We emit at DEBUG so the
    typical run is silent; turn the logger up to debug to audit.
    """
    if not _log.isEnabledFor(logging.DEBUG):
        return
    dropped: list[str] = []

    # Loss-shaping knobs dp doesn't honour (only `pseudo_huber_c` survives).
    loss = cfg.loss
    _track(dropped, "loss.min_snr_gamma", loss.min_snr_gamma)
    _track(dropped, "loss.noise_offset", loss.noise_offset, default=0.0)
    _track(dropped, "loss.noise_offset_random_strength", loss.noise_offset_random_strength)
    _track(dropped, "loss.multires_noise_iterations", loss.multires_noise_iterations)
    _track(
        dropped, "loss.multires_noise_discount", loss.multires_noise_discount, default=0.3
    )
    _track(dropped, "loss.adaptive_noise_scale", loss.adaptive_noise_scale)
    _track(dropped, "loss.ip_noise_gamma", loss.ip_noise_gamma)
    _track(dropped, "loss.ip_noise_gamma_random_strength", loss.ip_noise_gamma_random_strength)
    _track(dropped, "loss.zero_terminal_snr", loss.zero_terminal_snr)
    _track(dropped, "loss.min_timestep", loss.min_timestep)
    _track(dropped, "loss.max_timestep", loss.max_timestep)
    _track(dropped, "loss.huber_schedule", loss.huber_schedule)
    _track(dropped, "loss.huber_c", loss.huber_c)
    _track(dropped, "loss.huber_scale", loss.huber_scale)
    _track(dropped, "loss.v_pred_like_loss", loss.v_pred_like_loss)
    _track(dropped, "loss.debiased_estimation", loss.debiased_estimation)
    _track(dropped, "loss.scale_v_pred_loss_like_noise_pred", loss.scale_v_pred_loss_like_noise_pred)

    # Optimizer scheduler override (kohya --lr_scheduler_type / args).
    o = cfg.optimizer
    _track(dropped, "optimizer.scheduler_module", o.scheduler_module)
    _track(dropped, "optimizer.scheduler_args", o.scheduler_args, default={})
    _track(dropped, "optimizer.scheduler_num_cycles", o.scheduler_num_cycles, default=1)
    _track(dropped, "optimizer.scheduler_power", o.scheduler_power, default=1.0)
    _track(dropped, "optimizer.scheduler_timescale", o.scheduler_timescale)
    _track(dropped, "optimizer.scheduler_min_lr_ratio", o.scheduler_min_lr_ratio)
    _track(dropped, "optimizer.max_grad_norm", o.max_grad_norm, default=1.0)

    # Output metadata + retention (kohya only).
    out = cfg.output
    _track(dropped, "output.save_last_n_epochs", out.save_last_n_epochs)
    _track(dropped, "output.save_last_n_steps", out.save_last_n_steps)
    _track(dropped, "output.training_comment", out.training_comment)
    _track(dropped, "output.no_metadata", out.no_metadata)
    _track(dropped, "output.metadata", out.metadata, default={})

    # Resume knobs that only kohya understands. resume.resume_from is now
    # forwarded above (compile_config -> --resume_from_checkpoint), so it is
    # NOT dropped here.
    r = cfg.resume
    _track(dropped, "resume.save_last_n_epochs_state", r.save_last_n_epochs_state)
    _track(dropped, "resume.save_last_n_steps_state", r.save_last_n_steps_state)
    _track(dropped, "resume.skip_until_initial_step", r.skip_until_initial_step)
    _track(dropped, "resume.initial_epoch", r.initial_epoch)
    _track(dropped, "resume.initial_step", r.initial_step)

    # Network knobs unique to kohya (LoRA fusion / module LR / weighted load).
    n = cfg.network
    _track(dropped, "network.base_weights", n.base_weights, default=[])
    _track(dropped, "network.base_weights_multiplier", n.base_weights_multiplier, default=[])
    _track(dropped, "network.dim_from_weights", n.dim_from_weights)
    _track(dropped, "network.module_lr", n.module_lr)

    # Caption / bucket / schedule knobs that aren't on dp.
    cap = cfg.dataset.caption
    _track(dropped, "dataset.caption.keep_tokens", cap.keep_tokens, default=0)
    _track(dropped, "dataset.caption.keep_tokens_separator", cap.keep_tokens_separator)
    _track(dropped, "dataset.caption.secondary_separator", cap.secondary_separator)
    _track(dropped, "dataset.caption.enable_wildcard", cap.enable_wildcard)
    _track(dropped, "dataset.caption.prefix", cap.prefix)
    _track(dropped, "dataset.caption.suffix", cap.suffix)
    _track(dropped, "dataset.caption.max_token_length", cap.max_token_length)
    _track(dropped, "dataset.caption.token_warmup_min", cap.token_warmup_min)
    _track(dropped, "dataset.caption.token_warmup_step", cap.token_warmup_step)
    _track(dropped, "dataset.caption.weighted", cap.weighted)
    _track(dropped, "dataset.caption.dropout_every_n_epochs", cap.dropout_every_n_epochs, default=0)
    _track(dropped, "dataset.caption.tag_dropout_rate", cap.tag_dropout_rate, default=0.0)

    bk = cfg.dataset.bucket
    _track(dropped, "dataset.bucket.no_upscale", bk.no_upscale)
    _track(dropped, "dataset.bucket.skip_image_resolution", bk.skip_image_resolution)
    _track(dropped, "dataset.bucket.resize_interpolation", bk.resize_interpolation)

    sched = cfg.schedule
    _track(dropped, "schedule.seed", sched.seed)
    _track(dropped, "schedule.lr_decay_steps", sched.lr_decay_steps)

    # DataLoader / sampling / validation cadence are sd-scripts only.
    dl = cfg.dataloader
    _track(dropped, "dataloader.num_workers", dl.num_workers, default=8)
    _track(dropped, "dataloader.persistent_workers", dl.persistent_workers)
    _track(dropped, "dataloader.vae_batch_size", dl.vae_batch_size, default=1)
    _track(dropped, "dataloader.text_encoder_batch_size", dl.text_encoder_batch_size)
    _track(dropped, "dataloader.cache_shuffle_num", dl.cache_shuffle_num, default=0)

    samp = cfg.sampling
    _track(dropped, "sampling.every_n_steps", samp.every_n_steps)
    _track(dropped, "sampling.at_first", samp.at_first)

    val = cfg.validation
    _track(dropped, "validation.every_n_steps", val.every_n_steps)
    _track(dropped, "validation.seed", val.seed)

    # Augmentation block is kohya only.
    aug = cfg.augmentation
    _track(dropped, "augmentation.flip", aug.flip)
    _track(dropped, "augmentation.color", aug.color)
    _track(dropped, "augmentation.random_crop", aug.random_crop)
    _track(dropped, "augmentation.face_crop_aug_range", aug.face_crop_aug_range)
    _track(dropped, "augmentation.alpha_mask", aug.alpha_mask)

    # Optimization fields with no dp equivalent (fp8 / lowram / cpu offload /
    # caches). `full_bf16` is consumed via `optim_dtype`; everything else
    # is a no-op.
    opt = cfg.optimization
    _track(dropped, "optimization.full_fp16", opt.full_fp16)
    _track(dropped, "optimization.fp8_base", opt.fp8_base)
    _track(dropped, "optimization.fp8_base_unet", opt.fp8_base_unet)
    _track(dropped, "optimization.fp8_scaled", opt.fp8_scaled)
    _track(dropped, "optimization.fp8_vl_text_encoder", opt.fp8_vl_text_encoder)
    _track(dropped, "optimization.lowram", opt.lowram)
    _track(dropped, "optimization.highvram", opt.highvram)
    _track(dropped, "optimization.no_half_vae", opt.no_half_vae)
    _track(dropped, "optimization.disable_mmap_load_safetensors", opt.disable_mmap_load_safetensors)
    _track(dropped, "optimization.cpu_offload_checkpointing", opt.cpu_offload_checkpointing)
    _track(dropped, "optimization.unsloth_offload_checkpointing", opt.unsloth_offload_checkpointing)
    _track(dropped, "optimization.cache_text_encoder_outputs", opt.cache_text_encoder_outputs)
    _track(
        dropped,
        "optimization.cache_text_encoder_outputs_to_disk",
        opt.cache_text_encoder_outputs_to_disk,
    )

    # Dataset-level kohya-only inputs.
    ds = cfg.dataset
    _track(dropped, "dataset.conditioning_dir", ds.conditioning_dir)
    _track(dropped, "dataset.reg_source", ds.reg_source)
    _track(dropped, "dataset.val_split", ds.val_split, default=0.0)

    # Top-level kohya-only flags.
    _track(dropped, "cache_latents_to_disk", cfg.cache_latents_to_disk)
    _track(dropped, "skip_cache_check", cfg.skip_cache_check)
    _track(dropped, "cache_info", cfg.cache_info)
    _track(dropped, "train_inpainting", cfg.train_inpainting)

    # FlowMatch settings are arch/backend-specific. dp uses the
    # `[model] timestep_sample_method` and shift fields exposed via
    # `DiffusionPipeOptions.timestep_sample_method` instead.
    fm = cfg.flow_match
    _track(dropped, "flow_match.timestep_sampling", fm.timestep_sampling)
    _track(dropped, "flow_match.sigmoid_scale", fm.sigmoid_scale)
    _track(dropped, "flow_match.model_prediction_type", fm.model_prediction_type)
    _track(dropped, "flow_match.discrete_flow_shift", fm.discrete_flow_shift)
    _track(dropped, "flow_match.training_shift", fm.training_shift)
    _track(dropped, "flow_match.weighting_scheme", fm.weighting_scheme)
    _track(dropped, "flow_match.logit_mean", fm.logit_mean)
    _track(dropped, "flow_match.logit_std", fm.logit_std)
    _track(dropped, "flow_match.mode_scale", fm.mode_scale)

    if dropped:
        _log.debug(
            "diffusion-pipe backend ignored %d kohya-only field(s): %s",
            len(dropped),
            ", ".join(dropped),
        )


def _track(bucket: list[str], path: str, value: Any, *, default: Any = None) -> None:
    """Append ``path`` to ``bucket`` when ``value`` differs from ``default``.

    Treats `False`, ``None``, empty containers, and empty strings as "unset"
    so flag-style booleans only show up in the dropped-fields log when the
    user actually flipped them on.
    """
    if value == default:
        return
    if value in (None, False, "", [], {}):
        return
    bucket.append(path)
