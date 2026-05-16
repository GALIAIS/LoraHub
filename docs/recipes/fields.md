---
title: Recipe field reference
description: Every field in the RecipeConfig schema, grouped by section.
---

# Recipe field reference

This page documents every field in the
[`RecipeConfig`](https://github.com/GALIAIS/LoraHub/blob/main/lorahub/core/config/schema.py)
Pydantic model. The schema is the source of truth — defaults below mirror the
code as of this writing.

!!! note "Authoritative source"
    Field names, types, and defaults are validated by Pydantic at load time.
    Anything not listed here is rejected by `RecipeConfig.model_config =
    {"extra": "forbid"}`.

## Top-level

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `schema_version` | `str` | `"1.0"` | Reserved for future migrations. |
| `precision` | `Literal["fp16", "bf16", "fp32"]` | `bf16` | Mixed-precision dtype for training. |
| `gradient_checkpointing` | `bool` | `true` | Trades compute for VRAM. |
| `cache_latents` | `bool` | `true` | Cache VAE latents on disk for the first epoch. |

## `base_model`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `arch` | `Literal["sd15", "sdxl", "flux", "sd3"]` | `sdxl` | Picks the kohya entry script. |
| `arch_variant` | `Literal["", "pony", "illustrious", "noobai", "animagine"]` | `""` | SDXL sub-variant; nudges defaults. Requires `arch == "sdxl"`. |
| `checkpoint` | `Path` | _(required)_ | `.safetensors` or `.ckpt` of the base model. |
| `vae` | `Path \| None` | `None` | Optional external VAE. |

## `dataset`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `source` | `Path` | _(required)_ | Directory of images and `.txt` captions. |
| `resolution` | `list[int]` | `[1024, 1024]` | `[size]` (square) or `[width, height]`. |
| `bucket.enabled` | `bool` | `true` | Multi-resolution bucketing. |
| `bucket.min_size` | `int` (alias `min`) | `256` | Smallest bucket edge. |
| `bucket.max_size` | `int` (alias `max`) | `2048` | Largest bucket edge. |
| `bucket.step` | `int` | `64` | Bucket size quantum. |
| `caption.strategy` | `Literal["tag_file", "filename", "none"]` | `tag_file` | Where captions come from. |
| `caption.ext` | `str` | `".txt"` | Caption file extension when `strategy == "tag_file"`. |
| `caption.shuffle` | `bool` | `true` | Shuffle comma-separated tags each step. |
| `caption.drop_rate` | `float` (0-1) | `0.0` | Random caption-drop probability. |
| `num_repeats` | `int (>=1)` | `1` | How many times each image is replayed per epoch. |

## `network`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["lora", "locon", "loha", "dora"]` | `lora` | Adapter family. |
| `rank` | `int` (1-512) | `32` | Adapter rank / dimension. |
| `alpha` | `int (>=1)` | `16` | LoRA scaling alpha (effective LR ≈ alpha / rank). |
| `target_unet` | `bool` | `true` | Train the UNet. |
| `target_text_encoder` | `bool` | `false` | Train the text encoder(s) — usually for style LoRAs. |

## `optimizer`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `str` | `"adamw8bit"` | Any optimizer name kohya understands. |
| `lr.unet` | `float` | `1e-4` | UNet learning rate. |
| `lr.text_encoder` | `float` | `5e-5` | Text encoder learning rate. |
| `schedule` | `str` | `"cosine_with_restarts"` | LR scheduler name. |
| `warmup_steps` | `int` | `100` | Linear warmup before the schedule. |

## `schedule`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `epochs` | `int (>=1)` | `10` | Number of training epochs. |
| `batch_size` | `int (>=1)` | `1` | Per-step batch size. |
| `grad_accum` | `int (>=1)` | `2` | Gradient accumulation steps. |
| `max_steps` | `int \| None` | `None` | Optional global step cap. |

## `sampling`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `enabled` | `bool` | `true` | Generate preview images during training. |
| `every_n_epochs` | `int (>=1)` | `1` | Cadence in epochs. |
| `prompts_file` | `Path \| None` | `None` | Plain-text prompt file (one per line). |
| `resolution` | `list[int]` | `[1024, 1024]` | Sample resolution. |
| `seed` | `int` | `42` | Sampling RNG seed. |

## `output`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `name` | `str` | `"lora_output"` | Filename stem for saved LoRA files. |
| `save_every_n_epochs` | `int (>=1)` | `1` | Checkpoint cadence. |
| `save_dtype` | `Literal["fp16", "bf16", "float"]` | `fp16` | dtype of the persisted weights. |
| `output_dir` | `Path \| None` | `None` | Override for the workspace output directory. |

## `backend`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["kohya", "diffusion-pipe"]` | `kohya` | Training engine to invoke. |
| `pin_version` | `str \| None` | `None` | Pin the backend version (kohya commit hash). |
| `sd_scripts_path` | `Path \| None` | `None` | Override for the kohya checkout path. |
| `python_executable` | `Path \| None` | `None` | Override for the kohya venv Python. |
| `extra_args` | `dict[str, Any]` | `{}` | Escape hatch for raw kohya CLI flags. |

## `resume`

`resume` controls checkpoint state writing for resume support. When
`save_state == true`, kohya writes optimizer + scheduler state next to the
safetensors so a later run can pick up exactly where the interrupted one left
off. State directories are large; use `save_state_every_n_epochs` to throttle
writes if disk is tight.

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `save_state` | `bool` | `true` | Write optimizer/scheduler state at every checkpoint. |
| `save_state_at_end` | `bool` | `true` | Force a state write when training ends cleanly. |
| `save_state_every_n_epochs` | `int (>=1) \| None` | `None` | Throttle state writes to every N epochs. |
