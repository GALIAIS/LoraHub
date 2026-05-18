---
title: Config field reference
description: Every field in the TrainingConfig schema, grouped by section.
---

# Config field reference

This page documents the most-used fields in the
[`TrainingConfig`](https://github.com/GALIAIS/LoraHub/blob/main/lorahub/core/config/schema.py)
Pydantic model. The schema is the source of truth — defaults below mirror
the code as of this writing. Field names are shown in **camelCase** (the
preferred wire form); `snake_case` is also accepted by the validator.

!!! note "Authoritative source"
    Field names, types, and defaults are validated by Pydantic at load
    time. The top-level model has `extra: forbid`, so unknown keys are
    rejected with a clear error.

## Top-level

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `schemaVersion` | `str` | `"1.0"` | Reserved for future migrations. |
| `precision` | `Literal["fp16", "bf16", "fp32"]` | `bf16` | Mixed-precision dtype for training. |
| `gradientCheckpointing` | `bool` | `true` | Trades compute for VRAM. |
| `cacheLatents` | `bool` | `true` | Cache VAE latents on disk for the first epoch. |

## `baseModel`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `arch` | `Literal[...]` | `sdxl` | Picks the backend entry path. Supported: `sd15`, `sd2`, `sdxl`, `sd3`, `flux`, `lumina`, `hunyuan_image`, `anima`, `flux2`, `chroma`, `hidream`, `omnigen2`, `auraflow`, `qwen_image`, `cosmos`, `cosmos_predict2`, `hunyuan_video`, `hunyuan_video_15`, `ltx_video`, `ltx2`, `wan`, `z_image`, `ernie_image`. |
| `archVariant` | `Literal["", "pony", "illustrious", "noobai", "animagine"]` | `""` | SDXL sub-variant; nudges defaults. Requires `arch == "sdxl"`. |
| `checkpoint` | `Path` | _(required)_ | Backbone weights (`.safetensors` or `.ckpt`). |
| `vae` | `Path \| None` | `None` | Optional external VAE. |
| `archPaths` | `ArchPathsConfig` | `{}` | Per-component checkpoint paths for multi-file arches (`clipL`, `clipG`, `t5xxl`, `ae`, `transformer`, `qwen3`, `t5Tokenizer`, `llmAdapter`, ...). |

## `dataset`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `source` | `Path` | _(required)_ | Directory of images and `.txt` captions. |
| `resolution` | `list[int]` | `[1024, 1024]` | `[size]` (square) or `[width, height]`. |
| `bucket.enabled` | `bool` | `true` | Multi-resolution bucketing. |
| `bucket.min` | `int` | `256` | Smallest bucket edge. |
| `bucket.max` | `int` | `2048` | Largest bucket edge. |
| `bucket.step` | `int` | `64` | Bucket size quantum. |
| `caption.strategy` | `Literal["tag_file", "filename", "none"]` | `tag_file` | Where captions come from. **Always snake_case** — Literal value. |
| `caption.ext` | `str` | `".txt"` | Caption file extension when `strategy == "tag_file"`. |
| `caption.shuffle` | `bool` | `true` | Shuffle comma-separated tags each step. |
| `caption.dropRate` | `float` (0-1) | `0.0` | Random caption-drop probability. |
| `numRepeats` | `int (>=1)` | `1` | How many times each image is replayed per epoch. |

## `network`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["lora", "locon", "loha", "dora"]` | `lora` | Adapter family. |
| `rank` | `int` (1-512) | `32` | Adapter rank / dimension. |
| `alpha` | `int (>=1)` | `16` | LoRA scaling alpha (effective LR ≈ alpha / rank). dp-side LoRA forces `alpha == rank`. |
| `targetUnet` | `bool` | `true` | Train the UNet / transformer. |
| `targetTextEncoder` | `bool` | `false` | Train the text encoder(s). dp/Anima ignores this — TE is frozen. |
| `networkDropout` | `float (0-1)` | `0.0` | Adapter dropout for regularisation. |
| `convDim` / `convAlpha` | `int \| None` | `None` | Only valid for `locon` / `loha`. |

## `optimizer`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `str` | `"adamw8bit"` | Any optimizer name the backend understands. dp accepts `adamw`, `adamw8bit`, `adamw_optimi`, `stableadamw`, `lion`, `prodigy`, ... |
| `lr.unet` | `float` | `1e-4` | UNet/transformer learning rate. |
| `lr.textEncoder` | `float` | `5e-5` | Text encoder learning rate. |
| `schedule` | `str` | `"cosine_with_restarts"` | LR scheduler name. |
| `warmupSteps` | `int` | `100` | Linear warmup before the schedule. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Adam betas. |
| `weightDecay` | `float` | `0.0` | L2 weight decay. |
| `eps` | `float` | `1e-8` | Optimizer eps. |
| `optimizerArgs` | `dict[str, str]` | `{}` | Free-form `key=value` bag forwarded to the optimizer. |
| `gradientRelease` | `bool` | `false` | dp gradient_release: chunk-wise grad release for memory savings. |

## `schedule`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `epochs` | `int (>=1)` | `10` | Number of training epochs. |
| `batchSize` | `int (>=1)` | `1` | Per-step batch size. |
| `gradAccum` | `int (>=1)` | `2` | Gradient accumulation steps. |
| `maxSteps` | `int \| None` | `None` | Optional global step cap. |

## `sampling`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `enabled` | `bool` | `true` | Generate preview images during training. |
| `everyNEpochs` | `int (>=1)` | `1` | Sample cadence in epochs. |
| `everyNSteps` | `int \| None` | `None` | Sample cadence in steps. Mutually exclusive with `everyNEpochs` for live preview. |
| `promptsFile` | `Path \| None` | `None` | Plain-text prompt file (one per line, kohya format). |
| `resolution` | `list[int]` | `[1024, 1024]` | Sample resolution. |
| `seed` | `int` | `42` | Sampling RNG seed. |
| `enableLiveInference` | `bool` | `false` | Turn on the lorahub-side preview worker (only useful with `backend.type == "diffusion-pipe"`). |
| `inferenceSteps` | `int (>=1)` | `24` | Default sample steps when the prompt line doesn't specify `--s`. |
| `inferenceCfg` | `float (>0)` | `5.0` | Default CFG scale when the prompt line doesn't specify `--l`. |

## `output`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `name` | `str` | `"lora_output"` | Filename stem for saved LoRA files. |
| `saveEveryNEpochs` | `int (>=1)` | `1` | Epoch-cadence checkpoint. |
| `saveEveryNSteps` | `int \| None` | `None` | Step-cadence checkpoint. **Wins over the epoch cadence**: when set, the compiler suppresses the epoch flag so dp / kohya don't double-save on aligned boundaries. |
| `saveDtype` | `Literal["fp16", "bf16", "float"]` | `fp16` | dtype of the persisted weights. |
| `outputDir` | `Path \| None` | `None` | Override for the workspace output directory. |

## `backend`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["kohya", "diffusion-pipe"]` | `kohya` | Training engine to invoke. |
| `pinVersion` | `str \| None` | `None` | Pin the backend version (commit hash). |
| `sdScriptsPath` | `Path \| None` | `None` | Override for the kohya checkout path. |
| `pythonExecutable` | `Path \| None` | `None` | Override for the kohya / dp venv Python. |
| `extraArgs` | `dict[str, Any]` | `{}` | Escape hatch for raw backend CLI flags. |
| `diffusionPipe` | `DiffusionPipeOptions \| None` | `None` | dp-specific knobs (see below). Ignored by kohya. |

## `backend.diffusionPipe`

Only consumed when `backend.type == "diffusion-pipe"`. Highlights:

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `pipelineStages` | `int (>=1)` | `1` | DeepSpeed pipeline parallelism degree. |
| `gradientClipping` | `float (>0)` | `1.0` | Grad-norm clip value. |
| `cachingBatchSize` | `int (>=1)` | `1` | Batch size while caching latents/text embeddings. |
| `stepsPerPrint` | `int (>=1)` | `1` | DeepSpeed log cadence. |
| `blocksToSwap` | `int (>=0)` | `0` | Blocks offloaded to RAM (0 = disabled). |
| `compile` | `bool` | `false` | **Keep off.** dp's pipeline already compiles internally; turning this on spawns an Inductor compile pool that has hung in `unix_stream_data_wait` at the tail of the cache pass on Anima runs. |
| `minAr` | `float (>0)` | `0.5` | Minimum aspect ratio bucket. |
| `maxAr` | `float (>0)` | `2.0` | Maximum aspect ratio bucket. |
| `numArBuckets` | `int (>=1)` | `7` | Number of aspect-ratio buckets. **Biggest knob for cache disk usage** — each bucket stores its own VAE-encoded copy of every image, so 200 images × 11 buckets fills ~35 GB of latent cache before training even starts. The Anima recipe ships with `5`. |
| `cacheShuffleNum` | `int (>=0)` | `0` | Shuffle the first N tags during caching (0 keeps order). |
| `skipEmptyCaption` | `bool` | `true` | Skip images without caption files; `false` trains them with an empty caption. |
| `checkpointEveryNMinutes` | `int (>=1) \| None` | `None` | Wall-clock cadence for **full DeepSpeed checkpoints** (optimizer state + LR scheduler step + dataloader epoch + a `latest` pointer). Required for `POST /api/jobs/{id}/resume` to work — `output.saveEveryNSteps` only writes the LoRA weights, not the optimiser state. The Anima recipes ship with `30`. |
| `checkpointEveryNEpochs` | `int (>=1) \| None` | `None` | Same as above but anchored on epoch boundaries. Use only one of the two (mins is usually preferable since it survives long single-epoch runs). |
| `modelPaths` | `dict[str, str]` | `{}` | Free-form per-arch path bag. **Keys are written verbatim to the dp TOML**, so they keep upstream's literal `snake_case` names (`transformer_path`, `vae_path`, `llm_path`). |

`wandb_api_key` is intentionally absent from the config; diffusion-pipe
reads `$WANDB_API_KEY` directly so secrets stay out of the on-disk TOML.

## `resume`

`resume` controls checkpoint state writing for resume support. When
`saveState == true`, kohya writes optimizer + scheduler state next to the
safetensors so a later run can pick up exactly where the interrupted one
left off. State directories are large; use `saveStateEveryNEpochs` to
throttle writes if disk is tight.

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `saveState` | `bool` | `true` | Write optimizer/scheduler state at every checkpoint. |
| `saveStateAtEnd` | `bool` | `true` | Force a state write when training ends cleanly. |
| `saveStateEveryNEpochs` | `int \| None` | `None` | Throttle state writes to every N epochs. |
| `resumeFrom` | `Path \| None` | `None` | Local resume path. |
