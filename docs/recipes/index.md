---
title: Configs
description: The TrainingConfig schema is the single semantic description of a LoRA training run.
---

# Configs

A *config* is a single YAML file that fully describes a LoRA training run.
LoraHub validates the file against the `TrainingConfig` Pydantic model, then
a backend-specific compiler translates it into the native arguments of the
chosen training engine (currently kohya-ss/sd-scripts or
tdrussell/diffusion-pipe).

The schema's job is to stay **semantic** — users describe *what* they want
to train, not *how* the backend should be invoked. New backends can plug in
without changing config files.

!!! note "Recipe → config rename"
    Earlier docs and code paths called these files "recipes" and stored
    them under `recipes/`. The on-disk directory is now `configs/` and the
    REST endpoints live under `/api/configs`. The Python type is still
    called `TrainingConfig` (it has always been). The aliases were updated
    in lockstep — old code paths that still spelt `recipe` survive only as
    abstract noun in comments and docstrings.

## Top-level structure

```yaml
schemaVersion: "1.0"

baseModel:        # which checkpoint and architecture
dataset:          # where the images live, resolution, bucket, captions
network:          # LoRA / LoCon / LoHa / DoRA shape (rank, alpha, targets)
optimizer:        # type, learning rates, schedule, warmup
schedule:         # epochs, batchSize, gradAccum
precision:        # fp16 / bf16 / fp32
sampling:         # optional preview images during training
output:           # filename, save cadence, dtype
backend:          # which training engine + extra_args escape hatch
resume:           # optimizer/scheduler state for resume support
```

Every section has tuned defaults aimed at SDXL on 8 GB VRAM, so a minimal
config only needs `baseModel.checkpoint` and `dataset.source`.

## camelCase / snake_case

The Pydantic schema applies a `to_camel` alias generator with
`populate_by_name=True`, so the validator accepts **either** form. New
configs emit `camelCase` (matches the front-end form fields and the
`camelCase` API wire format); legacy `snake_case` configs round-trip cleanly.

```yaml
# Both forms validate to the same TrainingConfig.
schemaVersion: "1.0"   # or schema_version
baseModel:             # or base_model
  arch: sdxl
  checkpoint: ./model.safetensors
schedule:
  batchSize: 2         # or batch_size
  gradAccum: 2         # or grad_accum
```

Two values stay literal regardless:

- `caption.strategy: tag_file` — Literal value, not a field name.
- `backend.diffusionPipe.modelPaths.transformer_path` /
  `vae_path` / `llm_path` — keys are passed verbatim to the diffusion-pipe
  TOML, which expects upstream's snake_case names.

## How a config becomes a training run

1. **Load** — `load_config(path)` parses YAML, applies defaults, validates
   types, and returns a `TrainingConfig`.
2. **Path normalisation** — at job launch every recipe-relative path
   (checkpoint, vae, archPaths.*, dataset.source, modelPaths.*, init_from,
   prompts_file, ...) is absolutised against the project root so the
   training subprocess can find them regardless of which cwd the backend
   chdirs into.
3. **Compile** — `compile_config(cfg, workspace)` returns the entry argv
   plus a dict of files to write into the workspace (`dataset.toml`,
   diffusion-pipe TOML, sample prompts, ...).
4. **Launch** — the selected backend spawns the compiled command, parses
   stdout into `TrainingEvent`s, and persists them to `events.jsonl` next
   to the checkpoints. SSE / WS streams replay the file on reconnect.

## Where to look next

- [Templates](templates.md) — the built-in starting points and the
  fill-in placeholders.
- [Field reference](fields.md) — every knob in the schema, grouped by
  section.
