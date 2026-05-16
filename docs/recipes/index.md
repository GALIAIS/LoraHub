---
title: Recipes
description: The RecipeConfig schema is the single semantic description of a LoRA training run.
---

# Recipes

A *recipe* is a single YAML file that fully describes a LoRA training run.
LoraHub validates the file against the `RecipeConfig` Pydantic model, then a
backend-specific compiler translates it into the native arguments of the
chosen training engine (currently kohya-ss/sd-scripts).

The schema's job is to stay **semantic** — users describe *what* they want to
train, not *how* the backend should be invoked. New backends can plug in
without changing recipe files.

## Top-level structure

```yaml
schema_version: "1.0"

base_model:    # which checkpoint and architecture
dataset:       # where the images live, resolution, bucket, captions
network:       # LoRA / LoCon / LoHa / DoRA shape (rank, alpha, targets)
optimizer:     # type, learning rates, schedule, warmup
schedule:      # epochs, batch size, gradient accumulation
precision:     # fp16 / bf16 / fp32
sampling:      # optional preview images during training
output:        # filename, save cadence, dtype
backend:       # which training engine + extra_args escape hatch
resume:        # optimizer/scheduler state for resume support
```

Every section has tuned defaults aimed at SDXL on 8 GB VRAM, so a minimal
recipe only needs `base_model.checkpoint` and `dataset.source`.

## How a recipe becomes a training run

1. **Load** — `load_recipe(path)` parses YAML, applies defaults, validates
   types, and returns a `RecipeConfig`.
2. **Compile** — `compile_recipe(cfg, workspace)` returns
   `(script, argv, files_to_write)` — the kohya entry script, its argv list,
   and any auxiliary files (`dataset.toml`, sample prompts) to write into the
   workspace.
3. **Launch** — `KohyaBackend.launch()` spawns the compiled command, parses
   stdout into `TrainingEvent`s, and persists them to `events.jsonl` next to
   the checkpoints.

## Where to look next

- [Templates](templates.md) — the four built-in starting points and their
  fill-in placeholders.
- [Field reference](fields.md) — every knob in the schema, grouped by
  section.
