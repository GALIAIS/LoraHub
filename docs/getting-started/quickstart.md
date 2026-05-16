---
title: Quick start
description: Scaffold a recipe and run your first training job.
---

# Quick start

Once LoraHub is installed and pointed at a kohya checkout, the path from zero
to a running job is four commands.

```powershell
# 1. Scaffold a recipe
lorahub init my_character

# 2. Edit my_character.yaml: point at your SDXL model and dataset
notepad my_character.yaml

# 3. Sanity check (no training yet)
lorahub validate my_character.yaml
lorahub info     my_character.yaml

# 4. Train
lorahub train    my_character.yaml
```

## A minimal recipe

```yaml
schema_version: "1.0"
base_model:
  arch: sdxl
  checkpoint: ./models/sdxl_base_1.0.safetensors
dataset:
  source: ./datasets/my_character
  resolution: [1024, 1024]
network:
  type: lora
  rank: 32
  alpha: 16
schedule:
  epochs: 10
  batch_size: 1
  grad_accum: 4
precision: bf16
gradient_checkpointing: true
output:
  name: my_character_v1
backend:
  type: kohya
```

See
[`recipes/sdxl_character_8gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/recipes/sdxl_character_8gb.yaml)
for a fully annotated example.

## Tune to your machine

`lorahub init --auto` probes `nvidia-smi` for VRAM, scans your dataset
directory, detects the architecture from the checkpoint filename, and writes a
recipe with rank/batch/grad_accum tuned per VRAM tier:

```powershell
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

Use `--vram-mib 8192` to override detection.

## What `lorahub info` shows

`lorahub info` is a dry run: it compiles the recipe to the kohya argv it would
launch, prints the entry script, and estimates VRAM — without touching the
GPU. Useful before a long training session.

## Next

- [Smoke test](smoke-test.md) — full pipeline from BangumiBase images to a
  trained LoRA.
- [Recipe field reference](../recipes/fields.md) — every knob in the schema.
