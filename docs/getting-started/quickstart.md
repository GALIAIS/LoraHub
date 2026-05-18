---
title: Quick start
description: Scaffold a config and run your first training job.
---

# Quick start

Once LoraHub is installed and a backend is on disk, the path from zero to a
running job is four commands.

```powershell
# 1. Scaffold a config
lorahub init my_character

# 2. Edit configs/my_character.yaml: point at your checkpoint and dataset
notepad configs/my_character.yaml

# 3. Sanity check (no training yet)
lorahub validate configs/my_character.yaml
lorahub info     configs/my_character.yaml

# 4. Train
lorahub train    configs/my_character.yaml
```

## A minimal config

Configs use camelCase on the wire (the validator still accepts the legacy
snake_case so old files keep loading):

```yaml
schemaVersion: "1.0"
baseModel:
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
  batchSize: 1
  gradAccum: 4
precision: bf16
gradientCheckpointing: true
output:
  name: my_character_v1
backend:
  type: kohya
```

See [`configs/sdxl_character_8gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/sdxl_character_8gb.yaml)
for a fully annotated example, and the bundled Anima configs at
[`configs/anima_style_24gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/anima_style_24gb.yaml)
and `configs/anima_character_24gb.yaml` for the diffusion-pipe path.

## Tune to your machine

`lorahub init --auto` probes `nvidia-smi` for VRAM, scans your dataset
directory, detects the architecture from the checkpoint filename, and writes a
config with rank / batch / grad_accum tuned per VRAM tier:

```powershell
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

Use `--vram-mib 8192` to override detection.

## What `lorahub info` shows

`lorahub info` is a dry run: it compiles the config to the backend argv it
would launch (kohya CLI flags, or a diffusion-pipe TOML), prints the entry
script, and estimates VRAM — without touching the GPU. Useful before a long
training session.

## Next

- [Smoke test](smoke-test.md) — full pipeline from BangumiBase images to a
  trained LoRA.
- [Config field reference](../recipes/fields.md) — every knob in the schema.
