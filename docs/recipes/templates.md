---
title: Config templates
description: Built-in starting points and the placeholder system that fills them in.
---

# Config templates

LoraHub ships several built-in templates as plain YAML under
[`configs/`](https://github.com/GALIAIS/LoraHub/tree/main/configs) (the old
`configs/builtin/` subtree was promoted to top-level). The web UI's config
wizard reads them through `GET /api/configs/templates`, shows a card per
template, and lets the user fill in just the placeholders before saving the
new config under `configs/<name>.yaml`.

## Catalogue

| Template | Architecture | Network | Notes |
| -------- | ------------ | ------- | ----- |
| `sdxl_character_8gb` | SDXL | LoRA, rank 32 / alpha 16 | 8 GB VRAM friendly, 1024 px, 10 epochs, UNet only |
| `anima_style_24gb` | Anima (dp) | LoRA, rank 16 / alpha 8 | Style LoRA on a 4090 / 24 GB card; 200-step checkpoints; live preview enabled |
| `anima_character_24gb` | Anima (dp) | LoRA, rank 32 / alpha 16 | Character LoRA on a 4090 / 24 GB card; 200-step checkpoints |

The Anima configs are the canonical example of the diffusion-pipe path —
they wire transformer + Qwen-Image VAE + Qwen3-0.6B text encoder together
and turn on the lorahub live-preview worker.

## Default sample prompt set

`configs/sample_prompts/anima_default.txt` ships with both Anima recipes —
8 prompts covering portrait / cowboy shot / full body / group / scene /
wide landscape, with `@Kiko.L` baked in. Edit the file in place; the live
preview worker re-reads it at the next checkpoint.

## Placeholder format

Each template YAML may carry two optional top-level metadata blocks that the
schema strips before validating against `TrainingConfig`:

- `_template` — UI card metadata (`name`, `description`, `arch`).
- `_placeholders` — list of fields the user must supply when instantiating
  the template, each with `key`, `label`, `path_field`, `placeholder`.

Example placeholder block:

```yaml
_placeholders:
  - key: name
    label: Config / output name
    path_field: output.name
    placeholder: my_character_v1
  - key: checkpoint
    label: SDXL base model checkpoint
    path_field: baseModel.checkpoint
    placeholder: C:\models\sdxl_base.safetensors
  - key: dataset
    label: Dataset directory
    path_field: dataset.source
    placeholder: ./datasets/my_character
```

The wizard renders one form field per placeholder, then `POST
/api/configs/templates/{template_id}/instantiate` deep-merges the values into
the template body, validates the result, and writes it to
`configs/<name>.yaml`.

## Adding a new template

1. Drop a new YAML file into `configs/`.
2. Add a `_template` block (otherwise the file stem is used as the name).
3. List every required path or label in `_placeholders`.
4. Restart the API server. Bad templates are logged and skipped — a typo
   in one file can't take the catalogue down.

## CLI shortcut

`lorahub init <name>` copies one of these templates from disk:

```powershell
lorahub init my_character                          # default: sdxl_character_8gb
lorahub init my_style --template anima_style_24gb  # picks configs/anima_style_24gb.yaml
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character           # tunes to detected VRAM
```
