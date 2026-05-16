---
title: Recipe templates
description: Built-in starting points and the placeholder system that fills them in.
---

# Recipe templates

LoraHub ships four built-in templates as plain YAML under
[`recipes/builtin/`](https://github.com/GALIAIS/LoraHub/tree/main/recipes/builtin).
The web UI's recipe wizard reads them through `GET /api/recipes/templates`,
shows a card per template, and lets the user fill in just the placeholders
before saving the new recipe under `recipes/<name>.yaml`.

## Catalogue

| Template | Architecture | Network | Notes |
| -------- | ------------ | ------- | ----- |
| `sdxl_character` | SDXL | LoRA, rank 32 / alpha 16 | 8 GB VRAM friendly, 1024 px, 10 epochs, UNet only |
| `sdxl_style` | SDXL | LoRA, rank 16 / alpha 8 | Trains text encoder, 20 epochs, lower repeats |
| `sd15_character` | SD 1.5 | LoRA, rank 16 / alpha 8 | 768 px, fp16 for older GPUs |
| `blank` | minimal | minimal | Empty starting point for fully custom recipes |

## Placeholder format

Each template YAML may carry two optional top-level metadata blocks that the
schema strips before validating against `RecipeConfig`:

- `_template` — UI card metadata (`name`, `description`, `arch`).
- `_placeholders` — list of fields the user must supply when instantiating
  the template, each with `key`, `label`, `path_field`, `placeholder`.

Example placeholder block from `sdxl_character.yaml`:

```yaml
_placeholders:
  - key: name
    label: Recipe / output name
    path_field: output.name
    placeholder: my_character_v1
  - key: checkpoint
    label: SDXL base model checkpoint
    path_field: base_model.checkpoint
    placeholder: C:\models\sdxl_base.safetensors
  - key: dataset
    label: Dataset directory
    path_field: dataset.source
    placeholder: ./datasets/my_character
```

The wizard renders one form field per placeholder, then `POST
/api/recipes/templates/{template_id}/instantiate` deep-merges the values into
the template body, validates the result, and writes it to
`recipes/<name>.yaml`.

## Adding a new template

1. Drop a new YAML file into `recipes/builtin/`.
2. Add a `_template` block (otherwise the file stem is used as the name).
3. List every required path or label in `_placeholders`.
4. Restart the API server. Bad templates are logged and skipped — a typo in
   one file can't take the catalogue down.

## CLI shortcut

`lorahub init <name>` copies one of these templates from disk:

```powershell
lorahub init my_character                          # default: sdxl_character_8gb
lorahub init my_style --template sdxl_style        # picks recipes/sdxl_style.yaml
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character           # tunes to detected VRAM
```
