# LoraHub recipes

Built-in recipe library. Each recipe is a self-contained YAML file documenting a training configuration that has been validated end-to-end.

## Naming convention

```
<arch>_<purpose>_<vram-tier>.yaml
```

Examples:
- `sdxl_character_8gb.yaml` — character LoRA on SDXL, fits 8GB VRAM
- `sdxl_style_12gb.yaml` — style LoRA on SDXL, requires 12GB VRAM
- `sd15_concept_6gb.yaml` — concept LoRA on SD1.5, fits 6GB VRAM

## Status

v0.1 ships with one tracer-bullet recipe: `sdxl_character_8gb.yaml`.
