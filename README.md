# LoraHub

> An open-source LoRA training workbench for diffusion models — data, training, evaluation, and recipes in one workflow.

LoraHub wraps mature training backends (kohya-ss/sd-scripts) behind a stable, semantic configuration layer and a unified workflow. Start with SDXL on Windows; SD1.5 / Pony / Flux / SD3 to follow.

## Status

**Pre-alpha.** v0.1 is a CLI-only tracer bullet that takes a recipe YAML and produces a LoRA file via kohya-ss.

## Project layout

```
lorahub/
├── core/                  Python library (CLI + future Web share this)
│   ├── config/            Recipe schema + YAML loader + compiler
│   ├── backends/          Training backend abstraction
│   ├── dataset/           Dataset management (v0.3+)
│   ├── tagging/           Auto-taggers (v0.4+)
│   └── events.py          Structured training events
├── cli/                   typer-based command line
├── api/                   FastAPI surface (v0.2+)
├── web/                   Vue3 UI (v0.2+)
├── recipes/               Built-in recipe library
├── tests/
└── docs/
```

## Quick start

Requires Python 3.11+ on Windows 10/11 with an NVIDIA GPU. Linux support is on the roadmap.

```powershell
pip install -e .
lorahub init my_first_lora
lorahub train recipes/sdxl_character_8gb.yaml
```

## License

Apache License 2.0. See `LICENSE`.
