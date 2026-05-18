---
title: LoraHub
description: Open-source LoRA training workbench for diffusion models.
hide:
  - navigation
---

# LoraHub

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

**Open-source LoRA training workbench for diffusion models** — datasets,
captioning, training, live previews, and analysis in one workflow.

LoraHub wraps two production training backends —
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and
[tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) — behind
a stable, semantic configuration layer and a unified CLI / API / web UI. The
goal is to make LoRA training reproducible, recipe-driven, and tool-agnostic.

```
+---------------------------------------------------+
|  CLI  /  React web UI                             |
+----------------------+----------------------------+
                       | TrainingConfig + SSE events
+----------------------v----------------------------+
|  Core: schema · backends · events · inference     |
+----------------------+----------------------------+
                       | subprocess + JSON
+----------------------v----------------------------+
|  KohyaBackend  ·  DiffusionPipeBackend            |
+---------------------------------------------------+
```

## Status

**Alpha — actively used for real LoRA work.**

What works today:

- React + FastAPI workbench: Dashboard, Jobs, Configs, Datasets, Image
  Studio, Sample Gallery, Sweeps, Settings (zh-CN UI). Live event streams
  over SSE with browser-native reconnect + `Last-Event-ID` resume; legacy
  WebSocket endpoints kept as fallback.
- Two training backends behind a single config schema:
    - **kohya** — 8 archs (SD1.5, SDXL, SD3, Flux, Lumina, HunyuanImage, Anima)
      with `arch_variant` for SDXL flavours (Pony, Illustrious, NoobAI,
      Animagine).
    - **diffusion-pipe** — 21 archs including Flux2, Chroma, HiDream,
      OmniGen2, AuraFlow, Qwen-Image, Cosmos, Cosmos Predict2, Wan, LTX,
      LTX2, HunyuanVideo, HunyuanVideo 1.5, HunyuanImage, Z-Image,
      ErnieImage, plus Anima (routed through the cosmos_predict2 pipeline
      upstream).
- Anima Base full pipeline: model downloader, paired transformer +
  Qwen-Image VAE + Qwen3-0.6B text encoder configs, training, and live
  preview rendering between checkpoints.
- Image Studio dataset manager: virtualized grid, multi-select,
  drag-and-drop upload, AR-bucket caption strategies (style / character /
  general), inline VLM smart caption (WD14 EVA02 + vision LLM),
  perceptual-hash de-duplication, batch quality scoring, trash + restore.
- Visual config editor exposes every advanced field; YAML uses camelCase on
  the wire and the validator still accepts legacy snake_case.
- Job runtime: per-slot `CUDA_VISIBLE_DEVICES`, checkpoint resume, automatic
  SSE event replay, GPU sampler thread, AI training analysis (Claude reads
  metrics + config, returns Markdown diagnosis), collapsible run-summary
  card on the job detail page.
- Live preview during diffusion-pipe training — lorahub watches
  `output/step{N}/` directories and renders one PNG per prompt for every
  new checkpoint via subprocess to sd-scripts' Anima inference. Stub
  fallback when sd-scripts isn't available.
- One-click bootstrap for both backends, uv-based dependency installs,
  portable CPython runtime, HF/ModelScope downloader, PyPI mirror probing.
- 778 tests covering schema, compilers, parsers, runners, API routers,
  scheduler, sweeps, taggers, captions, inference preview, and CLI.

!!! warning "Not yet"
    - Random / Bayesian sweep strategies on top of the existing grid expander.
    - Embedded Weights & Biases dashboard.
    - CI that runs an end-to-end LoRA training (currently only unit /
      integration tests).
    - Optional auth / multi-user mode for the API.

## Quick links

- [Install](getting-started/install.md) — system requirements, `pip install`,
  backend bootstrap.
- [Quick start](getting-started/quickstart.md) — first config → first
  training run in four commands.
- [Smoke test](getting-started/smoke-test.md) — full path from images to
  trained LoRA.
- [Configs overview](recipes/index.md) — `TrainingConfig` structure and
  field reference.
- [CLI reference](cli/index.md) — every `lorahub` command with one example.
- [API reference](api/index.md) — REST + SSE + WebSocket endpoints.
- [Roadmap](roadmap.md) — what is shipping next.

## License

Apache License 2.0. See [LICENSE](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE).
