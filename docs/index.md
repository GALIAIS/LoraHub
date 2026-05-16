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
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

**An open-source LoRA training workbench for diffusion models** — data,
training, evaluation, and recipes in one workflow.

LoraHub wraps mature training backends (currently
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)) behind a stable,
semantic configuration layer and a unified CLI / API. The goal is to make LoRA
training reproducible, recipe-driven, and tool-agnostic.

```
+-----------------------------------------+
|  CLI  /  Web UI (v0.2+)                 |
+-----------------+-----------------------+
                  | RecipeConfig + events
+-----------------v-----------------------+
|  Core: schema · backends · events       |
+-----------------+-----------------------+
                  | subprocess + JSON-RPC
+-----------------v-----------------------+
|  KohyaBackend  ·  DiffusersBackend (v0.7)
+-----------------------------------------+
```

## Status

**Pre-alpha (v0.2 workbench).** What works today:

- Semantic recipe schema (Pydantic) → kohya argv compiler
- KohyaBackend with subprocess management, SQLite job history, and event streaming
- CLI: `init`, `bootstrap-kohya`, `fetch-bangumi`, `tag`, `validate`, `info`, `train`, `serve`, `version`
- FastAPI server with settings, recipe browsing/editing, job CRUD, and WebSocket streams
- React web UI for dashboard, jobs, recipes, visual recipe editing, and workbench settings
- 140+ tests covering schema, compiler, parser, runner, backend, API, store, CLI, and tagger

!!! warning "Not yet"
    - Dataset management UI: import, thumbnails, and caption editor
    - Web auto-tagging workflow on top of the existing WD14 CLI/tagger
    - Job queue, multi-GPU scheduling, and resume orchestration
    - DiffusersBackend and non-kohya training backends

## Quick links

- [Install](getting-started/install.md) — system requirements, `pip install`, kohya bootstrap
- [Quick start](getting-started/quickstart.md) — first recipe → first training run in four commands
- [Smoke test](getting-started/smoke-test.md) — full path from images to trained LoRA
- [Recipes overview](recipes/index.md) — RecipeConfig structure and field reference
- [CLI reference](cli/index.md) — every `lorahub` command with one example
- [API reference](api/index.md) — REST + WebSocket endpoints for the web UI
- [Roadmap](roadmap.md) — what is shipping next

## License

Apache License 2.0. See [LICENSE](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE).
