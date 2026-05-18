---
title: Roadmap
description: LoraHub release plan and current progress.
---

# Roadmap

LoraHub releases ship in tight, scoped slices. Each version is a working
end-to-end deliverable rather than a feature dump.

| Version | Scope | Status |
| ------- | ----- | ------ |
| v0.1 | CLI tracer bullet: recipe -> kohya -> LoRA file | shipped |
| v0.2 | FastAPI + React UI, config editor, settings, job monitor | shipped |
| v0.3 | Dataset module: import, thumbnails, caption editor | shipped |
| v0.4 | Auto-taggers: WD14, JoyTag | shipped |
| v0.5 | Job queue + multi-GPU + resume from checkpoint | shipped |
| v0.6 | Config library + sample image gallery | shipped |
| v0.7 | SD1.5 + Pony/Illustrious sub-variants | shipped |
| v0.8 | Flux / SD3 / diffusion-pipe backend (21 architectures) | shipped |
| v0.9 | SSE event streams, camelCase config schema, Image Studio | shipped |
| v0.10 | Live preview (Anima), AI training analysis, GPU resource trends | shipped |
| v1.0 | Hyperparameter sweeps v2, multi-node, public docs site | in progress |

## What's next

The current focus is the v1.0 line:

- **Hyperparameter sweeps v2** — the existing `SweepPlan` grid works but
  lacks Bayesian search and early stopping. v1.0 adds Optuna integration
  and a sweep dashboard with Pareto-front visualisation.
- **Multi-node training** — DeepSpeed ZeRO-3 and pipeline parallelism are
  already wired in diffusion-pipe; the missing piece is a multi-machine
  launcher and a node-health dashboard.
- **Public docs site** — this mkdocs-material site ships with the repo;
  v1.0 publishes it to GitHub Pages with versioned snapshots.

Recent v0.9/v0.10 highlights that landed:

- SSE event streams with `Last-Event-ID` resume (WebSocket kept as fallback).
- camelCase config schema (snake_case still accepted by the validator).
- Image Studio with AI smart-caption (WD14 + VLM), deduplication, and
  similarity scanning.
- Live preview for diffusion-pipe training (Anima): event-driven checkpoint
  watcher, PEFT-to-kohya LoRA conversion, GPU budget control.
- AI training analysis endpoint with loss-curve interpretation.
- Run-summary card with GPU resource trends and overfit signal.

## Tracking work

- Day-to-day changes are recorded in
  [`CHANGELOG.md`](https://github.com/GALIAIS/LoraHub/blob/main/CHANGELOG.md).
- Larger pieces of work are tracked as GitHub Issues at
  [github.com/GALIAIS/LoraHub/issues](https://github.com/GALIAIS/LoraHub/issues).
