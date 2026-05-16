---
title: Roadmap
description: LoraHub release plan and current progress.
---

# Roadmap

LoraHub releases ship in tight, scoped slices. Each version is a working
end-to-end deliverable rather than a feature dump.

| Version | Scope | Status |
| ------- | ----- | ------ |
| v0.1 | CLI tracer bullet: recipe → kohya → LoRA file | :white_check_mark: shipped |
| v0.2 | FastAPI + minimal React UI, recipe editor, settings, job monitor | :white_check_mark: shipped |
| v0.3 | Dataset module: import, thumbnails, caption editor | :white_check_mark: shipped |
| v0.4 | Auto-taggers: WD14, JoyTag | :white_check_mark: shipped |
| v0.5 | Job queue + multi-GPU + resume from checkpoint | :white_check_mark: shipped |
| v0.6 | Recipe library + sample image gallery | :white_check_mark: shipped |
| v0.7 | SD1.5 + Pony/Illustrious; DiffusersBackend (self-written) starts | :hourglass_flowing_sand: in progress |
| v0.8 | Flux / SD3 support | :white_large_square: planned |
| v1.0 | Hyperparameter sweeps, overfit detection, docs site | :white_large_square: planned |

## What's next

The current focus is the v0.7 line: bringing DiffusersBackend up to parity
with KohyaBackend so SDXL sub-variants (Pony, Illustrious, NoobAI, Animagine)
have a non-kohya path, and broadening SD 1.5 support beyond the existing
template.

The v1.0 cut adds hyperparameter sweeps and overfit detection on top of the
job queue, plus the public documentation site (this site).

## Tracking work

- Day-to-day changes are recorded in
  [`CHANGELOG.md`](https://github.com/GALIAIS/LoraHub/blob/main/CHANGELOG.md).
- Larger pieces of work are tracked as GitHub Issues at
  [github.com/GALIAIS/LoraHub/issues](https://github.com/GALIAIS/LoraHub/issues).
