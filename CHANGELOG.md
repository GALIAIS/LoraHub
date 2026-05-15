# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **WD14 / WD-v3 auto-tagger** (`lorahub.core.tagging.wd14`) — ONNX-based multi-label image classifier that produces Danbooru-style tags. Lazy-loads the model on first use, supports configurable `general` / `character` thresholds (defaults `0.35` / `0.85`), and writes kohya-style comma-separated `.txt` captions next to each image.
- **CLI: `lorahub tag`** — auto-tag a directory of images. Skips images that already have non-empty captions unless `--overwrite` is passed; supports `--recursive`, `--include-character / --no-include-character`, and threshold overrides.
- **BangumiBase fetcher** (`lorahub.core.dataset.sources.bangumi_base`) — pulls a single character's image set from a Hugging Face BangumiBase dataset, unpacks `dataset.zip`, and seeds empty caption files for kohya tag-file mode.
- **CLI: `lorahub fetch-bangumi`** — list characters, download previews, or grab a character's full image set with `--limit` capping.
- **`.env` auto-loading** — the CLI reads variables from a `.env` file in the working directory on startup, so `LORAHUB_KOHYA_SD_SCRIPTS` and similar settings persist without exports. `.env.example` ships as a template.
- **kohya venv auto-detection** — `bootstrap.resolve()` now finds `<sd-scripts>/venv/Scripts/python.exe` (Windows) or `<sd-scripts>/venv/bin/python` (Unix) automatically, matching kohya's official install layout.

### Changed

- CLI uses ASCII status markers (`OK`, `->`) instead of Unicode glyphs for Windows GBK console compatibility.
- Added runtime dependencies: `huggingface_hub`, `onnxruntime`, `pillow`, `numpy`.

## [0.1.0-dev] - 2026-05-15

Initial pre-alpha release. CLI tracer bullet: from a YAML recipe to a trained LoRA file via kohya-ss/sd-scripts.

### Added

- **Recipe schema** (`lorahub.core.config.schema`) — Pydantic v2 semantic configuration covering `base_model`, `dataset`, `network`, `optimizer`, `schedule`, `precision`, `sampling`, `output`, and `backend`. Defaults tuned for SDXL on 8GB VRAM.
- **Recipe loader** (`lorahub.core.config.loader`) — YAML round-trip plus JSON Schema export for future UI form generation.
- **Event bus** (`lorahub.core.events`) — `TrainingEvent`, `EventType`, thread-safe `EventBus`, and `JsonlEventSink` for durable replay.
- **Backend protocol** (`lorahub.core.backends.base`) — `TrainingBackend` `Protocol`, `TrainingHandle` with `stop()`/`wait()`, plus `ValidationIssue` and `VRAMEstimate`.
- **KohyaBackend** (`lorahub.core.backends.kohya`):
  - `bootstrap` resolves an existing kohya checkout via recipe field, env var, or default user-data path.
  - `compiler` translates a `RecipeConfig` into the `(script, argv)` pair that `sd-scripts` expects. Covers SDXL/SD1.5/Flux/SD3 entry scripts, bucket/caption/lora variants, optimizer/precision/sampling/output flags, and an `extra_args` escape hatch.
  - `parser` recognizes tqdm progress bars (steps + `avr_loss`), epoch markers, checkpoint saves, sample image generation, and tracebacks/CUDA errors.
  - `runner` owns the child process plus stdout/stderr pump threads. Stop is platform-aware (`CTRL_BREAK_EVENT` on Windows, `SIGINT` elsewhere) with terminate/kill escalation.
- **CLI** (`lorahub.cli.main`) — `init`, `validate`, `info`, `train`, `version` commands powered by typer + rich.
- **Built-in recipe** — `recipes/sdxl_character_8gb.yaml` for character LoRAs on 8GB VRAM cards.
- **Test suite** — 62 tests covering schema, events, compiler, parser, bootstrap, runner, backend, and CLI.

### Known limitations

- No automatic clone/install of kohya-ss; you must point at an existing checkout.
- No Web UI yet — planned for v0.2.
- VRAM estimator is a coarse first-pass approximation.
- End-to-end smoke test against a real SDXL model is not yet automated.

[Unreleased]: https://github.com/GALIAIS/LoraHub/compare/v0.1.0-dev...HEAD
[0.1.0-dev]: https://github.com/GALIAIS/LoraHub/releases/tag/v0.1.0-dev
