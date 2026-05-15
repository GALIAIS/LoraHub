# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BangumiBase fetcher** (`lorahub.core.dataset.sources.bangumi_base`) — pulls a single character's image set from a Hugging Face BangumiBase dataset, unpacks `dataset.zip`, and seeds empty caption files for kohya tag-file mode.
- **CLI: `lorahub fetch-bangumi`** — list characters, download previews, or grab a character's full image set with `--limit` capping.

### Changed

- CLI uses ASCII status markers (`OK`, `->`) instead of Unicode glyphs for Windows GBK console compatibility.
- Added `huggingface_hub>=0.24` as a runtime dependency.

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
