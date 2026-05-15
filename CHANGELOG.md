# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **HTTP API (v0.2 starter)** — `lorahub.api` module exposing FastAPI routes for `/health`, `/recipes/schema`, `/jobs` (CRUD), `/jobs/{id}/events`, and `/jobs/{id}/stream` (WebSocket). In-process `JobRegistry` keeps live state and a per-job event ring buffer so reconnecting WebSocket clients can replay. Launch with `lorahub serve [--host ... --port ... --reload]`. Install API extras: `pip install lorahub[api]`. Auth/persistence deferred to v0.3+.
- **Auto-emitted `done` events** — `KohyaRunner` now spawns a reaper thread on `start()` so the `done` event fires whenever the child exits, not only when something calls `wait()`. The HTTP API and any fire-and-forget caller now see job completion correctly.
- **`lorahub init --auto`** — probes `nvidia-smi` for VRAM, scans the dataset directory for image count, detects the architecture from the checkpoint filename (SDXL / Flux / SD3 / SD1.5, with IllustriousXL / Pony / NoobAI / Animagine matched as SDXL), and writes a recipe with rank/batch/grad_accum tuned per VRAM tier and num_repeats inversely scaled to dataset size. `--checkpoint` and `--dataset` are required; `--vram-mib` overrides detection. Replaces hand-editing the template recipe.
- **WD14 GPU acceleration** — `WD14Tagger` and `lorahub tag` accept `--device auto/cpu/cuda`. Auto picks `CUDAExecutionProvider` when available, falls back to CPU silently. Explicit `--device cuda` raises an actionable error pointing at `onnxruntime-gpu` install. The CLI prints which provider the session is actually using. `pip install lorahub[gpu]` opts into the GPU runtime.
- **`lorahub bootstrap-kohya`** — one-shot kohya install. Clones `kohya-ss/sd-scripts`, creates a venv, installs PyTorch + torchvision (`--cuda cu121/cu124/cu128`, `--torch X.Y.Z`), runs `pip install -r requirements.txt`, and installs xformers (skip with `--no-xformers`). `--force` wipes a half-installed target. Replaces ~30 minutes of manual command typing.
- **End-to-end smoke test passes** — full path from `fetch-bangumi` -> `tag` -> `train` produces a 21 MB SDXL LoRA file in under 3 minutes on an RTX 4070 Laptop (8 GB VRAM) using IllustriousXL as the base. Verified with 3 BangumiBase images of "laffey (azur lane)" at 512x512, 2 steps.
- **WD14 / WD-v3 auto-tagger** (`lorahub.core.tagging.wd14`) — ONNX-based multi-label image classifier that produces Danbooru-style tags. Lazy-loads the model on first use, supports configurable `general` / `character` thresholds (defaults `0.35` / `0.85`), and writes kohya-style comma-separated `.txt` captions next to each image.
- **CLI: `lorahub tag`** — auto-tag a directory of images. Skips images that already have non-empty captions unless `--overwrite` is passed; supports `--recursive`, `--include-character / --no-include-character`, and threshold overrides.
- **BangumiBase fetcher** (`lorahub.core.dataset.sources.bangumi_base`) — pulls a single character's image set from a Hugging Face BangumiBase dataset, unpacks `dataset.zip`, and seeds empty caption files for kohya tag-file mode.
- **CLI: `lorahub fetch-bangumi`** — list characters, download previews, or grab a character's full image set with `--limit` capping.
- **`.env` auto-loading** — the CLI reads variables from a `.env` file in the working directory on startup, so `LORAHUB_KOHYA_SD_SCRIPTS` and similar settings persist without exports. `.env.example` ships as a template.
- **kohya venv auto-detection** — `bootstrap.resolve()` now finds `<sd-scripts>/venv/Scripts/python.exe` (Windows) or `<sd-scripts>/venv/bin/python` (Unix) automatically, matching kohya's official install layout.

### Changed

- **Recipe -> kohya translation now goes through `dataset.toml`** instead of `--train_data_dir` plus a slew of resolution/bucket/caption flags. kohya's flat-dir `--train_data_dir` mode requires `<n>_<concept>/` subdirectories, which the recipe schema doesn't model; switching to the TOML config lets users keep flat directories and works with `num_repeats` directly. `compile_recipe()` now returns `(script, argv, files_to_write)`.
- `KohyaBackend.launch()` and the `train` CLI now resolve the workspace path to absolute, so kohya output never accidentally lands under the sd-scripts checkout.
- The kohya stdout parser handles checkpoint and sample paths that contain spaces (Windows install paths like `E:\WorkSpace\Lora Scripts\...`) and supports `at`/`as`/`to` keyword separators in addition to colon-prefixed paths.
- CLI uses ASCII status markers (`OK`, `->`) instead of Unicode glyphs for Windows GBK console compatibility.
- Added runtime dependencies: `huggingface_hub`, `onnxruntime`, `pillow`, `numpy`, `python-dotenv`.

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
