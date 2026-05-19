# 更新日志

本文件记录项目所有值得关注的变更。

## 格式

格式遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)。每个版本固定使用五个小节：Added、Changed、Fixed、Removed、Tests。

不变规则：

- 已发布的 entry 不修改、不删除；笔误只在 Unreleased 段补充说明。
- 一条 entry 描述一项独立变更，并列出受影响的模块或文件。
- 已发布版本的 entry 用英文记录（保持与 commit message 一致）；本节顶部说明使用中文。

## [Unreleased]

## [0.3.0] - 2026-05-18

### Added

- SSE event channels at `/api/jobs/{id}/sse`, `/api/backend/bootstrap/sse`, and `/api/system/sse`, with monotone sequence IDs and `Last-Event-ID` resume; WebSocket endpoints retained as fallback.
- Live preview for diffusion-pipe: `PreviewWorker` watches checkpoint directories, converts PEFT LoRA to kohya format, and runs Anima inference subprocesses with a per-render budget capped at `min(300s, 0.3 * delta_since_last)`.
- AI training analysis on `POST /api/jobs/{id}/analyze`, returning Markdown notes on loss curves, learning-rate schedules, and overfit signals through the AI router.
- GPU resource trends in `/api/jobs/{id}/metrics`, exposing aggregated `gpu_samples` for VRAM and utilisation charts.
- Run-summary card on the job detail page: collapsible single-line digest of step / loss / drop% / convergence trend; expanded view shows progress, loss, and the hyperparameter snapshot.
- Event timeline: dual-pane milestone rail plus a detail pane, replacing the flat log list.
- Image Studio: dedicated dataset manager with virtualized grid, multi-select, drag-and-drop upload, and batch endpoints for smart-caption, quality scoring, deduplication, and similarity scan.
- Smart caption pipeline combining WD14 with a vision LLM, in `style` / `character` / `general` modes, with trigger-word injection and style-tag stripping.
- Default sample prompts at `configs/sample_prompts/anima_default.txt` (8 prompts: portrait, cowboy, full body, group, scene, landscape).
- Anima config templates: `anima_style_24gb.yaml` and `anima_character_24gb.yaml` for 24 GB cards, with 200-step checkpoints and live preview enabled.
- Network presets: `GET /api/network/presets` and `POST /api/network/probe` suggest network targets from a base-model checkpoint.

### Changed

- camelCase config schema: all config fields now use camelCase as the preferred wire form; `snake_case` remains accepted via Pydantic `populate_by_name`. YAML files, API payloads, and frontend forms all emit camelCase.
- `recipes/` renamed to `configs/`: on-disk directory, REST endpoints (`/api/recipes` -> `/api/configs`), and frontend routes follow the new name. The Python type stays `TrainingConfig`.
- Anima templates default to `saveEveryNSteps: 200` instead of per-epoch saves.
- diffusion-pipe compiler: `optim_dtype` is no longer emitted for quantized optimizers (`adamw8bit`, `lion8bit`, etc.) to avoid a bitsandbytes `TypeError`.
- diffusion-pipe compiler: step-based save cadence suppresses the epoch flag so dp / kohya do not double-save on aligned boundaries.
- Frontend SSE preference: `useJobStream`, `useBootstrapStream`, and `useSystemStream` now use `EventSource` with WebSocket fallback.
- Status bar: top-bar connection indicator now reads "SSE" instead of the stale "WS" label.
- Documentation: full rewrite of README and `docs/` to match the SSE / dp / Anima / Image Studio surface.

### Fixed

- diffusion-pipe `steps: N loss: X` parsing: PipelineEngine stdout was not recognised; added `_STEPS_LOSS_RE` to the dp parser.
- Relative path resolution: dp training subprocess could not find model files because paths were relative to the project root rather than the dp checkout. `_normalize_recipe_paths` now absolutises all path fields at launch.
- `torch.compile` Inductor hang: `compile: true` spawned a 32-worker Inductor pool that deadlocked on `unix_stream_data_wait`; both Anima configs ship with `compile: false`.
- Cancel traceback rendering: `KeyboardInterrupt` and `sigkill_handler` lines are now filtered via `_CANCEL_HINTS` instead of being shown as red error events.
- Rerun not clearing old events: `_relaunch_job_in_place` unlinks `events.jsonl` and clears the in-memory deque for plain reruns.
- `selectstart` swallowing clicks: removed a document-level `selectstart` `preventDefault` that was eating sidebar navigation clicks; CSS `user-select: none` already handles text selection.
- Archive stealing the active workspace: added a workspace-sharing check before allowing archive of a finished job.

## [0.2.0] - 2026-05-16

First production-leaning release. LoraHub moves from a CLI tracer bullet to a usable workbench: React + FastAPI control plane, two backends (kohya `sd-scripts` with 8 architectures and `diffusion-pipe` with 21), an end-to-end data preparation pipeline, hyperparameter sweep tooling, and a published mkdocs site. 474 tests cover the new surface area.

### Added — Web workbench

- React + Vite SPA served from `web/dist` by the FastAPI app; navigation covers Dashboard, Jobs, Recipes, Datasets, Settings, Sample Gallery, and Sweeps, with zh-CN labels.
- Dashboard with live hardware tiles (CPU, RAM, GPU, VRAM, battery), a workbench status strip, collapsible "cores", and inline preflight checks for the active recipe.
- Jobs page: tabs for live log, loss / learning-rate charts, artifacts, and sample images; terminal-style log view; side-by-side compare; collapsible sidebar; launch-time recipe override dialog.
- Recipes page: toolbar, template library, import / export, file sidebar, read-only structured preview, and a visual editor split per schema section (base model, dataset, network, optimizer, schedule, loss, validation, resume, sampling, output, backend, diffusion-pipe).
- Visual recipe form exposing every advanced field: arch + arch_variant, conv_dim / conv_alpha / dropout, optimizer betas / weight_decay / eps / `optimizer_args` via key-value editor, loss weighting, validation split, resume from checkpoint, diffusion-pipe options.
- Wizard flow that instantiates parameterized recipe templates (`_placeholders`) into a saved YAML in a few clicks.
- Datasets page with thumbnail grid, inline caption editor, and a WD14 auto-tagging dialog wired to `/api/tagging/tag`.
- Sample Gallery: cross-job lightbox that filters by job and pages through generated previews.
- Settings page with tabs for backends (kohya / diffusion-pipe path + auto-detect), Python runtimes, dependencies (uv-based install with force reinstall and live progress), PyPI mirror probing, model downloader (Hugging Face + ModelScope), tagger device, and `max_concurrent_jobs`.
- One-click kohya bootstrap from Settings, with a force-retry toggle, Windows `.git` read-only handling, and surfaced install errors.
- Loss chart that overlays validation loss, shades the train / val gap, and badges runs that trip the overfit heuristic.

### Added — Training backends

- Pluggable backend protocol with kohya and diffusion-pipe implementations selectable per recipe; shared `SubprocessRunner` and bootstrap / installer plumbing in `lorahub.core.backends._common`.
- diffusion-pipe backend compiles recipes to TOML, launches `train.py`, and parses progress / checkpoint events; covers all 21 upstream architectures including SDXL, SD3, FLUX, HunyuanVideo, Wan2.1, LTX, Cosmos, Lumina, and Chroma.
- Architecture matrix aligned with both upstreams: 8 kohya entry-point archs (`sdxl`, `sd15`, `sd3`, `flux`) and 21 diffusion-pipe archs surfaced in the schema, with `arch_variant` for SDXL sub-architectures (`pony`, `illustrious`, `noobai`, `animagine`); Pony auto-injects `--clip_skip=2`.
- VRAM estimator centralised across the 23-arch matrix so dashboard, recipe info, and sweep planner report identical numbers.
- Recipe schema extensions: `LossConfig` (loss weighting), `ValidationConfig` (`val_split` and emit-kohya-validation-argv), `ResumeConfig` (resume from checkpoint), advanced `NetworkConfig` (conv_dim / conv_alpha / dropout), extended `OptimizerConfig` (betas / weight_decay / eps / `optimizer_args` passthrough), `BackendConfig.diffusion_pipe` options including `model_paths`.
- kohya argv compiler emits loss flags and forwards optimizer betas / eps / weight_decay to both kohya and diffusion-pipe.
- kohya stdout parser emits OOM and cache-progress events, validation-loss events from `val_loss` prints, and aggregates traceback blocks.

### Added — Dataset and data preparation

- BangumiBase fetcher (`lorahub fetch-bangumi`) pulls a single character's pre-clustered image set from Hugging Face and seeds empty caption files.
- WD14 / WD-v3 ONNX tagger with `general` / `character` thresholds, `--device auto/cpu/cuda`, and a CLI `lorahub tag` that skips already-captioned images.
- JoyTag PyTorch tagger with a vendored inference-only ViT (no `timm` / `einops` runtime deps), routed through a shared `BaseTagger` protocol; `tagger='wd14'|'joytag'` selectable per CLI / API call and via Settings.
- Anime caption pipeline (`lorahub.core.dataset.captions`) with atomic transforms, dropout anchoring, batch I/O, and a high-level Anima formatter for booru-style cleanup.
- CLI: `lorahub caption normalize` and `lorahub anima-caption` for offline cleanup; `/api/captions/normalize` and `/api/anima/caption` expose the same pipeline as background sessions.
- Booru alias: opt-in remapping of Danbooru tags to Gelbooru-compatible spellings.
- Dataset thumbnail cache and path allowlist plumbing for the workbench grid; `/api/datasets/thumb` and caption GET / PUT endpoints back the inline editor.

### Added — Scheduling, queue, and observability

- Single-slot job queue with a `max_concurrent_jobs` setting and runtime scheduler resize; multi-GPU via per-slot `CUDA_VISIBLE_DEVICES` injection (covered by a parallel-execution regression test).
- Checkpoint resume orchestrated by the scheduler (v0.5 slice).
- Hyperparameter sweep grid: `SweepPlan` expander, `lorahub sweep` CLI with dry-run variant enumeration, `POST /api/sweeps` fan-out, and `GET /api/sweeps[/{id}]` aggregation.
- `JobRecord.metadata` bag carries sweep linkage and arbitrary launch context.
- Persisted job event replay on API restart; non-terminal jobs surface as `interrupted`.
- Cross-job sample gallery endpoint with per-job filter.
- Hugging Face / ModelScope model downloads with progress sessions and per-file granularity.
- Network acceleration: PyPI mirror probing and a configurable mirror across the install pipeline.

### Added — Toolchain and packaging

- uv-based install pipeline replaces pip for backend dependency installs; deepspeed is skipped on Windows where its build is broken.
- Portable CPython runtime management in the Settings dependencies tab; runtime list deduped and backend paths auto-filled.
- Cross-platform launcher at `scripts/launch.{ps1,bat,sh}` resolves `.venv/` and `web/node_modules/`, runs `pip install -e ".[api,dev]"` and `npm install` only when missing, and brings up API + Vite together with the right `/api` proxy.

### Added — Documentation site

- mkdocs-material site with five top-level sections (Overview, Install, Recipes, Workbench, Reference) and a CI workflow that publishes to GitHub Pages on push to `main`.
- `docs` extras (`pip install lorahub[docs]`) and a `.gitignore` entry for the build output.

### Added — Earlier v0.2 starter (folded in)

- HTTP API starter under `/api` for health, settings, recipe schema / list / detail / validate / save, jobs CRUD, events ring buffer, and a `/api/jobs/{id}/stream` WebSocket; install with `pip install lorahub[api]` and run `lorahub serve`.
- SQLite job store at `runs/.lorahub.sqlite` (WAL mode) mirrors job metadata, with orphan recovery on startup.
- Auto-emitted `done` events via a reaper thread in `KohyaRunner` so fire-and-forget callers see completion.
- `lorahub init --auto` infers VRAM, dataset size, and architecture, then writes a tuned recipe.
- `lorahub bootstrap-kohya`: one-shot kohya install with `--cuda`, `--torch`, `--no-xformers`, `--force`.
- WD14 GPU acceleration via `onnxruntime-gpu`, opt-in with `pip install lorahub[gpu]`.
- `.env` auto-loading and kohya venv auto-detection (`venv/Scripts/python.exe` on Windows, `venv/bin/python` elsewhere).

### Changed

- `BackendConfig.type` accepts `kohya` or `diffusion_pipe`; the API and web client speak the multi-backend shape.
- Recipe-to-kohya translation goes through a generated `dataset.toml` instead of `--train_data_dir` flags, so flat dataset directories work with `num_repeats` directly. `compile_recipe()` now returns `(script, argv, files_to_write)`.
- Backend installs moved from pip to uv; `KohyaBackend.launch()` and `lorahub train` resolve workspace paths to absolute so output never lands under the sd-scripts checkout.
- kohya stdout parser tolerates checkpoint and sample paths with spaces and `at` / `as` / `to` keyword separators.
- CLI status markers use ASCII (`OK`, `->`) for Windows GBK console compatibility.
- API layout: `app.py` split into per-domain routers; web `Recipes` page and recipe form split into per-section modules.
- Runtime dependencies: `huggingface_hub`, `onnxruntime`, `pillow`, `numpy`, `python-dotenv`.

### Fixed

- Workbench installs surface real errors instead of silent failures; force-overwrite is sticky across retries; Windows `.git` read-only files no longer block reinstall; wipe is verified before retry.
- Dataset thumbnail and caption endpoints reject path traversal and round-trip cleanly.
- Recipes split layout is locked to viewport height; per-page scrolling stays inside the page rather than the shell.
- Select trigger displays the chosen option label (was showing the raw value); live numbers no longer twitch on idle ticks.
- Battery tone uses full = green, low = red (was inverted).
- System fields tolerate `undefined` and `null` interchangeably across the API client.

### Tests

- 474 tests covering schema, compiler, parser, runner, backends (kohya + diffusion-pipe), API routers, store, scheduler concurrency, sweeps, taggers (WD14 + JoyTag), caption pipeline, recipe templates, and CLI commands.

## [0.1.0-dev] - 2026-05-15

Initial pre-alpha release. CLI tracer bullet: from a YAML recipe to a trained LoRA file via kohya-ss/sd-scripts.

### Added

- Recipe schema (`lorahub.core.config.schema`): Pydantic v2 semantic configuration covering `base_model`, `dataset`, `network`, `optimizer`, `schedule`, `precision`, `sampling`, `output`, and `backend`. Defaults tuned for SDXL on 8 GB VRAM.
- Recipe loader (`lorahub.core.config.loader`): YAML round-trip and JSON Schema export for future UI form generation.
- Event bus (`lorahub.core.events`): `TrainingEvent`, `EventType`, thread-safe `EventBus`, and `JsonlEventSink` for durable replay.
- Backend protocol (`lorahub.core.backends.base`): `TrainingBackend` `Protocol`, `TrainingHandle` with `stop()` / `wait()`, `ValidationIssue`, and `VRAMEstimate`.
- KohyaBackend (`lorahub.core.backends.kohya`):
  - `bootstrap` resolves an existing kohya checkout via recipe field, env var, or default user-data path.
  - `compiler` translates a `RecipeConfig` into the `(script, argv)` pair that `sd-scripts` expects. Covers SDXL / SD1.5 / FLUX / SD3 entry scripts, bucket / caption / lora variants, optimizer / precision / sampling / output flags, and an `extra_args` escape hatch.
  - `parser` recognizes tqdm progress bars (steps + `avr_loss`), epoch markers, checkpoint saves, sample image generation, tracebacks, and CUDA errors.
  - `runner` owns the child process and stdout / stderr pump threads. Stop is platform-aware (`CTRL_BREAK_EVENT` on Windows, `SIGINT` elsewhere) with terminate / kill escalation.
- CLI (`lorahub.cli.main`): `init`, `validate`, `info`, `train`, `version` commands powered by typer + rich.
- Built-in recipe `recipes/sdxl_character_8gb.yaml` for character LoRAs on 8 GB cards.

### Tests

- 62 tests covering schema, events, compiler, parser, bootstrap, runner, backend, and CLI.

### Known limitations

- No automatic clone or install of kohya-ss; you must point at an existing checkout.
- No web UI yet; planned for v0.2.
- VRAM estimator is a coarse first-pass approximation.
- End-to-end smoke test against a real SDXL model is not yet automated.

[Unreleased]: https://github.com/GALIAIS/LoraHub/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/GALIAIS/LoraHub/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GALIAIS/LoraHub/compare/v0.1.0-dev...v0.2.0
[0.1.0-dev]: https://github.com/GALIAIS/LoraHub/releases/tag/v0.1.0-dev







