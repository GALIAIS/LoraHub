# LoraHub

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

**Open-source LoRA training workbench for diffusion models** — datasets, captioning, training, live previews, and analysis in one workflow.

LoraHub wraps two production training backends ([kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)) behind a single semantic configuration layer and a unified CLI / API / web UI. The goal is to make LoRA training reproducible, recipe-driven, and tool-agnostic — pick a config, hit train, watch the run.

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

- **React + FastAPI workbench** — Dashboard, Jobs, Configs, Datasets, Image Studio, Sample Gallery, Sweeps, Settings (zh-CN UI). Live event streams over SSE with browser-native reconnect + `Last-Event-ID` resume.
- **Two training backends** behind a single config schema: `kohya` (8 archs — SD1.5, SDXL, SD3, Flux, Lumina, HunyuanImage, Anima) and `diffusion-pipe` (21 archs — Flux2, Chroma, HiDream, OmniGen2, AuraFlow, Qwen-Image, Cosmos, Wan, LTX, HunyuanVideo, Z-Image, ErnieImage and more).
- **Anima Base full pipeline**: model downloader, recipe with paired transformer + Qwen-Image VAE + Qwen3 text encoder, training, and live preview rendering between checkpoints.
- **Image Studio** — dataset manager with virtualized grid, multi-select, drag-and-drop upload, AR-bucket caption strategies (style / character / general), inline VLM smart caption (WD14 EVA02 + vision LLM), perceptual-hash de-duplication, batch quality scoring, trash + restore.
- **Visual config editor** with every advanced field (loss, validation, resume, conv/dropout, optimizer betas / eps / weight_decay, dp options) plus a wizard for parameterized templates. YAML uses camelCase on the wire; legacy snake_case still loads.
- **Job runtime**: per-slot `CUDA_VISIBLE_DEVICES`, checkpoint resume, automatic SSE event replay on reconnect, GPU sampler thread, AI training analysis (Claude reads metrics + config, returns Markdown diagnosis).
- **Run-summary card** on the job detail page: collapsible single-line digest of step / loss / drop% / convergence trend, expands to progress + loss + hyperparam snapshot.
- **Dataset captioning**: WD14 / WD-v3 ONNX taggers (default `wd-eva02-large-v3`), JoyTag PyTorch backend, and the **smart-caption** pipeline that combines WD14 with a vision LLM via the AI router for Anima-format captions.
- **Live preview during dp training** — lorahub watches `output/step{N}/` directories and renders one PNG per prompt for every new checkpoint via subprocess to sd-scripts' Anima inference. Stub fallback when sd-scripts isn't available.
- **Hyperparameter sweep grid** via `lorahub sweep` and `POST /api/sweeps`, train/val loss chart with overfit signal.
- **One-click bootstrap** for both backends, uv-based dependency installs, portable CPython runtime, HF/ModelScope downloader, PyPI mirror probing.
- **Cross-platform launcher** (`scripts/launch.{ps1,bat,sh}`) and a published [mkdocs-material documentation site](https://galiais.github.io/LoraHub/).
- **778 tests** across schema, compilers, parsers, runners, API routers, scheduler, sweeps, taggers, captions, inference preview, and CLI.

Not yet:

- Random / Bayesian sweep strategies (only grid expansion today).
- Embedded Weights & Biases dashboard.
- CI that runs an end-to-end LoRA training (currently only unit/integration tests).
- Optional auth / multi-user mode for the API.

## Install

Requires Python 3.11 or 3.12, an NVIDIA GPU with 8 GB+ VRAM. At least one training backend must be present — either `kohya-ss/sd-scripts` or `tdrussell/diffusion-pipe`.

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
```

### Bootstrap a backend

Install one of the upstreams from inside LoraHub:

```powershell
# kohya — handles SD1.5 / SDXL / SD3 / Flux / Lumina / HunyuanImage / Anima.
lorahub bootstrap-kohya              # clones sd-scripts, builds a venv, installs PyTorch + deps

# diffusion-pipe — handles dp's 21-arch matrix including Flux2, Wan2.1, Cosmos, Anima.
lorahub bootstrap-diffusion-pipe
```

Or point LoraHub at existing checkouts:

```powershell
$env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
$env:LORAHUB_DIFFUSION_PIPE   = "C:\path\to\diffusion-pipe"
```

`.env` in the project root is auto-loaded on startup.

### Optional extras

| Extra     | When to install                                  | Command                              |
| --------- | ------------------------------------------------ | ------------------------------------ |
| `api`     | FastAPI server (`lorahub serve`)                 | `pip install -e ".[api]"`            |
| `gpu`     | WD14 tagger on CUDA via `onnxruntime-gpu`        | `pip install -e ".[gpu]"`            |
| `tagging` | JoyTag (PyTorch) tagger backend                  | `pip install -e ".[tagging]"`        |
| `dev`     | Tests, lint, mypy, httpx                         | `pip install -e ".[dev]"`            |
| `docs`    | Build the mkdocs documentation site              | `pip install -e ".[docs]"`           |

## Quick start

```powershell
# 1. Scaffold a config
lorahub init my_character

# 2. Edit configs/my_character.yaml: point at your checkpoint and dataset
notepad configs/my_character.yaml

# 3. Sanity check (no training yet)
lorahub validate configs/my_character.yaml
lorahub info     configs/my_character.yaml

# 4. Train
lorahub train    configs/my_character.yaml
```

A minimal config (camelCase, the new wire format):

```yaml
schemaVersion: "1.0"
baseModel:
  arch: sdxl
  checkpoint: ./models/sdxl_base_1.0.safetensors
dataset:
  source: ./datasets/my_character
  resolution: [1024, 1024]
network:
  type: lora
  rank: 32
  alpha: 16
schedule:
  epochs: 10
  batchSize: 1
  gradAccum: 4
precision: bf16
gradientCheckpointing: true
output:
  name: my_character_v1
backend:
  type: kohya
```

Both `camelCase` and `snake_case` are accepted by the validator — old configs still load. New defaults emit camelCase.

## Anima Base — first-class workflow

LoraHub ships annotated configs for [Anima Base](https://huggingface.co/circlestone-labs/Anima) (Qwen-Image-style stack: Anima transformer + Qwen-Image VAE + Qwen3-0.6B text encoder).

```powershell
# Pull the full Anima stack (~5.5 GB) into ./models/circlestone-labs__Anima/
bash scripts/_download_anima.sh

# Use the bundled config — already wired for diffusion-pipe + 200-step checkpoints + the 8-prompt default preview set
lorahub train configs/anima_style_24gb.yaml
# or the character variant
lorahub train configs/anima_character_24gb.yaml
```

The configs at `configs/anima_style_24gb.yaml` and `configs/anima_character_24gb.yaml` cover the common knobs with comments explaining every choice. Default sample prompts live at `configs/sample_prompts/anima_default.txt` — edit the file in place; the worker re-reads it at the next checkpoint.

## End-to-end smoke test

Once a backend is installed and a base model is on disk:

```powershell
# 1. Pull a character's images from BangumiBase
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. Smart-caption every image (WD14 + vision LLM)
#    — or use the classic auto-tagger:
#      lorahub tag ./datasets/laffey
curl -X POST http://127.0.0.1:18765/api/image-studio/ai/smart-caption \
     -H 'Content-Type: application/json' \
     -d '{"path":"./datasets/laffey","captionMode":"character","triggerWord":"laffey"}'

# 3. Scaffold a config and edit it
lorahub init smoke

# 4. Sanity check
lorahub validate configs/smoke.yaml
lorahub info     configs/smoke.yaml

# 5. Train
lorahub train    configs/smoke.yaml
```

## HTTP API

LoraHub ships a FastAPI server. Install API extras and start it:

```powershell
pip install -e ".[api]"
lorahub serve --port 18765
```

### One-shot launcher

If you'd rather not memorise `pip install` and `npm run dev` separately, the `scripts/` folder has a cross-platform launcher that resolves the project venv (or system Python), installs missing dependencies on first run, and brings up the API and the React dev server side by side:

```powershell
# Windows
scripts\launch-dev.ps1                    # quick dev: API + Vite each in its own console
scripts\launch.bat                        # default: dev mode (API + Vite)
scripts\launch.bat -Mode prod             # API only, serves prebuilt web/dist
scripts\launch.bat -Mode build            # one-shot npm install + vite build
```

```bash
# macOS / Linux
chmod +x scripts/launch.sh
scripts/launch.sh                         # default: dev mode
scripts/launch.sh --mode prod --port 8080
scripts/launch.sh --mode build
```

### Endpoint overview

All endpoints live under `/api`. Highlights:

- `GET /api/health` — server status, version, backend probes.
- `GET /api/system/sse` — SSE telemetry: 1 Hz host snapshot (CPU / RAM / GPU / temp / network).
- `GET /api/configs` / `GET /api/configs/{name}` — list and inspect training configs.
- `POST /api/configs/validate` — validate an in-memory config and return field errors.
- `POST /api/configs` — save a validated config to `configs/<name>.yaml`.
- `GET /api/jobs` / `GET /api/jobs/{id}` — list / inspect training jobs.
- `POST /api/jobs` `{config, workspace?}` — start a job.
- `POST /api/jobs/{id}/rerun` — re-launch.
- `POST /api/jobs/{id}/resume` — continue from the last checkpoint.
- `DELETE /api/jobs/{id}` — cancel a running job (or `?archive=true` for finished ones).
- `POST /api/jobs/{id}/kill` — last-resort SIGKILL.
- `GET /api/jobs/{id}/metrics` — aggregated per-step metrics for charts.
- `GET /api/jobs/{id}/sse` — live `TrainingEvent` stream with `Last-Event-ID` resume; the legacy WS endpoint at `/api/jobs/{id}/stream` is preserved as fallback.
- `POST /api/jobs/{id}/analyze` — ask the AI router for a Markdown diagnosis.
- `POST /api/image-studio/ai/smart-caption` — WD14 + vision LLM caption pipeline (style / character / general modes; supports trigger words and Anima header).
- `POST /api/image-studio/dedupe/scan` — phash-based duplicate detection.
- `POST /api/tagging/tag` — async tagging session over a directory.
- `POST /api/models/download` — async base-model download from HF / ModelScope.

The API binds to `127.0.0.1` by default and has no auth — safe for localhost only. Job metadata persists to SQLite at `runs/jobs.sqlite`; live event rings remain process-local.

## Project layout

```
lorahub/
  core/
    config/      Recipe schema (camelCase aliases) + YAML loader + JSON Schema export
    backends/    TrainingBackend protocol + KohyaBackend + DiffusionPipeBackend
    inference/   Live-preview worker + Anima inference subprocess wrapper
    tagging/     WD14 / JoyTag tagger backends
    dataset/     Caption pipeline + BangumiBase fetcher
    events.py    Structured training event bus + JSONL persistence
  cli/           typer + rich command line
  api/           FastAPI app, routers, scheduler, stores
  ...
configs/         Built-in config library (anima_style/character, sdxl_character, ...)
configs/sample_prompts/   Default preview prompt sets
web/             React + Vite workbench
tests/           pytest suite — 778 tests
docs/            mkdocs-material documentation site
```

## Roadmap

| Version | Scope                                                                       | Status |
| ------- | --------------------------------------------------------------------------- | ------ |
| v0.1    | CLI tracer bullet: recipe → kohya → LoRA file                               | done   |
| v0.2    | FastAPI + React workbench, recipe editor, settings, job monitor             | done   |
| v0.3    | Dataset module: import, thumbnails, caption editor                          | done   |
| v0.4    | Auto-taggers: WD14, JoyTag                                                  | done   |
| v0.5    | Job queue + multi-GPU + resume from checkpoint                              | done   |
| v0.6    | Recipe library + sample image gallery                                       | done   |
| v0.7    | Second backend (diffusion-pipe), 21-arch matrix                             | done   |
| v0.8    | Flux / SD3 / Anima support                                                  | done   |
| v0.9    | Image Studio + smart caption + AI training analysis + SSE event channel    | done   |
| v0.10   | dp live preview (3-cut: scaffold + Anima inference + GPU budget)            | done   |
| v1.0    | Hyperparameter sweeps, overfit detection, docs site                         | done   |
| v1.x    | Random/Bayesian sweeps, wandb integration, end-to-end CI training           | next   |

## Contributing

Pull requests welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening an issue or PR.

## Acknowledgements

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — primary training engine LoraHub wraps.
- [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) — second training engine, covers the modern DiT zoo.
- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) — Anima Base model (DiT + Qwen3 TE).
- [Pydantic](https://docs.pydantic.dev/), [typer](https://typer.tiangolo.com/), [rich](https://rich.readthedocs.io/), [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/) — foundations of the LoraHub stack.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
