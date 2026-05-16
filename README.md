# LoraHub

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

**An open-source LoRA training workbench for diffusion models** — data, training, evaluation, and recipes in one workflow.

LoraHub wraps mature training backends (currently [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)) behind a stable, semantic configuration layer and a unified CLI / API. The goal is to make LoRA training reproducible, recipe-driven, and tool-agnostic.

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

Not yet:

- Dataset management UI: import, thumbnails, and caption editor
- Web auto-tagging workflow on top of the existing WD14 CLI/tagger
- Job queue, multi-GPU scheduling, and resume orchestration
- DiffusersBackend and non-kohya training backends

See [Roadmap](#roadmap) for the full picture.

## Install

Requires Python 3.11 or 3.12, an NVIDIA GPU with 8GB+ VRAM, and an existing [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) checkout with its own dependencies installed.

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[dev]"
```

Tell LoraHub where your kohya checkout lives — either via env var or directly in your recipe:

```powershell
# Option A: install kohya inside the LoraHub working tree (this is the default lookup path)
lorahub bootstrap-kohya         # clones sd-scripts and installs PyTorch + deps in ~10 min

# Option B: point at an existing checkout
$env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
# or copy .env.example to .env and edit
```

`lorahub bootstrap-kohya` defaults to PyTorch 2.6.0 + CUDA 12.4. Use `--cuda cu121` (or `cu118` / `cu128`) and `--torch 2.6.0` to switch versions, `--no-xformers` to skip the optional xformers install, or `--force` to wipe a half-installed target.

LoraHub auto-loads `.env` from the project root on startup, so once `.env` has `LORAHUB_KOHYA_SD_SCRIPTS=./sd-scripts` you don't need to export it in every shell.

## Quick start

```powershell
# 1. Scaffold a recipe
lorahub init my_character

# 2. Edit my_character.yaml: point at your SDXL model and dataset
notepad my_character.yaml

# 3. Sanity check (no training yet)
lorahub validate my_character.yaml
lorahub info     my_character.yaml

# 4. Train
lorahub train    my_character.yaml
```

A minimal recipe:

```yaml
schema_version: "1.0"
base_model:
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
  batch_size: 1
  grad_accum: 4
precision: bf16
gradient_checkpointing: true
output:
  name: my_character_v1
backend:
  type: kohya
```

See [`recipes/sdxl_character_8gb.yaml`](recipes/sdxl_character_8gb.yaml) for a fully annotated example.

## End-to-end smoke test

Once you have kohya-ss/sd-scripts installed (set `LORAHUB_KOHYA_SD_SCRIPTS` or copy `.env.example`) and an SDXL base model on disk, the full path from zero to a trained LoRA looks like this:

```powershell
# 1. Pull a character's images from BangumiBase
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. Auto-tag every image
lorahub tag ./datasets/laffey

# 3. Scaffold a recipe and edit it (point base_model.checkpoint at your SDXL .safetensors)
lorahub init smoke
notepad smoke.yaml

# 4. Sanity check
lorahub validate smoke.yaml
lorahub info     smoke.yaml

# 5. Train
lorahub train    smoke.yaml
```

## Need test data fast?

`lorahub fetch-bangumi` pulls a single character's image set from the [BangumiBase](https://huggingface.co/BangumiBase) Hugging Face datasets — pre-clustered, MIT-licensed, ready for smoke testing.

```powershell
# List characters in a show
lorahub fetch-bangumi azurlaneanime

# Grab character 5 with up to 50 images
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50

# Or pull the 8 preview thumbnails first to identify the character
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

Each image lands next to an empty `.txt` caption file — fill them in (or auto-tag with `lorahub tag`) before training.

## Auto-tag a dataset

`lorahub tag` runs the WD14 / WD-v3 ONNX tagger over a directory and writes kohya-style `.txt` captions next to each image.

```powershell
# Default thresholds (general=0.35, character=0.85), skips images that already have a non-empty caption
lorahub tag ./datasets/akagi

# Re-tag everything from scratch with a tighter general threshold
lorahub tag ./datasets/akagi --overwrite --general 0.45

# Skip the character tag if you're training a style or concept LoRA
lorahub tag ./datasets/akagi --no-include-character
```

The first run downloads ~400 MB of ONNX weights from Hugging Face (cached for subsequent runs). CPU inference handles hundreds of images at ~1 s/image; for batch throughput install the GPU runtime:

```powershell
pip uninstall onnxruntime
pip install lorahub[gpu]              # or: pip install onnxruntime-gpu
lorahub tag ./datasets/akagi --device cuda
```

`--device auto` picks GPU when `onnxruntime-gpu` and a CUDA 12.x runtime are present, otherwise falls back to CPU. `--device cuda` forces GPU and errors out with an actionable message if it isn't available.

## HTTP API (v0.2 starter)

LoraHub ships a FastAPI server for programmatic access. Install API extras and start it:

```powershell
pip install lorahub[api]
lorahub serve --port 18765
```

### One-shot launcher

If you'd rather not memorise `pip install` and `npm run dev` separately, the
`scripts/` folder ships a cross-platform launcher that resolves the project
venv (or system Python), installs missing dependencies on first run, and brings
up the API and the React dev server side by side:

```powershell
# Windows (PowerShell or double-click in Explorer)
scripts\launch.bat              # default: dev mode (API + Vite)
scripts\launch.bat -Mode prod   # API only, serves prebuilt web/dist
scripts\launch.bat -Mode build  # one-shot npm install + vite build
```

```bash
# macOS / Linux
chmod +x scripts/launch.sh
scripts/launch.sh                       # default: dev mode
scripts/launch.sh --mode prod --port 8080
scripts/launch.sh --mode build
```

The launcher auto-detects `.venv/` and `web/node_modules/`, runs `pip install -e ".[api,dev]"` and `npm install` only when something's missing, and forwards Vite's `/api` proxy to whichever port the API ended up on. Pass `--no-install` to skip the dependency check, `--reload` for uvicorn auto-reload.

Endpoints live under `/api`:

- `GET /api/health` — server status, version, and backend probe
- `GET /api/settings` / `PUT /api/settings` — workbench defaults for kohya and tagger device
- `GET /api/recipes/schema` — recipe JSON Schema used by the visual editor
- `GET /api/recipes` / `GET /api/recipes/{name}` — list and inspect recipe YAML files
- `POST /api/recipes/validate` — validate an in-memory recipe and return field errors
- `POST /api/recipes` — save a validated recipe to `recipes/<name>.yaml`
- `GET /api/jobs` / `GET /api/jobs/{id}` — list / inspect training jobs
- `POST /api/jobs` `{recipe, workspace?}` — start a job
- `DELETE /api/jobs/{id}` — stop a running job
- `GET /api/jobs/{id}/events` — recent events from the in-memory ring buffer
- `WS /api/jobs/{id}/stream` — live event stream (replays the buffer first)

The API binds to `127.0.0.1` by default and has no auth — safe for localhost only. Job metadata persists to SQLite at `runs/.lorahub.sqlite`; live handles and the recent event ring remain process-local.

## Project layout

```
lorahub/
  core/
    config/      Recipe schema + YAML loader + JSON Schema export
    backends/    TrainingBackend protocol + KohyaBackend implementation
    events.py    Structured training event bus + JSONL persistence
  cli/           typer + rich command line
  api/           FastAPI (v0.2+)
  web/           React UI (v0.2+)
recipes/         Built-in recipe library
tests/           pytest suite
```

## Roadmap

| Version | Scope                                                                |
| ------- | -------------------------------------------------------------------- |
| v0.1    | CLI tracer bullet: recipe → kohya → LoRA file (this release)         |
| v0.2    | FastAPI + minimal React UI, recipe editor, settings, job monitor        |
| v0.3    | Dataset module: import, thumbnails, caption editor                   |
| v0.4    | Auto-taggers: WD14, JoyTag                                            |
| v0.5    | Job queue + multi-GPU + resume from checkpoint                       |
| v0.6    | Recipe library + sample image gallery                                |
| v0.7    | SD1.5 + Pony/Illustrious; DiffusersBackend (self-written) starts     |
| v0.8    | Flux / SD3 support                                                    |
| v1.0    | Hyperparameter sweeps, overfit detection, docs site                  |

## Contributing

Pull requests welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening an issue or PR.

## Acknowledgements

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — the training engine LoraHub wraps.
- [Pydantic](https://docs.pydantic.dev/), [typer](https://typer.tiangolo.com/), [rich](https://rich.readthedocs.io/) — the foundations of LoraHub's CLI.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
