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

**Pre-alpha (v0.1).** What works today:

- Semantic recipe schema (Pydantic) → kohya argv compiler
- KohyaBackend with subprocess management and structured event stream
- CLI: `init`, `validate`, `info`, `train`, `version`
- 60+ tests covering compiler, parser, runner, backend, CLI

Not yet:

- Web UI (planned for v0.2)
- Dataset management & auto-tagging (v0.3 / v0.4)
- Job queue & multi-GPU (v0.5)
- Self-bootstrapping kohya install (today you point at an existing checkout)

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
$env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
```

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

## Project layout

```
lorahub/
  core/
    config/      Recipe schema + YAML loader + JSON Schema export
    backends/    TrainingBackend protocol + KohyaBackend implementation
    events.py    Structured training event bus + JSONL persistence
  cli/           typer + rich command line
  api/           FastAPI (v0.2+)
  web/           Vue3 UI (v0.2+)
recipes/         Built-in recipe library
tests/           pytest suite
```

## Roadmap

| Version | Scope                                                                |
| ------- | -------------------------------------------------------------------- |
| v0.1    | CLI tracer bullet: recipe → kohya → LoRA file (this release)         |
| v0.2    | FastAPI + minimal Vue UI, single-task form                            |
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
