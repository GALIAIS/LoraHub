---
title: CLI commands
description: Per-command reference for the lorahub CLI.
---

# CLI commands

Each command supports `--help` for the full flag list. The examples below
cover the common path.

## `init`

Scaffold a starter config in `configs/`.

```powershell
lorahub init my_character
lorahub init my_style --template sdxl_style
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

`--auto` probes `nvidia-smi` for VRAM, scans the dataset directory for image
count, detects the architecture from the checkpoint filename (SDXL / Flux /
SD3 / SD1.5, with IllustriousXL / Pony / NoobAI / Animagine matched as SDXL),
and writes a config with rank / batch / grad_accum tuned per VRAM tier and
`numRepeats` inversely scaled to dataset size. `--vram-mib` overrides
detection.

## `bootstrap-kohya`

One-shot install of kohya-ss/sd-scripts. Clones the repo, creates a venv,
installs PyTorch + torchvision (`--cuda cu121/cu124/cu128`, `--torch X.Y.Z`),
runs `pip install -r requirements.txt`, and installs xformers (skip with
`--no-xformers`). `--force` wipes a half-installed target.

```powershell
lorahub bootstrap-kohya                 # default: ./sd-scripts, cu124, torch 2.6.0
lorahub bootstrap-kohya --cuda cu121
lorahub bootstrap-kohya --no-xformers --force
```

## `bootstrap-diffusion-pipe`

One-shot install of tdrussell/diffusion-pipe. Mirrors the kohya bootstrap
flow: clones the upstream, creates a venv, installs PyTorch + DeepSpeed +
backend deps via uv. Use this when you want the dp-only architectures
(Flux2, Wan, Cosmos, Z-Image, ...) or to benchmark against kohya on a
shared arch like Anima.

```powershell
lorahub bootstrap-diffusion-pipe
```

## `fetch-bangumi`

Download a single character's image set from a Hugging Face BangumiBase
dataset. Used to quickly seed a smoke-test dataset.

```powershell
lorahub fetch-bangumi azurlaneanime                            # list characters
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

Each image lands next to an empty `.txt` caption file unless
`--no-seed-captions` is passed.

## `tag`

Auto-tag a directory of images and write kohya-style `.txt` captions next to
each one. Supports WD14 / WD-v3 (ONNX, default `wd-eva02-large-tagger-v3`)
and JoyTag (PyTorch).

```powershell
# Default thresholds (general=0.35, character=0.85), skips images with non-empty captions
lorahub tag ./datasets/akagi

# Re-tag everything; tighter general threshold; recurse
lorahub tag ./datasets/akagi --overwrite --general 0.45 -r

# JoyTag backend (needs `pip install lorahub[tagging]`)
lorahub tag ./datasets/akagi --tagger joytag --joytag-threshold 0.4
```

`--device auto` picks GPU when `onnxruntime-gpu` is available; `--device
cuda` forces it; `--device cpu` always uses CPU.

## `caption normalize`

Apply the caption pipeline in batch — atomic transforms, dropout anchoring,
booru alias remapping. Runs offline against a directory of paired
`{image}.txt` caption files.

```powershell
lorahub caption normalize ./datasets/akagi --shuffle --keep-tokens 1 --booru-alias
```

## `anima-caption`

High-level Anima caption formatter. Restructures existing captions into the
official Anima format:

```
masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
<1girl/solo/character>, @<trigger>,
<2-3 sentence natural-language description>,
<remaining general tags>
```

```powershell
lorahub anima-caption ./datasets/akagi --trigger akagi --mode character
```

## `validate`

Check a config without launching training. Exits non-zero when any
`Severity.error` issue is reported.

```powershell
lorahub validate configs/my_character.yaml
```

## `info`

Dry-run: load the config, compile it to backend argv (kohya CLI flags or a
diffusion-pipe TOML), and print the entry script + estimated VRAM. Does not
touch the GPU.

```powershell
lorahub info configs/my_character.yaml
```

## `train`

Run a training job to completion. Press `Ctrl+C` to stop gracefully — the
runner sends `CTRL_BREAK_EVENT` (Windows) or `SIGINT` (Unix) and escalates to
terminate/kill if the child doesn't exit. Cancel-shaped tracebacks
(KeyboardInterrupt, sigkill_handler) are recognised by the parser and not
rendered as red errors.

```powershell
lorahub train configs/my_character.yaml
lorahub train configs/my_character.yaml --workspace .\runs\my_character_v1
```

Job artifacts (logs, checkpoints, samples, `events.jsonl`) land under
`runs/<output.name>/` by default. Step-cadence checkpoints
(`saveEveryNSteps`) and epoch-cadence checkpoints (`saveEveryNEpochs`) are
mutually exclusive — if both backends are emitted with a step setting, the
epoch flag is dropped to avoid double-saves on aligned step boundaries.

## `sweep`

Plan or run a hyperparameter sweep. The current expander is grid-only;
random / Bayesian strategies are on the roadmap.

```powershell
lorahub sweep configs/my_character.yaml --axis lr=1e-4,5e-5 --axis rank=16,32 --dry-run
lorahub sweep configs/my_character.yaml --axis lr=1e-4,5e-5
```

`--dry-run` enumerates the variant cells without launching jobs. Without
`--dry-run`, each variant is written as its own config and submitted to
`POST /api/sweeps`.

## `serve`

Start the LoraHub FastAPI server. Requires the `api` extras
(`pip install lorahub[api]`).

```powershell
lorahub serve --port 18765
lorahub serve --host 0.0.0.0 --port 8000 --reload
```

The server binds to `127.0.0.1` by default and has no auth — safe for
localhost only.

## `version`

Print the installed `lorahub` version.

```powershell
lorahub version
```
