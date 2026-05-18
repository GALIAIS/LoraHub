---
title: Install
description: Install LoraHub and bootstrap a training backend.
---

# Install

Requires Python 3.11 or 3.12, an NVIDIA GPU with 8 GB+ VRAM. At least one
training backend must be present — either `kohya-ss/sd-scripts` (covers SD /
SDXL / Flux / Lumina / HunyuanImage / Anima) or `tdrussell/diffusion-pipe`
(covers the modern DiT zoo: Flux2, Chroma, Wan, Cosmos, Anima, ...). Both can
coexist; pick per-config in the `backend.type` field.

## Get the source

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
```

## Bootstrap a backend

=== "Option A: bootstrap inside the working tree"

    ```powershell
    # kohya — SD / SDXL / Flux / Lumina / HunyuanImage / Anima
    lorahub bootstrap-kohya              # ~10 min: clone + venv + PyTorch + deps
    # diffusion-pipe — DiT zoo
    lorahub bootstrap-diffusion-pipe
    ```

    `bootstrap-kohya` defaults to PyTorch 2.6.0 + CUDA 12.4. Use
    `--cuda cu121` (or `cu118` / `cu128`) and `--torch 2.6.0` to switch
    versions, `--no-xformers` to skip the optional xformers install, or
    `--force` to wipe a half-installed target.

=== "Option B: point at existing checkouts"

    ```powershell
    $env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
    $env:LORAHUB_DIFFUSION_PIPE   = "C:\path\to\diffusion-pipe"
    # or copy .env.example to .env and edit
    ```

!!! tip ".env auto-loading"
    LoraHub auto-loads `.env` from the project root on startup, so once
    `.env` contains `LORAHUB_KOHYA_SD_SCRIPTS=./sd-scripts` you don't need
    to export it in every shell.

## Optional extras

| Extra     | When to install                                  | Command                              |
| --------- | ------------------------------------------------ | ------------------------------------ |
| `api`     | FastAPI server (`lorahub serve`)                 | `pip install -e ".[api]"`            |
| `gpu`     | WD14 tagger on CUDA via `onnxruntime-gpu`        | `pip install -e ".[gpu]"`            |
| `tagging` | JoyTag (PyTorch) tagger backend                  | `pip install -e ".[tagging]"`        |
| `dev`     | Tests, lint, mypy, httpx                         | `pip install -e ".[dev]"`            |
| `docs`    | Build this documentation site                    | `pip install -e ".[docs]"`           |

## Anima Base downloader

LoraHub ships a one-shot downloader for the Anima full stack (~5.5 GB,
transformer + Qwen-Image VAE + Qwen3-0.6B text encoder):

```powershell
bash scripts/_download_anima.sh        # uses hf-mirror.com by default; safe in CN
```

Files end up under `models/circlestone-labs__Anima/split_files/`. Both
`configs/anima_style_24gb.yaml` and `configs/anima_character_24gb.yaml`
already point at this layout.

## Next

- [Quick start](quickstart.md) — first config in four commands.
- [Smoke test](smoke-test.md) — full pipeline against real images.
