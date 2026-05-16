---
title: Install
description: Install LoraHub and bootstrap the kohya-ss training backend.
---

# Install

Requires Python 3.11 or 3.12, an NVIDIA GPU with 8 GB+ VRAM, and an existing
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) checkout with its
own dependencies installed.

## Get the source

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[dev]"
```

## Point LoraHub at kohya

Tell LoraHub where your kohya checkout lives — either via env var or directly
in your recipe.

=== "Option A: bootstrap inside the working tree"

    ```powershell
    # Clones sd-scripts and installs PyTorch + deps in ~10 min.
    lorahub bootstrap-kohya
    ```

    `lorahub bootstrap-kohya` defaults to PyTorch 2.6.0 + CUDA 12.4. Use
    `--cuda cu121` (or `cu118` / `cu128`) and `--torch 2.6.0` to switch
    versions, `--no-xformers` to skip the optional xformers install, or
    `--force` to wipe a half-installed target.

=== "Option B: point at an existing checkout"

    ```powershell
    $env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
    # or copy .env.example to .env and edit
    ```

!!! tip ".env auto-loading"
    LoraHub auto-loads `.env` from the project root on startup, so once
    `.env` contains `LORAHUB_KOHYA_SD_SCRIPTS=./sd-scripts` you don't need to
    export it in every shell.

## Optional extras

| Extra | When to install | Command |
| ----- | --------------- | ------- |
| `api` | Run the FastAPI server (`lorahub serve`) | `pip install -e ".[api]"` |
| `gpu` | WD14 tagger on CUDA via `onnxruntime-gpu` | `pip install -e ".[gpu]"` |
| `tagging` | JoyTag (PyTorch) tagger backend | `pip install -e ".[tagging]"` |
| `dev` | Tests, lint, mypy, httpx | `pip install -e ".[dev]"` |
| `docs` | Build this documentation site | `pip install -e ".[docs]"` |

## Next

- [Quick start](quickstart.md) — first recipe in four commands.
- [Smoke test](smoke-test.md) — full pipeline against real images.
