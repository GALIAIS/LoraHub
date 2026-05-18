---
title: CLI
description: The lorahub command-line interface.
---

# CLI

LoraHub ships a single `lorahub` entry point built on
[typer](https://typer.tiangolo.com/) and [rich](https://rich.readthedocs.io/).
Every command runs against a config YAML or against the global state
managed by `.env` and the workbench settings file.

## Command list

| Command | Purpose |
| ------- | ------- |
| [`init`](commands.md#init) | Scaffold a starter config in `configs/`. |
| [`bootstrap-kohya`](commands.md#bootstrap-kohya) | One-shot install of kohya-ss/sd-scripts (clone + venv + PyTorch). |
| [`bootstrap-diffusion-pipe`](commands.md#bootstrap-diffusion-pipe) | One-shot install of tdrussell/diffusion-pipe. |
| [`fetch-bangumi`](commands.md#fetch-bangumi) | Download a single character's images from BangumiBase. |
| [`tag`](commands.md#tag) | Auto-tag a directory of images with WD14 or JoyTag. |
| [`caption normalize`](commands.md#caption-normalize) | Apply caption transforms in batch. |
| [`anima-caption`](commands.md#anima-caption) | High-level Anima caption formatter. |
| [`validate`](commands.md#validate) | Check a config without running training. |
| [`info`](commands.md#info) | Dry-run: show compiled argv + estimated VRAM. |
| [`train`](commands.md#train) | Run a training job to completion. |
| [`sweep`](commands.md#sweep) | Plan or run a hyperparameter sweep. |
| [`serve`](commands.md#serve) | Start the FastAPI HTTP API server. |
| [`version`](commands.md#version) | Print the installed `lorahub` version. |

## Conventions

- All commands accept `--help` for the full flag list.
- Status messages use ASCII markers (`OK`, `->`) so they render correctly in
  Windows GBK consoles.
- `lorahub` auto-loads `.env` from the working directory at startup; existing
  environment variables take precedence.
- Job artifacts land under `runs/<output.name>/` by default; SQLite metadata
  is at `runs/jobs.sqlite`, AI store at `runs/ai.sqlite`, image-studio store
  at `runs/image_studio.sqlite`.

See the full per-command reference on the [Commands](commands.md) page.
