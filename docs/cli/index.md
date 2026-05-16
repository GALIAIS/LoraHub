---
title: CLI
description: The lorahub command-line interface.
---

# CLI

LoraHub ships a single `lorahub` entry point built on
[typer](https://typer.tiangolo.com/) and [rich](https://rich.readthedocs.io/).
Every command runs against a recipe YAML or against the global state managed
by `.env` and the workbench settings file.

## Command list

| Command | Purpose |
| ------- | ------- |
| [`init`](commands.md#init) | Scaffold a starter recipe in the current directory. |
| [`bootstrap-kohya`](commands.md#bootstrap-kohya) | One-shot install of kohya-ss/sd-scripts (clone + venv + PyTorch). |
| [`fetch-bangumi`](commands.md#fetch-bangumi) | Download a single character's images from BangumiBase. |
| [`tag`](commands.md#tag) | Auto-tag a directory of images with WD14 or JoyTag. |
| [`validate`](commands.md#validate) | Check a recipe without running training. |
| [`info`](commands.md#info) | Dry-run: show compiled argv + estimated VRAM. |
| [`train`](commands.md#train) | Run a training job to completion. |
| [`serve`](commands.md#serve) | Start the FastAPI HTTP API server. |
| [`version`](commands.md#version) | Print the installed `lorahub` version. |

## Conventions

- All commands accept `--help` for the full flag list.
- Status messages use ASCII markers (`OK`, `->`) so they render correctly in
  Windows GBK consoles.
- `lorahub` auto-loads `.env` from the working directory at startup; existing
  environment variables take precedence.
- Job artifacts land under `runs/<output.name>/` by default; the SQLite
  history file is `runs/.lorahub.sqlite`.

See the full per-command reference on the [Commands](commands.md) page.
