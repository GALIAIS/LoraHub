---
title: API endpoints
description: REST + WebSocket endpoint reference.
---

# API endpoints

All endpoints live under `/api`. Quick summary follows; the live OpenAPI
document at `/openapi.json` (served by `lorahub serve`) is authoritative.

## Health & settings

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/health` | Server status, version, and backend probe. |
| GET    | `/api/settings` | Workbench defaults (kohya path, Python exe, tagger device). |
| PUT    | `/api/settings` | Update workbench defaults. |

## Recipes

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/recipes/schema` | Recipe JSON Schema used by the visual editor. |
| GET    | `/api/recipes` | List recipe YAML files. |
| GET    | `/api/recipes/{name}` | Inspect a single recipe. |
| POST   | `/api/recipes/validate` | Validate an in-memory recipe; returns Pydantic field errors. |
| POST   | `/api/recipes` | Save a validated recipe to `recipes/<name>.yaml`. |
| POST   | `/api/recipes/import` | Upload an existing recipe file. |
| POST   | `/api/recipes/{name}/duplicate` | Duplicate a recipe under a new name. |
| POST   | `/api/recipes/{name}/rename` | Rename a recipe file. |
| DELETE | `/api/recipes/{name}` | Delete a recipe. |
| GET    | `/api/recipes/templates` | List built-in templates with placeholder metadata. |
| POST   | `/api/recipes/templates/{template_id}/instantiate` | Fill placeholders + persist a new recipe. |

## Jobs

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/jobs` | List training jobs (live + historical). |
| GET    | `/api/jobs/{id}` | Inspect a single job. |
| POST   | `/api/jobs` | Start a job — body: `{recipe, workspace?}`. |
| POST   | `/api/jobs/{id}/rerun` | Re-launch a previously executed job. |
| POST   | `/api/jobs/{id}/resume` | Resume an interrupted job from its saved state. |
| POST   | `/api/jobs/{id}/reveal` | Open the workspace folder in the host OS file manager. |
| GET    | `/api/jobs/{id}/events` | Recent events from the in-memory ring buffer. |
| GET    | `/api/jobs/{id}/files` | List files written into the workspace. |
| GET    | `/api/jobs/{id}/files/raw` | Stream a single workspace file. |
| GET    | `/api/jobs/{id}/metrics` | Aggregated per-step metrics for charts. |
| DELETE | `/api/jobs/{id}` | Stop a running job (or archive a finished one with `?archive=true`). |

## Datasets

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/datasets/scan` | Walk a directory and return image + caption metadata. |
| GET    | `/api/datasets/thumb` | Stream a cached thumbnail for a dataset image. |
| GET    | `/api/datasets/caption` | Read the `.txt` caption next to an image. |
| PUT    | `/api/datasets/caption` | Write the `.txt` caption next to an image. |

## Tagging

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/api/tagging/tag` | Start an async tagging session over a directory. |
| GET    | `/api/tagging/tag/{session_id}` | Poll the running tagging session. |

## Models

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/api/models/download` | Start an async base-model download from Hugging Face. |
| GET    | `/api/models/download/{session_id}` | Poll the running download session. |

## Samples

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/samples` | List or stream sample images written during training. |

## System

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/system/stats` | One-shot host snapshot: CPU, RAM, GPU. |

## Backend bootstrap

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/backends` | Catalogue of supported training backends. |
| GET    | `/api/backend/bootstrap/status` | Probe whether the kohya bootstrap is healthy. |
| POST   | `/api/backend/bootstrap` | Kick off `lorahub bootstrap-kohya` in-process. |
| GET    | `/api/runtime/python` | Probe the Python runtime kohya should use. |
| POST   | `/api/runtime/python/install` | Install a managed Python runtime. |

## Network presets

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/network/presets` | Built-in network presets (LoRA / LoCon / LoHa / DoRA). |
| POST   | `/api/network/probe` | Probe a base-model checkpoint to suggest network targets. |

## WebSocket streams

| Path | Payload |
| ---- | ------- |
| `/api/jobs/{id}/stream` | Live `TrainingEvent` stream; replays the per-job ring buffer first, then forwards new events. |
| `/api/backend/bootstrap/stream` | Live progress for the running `bootstrap-kohya` session. |
| `/api/system/stream` | Hardware/host snapshot pushed every second until the client disconnects. |
