---
title: API endpoints
description: REST + SSE + WebSocket endpoint reference.
---

# API endpoints

All endpoints live under `/api`. Quick summary follows; the live OpenAPI
document at `/openapi.json` (served by `lorahub serve`) is authoritative.

## Health & settings

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/health` | Server status, version, and backend probe. |
| GET    | `/api/settings` | Workbench defaults (kohya / dp paths, Python exe, tagger device, max_concurrent_jobs). |
| PUT    | `/api/settings` | Update workbench defaults. |

## Configs

The endpoint set was renamed from `/api/recipes` to `/api/configs` to match
the on-disk `configs/` directory. The validator and schema accept both
`camelCase` (the new wire format) and `snake_case`.

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/configs/schema` | Config JSON Schema used by the visual editor. |
| GET    | `/api/configs` | List config YAML files. |
| GET    | `/api/configs/{name}` | Inspect a single config (returns parsed `camelCase` body). |
| POST   | `/api/configs/validate` | Validate an in-memory config; returns Pydantic field errors. |
| POST   | `/api/configs` | Save a validated config to `configs/<name>.yaml`. |
| POST   | `/api/configs/import` | Upload an existing config file. |
| POST   | `/api/configs/{name}/duplicate` | Duplicate a config under a new name. |
| POST   | `/api/configs/{name}/rename` | Rename a config file. |
| DELETE | `/api/configs/{name}` | Delete a config. |
| GET    | `/api/configs/templates` | List built-in templates with placeholder metadata. |
| POST   | `/api/configs/templates/{template_id}/instantiate` | Fill placeholders + persist a new config. |

## Jobs

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/jobs` | List training jobs (live + historical). |
| GET    | `/api/jobs/{id}` | Inspect a single job. |
| POST   | `/api/jobs` | Start a job — body: `{config, workspace?}`. |
| POST   | `/api/jobs/{id}/rerun` | Re-launch the job in place. Clears `events.jsonl` for plain rerun; preserves it for resume. |
| POST   | `/api/jobs/{id}/resume` | Resume from the latest checkpoint. |
| POST   | `/api/jobs/{id}/reveal` | Open the workspace folder in the host OS file manager. |
| GET    | `/api/jobs/{id}/events` | Recent events from the in-memory ring buffer. |
| GET    | `/api/jobs/{id}/files` | List files written into the workspace. |
| GET    | `/api/jobs/{id}/files/raw` | Stream a single workspace file. |
| GET    | `/api/jobs/{id}/metrics` | Aggregated per-step metrics for charts (loss, val_loss, gpu_samples, samples, checkpoints, overfit_signal). |
| GET    | `/api/jobs/{id}/analysis` | Cached AI training analysis (Markdown). |
| POST   | `/api/jobs/{id}/analyze` | Generate (or regenerate) the AI training analysis. |
| DELETE | `/api/jobs/{id}` | Cancel a running job (or archive a finished one with `?archive=true`). |
| POST   | `/api/jobs/{id}/kill` | SIGKILL the process group — last resort for hung trainers. |

## Datasets

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/datasets/scan` | Walk a directory and return image + caption metadata. |
| GET    | `/api/datasets/thumb` | Stream a cached thumbnail for a dataset image. |
| GET    | `/api/datasets/caption` | Read the `.txt` caption next to an image. |
| PUT    | `/api/datasets/caption` | Write the `.txt` caption next to an image. |

## Image Studio

The Image Studio is a dedicated dataset manager with virtualized grids,
multi-select, drag-and-drop upload, and several batch AI endpoints. All paths
below resolve under `LORAHUB_DATASETS_ROOT` (or the project's `./datasets`).

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/image-studio/datasets` | List all datasets in the configured roots. |
| GET    | `/api/image-studio/list` | Paged listing of images in a dataset directory. |
| GET    | `/api/image-studio/image` | Inspect a single image (resolution, caption, annotation). |
| POST   | `/api/image-studio/annotation` | Upsert favourite / soft-delete / quality fields. |
| POST   | `/api/image-studio/op/add` | Queue an op (rotate / flip / delete / caption / favorite). |
| POST   | `/api/image-studio/op/apply` | Apply queued ops for a single image. |
| POST   | `/api/image-studio/batch/delete` | Soft-delete by path list (move to `_image_studio_trash/`). |
| POST   | `/api/image-studio/upload` | Multipart upload into a dataset; auto-extracts archives. |
| POST   | `/api/image-studio/ai/caption` | Pure VLM caption (vision LLM only). |
| POST   | `/api/image-studio/ai/quality` | Batch quality scoring via the AI router. |
| POST   | `/api/image-studio/ai/smart-caption` | WD14 + vision LLM combined caption (Anima format). Modes: `style` / `character` / `general`; `triggerWord` and `stripStyleTags` are honoured. |
| POST   | `/api/image-studio/ai/smart-caption/single` | Single-image variant of the above. |
| POST   | `/api/image-studio/dedupe/scan` | Async perceptual-hash duplicate scan. |
| GET    | `/api/image-studio/dedupe/clusters` | Read clusters produced by the scan. |
| POST   | `/api/image-studio/similarity/scan` | Async embedding-based similarity scan. |
| GET    | `/api/image-studio/similarity/results` | Read pairwise similarity results. |

## Tagging

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/api/tagging/tag` | Start an async tagging session. Default tagger model: `wd-eva02-large-v3`. |
| GET    | `/api/tagging/tag/{session_id}` | Poll the running tagging session. |

## AI providers + routes

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/ai/providers` / `/api/ai/providers/{id}` | List / inspect AI providers. |
| POST   | `/api/ai/providers` | Create a provider (Anthropic, OpenAI, Vertex, custom OpenAI-compat). |
| GET    | `/api/ai/providers/{id}/models` | Discover models from the provider. |
| POST   | `/api/ai/providers/{id}/test` | Smoke-test connectivity + auth. |
| GET    | `/api/ai/routes` | List task routes (`global.default`, `tagging.assist`, `training.analyze`, ...). |
| PUT    | `/api/ai/routes/{task}` | Bind a task to a `(provider, model)` pair. |
| POST   | `/api/ai/invoke` | Direct synchronous invocation against a route. |

## Models

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/api/models/download` | Start an async base-model download from Hugging Face / ModelScope. |
| GET    | `/api/models/download/{session_id}` | Poll the running download session. |

## Captions pipeline

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/api/captions/normalize` | Run the offline caption transforms (shuffle / dropout / booru-alias / keep-tokens). |
| POST   | `/api/anima/caption` | Anima-format caption normaliser. |

## Samples

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/samples` | List or stream sample images written during training. |

## System

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/system/stats` | One-shot host snapshot: CPU, RAM, GPU, network, disks. |
| GET    | `/api/system/attention-backends` | Probe which attention kernels (sdpa / xformers / flash / sageattn) are available on this GPU. |

## Backend bootstrap

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/backends` | Catalogue of supported training backends. |
| GET    | `/api/backend/bootstrap/status` | Probe whether the kohya / dp bootstrap is healthy. |
| POST   | `/api/backend/bootstrap` | Kick off `lorahub bootstrap-*` in-process. |
| GET    | `/api/runtime/python` | Probe the Python runtime kohya / dp should use. |
| POST   | `/api/runtime/python/install` | Install a managed Python runtime. |

## Network presets

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/api/network/presets` | Built-in network presets (LoRA / LoCon / LoHa / DoRA). |
| POST   | `/api/network/probe` | Probe a base-model checkpoint to suggest network targets. |

## SSE streams (preferred)

Each event is tagged with a monotone `id: <seq>`. Browsers' `EventSource`
forwards `Last-Event-ID` on reconnect and the server resumes from that index
so reconnects are gap-free.

| Path | Payload |
| ---- | ------- |
| `/api/jobs/{id}/sse` | Live `TrainingEvent` stream; replays the per-job ring buffer first, then forwards new events. Includes `step` / `epoch_end` / `validation` / `checkpoint_saved` / `sample_ready` / `cache_progress` / `oom` / `gpu_sample` / `done` events. |
| `/api/backend/bootstrap/sse` | Live progress for the running `bootstrap-*` session. |
| `/api/system/sse` | Hardware/host snapshot pushed every second; idle pings every 25 s as `: ping` SSE comments. |

## Legacy WebSocket streams (fallback)

| Path | Payload |
| ---- | ------- |
| `/api/jobs/{id}/stream` | Same payload as the SSE counterpart; sends `{type: "ping"}` heartbeats every 25 s. |
| `/api/backend/bootstrap/stream` | Same payload as the SSE counterpart. |
| `/api/system/stream` | Same payload as the SSE counterpart; pre-SSE web client falls back here. |
