# Background Task Persistence Design

## Goal
Make long-running non-training tasks durable and recoverable across page refreshes and server restarts, starting with model downloads, Anima model downloads, backend installs, MSVC installs, and self-update runs.

## Scope
This design covers API-visible background sessions that currently keep most or all state in process memory. It does not change the training job scheduler, sweep runtime, or Image Studio task queue in the first implementation batch.

## Current State
LoraHub already persists training jobs and sweeps, but several operational workflows are in-memory:

- Generic model downloads keep `_sessions` in `lorahub/api/routers/models.py`.
- Anima model download and MSVC install sessions keep dictionaries in `lorahub/api/routers/backends.py`.
- Backend bootstrap has a dedicated session object and status endpoint.
- System update streams progress through SSE and has concurrency guards, but its progress history is not represented as a common task record.

The UI has started to compensate with `localStorage`, but localStorage only survives browser refreshes. It cannot recover state after API restart and it fragments recovery behavior by page.

## Proposed Architecture
Introduce a small generic task-session layer under `lorahub/api/task_sessions.py` backed by a SQLite database under the existing user state/data path. It stores a bounded event log, status fields, timestamps, lightweight metadata, and optional result/error payloads. Routers can continue to own execution, but they write session state through this shared store.

The first batch should keep execution semantics unchanged. For tasks that cannot safely resume subprocess work after process death, restart recovery marks previously running sessions as `interrupted` with an explanatory event. For tasks that are idempotent and internally skip existing files, the UI can offer a retry button that starts a fresh session with the same request metadata.

## Session Model
Each task session has:

- `id`: stable UUID/hex string.
- `kind`: e.g. `model_download`, `anima_model_download`, `msvc_install`, `backend_bootstrap`, `system_update`.
- `status`: `queued`, `running`, `succeeded`, `failed`, `canceled`, `interrupted`.
- `title`: human-readable summary for UI lists.
- `metadata`: JSON object containing request shape, source, repo id, backend id, paths, target dir, etc.
- `result`: JSON object or null.
- `error`: string or null.
- `percent`: float 0-100.
- `started_at`, `updated_at`, `finished_at`: unix timestamps.
- `events`: append-only bounded event rows with `level`, `message`, `percent`, and payload JSON.

## API Shape
Add internal Python APIs first, then expose small HTTP endpoints:

- `GET /api/tasks?kind=&limit=` lists recent task summaries.
- `GET /api/tasks/{id}` returns one task with recent events.
- `GET /api/tasks/latest?kind=` returns latest task for a kind.

Existing endpoints remain for compatibility. They can be backed by the common store and return their current response shapes.

## Migration Strategy
No data migration is required. On startup, the store initializes tables if missing and marks stale `queued`/`running` records as `interrupted`. Existing in-memory dictionaries can remain as runtime accelerators during the first batch but must not be the source of truth for status reads.

## Error Handling
All task writer methods are best-effort but should log failures. If persistence fails, the live task should continue and the route can still return in-memory state. Store writes are small and synchronous; this is acceptable for low-frequency progress events but events should be capped to avoid unbounded growth.

## Testing
Add focused tests for:

- Creating, updating, appending events, and reading sessions.
- Startup recovery marking stale running sessions interrupted.
- Generic model download writing to the store and latest endpoint surviving in-memory dictionary loss.
- Anima model download preserving current response shape while using the common store.

## First Implementation Batch
Batch 1 should implement the store and migrate generic model download plus Anima model download. MSVC/bootstrap/system update should follow after the store proves stable, because they have different execution semantics and more process-launch edge cases.
