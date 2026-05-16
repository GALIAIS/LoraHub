---
title: HTTP API
description: REST + WebSocket surface for the LoraHub workbench.
---

# HTTP API

LoraHub ships a FastAPI server for programmatic access. Install API extras
and start it:

```powershell
pip install lorahub[api]
lorahub serve --port 18765
```

The server binds to `127.0.0.1` by default and has no auth — safe for
localhost only. Job metadata persists to SQLite at `runs/.lorahub.sqlite`;
live handles and the recent event ring remain process-local.

## Layout

- All API routes live under `/api`.
- The site root and `/{spa-path}` are reserved for the React frontend, mounted
  from `web/dist` when the build artifact is present.
- Per-domain endpoint logic lives under `lorahub.api.routers.*` (one router
  per resource: jobs, recipes, datasets, settings, ...).
- WebSocket endpoints stay on the top-level FastAPI app to side-step
  historical caveats with `APIRouter` + WS routes.

## One-shot launcher

If you'd rather not memorise `pip install` and `npm run dev` separately, the
`scripts/` folder ships a cross-platform launcher that resolves the project
venv (or system Python), installs missing dependencies on first run, and
brings up the API and the React dev server side by side:

=== "Windows"

    ```powershell
    scripts\launch.bat              # default: dev mode (API + Vite)
    scripts\launch.bat -Mode prod   # API only, serves prebuilt web/dist
    scripts\launch.bat -Mode build  # one-shot npm install + vite build
    ```

=== "macOS / Linux"

    ```bash
    chmod +x scripts/launch.sh
    scripts/launch.sh                       # default: dev mode
    scripts/launch.sh --mode prod --port 8080
    scripts/launch.sh --mode build
    ```

The launcher auto-detects `.venv/` and `web/node_modules/`, runs
`pip install -e ".[api,dev]"` and `npm install` only when something's missing,
and forwards Vite's `/api` proxy to whichever port the API ended up on. Pass
`--no-install` to skip the dependency check, `--reload` for uvicorn
auto-reload.

## Reading the OpenAPI

When the server is running, FastAPI serves the live OpenAPI document at
`/openapi.json` and an interactive Swagger UI at `/docs`. Those are
authoritative — the [Endpoints](endpoints.md) page summarises the surface for
quick reference.
