# LoraHub scripts

Two entry points: local dev and VPS deploy.

## Local launcher

| Platform | Script | Modes |
|----------|--------|-------|
| Windows  | `scripts\run.bat` | `prod` (default) · `dev` · `api` |
| Linux / macOS / WSL | `scripts/run.sh` | `prod` (default) · `dev` · `api` |

```pwsh
# Windows
scripts\run.bat              # prod: API serves web/dist on :18765
scripts\run.bat dev          # dev: API + Vite HMR on :6006
scripts\run.bat api          # API only

# Linux / macOS / WSL
./scripts/run.sh
./scripts/run.sh dev
```

`run.*` looks for the project-local toolchain at `.lorahub/uv` and
`.node/`, falling back to system PATH if missing. First run requires
`scripts/install.{sh,bat}` to set those up.

## First-time install

Two flavours per platform — pick the one matching your network.

```pwsh
# Windows
scripts\install.bat              # upstream sources (GitHub / PyPI / nodejs.org)
scripts\install-cn.bat           # China mirrors (gh-proxy + TUNA + npmmirror)

# Linux / macOS / WSL
./scripts/install.sh
./scripts/install-cn.sh
```

Installs into the repo:
- `.lorahub/uv/` — uv toolchain
- `.lorahub/python/` — portable CPython (3.11 default; 3.13 needed for anima_lora)
- `.venv/` — main API venv
- `.node/` — portable Node 20
- `web/node_modules/` — frontend deps

## VPS deploy

Two scripts cooperate. Run them from a remote shell on the VPS — there
is no local-side wrapper script anymore (use ssh + run them directly).

| Script | Purpose |
|--------|---------|
| `scripts/remote_setup.sh` | Idempotent installer on the VPS: portable Node 20, uv, `.venv`, `uv pip install -e .[api,tagging]`, `npm install`, `vite build`. |
| `scripts/remote_serve.sh` | Kill prior uvicorn → start on `0.0.0.0:6006` → wait for `/api/health` → print log tail. |

Typical flow on the VPS:

```bash
cd /root/autodl-tmp/LoraHub

# First-time setup
bash scripts/remote_setup.sh

# Each deploy after pushing new commits
git pull
bash scripts/remote_setup.sh   # idempotent — only redoes what's stale
bash scripts/remote_serve.sh   # restart uvicorn

# Logs
tail -f /root/uvicorn.log
```

`remote_setup.sh` already defaults to in-China mirrors (gh-proxy.org +
TUNA + npmmirror) since the script targets AutoDL-style boxes; users
outside China can override with `LORAHUB_PYPI_INDEX=...`,
`LORAHUB_NPM_REGISTRY=...`, `LORAHUB_GH_PROXY=` (empty to disable).
