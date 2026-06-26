# LoraHub scripts

Two entry points: local dev and VPS deploy. After first install you can
also use the `lorahub` CLI directly — see [the CLI reference](#cli-reference)
below.

## Local launcher

| Platform | Script | Modes |
|----------|--------|-------|
| Windows  | `scripts\run.bat` | `prod` (default) · `dev` · `api` |
| Linux / macOS / WSL | `scripts/run.sh` | `prod` (default) · `dev` · `api` |

```pwsh
# Windows
scripts\run.bat              # prod: API serves web/dist on :18765
scripts\run.bat dev          # dev: API + Vite HMR on :6006

# Linux / macOS / WSL
./scripts/run.sh
./scripts/run.sh dev
```

`run.*` is now a thin wrapper around the `lorahub` CLI: prod/api
modes call `lorahub service start --foreground`; dev mode keeps the
two-process (uvicorn + vite) flow because vite HMR doesn't fit the
service-daemon shape.

## SSH tunnel

Use this when the VPS web port is not exposed publicly. Keep the terminal
open, then browse to the printed local URL.

```pwsh
# Windows
scripts\tunnel.bat cwadmin@113.108.63.33 13122 18080 18765

# Linux / macOS / WSL
./scripts/tunnel.sh root@1.2.3.4 22 18080 18765
```

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

Two scripts cooperate. Run them from a remote shell on the VPS. If the web
port is not public, use `scripts/tunnel.*` from your local machine.

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

## CLI reference

After running `scripts/install.{sh,bat}` once, the `lorahub` console
script lives in `.venv/bin/`. Add it to your user PATH with:

```bash
# Linux / macOS / WSL
.venv/bin/lorahub manage install   # creates ~/.local/bin/lorahub symlink

# Windows
.venv\Scripts\lorahub manage install   # writes %LOCALAPPDATA%\lorahub\bin shim + setx PATH
```

After that, `lorahub` is available globally. Restart your shell to pick
up the new PATH.

### Day-to-day commands

| Command | What it does |
|---------|--------------|
| `lorahub doctor` | Print env health (venv / Node / web/dist / backends) |
| `lorahub service start [--port N]` | Start the API daemon (random port if not specified) |
| `lorahub service stop` | Stop the daemon |
| `lorahub service restart` | Stop + start |
| `lorahub service status` | Show pid + port + healthy/unhealthy |
| `lorahub service logs -f` | Tail the daemon log |
| `lorahub service enable` | Register as a systemd / launchd service (Linux/macOS, requires sudo) |
| `lorahub service disable` | Unregister |
| `lorahub service install-unit --print` | Print the unit file without writing |
| `lorahub manage update` | git pull + reinstall deps + rebuild SPA |
| `lorahub manage update --skip-build` | Backend-only update |
| `lorahub manage upgrade` | Switch to the newest `v*` tag |
| `lorahub manage build` | Rebuild the frontend (vite build) |
| `lorahub manage path` | Where is the active `lorahub` command? |
| `lorahub serve` | Foreground uvicorn (dev / explicit) — same as `service start --foreground` |
| `lorahub train / validate / info / sweep / init / ...` | Existing config + training commands |
| `lorahub --lang en <subcommand>` | Switch help / output to English (default zh, also LORAHUB_LANG env) |
