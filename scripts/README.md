# LoraHub scripts

Three entry points, one per environment. **Always invoke these from the
repo root.**

## Local launcher (developer machine)

| Platform | Script | Notes |
|----------|--------|-------|
| Windows (PowerShell) | `scripts\launch.ps1` | Full-featured launcher; modes: `dev` / `prod` / `api` / `web` / `build` |
| Windows (cmd / Explorer) | `scripts\launch.bat` | Thin shim that forwards to `launch.ps1` |
| Windows (two-window dev) | `scripts\launch-dev.ps1` | Splits API + Vite into two consoles |
| macOS / Linux / WSL | `scripts/launch.sh` | Same modes as `launch.ps1`, runs uvicorn + vite |

Common usage:

```pwsh
# Windows: dev mode (API on 18765 + Vite on 6006, default)
scripts\launch.bat

# Windows: production mode (API serves web/dist on 18765)
scripts\launch.bat -Mode prod

# macOS / Linux / WSL: dev mode
./scripts/launch.sh

# Build web/dist without launching anything
./scripts/launch.sh --mode build
```

The first run installs Python deps (`pip install -e ".[api,dev]"`) and npm
dependencies if it sees stale state. Pass `--no-install` to skip.

## Remote VPS deploy (single-host single-user)

Three scripts cooperate. Run **`wsl_remote.sh`** from the dev box; the
other two execute on the VPS.

| Where | Script | Purpose |
|-------|--------|---------|
| Dev box | `scripts/wsl_remote.sh` | sshpass-driven WSL wrapper. Reads password from stdin or `$LORAHUB_REMOTE_PASS`. Sanitises the remote PATH so WSL's Windows-mixed PATH never leaks via heredoc expansion. |
| VPS | `scripts/remote_setup.sh` | Idempotent installer: portable Node 20, uv, .venv, `uv pip install -e .[api,tagging]`, `npm install`, `vite build`. |
| VPS | `scripts/remote_serve.sh` | Kill prior uvicorn → start on `0.0.0.0:6006` → wait for `/api/health` → print log tail. |

### Subcommands

```
scripts/wsl_remote.sh <subcommand> [args]

setup              First-time install (or re-run after big bumps)
serve              Start uvicorn on :6006
restart            Alias for serve
stop               Kill uvicorn
deploy             Full happy path: pull -> setup -> serve
pull               git fetch + reset --hard origin/main on the VPS
status             Ports, uvicorn pid, venv version, dist presence
health             curl /api/health
clean [all|venv|web|logs]   Wipe build artefacts to force fresh setup
logs [setup|install|npm|vite|uvicorn]   Tail the matching log
shell              Interactive ssh shell at the repo root
exec '<cmd>'       Run an ad-hoc remote command
```

### Typical flows

```bash
# Inside any WSL bash (sshpass + curl required, all defaults from AutoDL)
export LORAHUB_REMOTE_PASS='...'
cd "/mnt/e/WorkSpace/Lora Scripts/LoraHub"

# First-time deploy on a fresh VPS
bash scripts/wsl_remote.sh deploy
bash scripts/wsl_remote.sh logs setup     # follow the install
bash scripts/wsl_remote.sh status

# After pushing new commits
bash scripts/wsl_remote.sh deploy

# Just restart the API (no code changes)
bash scripts/wsl_remote.sh restart
bash scripts/wsl_remote.sh logs uvicorn

# Force a fresh build (e.g. after pyproject.toml bump)
bash scripts/wsl_remote.sh clean all
bash scripts/wsl_remote.sh setup

# Wipe just the venv (keep node_modules / dist)
bash scripts/wsl_remote.sh clean venv
```

### Pointing at a different VPS

`wsl_remote.sh` reads these env vars; defaults match the current AutoDL
machine.

```bash
export LORAHUB_REMOTE_HOST=connect.westc.seetacloud.com
export LORAHUB_REMOTE_PORT=45300
export LORAHUB_REMOTE_USER=root
export LORAHUB_REMOTE_DIR=/root/autodl-tmp/LoraHub
export LORAHUB_REMOTE_NODE=/root/autodl-tmp/opt/node20/bin
```

The `remote_setup.sh` mirrors are also overridable for users outside
mainland China:

```bash
LORAHUB_PYPI_INDEX=https://pypi.org/simple \
LORAHUB_NPM_REGISTRY=https://registry.npmjs.org \
LORAHUB_GH_PROXY= \
  scripts/wsl_remote.sh setup
```

## utils/

One-off helpers that don't belong in the main flow:

- `utils/dl_anima.sh` — fetch the three Anima safetensors via the HF mirror
- `utils/inspect_anima.sh` — list HF-mirror inventory for the Anima + Qwen-Image bundles
