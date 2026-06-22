#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# LoRaHub Launcher (Linux / macOS)
#
# Thin wrapper around the `lorahub` CLI. After running
# `scripts/install.sh` once, you can invoke `lorahub service start`
# directly — this script exists for muscle-memory and for setting
# up the project-local PATH so a fresh shell doesn't need to source
# the venv first.
#
# Usage:
#   scripts/run.sh              prod mode (foreground uvicorn on :18765)
#   scripts/run.sh dev          dev mode  (uvicorn :18765 + Vite :6006)
#   scripts/run.sh api          alias for prod
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${1:-prod}"
API_HOST="127.0.0.1"
API_PORT="18765"
WEB_PORT="6006"

# ---- Add project-local tools to PATH --------------------------------
[ -d "$ROOT/.lorahub/uv" ] && export PATH="$ROOT/.lorahub/uv:$PATH"
NODE_BIN=""
if [ -n "${NODE_DIR:-}" ] && [ -f "$NODE_DIR/bin/node" ]; then
    NODE_BIN="$NODE_DIR/bin"
elif [ -f "$ROOT/.node/bin/node" ]; then
    NODE_BIN="$ROOT/.node/bin"
elif [ -f "/root/autodl-tmp/opt/node20/bin/node" ]; then
    NODE_BIN="/root/autodl-tmp/opt/node20/bin"
fi
[ -n "$NODE_BIN" ] && export PATH="$NODE_BIN:$PATH"

# ---- Resolve Python --------------------------------------------------
if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] .venv not found. Run scripts/install.sh first."
    exit 1
fi
PYTHON="$ROOT/.venv/bin/python"

# ---- Sanity-check API deps ------------------------------------------
if ! "$PYTHON" -c "import lorahub, fastapi, uvicorn" 2>/dev/null; then
    echo "[ERROR] Python dependencies missing. Run scripts/install.sh first."
    exit 1
fi

# ---- Dev mode: vite + uvicorn in two foreground children ------------
# The CLI's `service start` only manages a single uvicorn daemon; vite
# HMR + auto-reload during dev is a different shape (two processes,
# both noisy on stdout, both quit together). Keep the legacy two-child
# logic for `dev` only.
if [[ "$MODE" == "dev" ]]; then
    if [[ -z "$NODE_BIN" ]]; then
        echo "[ERROR] Portable Node.js missing. Run scripts/install.sh first."
        exit 1
    fi
    PIDS=()
    cleanup() {
        echo ""; echo "[lorahub] Shutting down ..."
        for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
        wait 2>/dev/null
    }
    trap cleanup EXIT INT TERM

    echo "[lorahub] API:  http://${API_HOST}:${API_PORT}"
    "$PYTHON" -m uvicorn lorahub.api.app:app --host "$API_HOST" --port "$API_PORT" --reload &
    PIDS+=($!)

    echo "[lorahub] Web:  http://localhost:${WEB_PORT}"
    export LORAHUB_API_TARGET="http://${API_HOST}:${API_PORT}"
    (cd web && npm run dev -- --host 127.0.0.1 --port "$WEB_PORT") &
    PIDS+=($!)

    wait -n "${PIDS[@]}" 2>/dev/null || true
    exit 0
fi

# ---- Prod / api: build SPA if missing, then run via the CLI --------
if [[ ! -f "web/dist/index.html" ]]; then
    echo "[lorahub] Building frontend SPA ..."
    "$PYTHON" -m lorahub manage build || {
        echo "[ERROR] Frontend build failed."
        exit 1
    }
fi

echo "[lorahub] starting on http://${API_HOST}:${API_PORT}"
exec "$PYTHON" -m lorahub service start --host "$API_HOST" --port "$API_PORT" --foreground
