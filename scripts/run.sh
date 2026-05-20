#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# LoRaHub Launcher (Linux)
#
# Starts the API backend (uvicorn). In prod mode the API also serves
# the prebuilt SPA from web/dist; in dev mode a separate Vite HMR
# server is started on its own port.
#
# Usage:
#   scripts/run.sh              prod mode (default — API serves SPA)
#   scripts/run.sh dev          dev mode (API + Vite HMR)
#   scripts/run.sh api          API only (no SPA build, no Vite)
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${1:-prod}"
API_HOST="127.0.0.1"
API_PORT="18765"
WEB_PORT="6006"

# ---- Add project-local tools to PATH --------------------------------
# Hard requirement: every binary the launcher touches comes from the
# project tree. We never fall back to system installs — that's the
# whole point of the install.sh layout.
[ -d "$ROOT/.tools/uv" ] && export PATH="$ROOT/.tools/uv:$PATH"
if [ -f "$ROOT/.node/bin/node" ]; then
    export PATH="$ROOT/.node/bin:$PATH"
else
    echo "[ERROR] Portable Node.js not found at .node/bin/node."
    echo "         Run scripts/install.sh first."
    exit 1
fi

# ---- Resolve Python --------------------------------------------------
PYTHON=""
if [ -f ".venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    echo "[ERROR] .venv not found. Run scripts/install.sh first."
    exit 1
fi

# ---- Verify dependencies ---------------------------------------------
if ! "$PYTHON" -c "import lorahub, fastapi, uvicorn" 2>/dev/null; then
    echo "[ERROR] Python dependencies not installed. Run scripts/install.sh first."
    exit 1
fi

echo ""
echo "============================================================"
echo "  LoRaHub - $MODE mode"
echo "============================================================"
echo ""

# Track child PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "[lorahub] Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "[lorahub] Stopped."
}
trap cleanup EXIT INT TERM

# ---- Build SPA for prod mode (must run before API start so the
#      static mount in lorahub.api.app picks up web/dist) -----------
if [[ "$MODE" == "prod" ]]; then
    if [ ! -f "web/dist/index.html" ]; then
        echo "[lorahub] Building frontend SPA ..."
        cd web && npm run build && cd "$ROOT"
        if [ ! -f "web/dist/index.html" ]; then
            echo "[ERROR] Frontend build failed — web/dist/index.html missing."
            exit 1
        fi
    fi
fi

# ---- Start API -------------------------------------------------------
if [[ "$MODE" == "dev" || "$MODE" == "prod" || "$MODE" == "api" ]]; then
    if [[ "$MODE" == "prod" ]]; then
        echo "[lorahub] Open: http://${API_HOST}:${API_PORT}"
    else
        echo "[lorahub] API:  http://${API_HOST}:${API_PORT}"
    fi
    "$PYTHON" -m uvicorn lorahub.api.app:app \
        --host "$API_HOST" --port "$API_PORT" &
    PIDS+=($!)
fi

# ---- Start Web dev server (dev mode only) ---------------------------
if [[ "$MODE" == "dev" ]]; then
    echo "[lorahub] Web:  http://localhost:${WEB_PORT}"
    export LORAHUB_API_TARGET="http://${API_HOST}:${API_PORT}"
    cd web
    npm run dev -- --host 127.0.0.1 --port "$WEB_PORT" &
    PIDS+=($!)
    cd "$ROOT"
fi

echo ""
echo "[lorahub] Services running. Press Ctrl+C to stop."
echo ""

# ---- Wait for any child to exit --------------------------------------
wait -n "${PIDS[@]}" 2>/dev/null || true
