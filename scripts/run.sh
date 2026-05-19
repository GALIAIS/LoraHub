#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# LoRaHub Launcher (Linux)
#
# Starts the API backend (uvicorn) and frontend dev server (Vite).
# Uses project-local tools (.tools/, .node/, .venv/).
#
# Usage:
#   scripts/run.sh              dev mode (API + Vite HMR)
#   scripts/run.sh prod         production (API + prebuilt SPA)
#   scripts/run.sh api          API only
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${1:-dev}"
API_HOST="127.0.0.1"
API_PORT="18765"
WEB_PORT="6006"

# ---- Add project-local tools to PATH --------------------------------
[ -d "$ROOT/.tools/uv" ] && export PATH="$ROOT/.tools/uv:$PATH"
[ -d "$ROOT/.node/bin" ] && export PATH="$ROOT/.node/bin:$PATH"

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

# ---- Start API -------------------------------------------------------
if [[ "$MODE" == "dev" || "$MODE" == "prod" || "$MODE" == "api" ]]; then
    echo "[lorahub] API:  http://${API_HOST}:${API_PORT}"
    "$PYTHON" -m uvicorn lorahub.api.app:app \
        --host "$API_HOST" --port "$API_PORT" &
    PIDS+=($!)
fi

# ---- Build SPA for prod mode -----------------------------------------
if [[ "$MODE" == "prod" ]]; then
    if [ ! -f "web/dist/index.html" ]; then
        echo "[lorahub] Building frontend SPA ..."
        cd web && npm run build && cd "$ROOT"
    fi
fi

# ---- Start Web dev server --------------------------------------------
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
