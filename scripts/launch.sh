#!/usr/bin/env bash
# LoraHub launcher (macOS / Linux).
#
# Usage:
#   ./scripts/launch.sh                         # dev: API + Vite
#   ./scripts/launch.sh --mode prod             # API only, serves web/dist
#   ./scripts/launch.sh --mode api              # API only
#   ./scripts/launch.sh --mode web              # Vite only
#   ./scripts/launch.sh --mode build            # one-shot npm install + vite build
#   ./scripts/launch.sh --port 8080             # change API port
#   ./scripts/launch.sh --reload                # uvicorn --reload
#   ./scripts/launch.sh --no-install            # skip first-run installs
#
# Detects .venv/{Scripts,bin}/python first, falls back to python3 / python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${LORAHUB_MODE:-dev}"
API_HOST="${LORAHUB_API_HOST:-127.0.0.1}"
API_PORT="${LORAHUB_API_PORT:-18765}"
WEB_PORT="${LORAHUB_WEB_PORT:-6006}"
RELOAD=0
NO_INSTALL=0

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)        MODE="${2:?--mode requires a value}"; shift 2;;
    --host)        API_HOST="${2:?--host requires a value}"; shift 2;;
    --port)        API_PORT="${2:?--port requires a value}"; shift 2;;
    --web-port)    WEB_PORT="${2:?--web-port requires a value}"; shift 2;;
    --reload)      RELOAD=1; shift;;
    --no-install)  NO_INSTALL=1; shift;;
    -h|--help)     usage 0;;
    *)             echo "unknown arg: $1" >&2; usage 1;;
  esac
done

case "$MODE" in
  dev|prod|api|web|build) ;;
  *) echo "invalid --mode: $MODE (expected dev|prod|api|web|build)" >&2; exit 1;;
esac

info()  { printf '\033[36m[lorahub]\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m[lorahub]\033[0m %s\n' "$*"; }
errln() { printf '\033[31m[lorahub]\033[0m %s\n' "$*" >&2; }

# --- Python resolution ---------------------------------------------------
resolve_python() {
  if [ -x "$ROOT/.venv/bin/python" ];          then echo "$ROOT/.venv/bin/python"; return; fi
  if [ -x "$ROOT/.venv/Scripts/python.exe" ];  then echo "$ROOT/.venv/Scripts/python.exe"; return; fi
  if command -v python3 >/dev/null 2>&1;       then command -v python3; return; fi
  if command -v python  >/dev/null 2>&1;       then command -v python; return; fi
  errln 'No Python interpreter found. Install Python 3.11+ and re-run.'
  exit 1
}

python_ready() {
  local py="$1"
  "$py" -c 'import lorahub, fastapi, uvicorn' >/dev/null 2>&1
}

install_python_deps() {
  local py="$1"
  info 'Installing Python dependencies (lorahub[api,dev], editable)...'
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -e ".[api,dev]"
}

# --- Node ----------------------------------------------------------------
node_ready() { [ -d "$ROOT/web/node_modules/vite" ]; }

install_web_deps() {
  info 'Installing web dependencies (npm install)...'
  ( cd "$ROOT/web" && npm install )
}

build_web() {
  info 'Building web (vite build)...'
  ( cd "$ROOT/web" && npm run build )
}

# --- Port probe ----------------------------------------------------------
port_free() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ! ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .
  else
    # Fallback: optimistic. The actual listener will fail loudly.
    return 0
  fi
}

# --- Service launches ----------------------------------------------------
PIDS=()

start_api() {
  local py="$1"
  if ! port_free "$API_PORT"; then
    errln "Port $API_PORT is already in use. Pass --port <other> or stop the holder."
    exit 1
  fi
  info "API:  http://${API_HOST}:${API_PORT}"
  local args=(-m uvicorn lorahub.api.app:app --host "$API_HOST" --port "$API_PORT")
  if [ "$RELOAD" -eq 1 ]; then args+=(--reload); fi
  "$py" "${args[@]}" &
  PIDS+=($!)
}

start_web() {
  if ! port_free "$WEB_PORT"; then
    errln "Port $WEB_PORT is already in use. Pass --web-port <other> or stop the holder."
    exit 1
  fi
  export LORAHUB_API_TARGET="http://${API_HOST}:${API_PORT}"
  info "Web:  http://localhost:${WEB_PORT}  (proxying /api -> $LORAHUB_API_TARGET)"
  ( cd "$ROOT/web" && npm run dev -- --host 127.0.0.1 --port "$WEB_PORT" ) &
  PIDS+=($!)
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # Give children a moment, then escalate.
  sleep 0.5
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# --- Main ---------------------------------------------------------------
PY="$(resolve_python)"
info "Python: $PY"

if [ "$MODE" = 'dev' ] || [ "$MODE" = 'prod' ] || [ "$MODE" = 'api' ] || [ "$MODE" = 'build' ]; then
  if [ "$NO_INSTALL" -eq 0 ] && ! python_ready "$PY"; then
    install_python_deps "$PY"
  fi
fi

if [ "$MODE" = 'dev' ] || [ "$MODE" = 'web' ] || [ "$MODE" = 'build' ]; then
  if [ "$NO_INSTALL" -eq 0 ] && ! node_ready; then
    install_web_deps
  fi
fi

if [ "$MODE" = 'build' ]; then
  build_web
  info 'Build complete. Run --mode prod to serve the built SPA.'
  exit 0
fi

if [ "$MODE" = 'prod' ] && [ ! -f "$ROOT/web/dist/index.html" ]; then
  warn 'web/dist not found — running a build first.'
  if ! node_ready; then install_web_deps; fi
  build_web
fi

case "$MODE" in
  dev)  start_api "$PY"; start_web;;
  prod) start_api "$PY";;
  api)  start_api "$PY";;
  web)  start_web;;
esac

info 'All services launched. Press Ctrl+C to stop.'
# Wait on any child; if any exits we tear the rest down via the trap.
wait -n
