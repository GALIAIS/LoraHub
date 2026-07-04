#!/usr/bin/env bash
# LoraHub container entrypoint.
#
# Responsibilities (in order):
#   1. Create the runtime user matching PUID/PGID so volume file
#      ownership lines up with the host (linuxserver-style).
#   2. Ensure every LORAHUB_HOME subdirectory exists and is owned by
#      that user.
#   3. Re-affirm the XDG / HF / uv cache env redirects (the Dockerfile
#      sets them as ENV, but re-exporting here keeps the entrypoint
#      usable when someone overrides the entrypoint or runs the script
#      by hand).
#   4. Drop privileges and `exec uvicorn` against the FastAPI app.
#
# The container binds 0.0.0.0 internally; the compose port mapping is
# what keeps it at 127.0.0.1 on the host (LoraHub has no built-in auth
# — never expose the port directly to the public internet).
set -euo pipefail

# ── 0. Resolve config from env (with sane defaults) ──────────────────
LORAHUB_HOME="${LORAHUB_HOME:-/data}"
LORAHUB_PORT="${LORAHUB_PORT:-18765}"
LORAHUB_BIND_HOST="${LORAHUB_BIND_HOST:-0.0.0.0}"
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
BACKEND_VENV_DIRS=(
    "/app/external/anima_lora/.venv"
    "/app/external/ai_toolkit/venv"
)
BACKEND_MODELS_LINKS=(
    "/app/external/anima_lora/models"
    "/app/external/ai_toolkit/models"
)

# State redirects — must match the Dockerfile ENV block. Re-stated so
# a manual `docker run --entrypoint bash` invocation still lands state
# inside the volume.
export XDG_DATA_HOME="${XDG_DATA_HOME:-$LORAHUB_HOME/xdg/data}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$LORAHUB_HOME/xdg/state}"
export HF_HOME="${HF_HOME:-$LORAHUB_HOME/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$LORAHUB_HOME/.cache/uv}"

# ── 1. Runtime user ──────────────────────────────────────────────────
# Run as root only to chown the volume, then drop to the lorahub user.
# If the container is already non-root (e.g. `docker run --user`), skip
# the chown and trust the operator.
if [ "$(id -u)" = "0" ]; then
    # Create the group if missing (allow non-numeric PUID too).
    if ! getent group lorahub >/dev/null 2>&1; then
        groupadd -o -g "$PGID" lorahub
    fi
    # Create the user if missing.
    if ! id -u lorahub >/dev/null 2>&1; then
        useradd -o -u "$PUID" -g "$PGID" -d "$LORAHUB_HOME" -s /bin/bash lorahub
    fi
    # Align UID/GID to PUID/PGID in case the image defaults differ and
    # the operator passed custom values at runtime.
    usermod -o -u "$PUID" -g "$PGID" lorahub 2>/dev/null || true
    groupmod -o -g "$PGID" lorahub 2>/dev/null || true

    # ── 2. Ensure data tree exists & is owned by the runtime user ──
    mkdir -p \
        "$LORAHUB_HOME" \
        "$LORAHUB_HOME/runs" \
        "$LORAHUB_HOME/configs" \
        "$LORAHUB_HOME/datasets" \
        "$LORAHUB_HOME/models" \
        "$LORAHUB_HOME/workspaces" \
        "$LORAHUB_HOME/samples" \
        "$LORAHUB_HOME/checkpoints" \
        "$LORAHUB_HOME/.lorahub" \
        "$LORAHUB_HOME/.cache" \
        "$LORAHUB_HOME/xdg/data" \
        "$LORAHUB_HOME/xdg/state" \
        "$LORAHUB_HOME/hf-home"
    for dir in "${BACKEND_VENV_DIRS[@]}"; do
        mkdir -p "$dir"
    done
    for link in "${BACKEND_MODELS_LINKS[@]}"; do
        if [ ! -e "$link" ] && [ ! -L "$link" ]; then
            ln -s "$LORAHUB_HOME/models" "$link"
        fi
    done
    chown -R lorahub:lorahub "$LORAHUB_HOME" "${BACKEND_VENV_DIRS[@]}"
    for link in "${BACKEND_MODELS_LINKS[@]}"; do
        chown -h lorahub:lorahub "$link" 2>/dev/null || true
    done

    exec gosu lorahub:lorahub "$0" "$@"
fi

# ── 3. (non-root path) Re-mkdir as the runtime user — no-op if present
mkdir -p \
    "$LORAHUB_HOME/runs" \
    "$LORAHUB_HOME/configs" \
    "$LORAHUB_HOME/datasets" \
    "$LORAHUB_HOME/models" \
    "$LORAHUB_HOME/workspaces" \
    "$LORAHUB_HOME/samples" \
    "$LORAHUB_HOME/checkpoints" \
    "$LORAHUB_HOME/.lorahub" \
    "$LORAHUB_HOME/.cache" \
    "$XDG_DATA_HOME" \
    "$XDG_STATE_HOME" \
    "$HF_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "${BACKEND_VENV_DIRS[@]}"

# ── 4. Launch uvicorn ────────────────────────────────────────────────
# `lorahub.api.app:app` is the same target scripts/run.sh uses. The
# lifespan hook chdir's to LORAHUB_HOME itself, so cwd doesn't matter.
echo "[lorahub] starting on ${LORAHUB_BIND_HOST}:${LORAHUB_PORT}  (LORAHUB_HOME=${LORAHUB_HOME})"
exec uvicorn lorahub.api.app:app \
    --host "$LORAHUB_BIND_HOST" \
    --port "$LORAHUB_PORT"
