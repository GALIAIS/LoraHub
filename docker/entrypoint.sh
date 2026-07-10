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
# The container binds 0.0.0.0 internally. Its API token is generated in
# the persistent state volume unless LORAHUB_API_TOKEN is supplied.
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

# ``gosu`` changes credentials but deliberately preserves the environment.
# Without this override the process inherits the image's HOME=/root and some
# third-party libraries try to write caches/configuration into an unwritable
# directory after privileges are dropped.
export HOME="$LORAHUB_HOME"

# State redirects — must match the Dockerfile ENV block. Re-stated so
# a manual `docker run --entrypoint bash` invocation still lands state
# inside the volume.
export XDG_DATA_HOME="${XDG_DATA_HOME:-$LORAHUB_HOME/xdg/data}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$LORAHUB_HOME/xdg/state}"
export HF_HOME="${HF_HOME:-$LORAHUB_HOME/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$LORAHUB_HOME/.cache/uv}"

if ! [[ "$PUID" =~ ^[1-9][0-9]*$ ]] || ! [[ "$PGID" =~ ^[1-9][0-9]*$ ]]; then
    echo "[lorahub] PUID and PGID must be positive numeric IDs" >&2
    exit 2
fi

DATA_DIRS=(
    "$LORAHUB_HOME"
    "$LORAHUB_HOME/runs"
    "$LORAHUB_HOME/configs"
    "$LORAHUB_HOME/datasets"
    "$LORAHUB_HOME/models"
    "$LORAHUB_HOME/workspaces"
    "$LORAHUB_HOME/samples"
    "$LORAHUB_HOME/checkpoints"
    "$LORAHUB_HOME/.lorahub"
    "$LORAHUB_HOME/.cache"
    "$XDG_DATA_HOME"
    "$XDG_STATE_HOME"
    "$HF_HOME"
    "$HUGGINGFACE_HUB_CACHE"
    "$UV_CACHE_DIR"
)

# ── 1. Runtime user ──────────────────────────────────────────────────
# Run as root only to chown the volume, then drop to the lorahub user.
# If the container is already non-root (e.g. `docker run --user`), skip
# the chown and trust the operator.
if [ "$(id -u)" = "0" ]; then
    # Create the runtime identity using the requested numeric IDs.
    if getent group lorahub >/dev/null 2>&1; then
        groupmod -o -g "$PGID" lorahub
    else
        groupadd -o -g "$PGID" lorahub
    fi
    # Resolve the group before assigning it to the user. Doing this in the
    # opposite order fails when an existing container is restarted with a new
    # PGID, leaving the account attached to a group that no longer exists.
    if id -u lorahub >/dev/null 2>&1; then
        usermod -o -u "$PUID" -g lorahub -d "$LORAHUB_HOME" lorahub
    else
        useradd -o -u "$PUID" -g lorahub -d "$LORAHUB_HOME" -s /bin/bash lorahub
    fi

    # ── 2. Ensure data tree exists & is owned by the runtime user ──
    mkdir -p "${DATA_DIRS[@]}" "${BACKEND_VENV_DIRS[@]}"
    for link in "${BACKEND_MODELS_LINKS[@]}"; do
        if [ ! -e "$link" ] && [ ! -L "$link" ]; then
            ln -s "$LORAHUB_HOME/models" "$link"
        fi
    done
    # Never recursively chown the data volume during routine startup. It can
    # contain millions of model/dataset files or host bind mounts. New files
    # inherit the runtime identity; existing bind-mounted files keep the
    # ownership chosen by the host administrator.
    chown -h lorahub:lorahub "${DATA_DIRS[@]}" "${BACKEND_VENV_DIRS[@]}"
    for link in "${BACKEND_MODELS_LINKS[@]}"; do
        chown -h lorahub:lorahub "$link" 2>/dev/null || true
    done

    exec gosu lorahub:lorahub "$0" "$@"
fi

# ── 3. (non-root path) Re-mkdir as the runtime user — no-op if present
mkdir -p "${DATA_DIRS[@]}" "${BACKEND_VENV_DIRS[@]}"

if [ "$LORAHUB_BIND_HOST" != "127.0.0.1" ] && [ "$LORAHUB_BIND_HOST" != "localhost" ] && [ "$LORAHUB_BIND_HOST" != "::1" ]; then
    if [ -z "${LORAHUB_API_TOKEN:-}" ]; then
        LORAHUB_API_TOKEN=$(python -c \
            'from lorahub.api.auth import ensure_api_token; print(ensure_api_token())')
        export LORAHUB_API_TOKEN
        TOKEN_PATH=$(python -c \
            'from lorahub.api.auth import api_token_path; print(api_token_path())')
        echo "[lorahub] remote access authentication enabled; token file: ${TOKEN_PATH}"
    else
        export LORAHUB_API_TOKEN
        echo "[lorahub] remote access authentication enabled by LORAHUB_API_TOKEN"
    fi
fi

# ── 4. Launch uvicorn ────────────────────────────────────────────────
# `lorahub.api.app:app` is the same target scripts/run.sh uses. The
# lifespan hook chdir's to LORAHUB_HOME itself, so cwd doesn't matter.
echo "[lorahub] starting on ${LORAHUB_BIND_HOST}:${LORAHUB_PORT}  (LORAHUB_HOME=${LORAHUB_HOME})"
exec uvicorn lorahub.api.app:app \
    --host "$LORAHUB_BIND_HOST" \
    --port "$LORAHUB_PORT"
