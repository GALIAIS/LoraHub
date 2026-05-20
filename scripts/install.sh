#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# LoRaHub - Full environment installer (Linux)
#
# Installs EVERYTHING into the project directory:
#   1. uv -> .lorahub/uv/
#   2. Python 3.12 -> .lorahub/python/
#   3. Virtual environment -> .venv/
#   4. Python dependencies (lorahub[api,dev])
#   5. Node.js portable -> .node/
#   6. Frontend dependencies (npm install)
#
# No pre-existing Python/Node/uv needed. Fully self-contained.
# After completion, use scripts/run.sh to start.
#
# Mirror knobs (pass via env). Empty -> upstream default. Inside China
# users typically want the matching install-cn.sh wrapper which presets
# every variable below.
#
#   LORAHUB_GH_PROXY         GitHub proxy prefix (e.g. https://gh-proxy.org/)
#                            applied to uv release tarball download
#   UV_PYTHON_INSTALL_MIRROR python-build-standalone mirror, e.g.
#                            https://registry.npmmirror.com/-/binary/python-build-standalone
#                            (uv reads this env var natively for `uv python install`)
#   UV_INDEX_URL             PyPI index for `uv pip install` (e.g.
#                            https://pypi.tuna.tsinghua.edu.cn/simple)
#   LORAHUB_NODE_MIRROR      Node.js binary mirror base, default
#                            https://nodejs.org/dist (popular alt:
#                            https://npmmirror.com/mirrors/node)
#   NPM_CONFIG_REGISTRY      npm registry, e.g. https://registry.npmmirror.com
#                            (npm reads this env var natively)
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Single managed-tools home, shared with the LoraHub API. Both write to
# the same .lorahub/ directory so a runtime installed by this script is
# visible to the Settings UI (and vice versa).
TOOLS_DIR="$ROOT/.lorahub"
UV_DIR="$TOOLS_DIR/uv"
PY_DIR="$TOOLS_DIR/python"
NODE_DIR="$ROOT/.node"

GH_PROXY="${LORAHUB_GH_PROXY:-}"
NODE_MIRROR="${LORAHUB_NODE_MIRROR:-https://nodejs.org/dist}"

echo ""
echo "============================================================"
echo "  LoRaHub Environment Installer (Linux)"
echo "============================================================"
echo "  Project: $ROOT"
echo "  Tools:   $TOOLS_DIR"
if [ -n "$GH_PROXY" ];               then echo "  GH proxy:  $GH_PROXY";               fi
if [ -n "${UV_PYTHON_INSTALL_MIRROR:-}" ]; then echo "  Python:    $UV_PYTHON_INSTALL_MIRROR"; fi
if [ -n "${UV_INDEX_URL:-}" ];       then echo "  PyPI:      $UV_INDEX_URL";           fi
if [ "$NODE_MIRROR" != "https://nodejs.org/dist" ]; then echo "  Node:      $NODE_MIRROR"; fi
if [ -n "${NPM_CONFIG_REGISTRY:-}" ]; then echo "  npm:       $NPM_CONFIG_REGISTRY";    fi
echo ""

# ---- [1/6] Install uv locally ---------------------------------------
echo "[1/6] Installing uv ..."
if [ -f "$UV_DIR/uv" ]; then
    echo "  OK uv already installed"
else
    mkdir -p "$UV_DIR"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  UV_ARCH="x86_64-unknown-linux-gnu" ;;
        aarch64) UV_ARCH="aarch64-unknown-linux-gnu" ;;
        *)       echo "  [ERROR] Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz"
    [ -n "$GH_PROXY" ] && UV_URL="${GH_PROXY%/}/${UV_URL}"
    echo "  fetching $UV_URL"
    # --connect-timeout bounds the TCP handshake so a black-holed mirror
    # fails fast instead of hanging the whole install. --max-time caps
    # the full transfer so a stalled-but-alive socket eventually errors
    # out (uv tarball is ~25MB; 300s covers even slow 4G).
    curl -L --fail --show-error --connect-timeout 10 --max-time 300 \
        "$UV_URL" -o "$UV_DIR/uv.tar.gz"
    tar -xzf "$UV_DIR/uv.tar.gz" -C "$UV_DIR" --strip-components=1
    rm -f "$UV_DIR/uv.tar.gz"
    if [ ! -f "$UV_DIR/uv" ]; then
        echo "  [ERROR] uv binary not found after extraction."
        exit 1
    fi
    chmod +x "$UV_DIR/uv"
    echo "  OK uv downloaded"
fi
UV="$UV_DIR/uv"
export PATH="$UV_DIR:$PATH"
echo "  $($UV --version)"
echo ""

# ---- [2/6] Install Python 3.12 locally ------------------------------
echo "[2/6] Installing Python 3.12 ..."
# uv lays out two entries per install: a real ``cpython-3.12.<patch>-...``
# directory and a symlink ``cpython-3.12-...`` pointing at it. The
# symlink is uv's stable minor-version alias — pinning the venv to it
# means a future ``uv python install 3.12`` (which repoints the symlink
# to a newer patch) keeps the venv working instead of breaking
# pyvenv.cfg.
_find_anima_python() {
    # Look for the minor-version alias first (most stable).
    local alias_dir
    alias_dir=$(find "$1" -maxdepth 1 -type l -name "cpython-3.12-*" 2>/dev/null | head -1)
    if [ -z "$alias_dir" ]; then
        # Older uv: the alias may not exist, fall back to the real
        # patched cpython-3.12.<X>-... directory.
        alias_dir=$(find "$1" -maxdepth 1 -type d -name "cpython-3.12.*-*" 2>/dev/null | head -1)
    fi
    if [ -z "$alias_dir" ]; then
        return 0
    fi
    for cand in "python3.12" "python3" "python"; do
        local exe="$alias_dir/bin/$cand"
        if [ -x "$exe" ]; then
            echo "$exe"
            return 0
        fi
    done
}
if [ -n "$(_find_anima_python "$PY_DIR")" ]; then
    echo "  OK Python already installed"
else
    mkdir -p "$PY_DIR"
    # ``--no-bin`` skips uv's per-user shim in ~/.local/bin. We invoke
    # python by full path so the shim adds no value, and skipping it
    # avoids a confusing warning when a prior global ``uv python
    # install`` already wrote one there.
    "$UV" python install 3.12 --install-dir "$PY_DIR" --no-bin
    echo "  OK Python 3.12 installed"
fi
PY_EXE=$(_find_anima_python "$PY_DIR")
if [ -z "$PY_EXE" ]; then
    echo "  [ERROR] python executable not found under $PY_DIR/cpython-3.12*"
    exit 1
fi
echo "  Python: $PY_EXE"
echo ""

# ---- [3/6] Create virtual environment -------------------------------
echo "[3/6] Creating virtual environment .venv ..."
# Detect a stale .venv whose pyvenv.cfg `home =` points at a Python
# that's been removed (typical after the .tools -> .lorahub move).
# uv pip install --python <stale-venv> fails with "No virtual
# environment found", so we wipe and rebuild from scratch instead.
_venv_valid() {
    [ -f ".venv/bin/python" ] || return 1
    local home
    home=$(awk -F= '/^home[[:space:]]*=/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' .venv/pyvenv.cfg 2>/dev/null)
    [ -n "$home" ] && [ -x "$home/python" -o -x "$home/python3" -o -x "$home/python.exe" ]
}
if _venv_valid; then
    echo "  OK .venv already exists"
else
    if [ -d ".venv" ]; then
        echo "  stale .venv detected; rebuilding"
        rm -rf .venv
    fi
    "$UV" venv .venv --python "$PY_EXE"
    echo "  OK .venv created"
fi
VENV_PY="$ROOT/.venv/bin/python"
echo ""

# ---- [4/6] Install Python dependencies ------------------------------
echo "[4/6] Installing Python dependencies ..."
# uv pip install reads UV_INDEX_URL natively when set; explicit
# --index-url here would override that, so leave the env var to do
# its job.
"$UV" pip install -e ".[api,dev]" --python "$VENV_PY"
echo "  OK Python dependencies installed"
echo ""

# ---- [5/6] Install Node.js locally ----------------------------------
echo "[5/6] Installing Node.js ..."
# Always use a project-local portable Node, never the system one — same
# rationale as the Windows install.bat: a portable Node 20 is ~50 MB
# and avoids the entire class of "different node version on dev box vs
# CI vs VPS" problems.
if [ -f "$NODE_DIR/bin/node" ] && [ -f "$NODE_DIR/bin/npm" ]; then
    export PATH="$NODE_DIR/bin:$PATH"
    echo "  OK Node.js $(node --version) (portable, cached)"
else
    echo "  Downloading portable Node.js 20 (mirror: $NODE_MIRROR) ..."
    mkdir -p "$NODE_DIR"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  NODE_ARCH="x64" ;;
        aarch64) NODE_ARCH="arm64" ;;
        armv7l)  NODE_ARCH="armv7l" ;;
        *)       echo "  [ERROR] Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    NODE_VER="v20.18.1"
    NODE_TAR="node-${NODE_VER}-linux-${NODE_ARCH}.tar.xz"
    NODE_URL="${NODE_MIRROR%/}/${NODE_VER}/${NODE_TAR}"
    echo "  fetching $NODE_URL"
    curl -L --fail --show-error --connect-timeout 10 --max-time 600 \
        "$NODE_URL" -o "$NODE_DIR/$NODE_TAR"
    tar -xf "$NODE_DIR/$NODE_TAR" -C "$NODE_DIR" --strip-components=1
    rm -f "$NODE_DIR/$NODE_TAR"
    if [ ! -f "$NODE_DIR/bin/node" ] || [ ! -f "$NODE_DIR/bin/npm" ]; then
        echo "  [ERROR] node/npm not found after extraction."
        exit 1
    fi
    export PATH="$NODE_DIR/bin:$PATH"
    echo "  OK Node.js $(node --version) (portable, downloaded)"
fi
echo ""

# ---- [6/6] Install frontend dependencies ----------------------------
echo "[6/6] Installing frontend dependencies (web/) ..."
# npm reads NPM_CONFIG_REGISTRY natively when set.
if [ -d "web/node_modules/vite" ]; then
    echo "  OK web/node_modules already exists"
else
    cd web
    npm install
    cd "$ROOT"
    echo "  OK Frontend dependencies installed"
fi
echo ""

echo "============================================================"
echo "  Installation Complete"
echo "============================================================"
echo ""
echo "  All tools installed locally:"
echo "    uv:      .lorahub/uv/"
echo "    Python:  .lorahub/python/"
echo "    Node.js: .node/"
echo "    venv:    .venv/"
echo ""
echo "  To start LoRaHub:"
echo "    scripts/run.sh              (default prod: API serves built SPA)"
echo "    scripts/run.sh dev          (dev mode: API + Vite HMR)"
echo ""
