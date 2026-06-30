#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# LoRaHub - Full environment installer (Linux)
#
# !! Mirror script: scripts/install.bat (Windows cmd).
# !! The 6-step contract is documented in scripts/INSTALL_DESIGN.md —
# !! every change here MUST land in install.bat in the same commit.
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
AMDGPU_TOP_DIR="$TOOLS_DIR/amdgpu_top"
NODE_VERSION="20.19.0"
NODE_MIN_VERSION="20.19.0"
AMDGPU_TOP_VERSION="0.11.5"

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

if [ -n "${UV_INDEX_URL:-}" ] && [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    export UV_DEFAULT_INDEX="$UV_INDEX_URL"
fi

version_ge() {
    # Return true when $1 >= $2 for dotted numeric versions.
    local a b IFS=.
    read -r -a a <<< "${1#v}"
    read -r -a b <<< "${2#v}"
    for i in 0 1 2; do
        local ai="${a[$i]:-0}"
        local bi="${b[$i]:-0}"
        if ((10#$ai > 10#$bi)); then return 0; fi
        if ((10#$ai < 10#$bi)); then return 1; fi
    done
    return 0
}

download_github_asset() {
    local upstream="$1"
    local output="$2"
    local max_time="${3:-300}"
    local candidates=()
    local seen="|"
    if [ -n "$GH_PROXY" ]; then candidates+=("$GH_PROXY"); fi
    if [ -n "${LORAHUB_GH_PROXY_FALLBACKS:-}" ]; then
        IFS=';' read -r -a candidates_from_env <<< "$LORAHUB_GH_PROXY_FALLBACKS"
        candidates+=("${candidates_from_env[@]}")
    else
        candidates+=("" "https://v4.gh-proxy.org/" "https://gh-proxy.com/" "https://gh.ddlc.top/" "https://gh.jasonzeng.dev/" "https://gh.zwy.one/" "https://ghfast.top/" "https://gh-proxy.org/")
    fi
    for proxy in "${candidates[@]}"; do
        proxy="${proxy%/}"
        case "$seen" in *"|$proxy|"*) continue ;; esac
        seen="$seen$proxy|"
        local url="$upstream"
        if [ -n "$proxy" ]; then url="$proxy/$upstream"; fi
        echo "  fetching $url"
        if curl -L --fail --show-error --connect-timeout 10 --max-time "$max_time" \
            "$url" -o "$output"; then
            return 0
        fi
        rm -f "$output"
    done
    return 1
}

download_url() {
    local output="$1"
    local max_time="$2"
    shift 2
    local url
    for url in "$@"; do
        [ -n "$url" ] || continue
        echo "  fetching $url"
        if curl -L --fail --show-error --connect-timeout 10 --max-time "$max_time" \
            "$url" -o "$output"; then
            return 0
        fi
        rm -f "$output"
    done
    return 1
}

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
    # --connect-timeout bounds the TCP handshake so a black-holed mirror
    # fails fast instead of hanging the whole install. --max-time caps
    # the full transfer so a stalled-but-alive socket eventually errors
    # out (uv tarball is ~25MB; 300s covers even slow 4G).
    UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz"
    download_github_asset "$UV_URL" "$UV_DIR/uv.tar.gz" 300
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
    # ``--seed`` makes uv install pip / setuptools / wheel into the
    # fresh venv. Without it the venv has no ``pip`` binary, so users
    # who ``pip install <pkg>`` from the LoraHub in-app terminal hit
    # the auto-fallback to ``uv pip``. Seeding pip directly is more
    # intuitive (and lets third-party tools that subprocess
    # ``pip install`` keep working).
    "$UV" venv .venv --python "$PY_EXE" --seed
    echo "  OK .venv created (seeded with pip / setuptools / wheel)"
fi
VENV_PY="$ROOT/.venv/bin/python"
echo ""

# ---- [4/6] Install Python dependencies ------------------------------
echo "[4/6] Installing Python dependencies ..."
PY_DEPS_LOG="$ROOT/_uv_python_deps.log"
PY_DEPS_ARGS=(pip install)
if [ "${LORAHUB_INSTALL_VERBOSE:-}" = "1" ]; then
    PY_DEPS_ARGS+=(-v)
fi
PY_DEPS_ARGS+=(-e ".[api,dev]" --python "$VENV_PY" --link-mode=copy)
if [ -n "${UV_DEFAULT_INDEX:-}" ]; then
    PY_DEPS_ARGS+=(--index-url "$UV_DEFAULT_INDEX")
fi
echo "  uv default index: ${UV_DEFAULT_INDEX:-default}"
echo "  running uv ${PY_DEPS_ARGS[*]} (log: _uv_python_deps.log) ..."
"$UV" "${PY_DEPS_ARGS[@]}" 2>&1 | tee "$PY_DEPS_LOG"
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
    CACHED_NODE_VERSION="$(node --version)"
    if version_ge "$CACHED_NODE_VERSION" "$NODE_MIN_VERSION"; then
        echo "  OK Node.js $CACHED_NODE_VERSION (portable, cached)"
    else
        echo "  Cached Node.js $CACHED_NODE_VERSION is below required v$NODE_MIN_VERSION; reinstalling ..."
        rm -rf "$NODE_DIR"
        mkdir -p "$NODE_DIR"
        NEED_NODE_INSTALL=1
    fi
else
    NEED_NODE_INSTALL=1
fi

if [ "${NEED_NODE_INSTALL:-0}" = "1" ]; then
    echo "  Downloading portable Node.js 20 (mirror: $NODE_MIRROR) ..."
    mkdir -p "$NODE_DIR"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  NODE_ARCH="x64" ;;
        aarch64) NODE_ARCH="arm64" ;;
        armv7l)  NODE_ARCH="armv7l" ;;
        *)       echo "  [ERROR] Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    NODE_VER="v${NODE_VERSION}"
    NODE_TAR="node-${NODE_VER}-linux-${NODE_ARCH}.tar.xz"
    NODE_URL="${NODE_MIRROR%/}/${NODE_VER}/${NODE_TAR}"
    download_url "$NODE_DIR/$NODE_TAR" 600 \
        "$NODE_URL" \
        "https://npmmirror.com/mirrors/node/${NODE_VER}/${NODE_TAR}" \
        "https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/${NODE_VER}/${NODE_TAR}" \
        "https://mirrors.aliyun.com/nodejs-release/${NODE_VER}/${NODE_TAR}" \
        "https://nodejs.org/dist/${NODE_VER}/${NODE_TAR}"
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
needs_npm_install=0
if [ ! -d "web/node_modules" ]; then
    needs_npm_install=1
elif [ ! -f "web/node_modules/.package-lock.json" ]; then
    needs_npm_install=1
elif [ "web/package-lock.json" -nt "web/node_modules/.package-lock.json" ]; then
    needs_npm_install=1
elif [ "web/package.json" -nt "web/node_modules/.package-lock.json" ]; then
    needs_npm_install=1
elif ! (cd web && npm ls --depth=0 >/dev/null 2>&1); then
    needs_npm_install=1
fi

if [ "$needs_npm_install" = "0" ]; then
    echo "  OK web/node_modules already matches package lock"
else
    cd web
    echo "  npm registry: $(npm config get registry)"
    echo "  running npm ci (verbose log: web/_npm_install.log) ..."
    npm ci --verbose --no-audit --no-fund \
        --fetch-timeout=60000 \
        --fetch-retries=2 \
        --fetch-retry-mintimeout=5000 \
        --fetch-retry-maxtimeout=20000 \
        > _npm_install.log 2>&1 &
    npm_pid=$!
    while kill -0 "$npm_pid" 2>/dev/null; do
        echo "  npm ci still running ... ($(date '+%H:%M:%S'))"
        sleep 15
    done
    set +e
    wait "$npm_pid"
    npm_rc=$?
    set -e
    cd "$ROOT"
    if [ "$npm_rc" -ne 0 ]; then
        echo "  [ERROR] npm ci failed; tail of web/_npm_install.log:"
        tail -40 web/_npm_install.log || true
        exit 1
    fi
    echo "  OK Frontend dependencies installed"
fi
echo ""

# ---- [extra] install optional GPU telemetry helpers -----------------
echo "[extra] Installing GPU telemetry helpers ..."
if [ -f "$AMDGPU_TOP_DIR/amdgpu_top" ]; then
    echo "  OK amdgpu_top already installed"
else
    mkdir -p "$AMDGPU_TOP_DIR"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  AMDGPU_TOP_ARCH="x86_64-unknown-linux-gnu" ;;
        aarch64) AMDGPU_TOP_ARCH="aarch64-unknown-linux-gnu" ;;
        *)       AMDGPU_TOP_ARCH="" ;;
    esac
    if [ -z "$AMDGPU_TOP_ARCH" ]; then
        echo "  skipped: unsupported architecture $ARCH"
    else
        AMDGPU_TOP_TAR="amdgpu_top-${AMDGPU_TOP_VERSION}-${AMDGPU_TOP_ARCH}.tar.gz"
        AMDGPU_TOP_URL="https://github.com/Umio-Yasuno/amdgpu_top/releases/download/v${AMDGPU_TOP_VERSION}/${AMDGPU_TOP_TAR}"
        if download_github_asset "$AMDGPU_TOP_URL" "$AMDGPU_TOP_DIR/amdgpu_top.tar.gz" 300; then
            tar -xzf "$AMDGPU_TOP_DIR/amdgpu_top.tar.gz" -C "$AMDGPU_TOP_DIR"
            rm -f "$AMDGPU_TOP_DIR/amdgpu_top.tar.gz"
            found="$(find "$AMDGPU_TOP_DIR" -type f -name amdgpu_top 2>/dev/null | head -1)"
            if [ -n "$found" ] && [ "$found" != "$AMDGPU_TOP_DIR/amdgpu_top" ]; then
                cp "$found" "$AMDGPU_TOP_DIR/amdgpu_top"
            fi
            chmod +x "$AMDGPU_TOP_DIR/amdgpu_top" 2>/dev/null || true
            if [ -f "$AMDGPU_TOP_DIR/amdgpu_top" ]; then
                echo "  OK amdgpu_top installed"
            else
                echo "  skipped: amdgpu_top binary not found in archive"
            fi
        else
            echo "  skipped: failed to download amdgpu_top"
        fi
    fi
fi
echo ""

# ---- [extra] register the `lorahub` CLI in the user PATH ----------
# .venv/bin/lorahub already exists thanks to ``uv pip install -e .``,
# but it isn't reachable without the venv being activated. Run the
# CLI's own self-install path so a fresh shell can call ``lorahub``
# from anywhere — silent when already installed, fails soft so a
# weird HOME / permission setup doesn't sink the whole installer.
echo "[extra] Registering lorahub CLI ..."
"$VENV_PY" -m lorahub manage install 2>&1 || true
LOCAL_BIN="$HOME/.local/bin"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if [ -d "$LOCAL_BIN" ]; then
    export PATH="$LOCAL_BIN:$PATH"
    shell_rc="$HOME/.profile"
    [ -n "${BASH_VERSION:-}" ] && shell_rc="$HOME/.bashrc"
    if [ -w "$(dirname "$shell_rc")" ] && ! grep -Fqs "$PATH_LINE" "$shell_rc" 2>/dev/null; then
        printf '\n# LoRaHub CLI\n%s\n' "$PATH_LINE" >> "$shell_rc"
        echo "已添加 $LOCAL_BIN 到 $shell_rc"
    fi
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
echo "    lorahub service start       (background daemon — random port)"
echo ""
echo "  CLI PATH is configured for new shells when possible:"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
