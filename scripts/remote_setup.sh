#!/usr/bin/env bash
# remote_setup.sh — one-shot installer that runs on the LoraHub VPS.
#
# Idempotent: re-runs are safe and will only redo the steps that need it.
# Designed to be invoked by `scripts/wsl_remote.sh setup` from the WSL host
# (which wraps the heavy lifting in setsid + tee logs to /root/_setup.log).
#
# Steps:
#   1. Ensure /root/autodl-tmp/opt/node20 exists (Node 20 portable binary).
#   2. Ensure ~/.local/bin/uv exists (uv via direct GitHub release tarball).
#   3. Create .venv with the host's miniconda3 Python (3.12).
#   4. uv pip install -e .[api,tagging] from the Tsinghua mirror.
#   5. npm install + vite build under web/.
#   6. Print a summary of what's in place.
#
# Network: all downloads go through gh-proxy.org for GitHub and
# pypi.tuna.tsinghua.edu.cn / npmmirror.com for package indexes. Override
# the mirrors via the LORAHUB_PYPI_INDEX / LORAHUB_NPM_REGISTRY env vars
# if you are running outside mainland China.

set -uo pipefail

LORAHUB_DIR="${LORAHUB_DIR:-/root/autodl-tmp/LoraHub}"
NODE_DIR="${NODE_DIR:-/root/autodl-tmp/opt/node20}"
UV_BIN="${UV_BIN:-/root/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PYPI_INDEX="${LORAHUB_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
NPM_REGISTRY="${LORAHUB_NPM_REGISTRY:-https://registry.npmmirror.com}"
GH_PROXY="${LORAHUB_GH_PROXY:-https://gh-proxy.org/}"

NODE_VERSION="20.18.1"
UV_VERSION="latest"

log() { printf '\033[36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[setup error]\033[0m %s\n' "$*" >&2; }

ensure_node() {
  if [[ -x "${NODE_DIR}/bin/node" ]]; then
    log "Node already at ${NODE_DIR} ($(${NODE_DIR}/bin/node -v))"
    return 0
  fi
  log "Installing Node ${NODE_VERSION} portable to ${NODE_DIR}"
  mkdir -p "$(dirname "${NODE_DIR}")"
  local tmp
  tmp="$(mktemp /tmp/node20.XXXX.tar.xz)"
  curl -fsSL "https://npmmirror.com/mirrors/node/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o "$tmp" || {
    err "node download failed"; return 1
  }
  tar -xJf "$tmp" -C "$(dirname "${NODE_DIR}")"
  mv "$(dirname "${NODE_DIR}")/node-v${NODE_VERSION}-linux-x64" "${NODE_DIR}"
  rm -f "$tmp"
  log "Node installed: $(${NODE_DIR}/bin/node -v)"
}

ensure_uv() {
  if [[ -x "${UV_BIN}" ]]; then
    log "uv already at ${UV_BIN} ($(${UV_BIN} --version))"
    return 0
  fi
  log "Installing uv via gh-proxy"
  local tmp
  tmp="$(mktemp /tmp/uv.XXXX.tar.gz)"
  curl -fsSL "${GH_PROXY}https://github.com/astral-sh/uv/releases/${UV_VERSION}/download/uv-x86_64-unknown-linux-gnu.tar.gz" -o "$tmp" || {
    err "uv download failed"; return 1
  }
  local extract
  extract="$(mktemp -d /tmp/uv-extract.XXXX)"
  tar -xzf "$tmp" -C "$extract"
  mkdir -p "$(dirname "${UV_BIN}")"
  mv "${extract}/uv-x86_64-unknown-linux-gnu/uv" "${UV_BIN}"
  mv "${extract}/uv-x86_64-unknown-linux-gnu/uvx" "$(dirname "${UV_BIN}")/uvx"
  rm -rf "$tmp" "$extract"
  chmod +x "${UV_BIN}" "$(dirname "${UV_BIN}")/uvx"
  log "uv installed: $(${UV_BIN} --version)"
}

ensure_venv() {
  if [[ -x "${LORAHUB_DIR}/.venv/bin/python" ]]; then
    log ".venv already present"
    return 0
  fi
  log "Creating .venv at ${LORAHUB_DIR}/.venv with ${PYTHON_BIN}"
  cd "${LORAHUB_DIR}"
  "${UV_BIN}" venv -p "${PYTHON_BIN}" .venv || {
    err "uv venv failed"; return 1
  }
}

install_python_deps() {
  log "uv pip install -e .[api,tagging] (mirror: ${PYPI_INDEX})"
  cd "${LORAHUB_DIR}"
  "${UV_BIN}" pip install --python .venv/bin/python -e ".[api,tagging]" \
    --index-url "${PYPI_INDEX}" \
    > /root/_uv_install.log 2>&1 || {
    err "pip install failed; tail of log:"
    tail -30 /root/_uv_install.log >&2
    return 1
  }
  log "Python deps installed: $(.venv/bin/python -c 'import lorahub; print(lorahub.__version__)' 2>/dev/null || echo unknown)"
}

build_frontend() {
  cd "${LORAHUB_DIR}/web"
  # Decide whether to npm install: not just "is node_modules there?" but
  # "is node_modules in sync with package.json?". Without the mtime
  # check we'd silently keep running against stale deps after every
  # `git pull` that bumped package.json — which already burned us once
  # when a fresh dep (`react-markdown`) made vite build fail and the
  # old web/dist kept getting served.
  local needs_install=0
  if [[ ! -d node_modules ]]; then
    needs_install=1
  elif [[ -f package-lock.json ]] && [[ ! -f node_modules/.package-lock.json ]]; then
    needs_install=1
  elif [[ -f package-lock.json ]] && [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    needs_install=1
  elif [[ package.json -nt node_modules/.package-lock.json ]]; then
    needs_install=1
  fi
  if (( needs_install )); then
    log "npm install (registry: ${NPM_REGISTRY})"
    "${NODE_DIR}/bin/npm" install --registry="${NPM_REGISTRY}" \
      > /root/_npm_install.log 2>&1 || {
      err "npm install failed; tail of log:"
      tail -30 /root/_npm_install.log >&2
      return 1
    }
  else
    log "npm modules up to date (skip; rm -rf web/node_modules to force)"
  fi
  # If the previous vite build failed (no dist/index.html on disk) but
  # node_modules looked fine, the old "skip if dist exists" branch
  # would never retry. Force a rebuild whenever dist is missing or
  # any source/manifest file is newer than dist/index.html.
  local needs_build=0
  if [[ ! -f dist/index.html ]]; then
    needs_build=1
  elif [[ -n "$(find ./src -newer dist/index.html -type f -print -quit 2>/dev/null)" ]]; then
    needs_build=1
  elif [[ package.json -nt dist/index.html ]]; then
    needs_build=1
  elif [[ -f vite.config.ts ]] && [[ vite.config.ts -nt dist/index.html ]]; then
    needs_build=1
  elif [[ -f index.html ]] && [[ index.html -nt dist/index.html ]]; then
    needs_build=1
  fi
  if (( needs_build )); then
    log "vite build"
    "${NODE_DIR}/bin/npx" vite build > /root/_vite_build.log 2>&1 || {
      err "vite build failed; tail of log:"
      tail -30 /root/_vite_build.log >&2
      return 1
    }
  else
    log "web/dist up to date (skip; rm -rf web/dist to force)"
  fi
}

summary() {
  log "Setup complete:"
  log "  python:  ${LORAHUB_DIR}/.venv/bin/python ($(${LORAHUB_DIR}/.venv/bin/python -V 2>&1))"
  log "  lorahub: $(${LORAHUB_DIR}/.venv/bin/python -c 'import lorahub; print(lorahub.__version__)' 2>/dev/null || echo missing)"
  log "  node:    $(${NODE_DIR}/bin/node -v)"
  log "  uv:      $(${UV_BIN} --version)"
  log "  dist:    $([[ -f ${LORAHUB_DIR}/web/dist/index.html ]] && echo present || echo missing)"
  log ""
  log "Start the API: scripts/remote_serve.sh"
}

main() {
  log "LoraHub remote setup starting"
  if [[ ! -d "${LORAHUB_DIR}" ]]; then
    err "LoraHub repo not found at ${LORAHUB_DIR}; clone it first."
    exit 1
  fi
  ensure_node || exit 1
  ensure_uv || exit 1
  ensure_venv || exit 1
  install_python_deps || exit 1
  build_frontend || exit 1
  summary
}

main "$@"
