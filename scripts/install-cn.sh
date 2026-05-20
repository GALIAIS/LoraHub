#!/usr/bin/env bash
# LoRaHub installer — China mirrors preset.
#
# Forwards to scripts/install.sh with all download endpoints flipped to
# in-China mirrors. No secret sauce here — every variable below is
# documented at the top of install.sh, and you can mix-and-match by
# exporting them yourself before invoking install.sh directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# uv release tarball: GitHub via gh-proxy.org
export LORAHUB_GH_PROXY="${LORAHUB_GH_PROXY:-https://gh-proxy.org/}"

# python-build-standalone: npmmirror's mirror that uv knows how to use.
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}"

# PyPI: TUNA (Tsinghua) — the largest and most consistently up-to-date
# of the in-China PyPI mirrors.
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# Node binary releases: Aliyun's npmmirror also hosts these.
export LORAHUB_NODE_MIRROR="${LORAHUB_NODE_MIRROR:-https://npmmirror.com/mirrors/node}"

# npm package registry: same mirror.
export NPM_CONFIG_REGISTRY="${NPM_CONFIG_REGISTRY:-https://registry.npmmirror.com}"

echo "[install-cn] using China mirrors:"
echo "  GitHub:  $LORAHUB_GH_PROXY"
echo "  Python:  $UV_PYTHON_INSTALL_MIRROR"
echo "  PyPI:    $UV_INDEX_URL"
echo "  Node:    $LORAHUB_NODE_MIRROR"
echo "  npm:     $NPM_CONFIG_REGISTRY"
echo ""

exec bash "$SCRIPT_DIR/install.sh" "$@"
