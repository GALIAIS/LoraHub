#!/usr/bin/env bash
# LoRaHub installer — China-region edition with auto mirror selection.
#
# Probes a small candidate pool for each download endpoint, picks the
# fastest reachable one, then forwards to scripts/install.sh. The user
# does nothing.
#
# Probe = HTTP HEAD with a 3s connect timeout + 5s total timeout, run
# twice and we keep the median. Failures (connect refused / DNS / 4xx /
# 5xx) drop the candidate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Candidate pools per endpoint. First entry is the upstream default —
# kept in the list so well-connected users still benefit from probing
# (sometimes the official site is faster than the in-China mirror).

GH_PROXIES=(
  ""
  "https://gh-proxy.org/"
  "https://hk.gh-proxy.org/"
  "https://cdn.gh-proxy.org/"
  "https://v6.gh-proxy.org/"
  "https://ghfast.top/"
)

PYTHON_BUILD_MIRRORS=(
  "https://registry.npmmirror.com/-/binary/python-build-standalone"
  "https://github.com/astral-sh/python-build-standalone/releases/download"
)

PYPI_INDEXES=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
  "https://mirrors.cloud.tencent.com/pypi/simple"
  "https://pypi.org/simple"
)

NODE_MIRRORS=(
  "https://npmmirror.com/mirrors/node"
  "https://mirrors.tuna.tsinghua.edu.cn/nodejs-release"
  "https://mirrors.aliyun.com/nodejs-release"
  "https://nodejs.org/dist"
)

NPM_REGISTRIES=(
  "https://registry.npmmirror.com"
  "https://registry.npmjs.org"
)

# Probe URLs — small payloads (HEAD on a tiny known file) so the test
# completes fast even on a slow mirror. We don't care about the body,
# only about TCP connect + first response.
probe_one() {
    # $1 = url, $2 = label
    local url="$1"
    local t1 t2
    t1=$(date +%s%3N)
    curl -fsI -o /dev/null --max-time 5 --connect-timeout 3 "$url" 2>/dev/null || return 1
    t2=$(date +%s%3N)
    echo $((t2 - t1))
}

probe_one_download() {
    # $1 = url. HEAD is not enough for gh-proxy release binaries: some
    # proxies answer fast, then EOF on the real tarball stream.
    local url="$1"
    local tmp t1 t2 bytes
    tmp="$(mktemp)"
    t1=$(date +%s%3N)
    if ! curl -fsL --range 0-262143 -o "$tmp" --max-time 10 --connect-timeout 3 "$url" 2>/dev/null; then
        rm -f "$tmp"
        return 1
    fi
    t2=$(date +%s%3N)
    bytes=$(wc -c < "$tmp" 2>/dev/null || echo 0)
    rm -f "$tmp"
    [ "${bytes:-0}" -ge 65536 ] || return 1
    echo $((t2 - t1))
}

# Pick the fastest URL from a candidate list, given a probe-URL builder.
# Args:  pool_name builder_function
# Output: chosen base URL (echoed)
pick_fastest() {
    local label="$1"
    shift
    local builder="$1"
    shift
    local candidates=("$@")
    local best_url="" best_ms=99999 have_any=0
    printf '  probing %s ...\n' "$label" >&2
    for cand in "${candidates[@]}"; do
        local probe_url
        probe_url=$("$builder" "$cand")
        local ms
        if [ "$label" = "GitHub proxy" ]; then
            ms=$(probe_one_download "$probe_url" || echo "")
        else
            ms=$(probe_one "$probe_url" || echo "")
        fi
        local display
        if [ -z "$cand" ]; then display="(direct)"; else display="$cand"; fi
        if [ -n "$ms" ]; then
            printf '    %4sms  %s\n' "$ms" "$display" >&2
            have_any=1
            if [ "$ms" -lt "$best_ms" ]; then
                best_ms="$ms"
                best_url="$cand"
            fi
        else
            printf '    fail    %s\n' "$display" >&2
        fi
    done
    if [ "$have_any" -eq 0 ]; then
        # Every candidate failed — fall back to the first entry so the
        # downstream install attempt at least produces a real error
        # message instead of silently skipping.
        best_url="${candidates[0]}"
        local fallback
        if [ -z "$best_url" ]; then fallback="(direct)"; else fallback="$best_url"; fi
        printf '  [!] all probes failed, falling back to %s\n' "$fallback" >&2
    fi
    echo "$best_url"
}

# --- Builders: each maps a candidate base URL to a small probe URL ---

probe_gh_proxy() {
    # Probe the actual download URL the installer will hit. gh-proxy
    # variants sometimes accept the API surface but reject release
    # binaries (or vice versa); using the same URL we'll really fetch
    # gives a faithful reachability signal.
    echo "$1https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
}

probe_python_build() {
    # python-build-standalone: probe the releases JSON or directory
    # listing.
    case "$1" in
        *npmmirror*)
            echo "$1/"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

probe_pypi() {
    # /pip/ exists on every PyPI-shaped index (tuna / aliyun / tencent /
    # pypi.org all serve it).
    echo "$1/pip/"
}

probe_node() {
    # node mirror's index.json is small and present on every mirror.
    echo "$1/index.json"
}

probe_npm() {
    # npm registry: the lodash package metadata is small and works on
    # every npm-compatible mirror.
    echo "$1/lodash"
}

echo "[install-cn] selecting fastest mirrors ..."
echo ""

LORAHUB_GH_PROXY=$(pick_fastest "GitHub proxy" probe_gh_proxy "${GH_PROXIES[@]}")
UV_PYTHON_INSTALL_MIRROR=$(pick_fastest "python-build-standalone" probe_python_build "${PYTHON_BUILD_MIRRORS[@]}")
UV_INDEX_URL=$(pick_fastest "PyPI" probe_pypi "${PYPI_INDEXES[@]}")
LORAHUB_NODE_MIRROR=$(pick_fastest "Node binary" probe_node "${NODE_MIRRORS[@]}")
NPM_CONFIG_REGISTRY=$(pick_fastest "npm registry" probe_npm "${NPM_REGISTRIES[@]}")

export LORAHUB_GH_PROXY UV_PYTHON_INSTALL_MIRROR UV_INDEX_URL LORAHUB_NODE_MIRROR NPM_CONFIG_REGISTRY

echo ""
echo "[install-cn] selected mirrors:"
echo "  GitHub:  ${LORAHUB_GH_PROXY:-(direct)}"
echo "  Python:  $UV_PYTHON_INSTALL_MIRROR"
echo "  PyPI:    $UV_INDEX_URL"
echo "  Node:    $LORAHUB_NODE_MIRROR"
echo "  npm:     $NPM_CONFIG_REGISTRY"
echo ""

exec bash "$SCRIPT_DIR/install.sh" "$@"
