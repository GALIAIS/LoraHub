# syntax=docker/dockerfile:1.7
# LoraHub — Docker image (GPU-default, CPU-capable)
#
# One image serves both profiles:
#   * GPU: docker compose --profile gpu up → nvidia runtime, WD14/onnxruntime-gpu
#   * CPU: docker compose --profile cpu up → same image, no GPU device attached
#
# The image carries the LoraHub *application* only — code, vendored
# `external/` backends, the built SPA, and the main venv with the
# `[api,gpu,tagging]` extras. Training-backend venvs (kohya /
# diffusion-pipe / anima_lora / ai-toolkit) are NOT preinstalled: they
# are bootstrapped at runtime into the mounted `/data/.lorahub/` volume,
# matching the project's "each backend owns its isolated venv" design
# and keeping the image small + backend versions swappable.
#
# Build args (all optional, empty = upstream default) let users in
# restricted networks route pip / npm / git through mirrors — same
# knobs the in-tree install scripts expose:
#
#   LORAHUB_APP_VERSION  Runtime/UI version string (e.g. 1.1.0-main-gabc12345)
#   LORAHUB_PYPI_INDEX   PyPI index URL            (e.g. https://pypi.tuna.tsinghua.edu.cn/simple)
#   LORAHUB_GH_PROXY     GitHub proxy prefix       (e.g. https://gh-proxy.org/)
#   NPM_CONFIG_REGISTRY  npm registry              (e.g. https://registry.npmmirror.com)
#
#   docker build -t lorahub:dev \
#     --build-arg LORAHUB_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
#     -f docker/Dockerfile .

# ──────────────────────────────────────────────────────────────────────
# Stage 1 — build the React SPA
# ──────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS web-build

WORKDIR /build/web

# Copy lockfile + manifests first for layer caching. package-lock.json is
# the source of truth; `npm ci` refuses to run without it.
COPY web/package.json web/package-lock.json ./
ARG NPM_CONFIG_REGISTRY=""
RUN --mount=type=cache,target=/root/.npm \
    if [ -n "$NPM_CONFIG_REGISTRY" ]; then \
        npm config set registry "$NPM_CONFIG_REGISTRY"; \
    fi \
    && npm ci --no-audit --no-fund

# Now copy the rest of the source and build. Vite needs the full tree
# (index.html, src/, public/, tsconfig, vite.config) to compile.
COPY web/ ./
ARG LORAHUB_APP_VERSION=""
ENV LORAHUB_APP_VERSION="${LORAHUB_APP_VERSION}"
RUN npm run build


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — runtime: Python + uv + LoraHub app
# ──────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ca-certificates + curl for healthcheck & bootstrap downloads;
# git for `lorahub manage update` / kohya clone; build-essential for
# any source-built wheel during backend bootstrap (rare, but keeps
# `pip install <sdist>` from failing opaquely).
# libgl1 / libglib2.0-0 satisfy Pillow's binary deps on slim images.
# tzdata lets the container honour TZ; ffmpeg is needed by some
# anima_lora / diffusion-pipe sample pipelines; gosu drops to the
# non-root lorahub user in entrypoint.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        libgl1 libglib2.0-0 \
        tzdata ffmpeg gosu \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 from deadsnakes — the project's primary target
# (pyproject.toml: requires-python = ">=3.11,<3.13"; install.sh pins
# 3.12). uv is the canonical installer across the project's own
# scripts, so we use it here too for parity.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Install uv standalone (mirrors scripts/install.sh): download the
# binary into /usr/local/bin so it's on PATH for both the build and
# for in-container `uv sync` during backend bootstrap.
ARG UV_ARCH=x86_64-unknown-linux-gnu
ARG LORAHUB_GH_PROXY=""
RUN case "${LORAHUB_GH_PROXY}" in \
      ""|*"github.com"*) BASE="" ;; \
      *) BASE="${LORAHUB_GH_PROXY%/}" ;; \
    esac ; \
    URL="${BASE:+$BASE/}https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz" ; \
    curl -fsSL "$URL" -o /tmp/uv.tar.gz \
    && tar -xzf /tmp/uv.tar.gz -C /tmp --strip-components=1 \
    && mv /tmp/uv /usr/local/bin/uv \
    && chmod +x /usr/local/bin/uv \
    && rm -rf /tmp/uv*

# A single project-owned venv keeps the image layer clean and lets the
# entrypoint `exec` the uvicorn binary directly. uv creates the venv
# and seeds pip / setuptools / wheel (same `--seed` rationale as
# install.sh: backend bootstrap subprocesses that shell out to
# `pip install` keep working).
RUN uv venv /opt/venv --python /usr/local/bin/python --seed
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /app

# Copy only the metadata first so dependency resolution caches on its
# own layer — source changes don't bust the pip cache.
COPY pyproject.toml ./
# hatch-vcs needs .git to derive a version; without it the editable
# install falls back to 0.0.0. Copy .git for the version probe, then
# the editable install picks up the source tree below.
# (We .dockerignore node_modules / runs / datasets etc., so COPY is lean.)
COPY . ./

# Install the app with the api/gpu/tagging extras. `--link-mode=copy`
# matches install.sh so the editable install works on overlayfs.
# UV_INDEX_URL / UV_DEFAULT_INDEX are uv-native; set when a mirror is
# requested.
ARG LORAHUB_PYPI_INDEX=""
ARG LORAHUB_PEP440_VERSION="0.0.0"
RUN if [ -n "$LORAHUB_PYPI_INDEX" ]; then \
        export UV_DEFAULT_INDEX="$LORAHUB_PYPI_INDEX"; \
    fi ; \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LORAHUB="$LORAHUB_PEP440_VERSION" ; \
    uv pip install -e ".[api,gpu,tagging]" --link-mode=copy

# Frontend from stage 1. The API serves this SPA directly
# (lorahub/api/app.py mounts web/dist), so no nginx sidecar is needed.
COPY --from=web-build /build/web/dist ./web/dist

# ── Runtime environment ───────────────────────────────────────────────
# LORAHUB_HOME is the single project root inside the container. All
# state (runs/, configs/, models/, .lorahub/, .cache/, XDG, HF) lives
# under /data, which is a named volume — survives `compose down`.
ENV LORAHUB_HOME=/data
ENV LORAHUB_DOCKER=1
ARG LORAHUB_APP_VERSION=""
ENV LORAHUB_APP_VERSION="${LORAHUB_APP_VERSION}"
ENV LORAHUB_WEB_DIST=/app/web/dist
# Container-internal bind is 0.0.0.0; the compose port mapping keeps
# the host side at 127.0.0.1 by default (no built-in auth — see README).
ENV LORAHUB_BIND_HOST=0.0.0.0
ENV LORAHUB_PORT=18765
# Default zh UI; override via -e LORAHUB_LANG=en.
ENV LORAHUB_LANG=zh
# Platformdirs state (settings.json / tasks.sqlite3 / runtime_bind /
# update-cache) — redirect into the volume so `down` doesn't drop it.
ENV XDG_DATA_HOME=/data/xdg/data
ENV XDG_STATE_HOME=/data/xdg/state
# HuggingFace model cache (WD14 / JoyTag first-load) — redirect into
# the volume too; otherwise it lands in ~/.cache and is lost on `down`.
ENV HF_HOME=/data/hf-home
ENV HUGGINGFACE_HUB_CACHE=/data/hf-home/hub
# Backend bootstrap writes here; keep it on the volume.
ENV UV_CACHE_DIR=/data/.cache/uv

# Non-root user. PUID/PGID let the host match volume file ownership
# (same pattern as linuxserver images). Created in entrypoint so the
# exact UID/GID can be chosen at runtime.
ENV PUID=1000
ENV PGID=1000

EXPOSE 18765

# Healthcheck hits the API's own /api/health endpoint. curl is already
# installed. --fail maps non-2xx to exit 1.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${LORAHUB_PORT}/api/health" || exit 1

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
