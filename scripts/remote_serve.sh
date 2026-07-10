#!/usr/bin/env bash
# remote_serve.sh — start uvicorn on port 6006 from the LoraHub VPS.
#
# Single-process production setup: uvicorn serves both web/dist (built by
# remote_setup.sh) and the /api routes from one worker. This is the right
# shape for a single-user box and matches what AutoDL maps to its public
# URL via container-internal port forwarding.
#
# Idempotent — kills any prior `uvicorn lorahub.api.app` before launching.
# Logs to /root/uvicorn.log and prints the tail when /api/health goes
# green so you can confirm the deploy from the same SSH command.

set -uo pipefail

LORAHUB_DIR="${LORAHUB_DIR:-/root/autodl-tmp/LoraHub}"
PORT="${LORAHUB_PORT:-6006}"
HOST="${LORAHUB_HOST:-0.0.0.0}"
LOG="${LORAHUB_LOG:-/root/uvicorn.log}"

log() { printf '\033[36m[serve]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[serve error]\033[0m %s\n' "$*" >&2; }

if [[ ! -x "${LORAHUB_DIR}/.venv/bin/python" ]]; then
  err ".venv not found at ${LORAHUB_DIR}/.venv. Run scripts/remote_setup.sh first."
  exit 1
fi
if [[ ! -f "${LORAHUB_DIR}/web/dist/index.html" ]]; then
  err "web/dist missing. Run scripts/remote_setup.sh (it builds the SPA)."
  exit 1
fi

cd "${LORAHUB_DIR}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" && "${HOST}" != "::1" ]]; then
  if [[ -z "${LORAHUB_API_TOKEN:-}" ]]; then
    LORAHUB_API_TOKEN=$("${LORAHUB_DIR}/.venv/bin/python" -c \
      'from lorahub.api.auth import ensure_api_token; print(ensure_api_token())')
    TOKEN_PATH=$("${LORAHUB_DIR}/.venv/bin/python" -c \
      'from lorahub.api.auth import api_token_path; print(api_token_path())')
    log "Remote access authentication enabled. Token file: ${TOKEN_PATH}"
  else
    log "Remote access authentication enabled by LORAHUB_API_TOKEN."
  fi
  export LORAHUB_API_TOKEN
fi

# 1. Stop any prior uvicorn pinned to the same port.
log "Stopping any prior uvicorn on :${PORT}"
existing_pids=$(ps -ef | grep "uvicorn lorahub.api.app" | grep -v grep | awk '{print $2}')
if [[ -n "${existing_pids}" ]]; then
  echo "${existing_pids}" | xargs -r kill -9
  sleep 2
fi
# Belt-and-suspenders: anything still bound to :${PORT}.
port_holders=$(ss -tlnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $0}' | grep -oE 'pid=[0-9]+' | sort -u | sed 's/pid=//')
if [[ -n "${port_holders}" ]]; then
  echo "${port_holders}" | xargs -r kill -9
  sleep 1
fi

# 2. Launch (detached, survives SSH disconnect via setsid).
log "Starting uvicorn on http://${HOST}:${PORT}"
setsid bash -c "
  cd '${LORAHUB_DIR}'
  exec .venv/bin/python -m uvicorn lorahub.api.app:app --host ${HOST} --port ${PORT} --log-level info
" </dev/null >"${LOG}" 2>&1 &
disown

# 3. Wait up to 30s for health to come up.
deadline=$((SECONDS + 30))
ready=""
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    ready="yes"
    break
  fi
  sleep 1
done

if [[ "${ready}" != "yes" ]]; then
  err "API did not respond at http://127.0.0.1:${PORT}/api/health within 30s"
  err "Tail of ${LOG}:"
  tail -30 "${LOG}" >&2
  exit 1
fi

log "API healthy. Tail of ${LOG}:"
tail -8 "${LOG}"
log ""
log "Listening on ${HOST}:${PORT} -- AutoDL maps it to its public 6006 URL."
log "Stop with: pkill -f 'uvicorn lorahub.api.app'"
