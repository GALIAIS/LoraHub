#!/usr/bin/env bash
# wsl_remote.sh — WSL-side wrapper that forwards a command to a remote host
# without leaking the local Windows-mixed PATH into the remote shell.
#
# Why this exists: when you run `wsl -d <distro> -- bash -lc 'ssh ...'` the
# WSL distro inherits PATH from Windows (interop is on by default), and any
# heredoc / inline command sent over ssh that references `$PATH` ends up
# resolving against THAT mess. We bypass the problem by:
#   1. Reading the password from stdin OR from $LORAHUB_REMOTE_PASS
#   2. Driving ssh via sshpass with non-interactive flags
#   3. Setting a single sane remote PATH on every command we launch
#
# Subcommands:
#   setup              First-time install: node20 + uv + venv + npm + vite build
#   serve              Start uvicorn on :6006 (kills prior, waits for /api/health)
#   restart            Alias for serve (re-launch uvicorn without re-setup)
#   stop               Kill the running uvicorn worker
#   deploy             pull -> setup -> serve, the full happy path
#   pull               git fetch + reset --hard origin/main on the VPS
#   status             Show ports, uvicorn pid, venv version, dist presence
#   health             curl /api/health and print the response
#   clean [all|venv|web|logs]   Wipe build artefacts to force a fresh setup
#   logs [setup|install|npm|vite|uvicorn]   Tail the relevant log file
#   shell              Open an interactive ssh shell at the repo root
#   exec '<cmd>'       Run an ad-hoc command with the sanitised remote PATH
#
# Usage examples:
#   echo '<password>' | scripts/wsl_remote.sh setup        # first-time install
#   LORAHUB_REMOTE_PASS=... scripts/wsl_remote.sh deploy   # pull + setup + serve
#   scripts/wsl_remote.sh shell                            # interactive shell
#   scripts/wsl_remote.sh exec 'tail -f /root/uvicorn.log'
#
# Required env (override on call site):
#   LORAHUB_REMOTE_HOST   default: connect.westc.seetacloud.com
#   LORAHUB_REMOTE_PORT   default: 45300
#   LORAHUB_REMOTE_USER   default: root
#   LORAHUB_REMOTE_DIR    default: /root/autodl-tmp/LoraHub
#   LORAHUB_REMOTE_NODE   default: /root/autodl-tmp/opt/node20/bin
#   LORAHUB_REMOTE_PASS   the SSH password (or pipe it in on stdin)

set -euo pipefail

REMOTE_HOST="${LORAHUB_REMOTE_HOST:-connect.westc.seetacloud.com}"
REMOTE_PORT="${LORAHUB_REMOTE_PORT:-45300}"
REMOTE_USER="${LORAHUB_REMOTE_USER:-root}"
REMOTE_DIR="${LORAHUB_REMOTE_DIR:-/root/autodl-tmp/LoraHub}"
REMOTE_NODE_BIN="${LORAHUB_REMOTE_NODE:-/root/autodl-tmp/opt/node20/bin}"

# Sanitised PATH the remote shell will see. Must NOT reference $PATH so the
# WSL-side $PATH never sneaks in via heredoc expansion.
REMOTE_PATH="${REMOTE_NODE_BIN}:/root/.local/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

usage() {
  sed -n '2,42p' "$0"
  exit "${1:-1}"
}

read_pass() {
  if [[ -n "${LORAHUB_REMOTE_PASS:-}" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    LORAHUB_REMOTE_PASS="$(cat)"
    LORAHUB_REMOTE_PASS="${LORAHUB_REMOTE_PASS%%$'\n'*}"
  fi
  if [[ -z "${LORAHUB_REMOTE_PASS:-}" ]]; then
    echo "remote password not provided. Set LORAHUB_REMOTE_PASS or pipe it on stdin." >&2
    exit 2
  fi
  export LORAHUB_REMOTE_PASS
}

require_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass not installed. apt-get install -y sshpass (run inside WSL)" >&2
    exit 3
  fi
}

ssh_args() {
  cat <<-EOF
		-o StrictHostKeyChecking=accept-new
		-o ConnectTimeout=20
		-o ServerAliveInterval=30
		-o ServerAliveCountMax=4
		-p ${REMOTE_PORT}
	EOF
}

run_remote() {
  # Push a command string to the remote shell. Stdin (if any) is forwarded
  # so heredocs / file uploads work. The remote PATH is exported before the
  # user command runs.
  local cmd="$*"
  SSHPASS="${LORAHUB_REMOTE_PASS}" sshpass -e ssh \
    $(ssh_args) \
    "${REMOTE_USER}@${REMOTE_HOST}" \
    "export PATH=${REMOTE_PATH}; cd ${REMOTE_DIR} 2>/dev/null || true; ${cmd}"
}

push_file() {
  local local_path="$1"
  local remote_path="$2"
  SSHPASS="${LORAHUB_REMOTE_PASS}" sshpass -e scp \
    -o StrictHostKeyChecking=accept-new \
    -P "${REMOTE_PORT}" \
    "${local_path}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}"
}

cmd_shell() {
  SSHPASS="${LORAHUB_REMOTE_PASS}" sshpass -e ssh \
    $(ssh_args) \
    -t "${REMOTE_USER}@${REMOTE_HOST}" \
    "export PATH=${REMOTE_PATH}; cd ${REMOTE_DIR} 2>/dev/null || true; bash -l"
}

cmd_exec() {
  if [[ $# -lt 1 ]]; then
    echo "usage: $0 exec '<command>'" >&2
    exit 1
  fi
  run_remote "$@"
}

cmd_setup() {
  # Push the setup script and run it. setsid means the install survives
  # the SSH socket dropping mid-run (large pip downloads can hang for
  # minutes and AutoDL kills idle sshds aggressively).
  local script_path="${REMOTE_DIR}/scripts/remote_setup.sh"
  run_remote "test -f ${script_path} || { echo 'remote_setup.sh missing — pull the repo first'; exit 1; }; chmod +x ${script_path}; setsid bash -c '${script_path} </dev/null >/root/_setup.log 2>&1; touch /root/_setup_done' </dev/null >/dev/null 2>&1 & disown; sleep 1; echo 'setup launched. follow with: $0 logs setup'"
}

cmd_serve() {
  local script_path="${REMOTE_DIR}/scripts/remote_serve.sh"
  run_remote "test -f ${script_path} || { echo 'remote_serve.sh missing'; exit 1; }; chmod +x ${script_path}; ${script_path}"
}

cmd_logs() {
  local which="${1:-uvicorn}"
  case "$which" in
    setup)   run_remote "tail -f /root/_setup.log" ;;
    install) run_remote "tail -f /root/_uv_install.log" ;;
    npm)     run_remote "tail -f /root/_npm_install.log" ;;
    vite)    run_remote "tail -f /root/_vite_build.log" ;;
    uvicorn|*) run_remote "tail -f /root/uvicorn.log" ;;
  esac
}

cmd_status() {
  run_remote "echo --- ports; ss -tln 2>/dev/null | grep -E ':(6006|6008|18765) ' || echo no_ports; echo --- uvicorn; ps -ef | grep 'uvicorn lorahub' | grep -v grep || echo no_uvicorn; echo --- venv; ls ${REMOTE_DIR}/.venv/bin/python 2>/dev/null && ${REMOTE_DIR}/.venv/bin/python -c 'import lorahub; print(lorahub.__version__)' 2>/dev/null || echo no_venv; echo --- web/dist; ls ${REMOTE_DIR}/web/dist/index.html 2>/dev/null && echo dist_present || echo no_dist"
}

cmd_pull() {
  run_remote "cd ${REMOTE_DIR} && git fetch origin main && git reset --hard origin/main && git log --oneline -3"
}

cmd_health() {
  # Check whether uvicorn responds. Prints the JSON or an error.
  run_remote "curl -fsS --max-time 5 http://127.0.0.1:6006/api/health 2>&1 || echo 'API not reachable on :6006'"
}

cmd_stop() {
  # Kill the uvicorn worker. Useful when you want to reclaim VRAM/RAM
  # without redeploying.
  run_remote "ps -ef | grep 'uvicorn lorahub' | grep -v grep | awk '{print \$2}' | xargs -r kill -9; sleep 1; ps -ef | grep 'uvicorn lorahub' | grep -v grep || echo stopped"
}

cmd_restart() {
  # Re-launch uvicorn without re-running setup. Equivalent to remote_serve.
  cmd_serve
}

cmd_deploy() {
  # The full happy path: pull latest, run setup (idempotent), restart
  # uvicorn. Use this after pushing a new commit from the dev box.
  echo "[deploy] step 1/3: git pull"
  cmd_pull
  echo "[deploy] step 2/3: setup (background; tail with: $0 logs setup)"
  cmd_setup
  echo "[deploy] step 3/3: waiting for setup to finish then restarting uvicorn"
  run_remote "until [ -f /root/_setup_done ] || ! pgrep -f remote_setup.sh > /dev/null; do sleep 3; done; tail -5 /root/_setup.log"
  cmd_serve
}

cmd_clean() {
  # Wipe per-step build artefacts. Keeps the repo + venv but forces the
  # next setup to redo deps and dist. Use when bumping pyproject.toml.
  case "${1:-all}" in
    venv)  run_remote "rm -rf ${REMOTE_DIR}/.venv && echo venv_removed" ;;
    web)   run_remote "rm -rf ${REMOTE_DIR}/web/node_modules ${REMOTE_DIR}/web/dist && echo web_artifacts_removed" ;;
    logs)  run_remote "rm -f /root/_setup.log /root/_uv_install.log /root/_npm_install.log /root/_vite_build.log /root/_setup_done /root/uvicorn.log && echo logs_cleared" ;;
    all|*) run_remote "rm -rf ${REMOTE_DIR}/.venv ${REMOTE_DIR}/web/node_modules ${REMOTE_DIR}/web/dist && rm -f /root/_setup.log /root/_uv_install.log /root/_npm_install.log /root/_vite_build.log /root/_setup_done /root/uvicorn.log && echo all_cleared" ;;
  esac
}

main() {
  if [[ $# -lt 1 ]]; then usage; fi
  local sub="$1"; shift
  # Help / usage subcommands don't need credentials or sshpass.
  case "$sub" in
    -h|--help|help) usage 0 ;;
  esac
  read_pass
  require_sshpass
  case "$sub" in
    shell)   cmd_shell ;;
    exec)    cmd_exec "$@" ;;
    setup)   cmd_setup ;;
    serve)   cmd_serve ;;
    restart) cmd_restart ;;
    stop)    cmd_stop ;;
    deploy)  cmd_deploy ;;
    clean)   cmd_clean "$@" ;;
    logs)    cmd_logs "$@" ;;
    status)  cmd_status ;;
    health)  cmd_health ;;
    pull)    cmd_pull ;;
    *) echo "unknown subcommand: $sub" >&2; usage ;;
  esac
}

main "$@"
