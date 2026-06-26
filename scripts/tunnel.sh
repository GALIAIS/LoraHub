#!/usr/bin/env bash
set -euo pipefail

# LoRaHub SSH tunnel (Linux / macOS)
# Usage:
#   scripts/tunnel.sh <user@host> [ssh_port] [local_port] [remote_port]
# Example:
#   scripts/tunnel.sh root@1.2.3.4 22 18080 18765

TARGET="${1:-}"
SSH_PORT="${2:-22}"
LOCAL_PORT="${3:-18080}"
REMOTE_PORT="${4:-18765}"

if [[ -z "$TARGET" ]]; then
    echo "Usage: scripts/tunnel.sh <user@host> [ssh_port] [local_port] [remote_port]"
    echo "Example: scripts/tunnel.sh root@1.2.3.4 22 18080 18765"
    exit 2
fi

if ! command -v ssh >/dev/null 2>&1; then
    echo "[ERROR] ssh not found. Install OpenSSH client first."
    exit 1
fi

cat <<EOF

============================================================
  LoRaHub SSH Tunnel
============================================================
  SSH:        ${TARGET}:${SSH_PORT}
  Browser:    http://127.0.0.1:${LOCAL_PORT}/
  Forward:    127.0.0.1:${LOCAL_PORT} -> 127.0.0.1:${REMOTE_PORT}
============================================================
Keep this terminal open while using LoRaHub. Press Ctrl+C to stop.

EOF

exec ssh -p "$SSH_PORT" -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$TARGET"
