#!/usr/bin/env bash
echo "=== listeners on 6006/18765/5173/22 ==="
ss -tlnp 2>/dev/null | head -1
ss -tlnp 2>/dev/null | awk '/:(6006|18765|5173|22)\s/'

echo
echo "=== node procs ==="
pgrep -af node | head -5 || echo "(no node)"
echo
echo "=== uvicorn procs ==="
pgrep -af uvicorn | head -5 || echo "(no uvicorn)"
echo
echo "=== curl localhost:6006 ==="
curl -sS --max-time 3 -o /dev/null -w "local 6006 -> http=%{http_code} time=%{time_total}s\n" http://127.0.0.1:6006/ 2>&1
echo
echo "=== web/dist exists? ==="
ls -la /root/lorahub/web/dist 2>&1 | head -5
