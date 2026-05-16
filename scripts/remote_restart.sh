#!/usr/bin/env bash
# Restart uvicorn + vite cleanly. Use Node 20 from /usr/local/bin.
set +e
export PATH="/usr/local/bin:$PATH"

echo "=== verifying lorahub package layout ==="
ls /root/lorahub/lorahub/core/models 2>&1 | head
ls /root/lorahub/lorahub/core/dataset 2>&1 | head

echo
echo "=== killing previous services ==="
pkill -f 'uvicorn.*lorahub' 2>/dev/null
pkill -f 'vite' 2>/dev/null
sleep 2

echo
echo "=== starting uvicorn 18765 ==="
cd /root/lorahub
source .venv/bin/activate
nohup python -m uvicorn lorahub.api.app:app --host 127.0.0.1 --port 18765 --log-level info > /root/uvicorn.log 2>&1 &
echo "uvicorn pid=$!"

echo
echo "=== starting vite 6006 (host 0.0.0.0, allowedHosts:true) ==="
cd /root/lorahub/web
export LORAHUB_API_TARGET=http://127.0.0.1:18765
nohup /usr/local/bin/npm run dev -- --host 0.0.0.0 --port 6006 > /root/vite.log 2>&1 &
echo "vite pid=$!"

echo
echo "=== waiting up to 60s ==="
for i in $(seq 1 60); do
  api=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18765/api/health 2>/dev/null || echo 000)
  web=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:6006/ 2>/dev/null || echo 000)
  if [ "$api" = "200" ] && [ "$web" = "200" ]; then
    echo "[${i}s] api=$api web=$web -> READY"
    break
  fi
  if [ $((i % 5)) -eq 0 ]; then echo "[${i}s] api=$api web=$web ..."; fi
  sleep 1
done

echo
echo "=== uvicorn log ==="
tail -8 /root/uvicorn.log
echo
echo "=== vite log ==="
tail -10 /root/vite.log
