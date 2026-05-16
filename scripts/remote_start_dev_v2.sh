#!/usr/bin/env bash
# Bypass apt entirely. Use pre-installed node 20 from /usr/local/bin and
# run uvicorn + vite directly. dpkg's broken state for python3.11 doesn't
# affect us — we use the .venv already created with python3.11.
set -e

# Make sure the new node 20 wins over the old apt-installed node 12.
export PATH="/usr/local/bin:$PATH"
echo "node $(node -v)  npm $(npm -v)  (PATH puts /usr/local/bin first)"

echo
echo "=== killing any stale services ==="
pkill -f 'uvicorn.*lorahub' 2>/dev/null || true
pkill -f 'vite' 2>/dev/null || true
pkill -f 'node.*vite' 2>/dev/null || true
sleep 1
ss -tln | awk '/:(6006|18765)\s/' || true

echo
echo "=== reinstalling web/node_modules with node 20 ==="
cd /root/lorahub/web
rm -rf node_modules package-lock.json
npm config set registry https://registry.npmmirror.com/
npm install --no-fund --no-audit 2>&1 | tail -5

echo
echo "=== starting uvicorn (background) ==="
cd /root/lorahub
source .venv/bin/activate
nohup python -m uvicorn lorahub.api.app:app --host 127.0.0.1 --port 18765 --log-level info > /root/uvicorn.log 2>&1 &
echo "uvicorn pid=$!"

echo
echo "=== starting vite dev (background, host 0.0.0.0) ==="
cd /root/lorahub/web
export LORAHUB_API_TARGET=http://127.0.0.1:18765
nohup /usr/local/bin/npm run dev -- --host 0.0.0.0 --port 6006 > /root/vite.log 2>&1 &
echo "vite pid=$!"

echo
echo "=== waiting up to 60s for both ==="
for i in $(seq 1 60); do
  api=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18765/api/health 2>/dev/null || echo 000)
  web=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:6006/ 2>/dev/null || echo 000)
  if [ "$api" = "200" ] && [ "$web" = "200" ]; then
    echo "[${i}s] api=$api web=$web  -> READY"
    break
  fi
  if [ $((i % 5)) -eq 0 ]; then echo "[${i}s] api=$api web=$web ..."; fi
  sleep 1
done

echo
echo "=== final listeners ==="
ss -tln | awk 'NR==1 || /:(6006|18765)\s/'
echo
echo "=== uvicorn log tail ==="
tail -10 /root/uvicorn.log
echo
echo "=== vite log tail ==="
tail -15 /root/vite.log
