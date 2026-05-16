#!/usr/bin/env bash
# Force Node 20 + start backend (uvicorn 18765) + frontend (Vite dev 6006).
set -e

echo "=== checking dpkg health (tzdata blockers) ==="
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime 2>/dev/null || true
echo Asia/Shanghai > /etc/timezone 2>/dev/null || true
dpkg --configure -a 2>&1 | tail -5 || true

echo
echo "=== node version ==="
node -v 2>&1 || echo "(no node)"

if ! node -v 2>/dev/null | grep -qE '^v(1[89]|2[0-9])'; then
  echo "=== removing old node 12 ==="
  apt-get remove -y --purge nodejs npm libnode72 2>&1 | tail -3 || true
  apt-get autoremove -y -qq 2>&1 | tail -3 || true
  echo "=== installing node 20 (NodeSource) ==="
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1 | tail -5
  apt-get install -y -qq nodejs 2>&1 | tail -3
fi
echo "node $(node -v)  npm $(npm -v)"

echo
echo "=== reinstalling web deps with new node ==="
cd /root/lorahub/web
rm -rf node_modules package-lock.json
npm config set registry https://registry.npmmirror.com/
npm install --no-fund --no-audit 2>&1 | tail -5

echo
echo "=== killing any prior services on 6006/18765 ==="
pkill -f 'uvicorn.*lorahub' 2>/dev/null || true
pkill -f 'vite' 2>/dev/null || true
sleep 1

echo
echo "=== starting uvicorn (background, listens 127.0.0.1:18765) ==="
cd /root/lorahub
source .venv/bin/activate
nohup python -m uvicorn lorahub.api.app:app --host 127.0.0.1 --port 18765 --log-level info > /root/uvicorn.log 2>&1 &
UVI_PID=$!
echo "uvicorn pid=$UVI_PID"

echo
echo "=== starting vite (background, listens 0.0.0.0:6006) ==="
cd /root/lorahub/web
export LORAHUB_API_TARGET=http://127.0.0.1:18765
# host 0.0.0.0 so the container's port-mapping can route external traffic
nohup npm run dev -- --host 0.0.0.0 --port 6006 > /root/vite.log 2>&1 &
VITE_PID=$!
echo "vite pid=$VITE_PID"

echo
echo "=== waiting up to 30s for both to come up ==="
for i in $(seq 1 30); do
  api_ok=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18765/api/health 2>/dev/null || echo 000)
  web_ok=$(curl -sS --max-time 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:6006/ 2>/dev/null || echo 000)
  if [ "$api_ok" = "200" ] && [ "$web_ok" = "200" ]; then
    echo "  [${i}s] api=$api_ok web=$web_ok  -- READY"
    break
  fi
  if [ $((i % 3)) -eq 0 ]; then echo "  [${i}s] api=$api_ok web=$web_ok"; fi
  sleep 1
done

echo
echo "=== final listeners ==="
ss -tln 2>/dev/null | awk 'NR==1 || /:(6006|18765)\s/'
echo
echo "=== uvicorn log tail ==="
tail -8 /root/uvicorn.log
echo
echo "=== vite log tail ==="
tail -10 /root/vite.log
