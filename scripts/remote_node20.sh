#!/usr/bin/env bash
# Force-replace stale Node 12 with Node 20, rebuild web/dist.
set -e

echo "=== current node ==="
node -v 2>&1 || true

echo "=== purging old node ==="
apt-get purge -y nodejs npm libnode72 2>&1 | tail -3 || true
apt-get autoremove -y -qq 2>&1 | tail -3 || true

echo "=== installing node 20 (NodeSource) ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1 | tail -5
apt-get install -y -qq nodejs 2>&1 | tail -3
node -v
npm -v

echo "=== rebuilding web/dist ==="
cd /root/lorahub/web
rm -rf node_modules
npm config set registry https://registry.npmmirror.com/
npm install --no-fund --no-audit 2>&1 | tail -5
npm run build 2>&1 | tail -15
echo
ls -la dist | head -10
