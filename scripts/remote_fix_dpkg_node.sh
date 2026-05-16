#!/usr/bin/env bash
# Aggressively repair dpkg + ensure node 20. Tolerates read-only /etc/timezone.
set +e

echo "=== inspecting dpkg state ==="
dpkg -l 2>/dev/null | awk '$1=="iU" {print}' | head -10

echo
echo "=== forcing tzdata to configure with Asia/Shanghai (no /etc/timezone write needed) ==="
# debconf seeds tzdata's choice without touching /etc/timezone (which is RO).
echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections
echo 'tzdata tzdata/Zones/Asia select Shanghai' | debconf-set-selections
DEBIAN_FRONTEND=noninteractive dpkg --configure tzdata 2>&1 | tail -10

echo
echo "=== retrying dpkg --configure -a ==="
DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>&1 | tail -15

echo
echo "=== still-broken packages ==="
dpkg -l 2>/dev/null | awk '$1=="iU" {print}' | head -10

echo
echo "=== current node ==="
which node npm 2>/dev/null
node -v 2>&1
npm -v 2>&1

if ! command -v npm >/dev/null 2>&1 || ! node -v 2>/dev/null | grep -qE '^v(1[89]|2[0-9])'; then
  echo
  echo "=== installing node 20 directly via tarball (avoid apt entirely) ==="
  cd /tmp
  rm -rf node-v20.18.0-linux-x64*
  curl -fsSL --max-time 60 https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz -o node20.tar.xz
  ls -la node20.tar.xz
  tar -xJf node20.tar.xz
  rm -rf /opt/node20
  mv node-v20.18.0-linux-x64 /opt/node20
  ln -sf /opt/node20/bin/node /usr/local/bin/node
  ln -sf /opt/node20/bin/npm  /usr/local/bin/npm
  ln -sf /opt/node20/bin/npx  /usr/local/bin/npx
fi

echo
echo "=== final node ==="
which node npm
node -v
npm -v
