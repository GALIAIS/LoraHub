#!/usr/bin/env bash
# Pin Python 3.11 from deadsnakes, since lorahub needs >=3.11. Then re-create
# the venv with python3.11 and reinstall.
set -e
cd /root/lorahub

echo "=== installing python 3.11 (deadsnakes) ==="
apt-get install -y -qq software-properties-common 2>&1 | tail -3
add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -3
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11 python3.11-venv python3.11-dev 2>&1 | tail -3
python3.11 --version

echo "=== rebuilding venv ==="
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -V

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -q --upgrade pip wheel setuptools

echo "=== installing lorahub[api,dev] ==="
pip install -e ".[api,dev]" 2>&1 | tail -10

echo "=== sanity ==="
python -c "import fastapi, uvicorn, lorahub; from lorahub.core.config.schema import RecipeConfig; arch = RecipeConfig.model_fields['base_model'].annotation.model_fields['arch']; print('OK; lorahub', lorahub.__version__ if hasattr(lorahub,'__version__') else 'imported')"

echo "=== node version ==="; node -v
echo "=== installing nodejs 20 (NodeSource) ==="
if ! node -v 2>/dev/null | grep -qE '^v(1[89]|2[0-9])'; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1 | tail -5
  apt-get install -y -qq nodejs 2>&1 | tail -3
fi
node -v
npm -v

echo "=== installing web deps ==="
cd web
npm config set registry https://registry.npmmirror.com/
npm install --no-fund --no-audit 2>&1 | tail -8

echo "=== building web/dist ==="
npm run build 2>&1 | tail -8
ls -la dist | head -10
echo DONE
