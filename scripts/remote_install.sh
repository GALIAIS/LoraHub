#!/usr/bin/env bash
# Run on the remote box. Provisions a venv, installs lorahub deps, builds web/dist.
set -e
cd /root/lorahub

echo "=== creating venv ==="
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -V
pip --version

echo "=== upgrading pip + setting PyPI mirror ==="
# tuna mirror — global pip default for this run
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -q --upgrade pip wheel setuptools

echo "=== installing lorahub[api,dev] ==="
pip install -e ".[api,dev]" 2>&1 | tail -15

echo "=== quick import sanity ==="
python -c "from lorahub.api.app import app; from lorahub.core.config.schema import RecipeConfig; print('lorahub import OK; arch lit count =', len(RecipeConfig.model_fields['base_model'].annotation.model_fields['arch'].annotation.__args__))"

echo "=== upgrading nodejs to 18 (LTS) via NodeSource ==="
node -v
if ! node -v | grep -qE '^v(1[89]|2[0-9])'; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
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
