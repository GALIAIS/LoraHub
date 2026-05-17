#!/usr/bin/env bash
# Download Anima v1.0 full stack to VPS for diffusion-pipe training.
# Files end up under: /root/autodl-tmp/LoraHub/models/circlestone-labs__Anima/split_files/
# Uses hf-mirror.com (China-friendly) and aria2c for multi-connection speed.
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/LoraHub/models/circlestone-labs__Anima/split_files}
MIRROR=${MIRROR:-https://hf-mirror.com}
REPO=circlestone-labs/Anima

declare -a FILES=(
  "diffusion_models/anima-base-v1.0.safetensors"
  "vae/qwen_image_vae.safetensors"
  "text_encoders/qwen_3_06b_base.safetensors"
)

mkdir -p "$ROOT"

# Pick a downloader: prefer aria2c (multi-connection, resumable), fall back
# to wget -c (single-connection, resumable) if aria2c is unavailable.
if command -v aria2c >/dev/null 2>&1; then
  FETCH=aria2c
else
  FETCH=wget
fi
echo "[anima] using downloader: $FETCH"
echo "[anima] mirror: $MIRROR"
echo "[anima] target: $ROOT"
echo

for rel in "${FILES[@]}"; do
  url="$MIRROR/$REPO/resolve/main/split_files/$rel"
  out="$ROOT/$rel"
  outdir=$(dirname "$out")
  fname=$(basename "$out")
  mkdir -p "$outdir"

  if [ -f "$out" ]; then
    sz=$(stat -c%s "$out")
    echo "[anima] EXISTS  $rel  ($sz bytes) — skip"
    continue
  fi

  echo "[anima] FETCH   $rel"
  if [ "$FETCH" = "aria2c" ]; then
    aria2c -x 16 -s 16 -k 1M --console-log-level=warn --summary-interval=10 \
      -d "$outdir" -o "$fname" "$url"
  else
    wget -c --show-progress -O "$out" "$url"
  fi
  echo "[anima] DONE    $rel"
  echo
done

echo "[anima] All files present:"
ls -lh "$ROOT"/*/*.safetensors
