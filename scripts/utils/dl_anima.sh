#!/usr/bin/env bash
# Pull the three Anima safetensors via hf-mirror with resume support.
set -e
mkdir -p /root/models/anima/diffusion_models
mkdir -p /root/models/anima/text_encoders
mkdir -p /root/models/anima/vae

BASE='https://hf-mirror.com/circlestone-labs/Anima/resolve/main/split_files'

dl() {
  local url="$1"; local out="$2"; local label="$3"
  echo "=== $label ==="
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" -gt $((100*1024*1024)) ]; then
    echo "already present: $out  size=$(du -h "$out" | cut -f1)"
    return 0
  fi
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$out" "$url"
  echo "-> $(du -h "$out" | cut -f1)  $out"
}

dl "$BASE/diffusion_models/anima-base-v1.0.safetensors" \
   /root/models/anima/diffusion_models/anima-base-v1.0.safetensors \
   "anima-base-v1.0 (3.9 GiB)"

dl "$BASE/text_encoders/qwen_3_06b_base.safetensors" \
   /root/models/anima/text_encoders/qwen_3_06b_base.safetensors \
   "qwen_3_06b_base (1.1 GiB)"

dl "$BASE/vae/qwen_image_vae.safetensors" \
   /root/models/anima/vae/qwen_image_vae.safetensors \
   "qwen_image_vae (242 MiB)"

echo "=== final layout ==="
du -ach /root/models/anima/* | sort
