#!/usr/bin/env bash
# List file inventory of Anima + Qwen-Image dependency repos via hf-mirror.
set -e

list_repo() {
  local repo="$1"
  local subpath="${2:-}"
  local label="${3:-$repo${subpath:+/$subpath}}"
  echo "=== $label ==="
  if [ -n "$subpath" ]; then
    url="https://hf-mirror.com/api/models/$repo/tree/main/$subpath"
  else
    url="https://hf-mirror.com/api/models/$repo/tree/main?recursive=true"
  fi
  curl -sS "$url" | python3 -c '
import sys, json
data = json.load(sys.stdin)
for x in data:
    if x.get("type") == "file":
        path = x["path"]
        size = x.get("size", 0) or 0
        if size >= 1024*1024:
            sz = f"{size/1024/1024:.1f}MiB"
        elif size >= 1024:
            sz = f"{size/1024:.1f}KiB"
        else:
            sz = f"{size}B"
        print(f"  {sz:>10s}  {path}")
'
  echo
}

list_repo "circlestone-labs/Anima"
list_repo "Comfy-Org/Qwen-Image_ComfyUI" "split_files/text_encoders" "Comfy-Org/Qwen-Image_ComfyUI :: text_encoders"
list_repo "Comfy-Org/Qwen-Image_ComfyUI" "split_files/vae" "Comfy-Org/Qwen-Image_ComfyUI :: vae"
list_repo "Comfy-Org/Qwen-Image_ComfyUI" "split_files/diffusion_models" "Comfy-Org/Qwen-Image_ComfyUI :: diffusion_models"
