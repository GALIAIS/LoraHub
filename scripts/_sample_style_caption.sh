#!/usr/bin/env bash
# Pick 5 random images from the lx dataset and run style-mode smart-caption
# on each, printing only the resulting caption. Used to spot-check the prompt
# before committing to a full-dataset run.
set -e
DIR=/root/autodl-tmp/LoraHub/datasets/lx
mapfile -t FILES < <(ls "$DIR" | grep -iE '\.(jpg|jpeg|png|webp)$' | shuf -n 5)
for f in "${FILES[@]}"; do
  echo "=== $f ==="
  curl -s http://127.0.0.1:6006/api/image-studio/ai/smart-caption/single \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"path\":\"$DIR/$f\",\"captionMode\":\"style\",\"triggerWord\":\"anima style\",\"mergeStrategy\":\"replace\"}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("caption",d))'
  echo
done
