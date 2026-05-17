#!/usr/bin/env bash
# Print 3 random captions to verify final output.
D=/root/autodl-tmp/LoraHub/datasets/lx
for f in $(ls "$D"/ | grep '\.txt$' | shuf -n 3); do
  echo "========== $f =========="
  cat "$D/$f"
  echo
done
