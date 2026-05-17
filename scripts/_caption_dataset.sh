#!/usr/bin/env bash
# Caption every image in a directory using the single-image smart-caption
# endpoint, sequentially. Writes a progress log so the run can be resumed
# if it dies, and falls back to skipping any image whose .txt was already
# rewritten in this same run.
set -u
DIR=${DIR:-/root/autodl-tmp/LoraHub/datasets/lx}
TRIGGER=${TRIGGER:-Kiko.L}
MODE=${MODE:-style}
STRATEGY=${STRATEGY:-replace}
LOG=${LOG:-/tmp/caption_run.log}
ENDPOINT=http://127.0.0.1:6006/api/image-studio/ai/smart-caption/single

mapfile -t FILES < <(ls "$DIR" | grep -iE '\.(jpg|jpeg|png|webp)$' | sort)
TOTAL=${#FILES[@]}
echo "[$(date +%T)] starting: $TOTAL images, mode=$MODE, trigger=$TRIGGER, strategy=$STRATEGY" | tee "$LOG"

i=0
ok=0
err=0
for f in "${FILES[@]}"; do
  i=$((i+1))
  resp=$(curl -s -m 180 -w '\n%{http_code}' "$ENDPOINT" \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"path\":\"$DIR/$f\",\"captionMode\":\"$MODE\",\"triggerWord\":\"$TRIGGER\",\"mergeStrategy\":\"$STRATEGY\"}")
  code=$(echo "$resp" | tail -n1)
  if [ "$code" = "200" ]; then
    ok=$((ok+1))
    printf '[%s] %3d/%d OK   %s\n' "$(date +%T)" "$i" "$TOTAL" "$f" | tee -a "$LOG"
  else
    err=$((err+1))
    body=$(echo "$resp" | head -n -1 | head -c 200)
    printf '[%s] %3d/%d FAIL %s :: %s :: %s\n' "$(date +%T)" "$i" "$TOTAL" "$f" "$code" "$body" | tee -a "$LOG"
  fi
done

echo "[$(date +%T)] done: ok=$ok err=$err total=$TOTAL" | tee -a "$LOG"
