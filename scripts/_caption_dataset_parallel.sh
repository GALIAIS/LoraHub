#!/usr/bin/env bash
# Concurrent caption driver: spawn N parallel workers against the
# single-image smart-caption endpoint. Network-bound (Claude vision call
# dominates), so concurrency translates almost linearly into throughput.
set -u
DIR=${DIR:-/root/autodl-tmp/LoraHub/datasets/lx}
TRIGGER=${TRIGGER:-Kiko.L}
MODE=${MODE:-style}
STRATEGY=${STRATEGY:-replace}
LOG=${LOG:-/tmp/caption_run.log}
ENDPOINT=http://127.0.0.1:6006/api/image-studio/ai/smart-caption/single
JOBS=${JOBS:-6}

caption_one() {
  local f=$1
  local resp code
  resp=$(curl -s -m 240 -w '\n%{http_code}' "$ENDPOINT" \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"path\":\"$DIR/$f\",\"captionMode\":\"$MODE\",\"triggerWord\":\"$TRIGGER\",\"mergeStrategy\":\"$STRATEGY\"}")
  code=$(echo "$resp" | tail -n1)
  if [ "$code" = "200" ]; then
    printf '[%s] OK   %s\n' "$(date +%T)" "$f"
  else
    body=$(echo "$resp" | head -n -1 | head -c 200)
    printf '[%s] FAIL %s :: %s :: %s\n' "$(date +%T)" "$f" "$code" "$body"
  fi
}
export -f caption_one
export DIR TRIGGER MODE STRATEGY ENDPOINT

mapfile -t FILES < <(ls "$DIR" | grep -iE '\.(jpg|jpeg|png|webp)$' | sort)
TOTAL=${#FILES[@]}
echo "[$(date +%T)] starting: $TOTAL images, mode=$MODE, trigger=$TRIGGER, strategy=$STRATEGY, jobs=$JOBS" | tee "$LOG"

# Warm the WD14 tagger by running one image first so the EVA02 weights
# are loaded in the parent process before workers fan out.
echo "[$(date +%T)] warming up tagger with first image..." | tee -a "$LOG"
caption_one "${FILES[0]}" | tee -a "$LOG"

# Now run the remaining files in parallel.
printf '%s\n' "${FILES[@]:1}" \
  | xargs -P "$JOBS" -I{} bash -c 'caption_one "$@"' _ {} \
  | tee -a "$LOG"

echo "[$(date +%T)] done" | tee -a "$LOG"
