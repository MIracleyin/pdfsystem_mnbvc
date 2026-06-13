#!/usr/bin/env bash
# Live progress bars for the model weights pulled by scripts/download_models.sh.
#
# Each row shows: model name, downloaded MB / target MB, ASCII bar, speed.
# Refreshes every 3 s. Ctrl-C to quit.
#
# Target sizes are approximate (HF reports nothing reliable before finish);
# adjust if a download finishes >100% — that means the target estimate
# is too low, not that anything is wrong.
#
# Usage:
#   bash scripts/watch_downloads.sh         # default 3 s refresh
#   bash scripts/watch_downloads.sh 1       # 1 s refresh

INTERVAL=${1:-3}
HF_HUB="$HOME/.cache/huggingface/hub"

# Parallel arrays — works on macOS bash 3.2 (no associative arrays).
NAMES=(  "PDF-Extract-Kit-1.0"   "MinerU2.5-Pro (VLM)"                    "ModernBERT quality" )
TGT_MB=( 2400                    2400                                     1600                  )
DIRS=(   "models--opendatalab--PDF-Extract-Kit-1.0"  "models--opendatalab--MinerU2.5-Pro-2605-1.2B"  "models--HuggingFaceFW--finepdfs_ocr_quality_classifier_eng_Latn" )
PREV_MB=( 0 0 0 )

dir_mb() {
  local d="$1"
  [ -d "$d" ] || { echo 0; return; }
  du -sm "$d" 2>/dev/null | awk '{ print $1 }'
}

bar() {
  local cur=$1 tgt=$2 width=$3
  local pct=$(( cur * 100 / tgt ))
  [ $pct -gt 100 ] && pct=100
  local filled=$(( cur * width / tgt ))
  [ $filled -gt $width ] && filled=$width
  local empty=$(( width - filled ))
  printf '['
  local i=0
  while [ $i -lt $filled ]; do printf '#'; i=$(( i + 1 )); done
  i=0
  while [ $i -lt $empty ]; do printf '.'; i=$(( i + 1 )); done
  printf '] %3d%%' $pct
}

clear_screen() { tput clear 2>/dev/null || printf '\033[2J\033[H'; }

trap 'echo; echo "[watch] stopped."; exit 0' INT

START=$(date +%s)

while true; do
  clear_screen
  NOW=$(date '+%H:%M:%S')
  ELAPSED=$(( $(date +%s) - START ))

  printf '%s   ⟳ refresh %ss   elapsed %dm%02ds   (Ctrl-C to quit)\n' \
    "$NOW" "$INTERVAL" $(( ELAPSED / 60 )) $(( ELAPSED % 60 ))
  printf 'cache: %s\n\n' "$HF_HUB"

  total_cur=0
  total_tgt=0
  N=${#NAMES[@]}
  i=0
  while [ $i -lt $N ]; do
    name="${NAMES[$i]}"
    target_mb="${TGT_MB[$i]}"
    dir="${DIRS[$i]}"
    cur_mb=$(dir_mb "$HF_HUB/$dir")
    prev=${PREV_MB[$i]}
    delta=$(( cur_mb - prev ))
    if [ $delta -gt 0 ]; then
      speed=$(awk -v d="$delta" -v t="$INTERVAL" 'BEGIN { printf "%5.2f MB/s", d / t }')
    elif [ $cur_mb -ge $target_mb ]; then
      speed="     done   "
    else
      speed="     idle   "
    fi
    PREV_MB[$i]=$cur_mb

    printf '  %-22s  %5d / %5d MB  ' "$name" "$cur_mb" "$target_mb"
    bar "$cur_mb" "$target_mb" 30
    printf '  %s\n' "$speed"

    total_cur=$(( total_cur + cur_mb ))
    total_tgt=$(( total_tgt + target_mb ))
    i=$(( i + 1 ))
  done

  printf '\n  %-22s  %5d / %5d MB  ' "TOTAL" "$total_cur" "$total_tgt"
  bar "$total_cur" "$total_tgt" 30
  printf '\n'

  printf '\n  hint: scripts/download_models.sh runs the actual downloader; this just watches.\n'

  sleep "$INTERVAL"
done
