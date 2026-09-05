#!/bin/bash
# Step 4 — ship the OCR-bound PDFs to xsy-02. Run on xsy-01.
#
# ~511 GB. Not the ~258 GB a count-proportional estimate gives: scanned PDFs
# average 9.47 MiB against 1.85 MiB for born-digital ones, so the OCR lane is
# 36% of the documents but 72% of the bytes. Shipping only what needs OCR
# saves 28% of the corpus, not 90% — the win is real but it is not the win
# the 10%-OCR assumption in the PRD implied.
#
# Do NOT run this while step 2 is still going: both read the same HDD.
set -e
cd "$(dirname "$0")" && source ./config.sh
_require_host "$CPU_HOST"

[ -s "$RUN/gpu_lane.txt" ] || { echo "run 03-handoff.sh first" >&2; exit 1; }

ssh "$GPUBOX" "mkdir -p $LANE/pdfs $LANE/p2/mineru"

# --partial so an interrupted transfer resumes instead of restarting.
# --files-from takes the relative list against the corpus root.
rsync -a --partial --info=progress2 \
  --files-from="$RUN/gpu_lane.txt" \
  "$CORPUS/" \
  "$GPUBOX:$LANE/pdfs/"

scp -q "$RUN/gpu_lane.txt" "$GPUBOX:$LANE/gpu_lane.txt"

want=$(wc -l < "$RUN/gpu_lane.txt")
got=$(ssh "$GPUBOX" "find $LANE/pdfs -type f | wc -l")
echo "  listed $want, landed $got"
[ "$want" -eq "$got" ] || echo "  WARNING $((want - got)) documents did not arrive" >&2
