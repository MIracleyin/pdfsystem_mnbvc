#!/bin/bash
# Step 2 — the CPU lane. Run on xsy-01. ~4 hours at 32 workers.
#
# One pass does both jobs: `router` classifies every document and `extract`
# pulls the text out of the ones that have a text layer. Measured on 400 real
# documents, router+extract took 184.6 s against 222.5 s for router alone —
# extraction is free once the router has already opened and parsed the PDF.
# So there is no reason to classify in a separate pass first.
#
# --extract-backends mupdf is what makes this a lane: OCR-bound documents are
# recorded as `skip_reason=lane-filter`, carrying their path. That record IS
# the GPU box's worklist. Without the flag they would really be sent to MinerU.
set -e
cd "$(dirname "$0")" && source ./config.sh

[ -f "$RUN/bucket-00" ] || { echo "run 01-inventory.sh first" >&2; exit 1; }

cd "$RUN"
for b in bucket-*; do
  # Each worker gets its own --out-dir: results.jsonl is append-only, so two
  # writers in one directory interleave into a file neither can resume from.
  #
  # --markdown-dir is shared and absolute. Files are named <sha256>.md, so
  # workers collide only on true duplicates, which write identical bytes.
  #
  # --resume appends and skips what is already done. Without it a restart
  # TRUNCATES the file that is also the other box's worklist.
  nohup uv --directory "$PDFSYS" run pdfsys run \
    --pdf-list  "$RUN/$b" \
    --out-dir   "$RUN/p1/$b" \
    --stages    router,extract \
    --extract-backends mupdf \
    --markdown-dir "$RUN/markdown" \
    --ocr-threshold "$OCR_THRESHOLD" \
    --resume \
    > "$RUN/logs/$b.log" 2>&1 &
done

sleep 10
echo "  launched, $(pgrep -fc -- "--pdf-list $RUN/bucket-" || true) workers alive"
echo "  watch with: ops/cmn-hani/status.sh"
