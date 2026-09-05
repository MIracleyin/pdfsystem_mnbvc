#!/bin/bash
# Step 5 — the GPU lane. Run ON xsy-02, after step 4.
#
# There is no `layout` stage here. MinerU does its own layout analysis on the
# bytes it receives; running layout on the CPU box would be paid for twice and
# thrown away once (LayoutCache is written but never read back).
set -e
cd "$(dirname "$0")" && source ./config.sh

[ -s "$LANE/gpu_lane.txt" ] || { echo "no worklist at $LANE/gpu_lane.txt" >&2; exit 1; }
mkdir -p "$LANE"/{p2,logs,markdown}

cd "$LANE"
rm -f gbucket-*
split -n "l/$GPU_WORKERS" -d -a 2 gpu_lane.txt gbucket-

for b in gbucket-*; do
  # MINERU_PIPELINE_URL is mandatory: `pdfsys run` has no --mineru-url flag
  # (that exists only on `pdfsys smoke`). Without it the parser tries to spawn
  # a local mineru-api and loads models on the wrong machine, or just fails.
  #
  # --parser-output-dir decides whether the output survives at all. MinerU's
  # own copy lives in the api container and is garbage-collected. Omitting it
  # does not abort the run — it warns once, keeps the markdown, and leaves
  # nothing for `pdfsys dataset --from-mineru` to read.
  MINERU_PIPELINE_URL="$MINERU_URL" NO_PROXY='*' no_proxy='*' \
  nohup uv --directory "$PDFSYS" run pdfsys run \
    --pdf-list  "$LANE/$b" \
    --path-root "$LANE/pdfs" \
    --out-dir   "$LANE/p2/$b" \
    --stages    router,extract \
    --extract-backends pipeline \
    --parser-output-dir "$LANE/p2/mineru" \
    --markdown-dir "$LANE/markdown" \
    --ocr-threshold "$OCR_THRESHOLD" \
    --resume \
    > "$LANE/logs/$b.log" 2>&1 &
done

sleep 10
echo "  launched $(pgrep -fc -- "--pdf-list $LANE/gbucket-" || echo 0) workers"
echo
echo "  Watch mineru-api's queue and raise GPU_WORKERS if it stays at 0:"
echo "    curl -s --noproxy '*' $MINERU_URL/health | jq '{queued_tasks,processing_tasks}'"
echo
echo "  A warning starting 'documents were routed to mupdf here but queued for"
echo "  lane' means the two boxes disagree — check OCR_THRESHOLD matches."
