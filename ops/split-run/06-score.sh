#!/bin/bash
# Step 6 — quality-score both lanes against ONE model. Run per box.
#
#   on xsy-01:  ./06-score.sh cpu
#   on xsy-02:  ./06-score.sh gpu
#
# Only text crosses the network, clipped to the 40000 characters the server
# truncates at anyway — so the markdown does not have to be moved. Note the
# client does not pass ensure_ascii=False, so CJK is escaped to \uXXXX at six
# bytes a character: budget up to ~240 KB per document, not the ~40 KB the
# character limit suggests.
set -e
cd "$(dirname "$0")" && source ./config.sh

case "${1:-}" in
  cpu) _require_host "$CPU_HOST"
       RESULTS="$RUN/results.jsonl";  MD="$RUN/markdown";  OUT="$RUN/results.scored.jsonl" ;;
  gpu) _require_host "$GPU_HOST"
       RESULTS="$LANE/results.jsonl"; MD="$LANE/markdown"; OUT="$LANE/results.scored.jsonl"
       # The GPU lane ran as several workers, each with its own out-dir.
       [ -f "$RESULTS" ] || cat "$LANE"/p2/gbucket-*/results.jsonl > "$RESULTS" ;;
  *) echo "usage: $0 cpu|gpu" >&2; exit 1 ;;
esac

# --model is checked against GET /health before any work — but ONLY when
# QUALITY_URL is set. Without it pdfsys starts a local scoring subprocess and
# the flag is not validated at all. Two lanes scored by two different models
# put two scales in one column and nothing in the data would say so.
QUALITY_URL="$QUALITY_URL_" NO_PROXY='*' no_proxy='*' \
uv --directory "$PDFSYS" run pdfsys score \
  --results "$RESULTS" \
  --markdown-dir "$MD" \
  --out "$OUT" \
  --model "$QUALITY_MODEL" \
  --workers 4 \
  --resume

# --workers 4 is a starting point, not a ceiling: the server is a threaded
# HTTP server with no lock around the forward pass, so requests really do run
# in parallel. They share one model on one GPU, so the return flattens
# somewhere — measure where rather than assuming it is 4.
