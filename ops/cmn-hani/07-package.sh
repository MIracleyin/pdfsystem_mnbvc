#!/bin/bash
# Step 7 — package each lane into ONE pdfsys.page/v2 dataset. Run per box.
#
#   on xsy-01:  ./07-package.sh cpu /hdd_common/dataset/v2
#   on xsy-02:  ./07-package.sh gpu /hdd_common/dataset/v2
#
# Both lanes write into the same --to directory under different --shard names.
# Sortedness is a per-file promise, so the two lanes' doc_id ranges may
# interleave; what must not happen is one doc_id in two shards, because
# (doc_id, page_index) is the primary key.
set -e
cd "$(dirname "$0")" && source ./config.sh

LANE_KIND="${1:-}"
TO="${2:-}"
[ -n "$TO" ] || { echo "usage: $0 cpu|gpu <dataset-dir>" >&2; exit 1; }

# --images is passed explicitly on purpose. The defaults are `pages` for
# --from-pdf-list and `crops` for --from-mineru, and `pages` means rasterising
# every page at 200 dpi (~311 KiB/page) — about 1 TB across this corpus, which
# does not fit. `none` leaves mnbvc-export's 图片 column null; that format's
# image IS the whole-page raster, so fill it only if you have the disk.
IMAGES=${IMAGES:-none}

case "$LANE_KIND" in
  cpu)
    # --from-pdf-list, not --from-pdf-dir: the corpus root also holds the GPU
    # lane's documents, and mupdf would extract those scans into pages of
    # nothing carrying doc_ids the GPU shard already owns.
    uv --directory "$PDFSYS" run pdfsys dataset \
      --from-pdf-list "$RUN/cpu_lane.txt" \
      --images "$IMAGES" \
      --to "$TO" --shard cpu-00 \
      --meta "$RUN/results.scored.jsonl" ;;
  gpu)
    uv --directory "$PDFSYS" run pdfsys dataset \
      --from-mineru "$LANE/p2/mineru" \
      --images "$IMAGES" \
      --to "$TO" --shard gpu-00 \
      --meta "$LANE/results.scored.jsonl" ;;
  *) echo "usage: $0 cpu|gpu <dataset-dir>" >&2; exit 1 ;;
esac

uv --directory "$PDFSYS" run pdfsys dataset-validate --shard "$TO"
