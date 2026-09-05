#!/bin/bash
# Step 8 — bring the two lanes' shards together. Run on the CPU box.
#
# Step 7 packages each lane into a directory of the same name on its own box.
# That is as far as it can get: the machines share no disk, so "both lanes
# write into one dataset directory" is a thing this step does, not step 7.
#
# It matters because of how the halves fail. `dataset-validate` on either box
# passes — each shard IS a valid dataset — while quietly describing a corpus
# missing the other lane's documents. Nothing says the other half exists.
set -e
cd "$(dirname "$0")" && source ./config.sh
_require_host "$CPU_HOST"

[ -d "$DATASET" ] || { echo "no dataset at $DATASET — run 07-package.sh cpu first" >&2; exit 1; }

remote_shards=$(ssh -o BatchMode=yes "$GPUBOX" "ls $DATASET/pages/*.parquet 2>/dev/null | wc -l" || echo 0)
if [ "${remote_shards:-0}" -eq 0 ]; then
  echo "no shard at $GPUBOX:$DATASET — run 07-package.sh gpu there first" >&2
  exit 1
fi

before=$(ls "$DATASET"/pages/*.parquet 2>/dev/null | wc -l)
rsync -a --info=progress2 "$GPUBOX:$DATASET/" "$DATASET/"
after=$(ls "$DATASET"/pages/*.parquet 2>/dev/null | wc -l)
echo "  shards: $before -> $after"

# Now validate the whole thing, which is the first time anything has.
uv --directory "$PDFSYS" run pdfsys dataset-validate --shard "$DATASET"

# And the invariant the split exists to protect: (doc_id, page_index) is the
# primary key, so a doc_id in two shards makes the merged dataset invalid.
# The lane filter should make this impossible; the point is to know, not hope.
uv --directory "$PDFSYS" run python - "$DATASET" <<'PY'
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

shards = defaultdict(set)
for f in sorted((Path(sys.argv[1]) / "pages").glob("*.parquet")):
    for doc_id in pq.read_table(f, columns=["doc_id"]).column("doc_id").to_pylist():
        shards[doc_id].add(f.stem)

names = {name for owners in shards.values() for name in owners}
both = {d: s for d, s in shards.items() if len(s) > 1}
print(f"  {len(shards)} distinct documents across {len(names)} shards")
if both:
    print(f"  ERROR {len(both)} document(s) in more than one shard:", file=sys.stderr)
    for doc_id, s in list(both.items())[:5]:
        print(f"    {doc_id[:12]}… in {', '.join(sorted(s))}", file=sys.stderr)
    sys.exit(1)
print("  no document is in two shards")
PY
