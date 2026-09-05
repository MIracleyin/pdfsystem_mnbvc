#!/bin/bash
# Step 1 — list every PDF, then cut the list into buckets. Run on xsy-01.
#
# 7 seconds warm, ~5 minutes cold. Worth doing as its own step because the
# count is the first thing that can be wrong: a glob for "*.pdf" finds 199,992
# of these 217,997 documents and reports success.
set -e
cd "$(dirname "$0")" && source ./config.sh
_require_host "$CPU_HOST"

mkdir -p "$RUN"/{p1,logs,markdown}

uv --directory "$PDFSYS" run python - "$CORPUS" "$RUN/all_paths.txt" <<'PY'
import sys
from pathlib import Path
from pdfsys_core.discovery import take_inventory

corpus, out = sys.argv[1], Path(sys.argv[2])
inv = take_inventory(corpus)
out.write_text("\n".join(str(p) for p in inv.paths) + "\n")
print(f"  {len(inv)} PDFs: {len(inv.by_suffix)} by suffix, "
      f"{len(inv.by_magic)} by %PDF- header")
if inv.unreadable_dirs:
    print(f"  WARNING {len(inv.unreadable_dirs)} directories could not be read "
          f"— the corpus is not smaller, it is partly invisible")
PY

# split -n l/N is GNU-only (BSD/macOS split has no -n). The boxes are Linux.
# This divides by line count, which has nothing to do with doc_id — buckets
# are a unit of parallelism and resumption, not of sharding.
cd "$RUN"
rm -f bucket-*
split -n "l/$WORKERS" -d -a 2 all_paths.txt bucket-
echo "  $(ls bucket-* | wc -l) buckets x ~$(wc -l < bucket-00) lines"
