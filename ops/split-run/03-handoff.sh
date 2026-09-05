#!/bin/bash
# Step 3 — derive the two lanes' worklists. Run on xsy-01, after step 2 ends.
set -e
cd "$(dirname "$0")" && source ./config.sh
_require_host "$CPU_HOST"

# The guard matches the workers' --pdf-list argument. Matching "pdfsys" would
# match this script's own ssh command line — that mistake has cost two
# debugging sessions here already.
running=$(pgrep -fc -- "--pdf-list $RUN/bucket-" || true)
if [ "${running:-0}" -gt 0 ]; then
  echo "refusing: $running workers still running, the lists would be partial" >&2
  exit 1
fi

cd "$RUN"

# Each worker owned its own --out-dir, so this is concatenation, not interleave.
cat p1/bucket-*/results.jsonl > results.jsonl
rows=$(wc -l < results.jsonl)
echo "  merged $rows rows"

# No worker is running, but that does not mean they finished — they may have
# been killed, or the box rebooted. The worklists derived below look identical
# either way, and a partial gpu_lane.txt means step 4 ships a subset and step
# 5 processes it: no error anywhere, just a corpus quietly missing a slice.
want=$(wc -l < all_paths.txt)
if [ "$rows" -lt "$want" ]; then
  echo "refusing: $rows of $want documents processed — the CPU lane did not" >&2
  echo "  finish. Re-run 02-cpu-lane.sh (it resumes), or set ALLOW_PARTIAL=1" >&2
  echo "  if you deliberately want to hand off a subset." >&2
  [ "${ALLOW_PARTIAL:-0}" = "1" ] || exit 1
  echo "  ALLOW_PARTIAL=1 — continuing with a partial handoff" >&2
fi

# The GPU worklist. Relative to the corpus root: that is what lets the same
# file be read on a box that mounted the corpus somewhere else (--path-root).
jq -r 'select(.skip_reason == "lane-filter") | .pdf_path' results.jsonl \
  | sed "s|^$CORPUS/||" > gpu_lane.txt

# What this box actually extracted, for step 7's packaging. Absolute: it stays.
jq -r 'select(.extract_backend=="mupdf" and .skip_reason==null and .error_class==null)
       | .pdf_path' results.jsonl > cpu_lane.txt

echo "  gpu_lane  $(wc -l < gpu_lane.txt) documents"
echo "  cpu_lane  $(wc -l < cpu_lane.txt) documents"
echo "  errored   $(jq -r 'select(.error_class != null) | .pdf_path' results.jsonl | wc -l)"

# Sanity: every row is in exactly one of the three.
tot=$(wc -l < results.jsonl)
acc=$(( $(wc -l < gpu_lane.txt) + $(wc -l < cpu_lane.txt) \
        + $(jq -r 'select(.error_class != null) | .pdf_path' results.jsonl | wc -l) ))
[ "$tot" -eq "$acc" ] || echo "  WARNING $tot rows but $acc accounted for" >&2
