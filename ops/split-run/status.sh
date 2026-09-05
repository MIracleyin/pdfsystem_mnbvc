#!/bin/bash
# Progress and rate for the CPU lane. Safe to run at any time.
cd "$(dirname "$0")" && source ./config.sh

# wc prints nothing when the file is missing, so this one does need a literal
# fallback — unlike pgrep -c, which prints 0 and *also* exits nonzero.
TOTAL=$(wc -l < "$RUN/all_paths.txt" 2>/dev/null || echo 0)
n1=$(cat "$RUN"/p1/*/results.jsonl 2>/dev/null | wc -l)
sleep 30
n2=$(cat "$RUN"/p1/*/results.jsonl 2>/dev/null | wc -l)
rate=$(( (n2 - n1) / 30 ))

echo "  done      $n2 / $TOTAL"
echo "  rate      ${rate} docs/s"
[ "$rate" -gt 0 ] && echo "  eta       $(( (TOTAL - n2) / rate / 60 )) min"
echo "  markdown  $(ls "$RUN/markdown" 2>/dev/null | wc -l) files"
# The pattern matches the worker's own --pdf-list argument, which this script
# does not have — checking for "pdfsys" would match the checker.
echo "  workers   $(pgrep -fc -- "--pdf-list $RUN/bucket-" || true) alive"
echo "  load      $(cut -d' ' -f1-3 /proc/loadavg)"
echo "  disk      $(df -h "$RUN" | tail -1 | awk '{print $4}') free"

echo "  errors:"
cat "$RUN"/p1/*/results.jsonl 2>/dev/null \
  | jq -r '.error_class // "ok"' | sort | uniq -c | sort -rn | sed 's/^/    /'

# MuPDF writes syntax complaints to stderr for perfectly processable files;
# they are not errors and there are a lot of them.
real=$(cat "$RUN"/logs/*.log 2>/dev/null | grep -icE "traceback|\[pdfsys\] error" || true)
echo "  log errors (excluding MuPDF noise): ${real:-0}"
