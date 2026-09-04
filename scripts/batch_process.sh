#!/usr/bin/env bash
# Batch-process a directory of PDFs against the running pdfsys stack.
#
# Talks to the local mineru + quality services (default
# http://localhost:8000 + :8765 — change via env if you're hitting a
# remote box). Calls scripts/extract_matrix.py to produce the parser ×
# PDF candidate matrix, then scripts/emit_quality_handoff_matrix.py to
# shape it for the downstream quality-score consumer.
#
# Usage:
#   bash scripts/batch_process.sh <pdf-dir> [out-dir] [extra args...]
#
# Examples:
#   # default: matrix all 3 parsers, write to out/batch-YYYYMMDD-HHMMSS/
#   bash scripts/batch_process.sh /data/pdfs
#
#   # custom output dir
#   bash scripts/batch_process.sh /data/pdfs out/runA
#
#   # smaller subset for smoke
#   bash scripts/batch_process.sh /data/pdfs out/smoke --limit 5
#
#   # skip VLM (no GPU available)
#   bash scripts/batch_process.sh /data/pdfs --skip-vlm
#
# Env overrides:
#   MINERU_URL=http://other-host:8000
#   QUALITY_URL=http://other-host:8765
#   VLM_ENGINE=vllm-engine            (default: from .deploy.env)

set -euo pipefail
cd "$(dirname "$0")/.."

PDF_DIR=${1:-}
if [ -z "$PDF_DIR" ]; then
  cat <<EOF
usage: bash scripts/batch_process.sh <pdf-dir> [out-dir] [args...]

  pdf-dir   directory of PDFs (recursive; *.pdf in any case, plus
            extensionless files whose header is %PDF-)
  out-dir   default: out/batch-YYYYMMDD-HHMMSS

passes any additional flags to scripts/extract_matrix.py
  --limit N            cap PDFs
  --skip-mupdf         skip mupdf lane
  --skip-pipeline      skip pipeline lane
  --skip-vlm           skip vlm lane
EOF
  exit 1
fi
shift

OUT_DIR=${1:-}
if [ -n "$OUT_DIR" ] && [[ "$OUT_DIR" != --* ]]; then
  shift
else
  OUT_DIR="out/batch-$(date +%Y%m%d-%H%M%S)"
fi

# Load .deploy.env if present (for VLM_ENGINE default).
if [ -f .deploy.env ]; then
  # shellcheck disable=SC1091
  source .deploy.env
fi

# Service URLs — env wins, .deploy.env or fallback otherwise.
MINERU_URL=${MINERU_URL:-http://localhost:8000}
QUALITY_URL=${QUALITY_URL:-http://localhost:8765}
VLM_ENGINE=${VLM_ENGINE:-transformers}

# Quick health probe so we fail fast if the stack is down.
echo "[batch] services:"
for kv in "mineru=$MINERU_URL/health" "quality=$QUALITY_URL/health"; do
  name=${kv%%=*}; url=${kv#*=}
  if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
    echo "  ✓ $name  $url"
  else
    echo "  ✗ $name  $url  (run scripts/deploy.sh)"
    exit 1
  fi
done

mkdir -p "$OUT_DIR/markdown"

# Both extract.py modules in pdfsys-parser-{pipeline,vlm} v0.2.0+ read
# MINERU_PIPELINE_URL / MINERU_VLM_URL. quality.py reads QUALITY_URL.
# Bypass macOS system proxy because we've seen it intercept these.
export MINERU_PIPELINE_URL="$MINERU_URL"
export MINERU_VLM_URL="$MINERU_URL"
export QUALITY_URL="$QUALITY_URL"
export NO_PROXY='*' no_proxy='*'

echo "[batch] config:"
echo "  pdf-dir:     $PDF_DIR"
echo "  out-dir:     $OUT_DIR"
echo "  vlm-engine:  $VLM_ENGINE"
echo "  extra args:  $*"
echo ""

# 1. parser × PDF matrix
uv run python scripts/extract_matrix.py \
  --pdf-dir "$PDF_DIR" \
  --out "$OUT_DIR/results.jsonl" \
  --markdown-dir "$OUT_DIR/markdown" \
  --vlm-engine "$VLM_ENGINE" \
  "$@"

# 2. reshape for downstream
uv run python scripts/emit_quality_handoff_matrix.py "$OUT_DIR/results.jsonl"

# 3. summary tarball
TAR_OUT="${OUT_DIR%.gz}.tar.gz"
tar czf "$TAR_OUT" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo ""
echo "[batch] ✓ done"
echo "  jsonl:      $OUT_DIR/results.jsonl"
echo "  markdown:   $OUT_DIR/markdown/"
echo "  handoff:    $OUT_DIR/quality_handoff_matrix.json"
echo "  tarball:    $TAR_OUT  ($(du -h "$TAR_OUT" | awk '{print $1}'))"
