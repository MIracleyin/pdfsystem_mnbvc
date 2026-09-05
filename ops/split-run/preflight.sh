#!/bin/bash
# Check a site config across BOTH boxes before committing hours to it.
# Run on the CPU box. Read-only: it starts nothing and writes nothing.
#
# Every check here exists because the thing it checks fails silently. A wrong
# --ocr-threshold, a stale checkout on one side, an unreachable scorer: none
# of them stop a run, they just make its output quietly wrong. This is the
# cheapest place to find them.
cd "$(dirname "$0")" && source ./config.sh

fail=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail+1)); }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }

echo "site: $SITE_FILE"
echo "      $CPU_HOST -> $GPU_HOST via $GPUBOX"
echo

# ── this box ─────────────────────────────────────────────────────────────
echo "CPU box ($(hostname)):"
case "$(hostname)" in
  $CPU_HOST) ok "hostname matches CPU_HOST" ;;
  *) bad "hostname is $(hostname), CPU_HOST says $CPU_HOST — run this there" ;;
esac

[ -d "$CORPUS" ] && ok "corpus $CORPUS" || bad "corpus $CORPUS does not exist"
[ -r "$CORPUS" ] || bad "corpus $CORPUS is not readable"
[ -d "$PDFSYS" ] && ok "checkout $PDFSYS" || bad "checkout $PDFSYS does not exist"

cpu_rev=$(git -C "$PDFSYS" rev-parse --short HEAD 2>/dev/null || echo none)
[ "$cpu_rev" = none ] && bad "$PDFSYS is not a git checkout" || ok "code $cpu_rev"

if uv --directory "$PDFSYS" run python -c "
from pdfsys_router.xgb_model import default_weights_path as p
import sys; sys.exit(0 if p().is_file() else 1)" 2>/dev/null; then
  ok "router weights present"
else
  bad "router weights missing — uv --directory $PDFSYS run python -m pdfsys_router.download_weights"
fi

free_cpu=$(df -BG --output=avail "$(dirname "$RUN")" 2>/dev/null | tail -1 | tr -dc 0-9)
ok "disk at $(dirname "$RUN"): ${free_cpu:-?} GB free"

# ── the other box ────────────────────────────────────────────────────────
echo
echo "GPU box ($GPUBOX):"
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$GPUBOX" true 2>/dev/null; then
  ok "ssh reachable without a password"

  remote_host=$(ssh -o BatchMode=yes "$GPUBOX" hostname 2>/dev/null)
  case "$remote_host" in
    $GPU_HOST) ok "hostname $remote_host matches GPU_HOST" ;;
    *) bad "$GPUBOX is $remote_host, GPU_HOST says $GPU_HOST" ;;
  esac

  gpu_rev=$(ssh -o BatchMode=yes "$GPUBOX" "git -C $PDFSYS rev-parse --short HEAD 2>/dev/null" || echo none)
  if [ "$gpu_rev" = none ]; then
    bad "no checkout at $PDFSYS on the GPU box — run 00-deploy.sh there"
  elif [ "$gpu_rev" != "$cpu_rev" ]; then
    # The lane semantics live in this code. Two versions is two pipelines.
    bad "code differs: $cpu_rev here, $gpu_rev there"
  else
    ok "code $gpu_rev matches"
  fi

  # The one value that must not disagree. Comparing the whole file's checksum
  # is stronger than comparing the value: it also catches a site config that
  # was edited on one box and not the other.
  remote_sum=$(ssh -o BatchMode=yes "$GPUBOX" "md5sum $SITE_FILE 2>/dev/null | cut -d' ' -f1" || true)
  if [ -z "$remote_sum" ]; then
    bad "site config $SITE_FILE not found on the GPU box — copy it there"
  elif [ "$remote_sum" != "$SITE_SUM" ]; then
    bad "site config differs between boxes (ocr-threshold could differ; a"
    bad "  document would then be handed off by one and skipped by the other)"
  else
    ok "site config identical on both boxes"
  fi

  free_gpu=$(ssh -o BatchMode=yes "$GPUBOX" "df -BG --output=avail $(dirname "$LANE") 2>/dev/null | tail -1 | tr -dc 0-9" || true)
  ok "disk at $(dirname "$LANE"): ${free_gpu:-?} GB free"
else
  bad "cannot ssh to $GPUBOX without a password"
fi

# ── services ─────────────────────────────────────────────────────────────
echo
echo "services:"
m=$(curl -s -m 8 --noproxy '*' "$MINERU_URL/health" 2>/dev/null || true)
case "$m" in
  *healthy*|*ok*) ok "mineru-api at $MINERU_URL" ;;
  "") bad "mineru-api at $MINERU_URL did not answer" ;;
  *) warn "mineru-api answered but not with a health status: ${m:0:60}" ;;
esac

q=$(curl -s -m 8 --noproxy '*' "$QUALITY_URL_/health" 2>/dev/null || true)
if [ -z "$q" ]; then
  bad "quality server at $QUALITY_URL_ did not answer"
else
  serving=$(printf '%s' "$q" | sed -n 's/.*"model"[: ]*"\([^"]*\)".*/\1/p')
  if [ "$serving" = "$QUALITY_MODEL" ]; then
    ok "quality server serving $serving"
  else
    # Two lanes scored by two models put two scales in one column, and
    # nothing in the data would say so.
    bad "quality server is serving '$serving', config expects '$QUALITY_MODEL'"
  fi
fi

# ── capacity ─────────────────────────────────────────────────────────────
# Only advisory: the real share is not known until the corpus is routed.
# The default assumes scanned PDFs dominate the bytes, which is what was
# measured on cmn_Hani (36% of documents, 72% of bytes).
echo
echo "capacity (rough):"
corpus_gb=$(du -sBG "$CORPUS" 2>/dev/null | tr -dc 0-9 || echo 0)
if [ "${corpus_gb:-0}" -gt 0 ]; then
  need=$(( corpus_gb * 72 / 100 ))
  echo "  corpus ${corpus_gb} GB; if it looks like cmn_Hani the OCR lane is"
  echo "  ~${need} GB to transfer, plus ~$(( corpus_gb / 3 )) GB of sidecars"
  if [ -n "${free_gpu:-}" ] && [ "$free_gpu" -lt "$need" ]; then
    bad "GPU box has ${free_gpu} GB free, needs roughly ${need} GB"
  fi
else
  warn "could not size $CORPUS (du was slow or denied); check disk yourself"
fi

echo
if [ "$fail" -gt 0 ]; then
  echo "$fail problem(s). Fix them before starting a run." >&2
  exit 1
fi
echo "ready."
