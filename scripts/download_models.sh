#!/usr/bin/env bash
# Download all model weights pdfsys needs for a full bench/run.
#
# Covers:
#   - mineru pipeline backend (PDF-Extract-Kit-1.0 + layoutreader + MFR/MFD)
#   - mineru VLM-mlx backend (MinerU2.5-2509-1.2B)
#   - pdfsys quality scorer (miracleyin/mnbvc-pdf-quality-scorer-modernbert,
#     plus the legacy FinePDFs classifier for comparison runs)
#   - pdfsys router XGBoost classifier (~250 KB, in-repo download)
#
# Picks the fastest HF endpoint by running a quick 30 MB benchmark against
# hf-mirror.com (a Hugging Face full-mirror, faster from many Asian POPs)
# vs huggingface.co. Bypasses any local HTTP/HTTPS proxy because we have
# seen Clash/V2ray on macOS stall HF connections silently (TCP CLOSED
# with no error to the python downloader).
#
# Usage:
#   bash scripts/download_models.sh
#
# Idempotent: skips files already in ~/.cache/huggingface.
# Total volume: ~5-10 GB. Time at 1.7 MB/s: 50-100 min.

set -euo pipefail

cd "$(dirname "$0")/.."

# Bypass any system proxy for HF — the proxy is often the bottleneck or hangs.
# `unset *_PROXY` only clears shell env; Python's requests/httpx on macOS will
# additionally read macOS System Preferences proxies via
# urllib.request.getproxies_macosx_sysconf(). NO_PROXY=* forces a hard bypass
# in both code paths.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

# Force HF online for this script (parsers force offline at runtime).
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

bench_url() {
  # Echo MB/s for a 30 MB range request to the given HF endpoint.
  # 0 on failure.
  local endpoint="$1"
  local path="opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
  local speed
  speed=$(curl -L --noproxy '*' -o /dev/null --max-time 25 \
    -r 0-30000000 \
    "${endpoint}/${path}" \
    -w '%{speed_download}' 2>/dev/null || echo 0)
  # bytes/sec -> MB/s with one decimal
  awk -v s="$speed" 'BEGIN { printf "%.2f", s/1048576 }'
}

echo "[download_models] benchmarking HF endpoints (30 MB sample, no proxy)..."
SPEED_HF=$(bench_url "https://huggingface.co")
SPEED_MIRROR=$(bench_url "https://hf-mirror.com")
echo "[download_models]   huggingface.co: ${SPEED_HF} MB/s"
echo "[download_models]   hf-mirror.com:  ${SPEED_MIRROR} MB/s"

# Pick the faster endpoint, but prefer huggingface.co unless the mirror is
# materially faster (>20%). hf-mirror.com has incomplete metadata for some
# files (e.g. opendatalab/PDF-Extract-Kit-1.0's PP-FormulaNet_plus-M.pth
# returns FileMetadataError), so we lean toward the canonical source.
if awk -v a="$SPEED_MIRROR" -v b="$SPEED_HF" 'BEGIN { exit !(a + 0 > 1.2 * (b + 0)) }'; then
  export HF_ENDPOINT="https://hf-mirror.com"
else
  export HF_ENDPOINT="https://huggingface.co"
fi
echo "[download_models] picked HF_ENDPOINT=$HF_ENDPOINT"

# Step 1 — mineru pipeline + VLM weights (the big ones).
echo
echo "[download_models] === step 1/3: mineru pipeline + vlm-mlx weights ==="
echo "[download_models] this is the long one (~5-10 GB). hit Ctrl-C to skip."
uv run mineru-models-download -s huggingface -m all

# Step 2 — quality scorer (ModernBERT, ~600 MB).
echo
echo "[download_models] === step 2/3: ModernBERT OCR quality scorer ==="
uv run python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Final scoring model first; FinePDFs legacy model kept for comparison
# runs (bench A/B, old bundles).
for m in (
    "miracleyin/mnbvc-pdf-quality-scorer-modernbert",
    "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn",
):
    print(f"  prefetching tokenizer + model: {m}")
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m)
print("  done")
PY

# Step 3 — router XGBoost weights (small, ~250 KB, in-repo download script).
echo
echo "[download_models] === step 3/3: pdfsys router XGBoost weights ==="
uv run python -m pdfsys_router.download_weights

# Quick sanity check on cache sizes.
echo
echo "[download_models] === cache summary ==="
du -sh ~/.cache/huggingface/hub/models--opendatalab--* 2>/dev/null || true
du -sh ~/.cache/huggingface/hub/models--HuggingFaceFW--* 2>/dev/null || true
ls -lh packages/pdfsys-router/models/xgb_classifier.ubj 2>/dev/null || true

echo
echo "[download_models] ready."
echo "[download_models] next: uv run python -m pdfsys_bench --pdf-dir <DIR> --out out/bench.jsonl --cascade --vlm --vlm-engine mlx-engine"
