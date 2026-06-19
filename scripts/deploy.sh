#!/usr/bin/env bash
# One-shot deploy: detect → ensure weights → build → up → smoke.
#
# Run on the target host (Linux NVIDIA box or macOS dev machine).
# Idempotent: re-runs after a git pull or compose-file edit just rebuild
# the changed layers.
#
# Env overrides:
#   FORCE_DETECT=1          re-run detect_gpu.sh even if .deploy.env exists
#   SKIP_WEIGHTS=1          assume ~/.cache/huggingface is already populated
#   SKIP_BUILD=1            assume images are present (deploy iter loop)
#   DOWN_FIRST=1            tear down running stack before bringing it up
#
# Steps:
#   1. detect_gpu.sh → .deploy.env
#   2. download_models.sh if HF cache empty
#   3. compose build (with the right override file)
#   4. compose up -d mineru quality
#   5. wait healthy
#   6. health probes (curl /health)

set -euo pipefail
cd "$(dirname "$0")/.."

FORCE_DETECT=${FORCE_DETECT:-0}
SKIP_WEIGHTS=${SKIP_WEIGHTS:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
DOWN_FIRST=${DOWN_FIRST:-0}

# 1. detect
if [ ! -f .deploy.env ] || [ "$FORCE_DETECT" = "1" ]; then
  bash scripts/detect_gpu.sh
fi
# shellcheck disable=SC1091
source .deploy.env
echo "[deploy] mode=$DEPLOY_MODE  vlm_engine=$VLM_ENGINE  gpu=${GPU_COUNT}×${GPU_NAME:-none}"

# 2. compose file set
COMPOSE_FILES=(-f docker-compose.yml)
if [ "$DEPLOY_MODE" = "gpu" ]; then
  COMPOSE_FILES+=(-f docker-compose.gpu.yml)
  if [ "$NVIDIA_RUNTIME_OK" != "1" ]; then
    echo "[deploy] ✗ GPU mode requested but docker lacks the nvidia runtime."
    echo "[deploy]   install NVIDIA Container Toolkit and retry, or run:"
    echo "[deploy]     DEPLOY_MODE=cpu bash scripts/deploy.sh"
    exit 1
  fi
fi

# 3. weights
HF_PIPELINE_DIR="$HOME/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0"
if [ "$SKIP_WEIGHTS" != "1" ] && [ ! -d "$HF_PIPELINE_DIR" ]; then
  echo "[deploy] weights not cached → running scripts/download_models.sh"
  echo "[deploy]   (this fetches ~6 GB from HuggingFace; slow on flaky network)"
  bash scripts/download_models.sh
else
  echo "[deploy] HF cache present ($(du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | awk '{print $1}'))"
fi

# 4. teardown if asked
if [ "$DOWN_FIRST" = "1" ]; then
  echo "[deploy] tearing down existing stack..."
  docker compose "${COMPOSE_FILES[@]}" down --remove-orphans || true
fi

# 5. build
if [ "$SKIP_BUILD" != "1" ]; then
  echo "[deploy] building images..."
  docker compose "${COMPOSE_FILES[@]}" build mineru quality
fi

# 6. up
echo "[deploy] starting mineru + quality ..."
docker compose "${COMPOSE_FILES[@]}" up -d mineru quality

# 7. wait healthy
wait_healthy() {
  local svc="$1" container="pdfsys-$1" max_secs="$2"
  printf "[deploy] waiting %s healthy " "$svc"
  local elapsed=0
  while [ "$elapsed" -lt "$max_secs" ]; do
    local state
    state=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "missing")
    case "$state" in
      healthy)   echo " ✓ (${elapsed}s)"; return 0 ;;
      unhealthy) echo " ✗ unhealthy after ${elapsed}s"; docker logs "$container" --tail 30; return 1 ;;
      missing)   echo " ✗ container missing"; return 1 ;;
    esac
    printf "."
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo " ✗ timed out after ${max_secs}s"
  docker logs "$container" --tail 30
  return 1
}

wait_healthy mineru  300 || exit 1
wait_healthy quality 180 || exit 1

# 8. smoke
echo "[deploy] smoke test..."
mineru_health=$(curl -fsS http://localhost:8000/health || echo "FAIL")
quality_health=$(curl -fsS http://localhost:8765/health || echo "FAIL")
echo "  mineru:  $mineru_health"
echo "  quality: $quality_health"

cat <<EOF

[deploy] ✓ READY.

services:
  mineru   http://localhost:8000   (POST /file_parse, GET /docs)
  quality  http://localhost:8765   (POST /score)

batch a directory of PDFs:
  bash scripts/batch_process.sh <pdf-dir>          # default --vlm-engine=$VLM_ENGINE
  bash scripts/batch_process.sh <pdf-dir> out/dir  # custom output

tail logs:
  docker compose ${COMPOSE_FILES[*]} logs -f mineru
  docker compose ${COMPOSE_FILES[*]} logs -f quality

teardown:
  docker compose ${COMPOSE_FILES[*]} down
EOF
