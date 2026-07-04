#!/usr/bin/env bash
# Self-contained server-side runner: rebuild mineru.gpu, replace container
# with correct mounts, run 2-PDF VLM smoke. Idempotent; safe to re-run.
#
# Detach with:   nohup bash scripts/server_deploy_chain.sh > deploy.log 2>&1 &
# Poll status:   tail -5 /root/pdfsys/deploy.log; ls /root/pdfsys/.deploy.state
#
# State markers (one file per phase, empty = pending, line = result):
#   .deploy.state/01_build.OK | .FAIL
#   .deploy.state/02_restart.OK | .FAIL
#   .deploy.state/03_smoke.OK | .FAIL
#   .deploy.state/DONE | .FAILED

set -uo pipefail
cd "${PDFSYS_ROOT:-/root/pdfsys}"
export PATH=$HOME/.local/bin:$PATH

mkdir -p .deploy.state
rm -f .deploy.state/* 2>/dev/null

stamp() {
  echo "[$(date "+%F %T")] $*"
}

# --- 1. build ---
stamp "=== phase 1: build mineru.gpu (devel base) ==="
if docker compose -f docker-compose.yml -f docker-compose.gpu.yml build \
     --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
     --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
     mineru 2>&1; then
  echo "ok $(date "+%F %T")" > .deploy.state/01_build.OK
  stamp "phase 1 OK"
else
  echo "fail $(date "+%F %T")" > .deploy.state/01_build.FAIL
  stamp "phase 1 FAIL — abort"
  echo "FAILED" > .deploy.state/DONE
  exit 1
fi

# --- 2. restart container ---
stamp "=== phase 2: replace mineru container ==="
docker rm -f pdfsys-mineru 2>&1 || true
if docker run -d --name pdfsys-mineru \
     --gpus all \
     --network pdfsys_default \
     -p 8000:8000 \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
     -e HF_HUB_OFFLINE=1 \
     -e TRANSFORMERS_OFFLINE=1 \
     -e HF_HOME=/root/.cache/huggingface \
     -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
     -v /root/mineru.json:/root/mineru.json:ro \
     pdfsys-mineru:gpu \
     --host 0.0.0.0 --port 8000 2>&1; then
  # Wait healthy
  for i in $(seq 1 60); do
    state=$(docker inspect pdfsys-mineru --format "{{.State.Health.Status}}" 2>/dev/null || echo missing)
    if [ "$state" = "healthy" ]; then
      stamp "mineru healthy after $((i*5))s"
      echo "healthy_after_$((i*5))s" > .deploy.state/02_restart.OK
      break
    fi
    sleep 5
  done
  if [ ! -f .deploy.state/02_restart.OK ]; then
    echo "fail timeout" > .deploy.state/02_restart.FAIL
    stamp "phase 2 FAIL — never healthy"
    echo "FAILED" > .deploy.state/DONE
    exit 1
  fi
else
  echo "fail run" > .deploy.state/02_restart.FAIL
  stamp "phase 2 FAIL — docker run errored"
  echo "FAILED" > .deploy.state/DONE
  exit 1
fi

# --- 3. smoke (2 PDFs through all 3 parsers) ---
stamp "=== phase 3: 2-PDF VLM smoke ==="
rm -rf out/test-smoke
if bash scripts/batch_process.sh /root/pdfsys/test-pdfs out/test-smoke 2>&1; then
  # Tally results
  errors=$(python3 -c "
import json
n = sum(1 for l in open(\"out/test-smoke/results.jsonl\") if json.loads(l).get(\"error\"))
print(n)")
  echo "errors=$errors" > .deploy.state/03_smoke.OK
  stamp "phase 3 done (errors=$errors)"
else
  echo "fail" > .deploy.state/03_smoke.FAIL
  stamp "phase 3 FAIL"
fi

# Final marker
echo "$(date "+%F %T")" > .deploy.state/DONE
stamp "=== chain complete ==="
