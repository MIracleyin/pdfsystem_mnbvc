# GPU server quick deploy

End-to-end flow for bringing pdfsys up on a fresh NVIDIA Linux box and
batching a directory of PDFs through it. The whole thing fits in three
scripts:

```
scripts/detect_gpu.sh     → emits .deploy.env (DEPLOY_MODE, VLM_ENGINE, ...)
scripts/deploy.sh         → build images, pull weights, compose up, smoke test
scripts/batch_process.sh  → run parser × PDF matrix against the live stack
```

The deploy script auto-selects between CPU and GPU images based on
detected hardware; you can force either with `DEPLOY_MODE=gpu|cpu`.

## Prerequisites

Fresh Ubuntu 22.04 or 20.04, NVIDIA T4 / L4 / A10 / 4090 / etc.,
~40 GB free disk:

```bash
# 1. NVIDIA driver (skip if already installed)
sudo apt-get install -y nvidia-driver-550
sudo reboot      # then verify: nvidia-smi

# 2. Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER" && newgrp docker

# 3. NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker info | grep -i nvidia       # expect "nvidia" in Runtimes
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Deploy

```bash
git clone --recurse-submodules https://github.com/<owner>/pdfsystem_mnbvc.git
cd pdfsystem_mnbvc

# Step 1: detect hardware → .deploy.env
bash scripts/detect_gpu.sh

# Step 2: download HF weights to host cache (~6 GB)
bash scripts/download_models.sh

# Step 3: build + start (mineru is GPU, quality stays CPU)
bash scripts/deploy.sh
```

`deploy.sh` runs ~10-30 min depending on bandwidth (vllm wheel is fat).
On success it prints the URLs and next-step commands.

### Final quality-scoring model

The quality service serves the project's final scoring model:

```
https://huggingface.co/miracleyin/mnbvc-pdf-quality-scorer-modernbert
```

A ModernBERT-base fine-tune with 4 ordinal quality classes (0..3),
8192-token context, scored as the softmax expectation over class
indices → continuous [0, 3] (so `parquet.quality_threshold: 2.0`
semantics are unchanged). It is the code default everywhere
(`--max-tokens 8192 --max-chars 40000`); `download_models.sh` step 2
pre-fetches it into the HF cache the container mounts, picking
hf-mirror.com automatically when huggingface.co is unreachable from
the host. Verify what the service is actually serving:

```bash
curl -s http://localhost:8765/health
# -> {"ok": true, "model": "miracleyin/mnbvc-pdf-quality-scorer-modernbert"}
```

The legacy FinePDFs regression model
`HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn` (512-token
budget, `max_chars 10000`) is still downloaded for comparison runs and
can be selected via `quality.model` in yaml.

### CN networking: pip mirror + docker.io throughput

If the host lives behind a network where `docker.io` and PyPI throughput
are limited (typical for CN clouds), the vanilla deploy stalls in the
build step. Two mitigations:

**1. Pre-configure a docker registry mirror in `/etc/docker/daemon.json`:**

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.m.daocloud.io"
  ]
}
```
Restart docker (`systemctl restart docker`). Manually `docker pull
python:3.12-slim` and `docker pull nvidia/cuda:12.4.0-devel-ubuntu22.04`
once to warm the layer cache — buildkit does NOT use daemon-level
mirrors, but docker-cli does, and the images are then cached locally
before the compose build starts.

**2. Pass a PyPI mirror as a build arg:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  mineru
```

Both build args are honored by `docker/mineru.gpu.Dockerfile`. On
mnbvcgpu3 this brought the `pip install torch+mineru+vllm` phase from
timeout to ~90 min.

### mineru.json bind-mount (fixes VLM path resolution)

mineru 3.4 defaults `model-source: modelscope` and rewrites
`~/mineru.json` on every container startup, pointing at
`/root/.cache/modelscope/...` which does not exist inside the
container. vllm then crashes trying to load MinerU2.5 from the wrong
path.

Fix: bind-mount the host's correctly-configured `/root/mineru.json`
(populated by `scripts/download_models.sh` with HF paths) into the
container:

```bash
docker rm -f pdfsys-mineru
docker run -d --name pdfsys-mineru \
  --gpus all --network pdfsys_default -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /root/mineru.json:/root/mineru.json:ro \
  pdfsys-mineru:gpu --host 0.0.0.0 --port 8000
```

Both mounts use the exact host paths (`/root/.cache/huggingface` and
`/root/mineru.json`) so mineru.json's model paths (which reference the
host layout) resolve cleanly inside the container. This replaces the
default `docker compose up mineru` for the GPU-with-vllm case until
the underlying mineru default is fixed upstream.

### Detached chain runner (`server_deploy_chain.sh`)

For unstable SSH links, run everything as a self-tracking chain on the
server side:

```bash
nohup bash scripts/server_deploy_chain.sh > deploy.log 2>&1 &
disown
```

Progress is written to `.deploy.state/` as `01_build.OK` /
`02_restart.OK` / `03_smoke.OK` / `DONE`. Poll from your laptop:

```bash
ssh <host> 'ls /root/pdfsys/.deploy.state/ ; tail -8 /root/pdfsys/deploy.log'
```

The chain rebuilds `mineru.gpu` (with the CN pip mirror), replaces the
container with the correct mounts, and runs a 2-PDF VLM smoke test.
Idempotent — re-run to iterate.

## Batch processing

After deploy, point batch at any PDF directory:

```bash
bash scripts/batch_process.sh /data/pdfs                 # default vlm-engine from .deploy.env
bash scripts/batch_process.sh /data/pdfs out/runA        # custom output dir
bash scripts/batch_process.sh /data/pdfs --limit 10      # smoke a subset
bash scripts/batch_process.sh /data/pdfs --skip-vlm      # mupdf + pipeline only
```

Each run produces:

```
<out-dir>/
  results.jsonl                   one row per (PDF, parser) tuple
  markdown/<sha>__<parser>.md     per successful extraction
  quality_handoff_matrix.json     downstream-ready, inlined markdown
<out-dir>.tar.gz                  packaged for handoff
```

## Sizing notes

| GPU memory | What works |
|---|---|
| 8 GB   | pipeline ok; vlm OOM at common batch sizes — skip vlm |
| 12 GB  | pipeline + vlm-engine, single-stream |
| 16 GB+ | pipeline + vlm-engine, multi-stream / batched |
| 24 GB+ | room for larger vllm batch + higher concurrency |

vllm + MinerU2.5-1.2B ≈ 6 GB resident. mineru pipeline backend lazily
loads layout + OCR sub-models (~2 GB more on first request). Both
caches stay resident inside the container until restart.

## Operations

### Tear down

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

### Logs

```bash
docker compose ... logs -f mineru     # vllm prints CUDA / batch info
docker compose ... logs -f quality    # ModernBERT request log
```

### Force rebuild after parsers submodule bump

```bash
git submodule update --remote external/parsers
docker compose ... build --no-cache mineru   # if mineru.gpu.Dockerfile pinned to a version
DOWN_FIRST=1 bash scripts/deploy.sh
```

### Skip detection (CI / known good config)

```bash
DEPLOY_MODE=gpu VLM_ENGINE=vllm-engine bash scripts/deploy.sh
```

## Troubleshooting

- **mineru container exits with `RuntimeError: CUDA out of memory`** —
  reduce concurrency or lower vllm batch size. The image's
  `ENTRYPOINT mineru-api` accepts `--vllm-gpu-memory-utilization 0.5`
  (override the CMD).
- **`/health` slow on first request** — vllm JIT-compiles CUDA kernels
  for your specific compute capability; first cold start can be 90 s.
  Subsequent restarts hit a kernel cache.
- **`nvidia-container-cli: requirement error`** — driver/toolkit
  mismatch; reinstall both pinning to the same major version.
- **`docker compose` sees no `nvidia` runtime** — restart docker
  daemon after `nvidia-ctk runtime configure`: it edits
  `/etc/docker/daemon.json` but doesn't reload.
- **HF download fails / stalls** — same fix as the
  `scripts/download_models.sh` work: `NO_PROXY=*` to bypass the host's
  HTTP proxy.
