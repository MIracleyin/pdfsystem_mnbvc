# pdfsys-mineru — GPU variant (NVIDIA CUDA + mineru pipeline + vlm-vllm).
#
# Drop-in replacement for docker/mineru.Dockerfile when an NVIDIA GPU is
# available. Both expose the same `mineru-api` HTTP surface on :8000,
# so clients (pdfsys-cli, scripts/extract_matrix.py) need no changes —
# just change `--vlm-engine` from `mlx-engine` (Apple Silicon) to
# `vllm-engine` (NVIDIA).
#
# Image: ~14-18 GB
#   nvidia/cuda runtime  ~3.0 GB
#   torch + cuda libs    ~5.5 GB
#   vllm                 ~3.5 GB
#   mineru + transitives ~2.0 GB
#
# Weights NOT baked — mount /cache/huggingface at runtime, same as CPU image.
#
# Host prereqs:
#   - NVIDIA driver
#   - nvidia-container-toolkit installed and registered with docker
#     (verify: `docker info | grep -i nvidia`)
#
# Build:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml build mineru
#
# Run:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d mineru
#   curl http://localhost:8000/health

# devel (not runtime) — vllm + flashinfer JIT-compile CUDA kernels at first
# inference and need nvcc + CUDA headers. The runtime base only ships
# libcudart; flashinfer crashes with `nvcc: not found` and vllm engine
# initialization fails. The size delta is ~1.5 GB.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Build-time mirror knobs (default = upstream). Override on CN hosts:
#   docker compose build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# Python 3.12 from deadsnakes PPA — ships only 3.10 by default on 22.04
# and we want consistency with the CPU image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      python3-pip \
      libgomp1 libglib2.0-0 libgl1 \
      gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip install for python3.12 (system pip is for older python; bootstrap fresh)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# torch CUDA 12.4 wheel + mineru[pipeline,vlm] + vllm.
# vllm is a separate top-level dep so the vlm-vllm-engine path resolves
# (mineru's `vlm-engine` flag dispatches by import availability).
RUN python -m pip install --no-cache-dir --upgrade pip \
      --index-url "$PIP_INDEX_URL" \
 && python -m pip install --no-cache-dir \
      --index-url "$PIP_INDEX_URL" \
      --extra-index-url "$PIP_EXTRA_INDEX_URL" \
      "torch>=2.5,<3.0" \
 && python -m pip install --no-cache-dir \
      --index-url "$PIP_INDEX_URL" \
      --extra-index-url "$PIP_EXTRA_INDEX_URL" \
      "mineru[pipeline,vlm]>=3.1,<4.0" \
      "vllm>=0.6,<1.0"

WORKDIR /app

ENV HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

# Long start period — vllm cold-start loads the MinerU2.5 weights into
# GPU memory (~2 GB) and JIT-compiles CUDA kernels for the host's
# compute capability. 30-120 s on first boot of a given driver version.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["mineru-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
