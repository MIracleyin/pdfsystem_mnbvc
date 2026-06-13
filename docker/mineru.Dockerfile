# pdfsys-mineru service — mineru-api HTTP server (pipeline backend, CPU).
#
# mineru-api ships its own HTTP surface, so this image is just a thin
# wrapper: install mineru[pipeline], expose port 8000, mount HF cache.
#
# Endpoints (provided by mineru-api itself):
#   GET  /health        → liveness
#   POST /file_parse    → parse PDF (backend=pipeline)
#   GET  /docs          → OpenAPI / Swagger UI
#
# The matching client lives in
# external/parsers/packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/
# extract.py — set MINERU_PIPELINE_URL there once the CLI image ships.
#
# Image: ~2.5 GB (CPU torch + opencv-headless + onnxruntime + mineru).
# Weights NOT baked — mount /cache/huggingface at runtime.
#
# VLM backend NOT included on CPU. For VLM use the docker/mineru.gpu.Dockerfile
# (vllm-engine, NVIDIA CUDA) or run mineru-api directly on Apple Silicon
# with the MLX backend (no Docker — MLX doesn't virtualize).
#
# Build:
#   docker build -f docker/mineru.Dockerfile -t pdfsys-mineru:cpu .
# Run:
#   docker run --rm -p 8000:8000 \
#     -v ~/.cache/huggingface:/cache/huggingface:ro \
#     pdfsys-mineru:cpu

FROM python:3.12-slim AS base

# OS deps:
#   libgomp1     — OpenMP runtime, needed by onnxruntime + opencv
#   libglib2.0-0 — opencv-python-headless transitive
#   curl         — healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      libgomp1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch wheel + mineru pipeline backend.
# Pin to the same major as the parsers' pyproject (mineru>=3.1,<4.0).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.5,<3.0" \
      "mineru[pipeline]>=3.1,<4.0"

WORKDIR /app

ENV HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Long start period — pipeline backend lazy-loads weights on first request,
# but the server itself starts in ~5 s. 60 s start_period covers both
# fast (server up) and slow (first-request lazy load) cases without
# false alarms.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["mineru-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
