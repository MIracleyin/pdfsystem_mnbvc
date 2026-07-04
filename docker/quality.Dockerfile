# pdfsys-quality service — ModernBERT OCR-quality classifier behind HTTP.
#
# The source server is packages/pdfsys-bench/src/pdfsys_bench/_quality_server.py.
# It only imports stdlib at module top; torch + transformers are loaded
# inside _init(). So the image needs nothing from the pdfsys workspace —
# just the script file and torch + transformers.
#
# Image: ~1.2 GB (CPU torch wheel + transformers + safetensors).
# Weights NOT baked — mount /cache/huggingface at runtime.
#
# Build:
#   docker build -f docker/quality.Dockerfile -t pdfsys-quality:dev .
#
# Run:
#   docker run --rm -p 8765:8765 \
#     -v ~/.cache/huggingface:/cache/huggingface \
#     pdfsys-quality:dev
#
# Health: curl http://localhost:8765/health
# Score:  curl -X POST http://localhost:8765/score \
#           -H 'content-type: application/json' \
#           -d '{"text":"some markdown extracted from a pdf"}'

FROM python:3.12-slim AS base

# OS deps: curl for healthcheck, ca-certificates for HF Hub TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch (CPU wheel) + transformers + safetensors.
# Versions match what packages/pdfsys-bench/pyproject.toml currently asks for
# (torch>=2.1, transformers>=4.44). The CPU-only wheel index keeps the image
# at ~1.2 GB instead of ~3 GB with the CUDA bundle that we'd never use here.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.5,<3.0" \
      "transformers>=4.46,<5.0" \
      "safetensors>=0.4"

# Copy just the server script. No pdfsys workspace needed — the server is
# self-contained (stdlib imports only at module top; torch/transformers
# loaded inside _init()).
COPY packages/pdfsys-bench/src/pdfsys_bench/_quality_server.py /app/quality_server.py

WORKDIR /app

# HF cache lives on a volume in production; bind-mount on dev. Both Hub
# clients (transformers, huggingface_hub) honor HF_HOME.
ENV HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

# Long start period — first invocation has to load the ModernBERT weights
# (~1.5 GB on disk) into the torch process. CPU bf16 load takes 20-40 s on
# typical hardware; allow up to 120 s before failing the first probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8765/health || exit 1

ENTRYPOINT ["python", "/app/quality_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8765"]
