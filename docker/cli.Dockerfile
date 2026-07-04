# pdfsys-cli — orchestrator + bench client.
#
# Talks to pdfsys-mineru and pdfsys-quality via HTTP (env vars). The
# parsers + bench env-override paths added in pdfsys-parsers v0.2.0
# (MINERU_PIPELINE_URL, MINERU_VLM_URL) and pdfsys-bench (QUALITY_URL)
# skip the in-container subprocess lifecycle when those vars are set,
# so this image never spawns mineru-api or _quality_server itself.
#
# Image: ~2.5 GB (CPU torch via pdfsys-bench is still a direct dep;
# slimming requires making torch an optional extra in
# packages/pdfsys-bench/pyproject.toml — see follow-up note in
# docs/deployment/docker.md).
#
# Build:
#   git submodule update --init --recursive          # first time only
#   docker compose build cli
#
# Run as one-shot CLI:
#   docker compose run --rm cli release verify
#
# Run as bench against the compose stack:
#   docker compose run --rm -v "$(pwd)/data:/data" cli \
#     -m pdfsys_bench --pdf-dir /data/in --out /data/out/results.jsonl \
#     --cascade --vlm

FROM python:3.12-slim AS base

# OS deps:
#   git    — uv falls back to git for some workspace deps if vendor-dir empty
#   curl   — healthcheck / debug
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      libgomp1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Workspace metadata + lockfile first → uv's dep resolution cache layer
# survives source-only edits.
COPY pyproject.toml uv.lock README.md system_release.toml ./

# All workspace members (main repo + submodule). The submodule must be
# initialized in the build context (`git submodule update --init`),
# otherwise external/parsers/ is empty and uv sync fails.
COPY packages packages
COPY external external

ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# Install project itself so `pdfsys` and `pdfsys-bench` console scripts
# are on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Default URLs for sibling services in the compose network. Override
# at run time with `-e MINERU_PIPELINE_URL=...` etc.
ENV MINERU_PIPELINE_URL=http://pdfsys-mineru:8000 \
    MINERU_VLM_URL=http://pdfsys-mineru:8000 \
    QUALITY_URL=http://pdfsys-quality:8765

# Mount points for batch I/O.
VOLUME ["/data/in", "/data/out"]

ENTRYPOINT ["pdfsys"]
CMD ["--help"]
