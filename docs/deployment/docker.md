# Docker / docker-compose deployment

pdfsys runs as three independent HTTP services that talk over a
docker-compose network. Each image installs only the deps its tier
actually needs, so per-image footprint stays small and tiers scale
independently.

```
┌────────────────────────────────────────────────────────────────────┐
│                    docker-compose default network                   │
│                                                                    │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐ │
│  │ pdfsys-mineru    │   │ pdfsys-quality   │   │ pdfsys-cli     │ │
│  │ docker/          │   │ docker/          │   │ docker/        │ │
│  │   mineru.        │   │   quality.       │   │   cli.         │ │
│  │   Dockerfile     │   │   Dockerfile     │   │   Dockerfile   │ │
│  │                  │   │                  │   │                │ │
│  │ mineru-api       │   │ _quality_server  │   │ pdfsys CLI     │ │
│  │ (CPU pipeline)   │   │ (ModernBERT)     │   │ + pdfsys_bench │ │
│  │                  │   │                  │   │                │ │
│  │ POST /file_parse │   │ POST /score      │   │ orchestrates   │ │
│  │ GET  /health     │   │ GET  /health     │   │ via env URLs   │ │
│  │ ~2.5 GB          │   │ ~1.5 GB          │   │ ~2.5 GB        │ │
│  │ :8000            │   │ :8765            │   │ (one-shot)     │ │
│  └────────▲─────────┘   └────────▲─────────┘   └────────┬───────┘ │
│           │                      │                      │         │
│           │ MINERU_PIPELINE_URL  │ QUALITY_URL          │         │
│           │ MINERU_VLM_URL       │                      │         │
│           └──────────────────────┴──────────────────────┘         │
│                                                                    │
│  Volumes:                                                          │
│     /cache/huggingface  ← bind-mount ~/.cache/huggingface (ro)    │
│     /data/in, /data/out ← bind-mount at `compose run` for cli     │
└────────────────────────────────────────────────────────────────────┘
```

## Why microservices

The pre-Docker architecture already wanted this shape: pipeline / vlm
parsers spawn a mineru-api subprocess and HTTP-talk to it; the quality
scorer is a separate subprocess serving HTTP at `/score`. The Docker
layout just lifts those subprocesses out of the parent process and into
sibling containers — exactly the same wire contracts.

Per-image dep separation:

| Service | Includes | Excludes |
|---|---|---|
| mineru | mineru[pipeline], torch (CPU), opencv-headless, onnxruntime | ModernBERT, pdfsys workspace, CUDA |
| quality | torch (CPU), transformers, safetensors | mineru, pdfsys workspace, CUDA |
| cli | pdfsys workspace (incl. external/parsers submodule), httpx | mineru runtime weights, ModernBERT runtime weights |

The cli image still pulls torch transitively because
`packages/pdfsys-bench/pyproject.toml` lists `torch>=2.1` as a direct
dep — see [Follow-up: slim cli image](#follow-up-slim-cli-image).

## First-time setup

1. **Install Docker.** On macOS, OrbStack or Docker Desktop both work.
   Verify with `docker --version`.

2. **Initialize the parsers submodule** (the cli image needs its files
   in the build context):
   ```bash
   git submodule update --init --recursive
   ```

3. **Pre-cache model weights on the host** — mineru and quality bind-
   mount the host's `~/.cache/huggingface` read-only at runtime. If you
   skip this, the services fail with `LocalEntryNotFoundError` on first
   request because both bake `HF_HUB_OFFLINE=1`.
   ```bash
   bash scripts/download_models.sh
   ```
   This downloads:
   - `opendatalab/PDF-Extract-Kit-1.0` (~2.3 GB, mineru pipeline)
   - `opendatalab/MinerU2.5-Pro-2605-1.2B` (~2.4 GB, mineru VLM-mlx)
   - `HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn` (~1.5 GB)
   - router XGBoost weights (~250 KB)

4. **Build images:**
   ```bash
   docker compose build
   ```
   First build pulls ~5 GB of base layers + Python deps; subsequent
   builds reuse the cache.

## Run modes

### Long-running HTTP services (mineru + quality)

```bash
docker compose up -d mineru quality
# wait for both to be healthy
docker compose ps
```

Healthchecks:
```bash
curl http://localhost:8000/health   # mineru
curl http://localhost:8765/health   # quality
```

### One-shot CLI commands

The `cli` service is configured `depends_on: [mineru, quality]` with
`service_healthy` gating, so any `docker compose run` against it
blocks until both backends are alive.

```bash
# Release manifest verification (no HTTP traffic)
docker compose run --rm cli release verify

# 150-PDF bench against the live stack
mkdir -p data/in data/out
# (drop your PDFs into data/in/)
docker compose run --rm \
  -v "$(pwd)/data:/data" \
  cli -m pdfsys_bench \
    --pdf-dir /data/in \
    --out /data/out/results.jsonl \
    --cascade --vlm

# Generate the quality-scorer handoff JSON from the results
docker compose run --rm \
  -v "$(pwd)/data:/data" \
  --entrypoint python cli \
  scripts/emit_quality_handoff.py /data/out/results.jsonl
```

The cli image overrides parser-side service-discovery via env vars:
```
MINERU_PIPELINE_URL=http://mineru:8000
MINERU_VLM_URL=http://mineru:8000
QUALITY_URL=http://quality:8765
```
These switch the parsers + bench from "spawn local subprocess" to
"HTTP to sibling container" — see
[the env-override implementation](../../external/parsers/packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/extract.py).

### Direct HTTP from the host

You can also call the services from outside the compose network for
debugging:

```bash
# Score arbitrary markdown
curl -X POST http://localhost:8765/score \
  -H 'content-type: application/json' \
  -d '{"text":"# Document title\n\nbody text..."}'

# Parse a PDF via mineru pipeline backend
curl -X POST http://localhost:8000/file_parse \
  -F "files=@sample.pdf" \
  -F "backend=pipeline" \
  -F "return_md=true"
```

## Volumes

```
~/.cache/huggingface  ←  /cache/huggingface  (ro)
   Both mineru and quality bind-mount the host's HF cache. Production
   should switch to a named volume by editing docker-compose.yml's
   volumes block. The `hf-cache:` named volume is already declared.

./data/in  →  /data/in  (rw)   bench input PDFs
./data/out →  /data/out (rw)   bench JSONL output + markdown dump
```

## GPU variants

CPU images are the default. For NVIDIA GPU:

- **mineru:gpu** — TBD (Phase D). Will use `nvidia/cuda:12.x-runtime`
  base + `mineru[pipeline,vllm-engine]`. Requires NVIDIA Container
  Toolkit on the host.
- **quality:gpu** — Not planned. ModernBERT is small; the CPU image
  scores at ~10 PDFs/sec which is already faster than the mineru tier.

Apple Silicon MLX (vlm-mlx-engine) is **not dockerizable** — MLX
doesn't virtualize. On Apple Silicon, run mineru-api directly outside
Docker and point `MINERU_VLM_URL` at it (host.docker.internal:PORT).

## Follow-up: slim cli image

`packages/pdfsys-bench/pyproject.toml` currently lists
`torch>=2.1`, `transformers>=4.44` as direct deps; the cli image
therefore drags them in even though only `_quality_server.py`
actually imports them (lazily, inside `_init()`). The quality service
is supposed to be the sole owner of that stack.

To slim the cli image to ~500 MB:

1. Move torch + transformers into a `[project.optional-dependencies]
   quality = [...]` extra in pdfsys-bench's pyproject.
2. Update `scripts/bootstrap.sh` to pass `--extra quality` so local
   development keeps the in-process scorer path.
3. Update `docker/cli.Dockerfile` to `uv sync --no-extra quality`.
4. Validate: the cli image should still pass `release verify`
   and the bench should run successfully against the compose stack.

Not done in Phase C because the pyproject change is workspace-wide
and worth its own commit + test cycle.

## Troubleshooting

- **`pdfsys-mineru` 409 status: failed** — model weights not in
  `~/.cache/huggingface`. Run `bash scripts/download_models.sh`.
- **`pdfsys-quality` health probe never goes healthy** — first
  ModernBERT load on a cold cache can take 60-120 s.
  `start_period: 120s` in compose covers this; if the failure
  persists, check `docker logs pdfsys-quality` for
  `LocalEntryNotFoundError`.
- **CLI image build fails at `COPY external external`** — the
  parsers submodule isn't initialized. Run
  `git submodule update --init --recursive` and rebuild.
- **macOS Clash/V2ray stalls model downloads** — see
  `scripts/download_models.sh`; it sets `NO_PROXY=*` to bypass the
  macOS system proxy, which Python's `requests`/`httpx` honors
  even when `HTTPS_PROXY` is unset.
