---
name: pdfsys-ops
description: Use when deploying, running, or batch-processing PDFs with this repo (pdfsystem_mnbvc) on a GPU server — bringing up the mineru + quality stack, running the parser matrix over a PDF folder, or evaluating/tuning the router. Covers the mnbvcgpu 4090 fleet gotchas (image reuse, long-job survival over VPN, GPU pinning, hf-mirror, router threshold).
---

# pdfsys-ops

Operate pdfsystem_mnbvc (PDF → Markdown extraction + ModernBERT quality scoring)
on a GPU host. Two long-lived HTTP services do the work:

| Service | Port | What | Image |
|---|---|---|---|
| mineru | 8000 | pipeline + VLM extraction (vllm) | `pdfsys-mineru:gpu` (~17.6 GB) |
| quality | 8765 | ModernBERT ordinal scorer, 0–3 | `pdfsys-quality:dev` (~1.2 GB) |

The full runbook is `docs/deployment/gpu-server.md`. This skill is the fast path
plus the non-obvious lessons that runbook doesn't cover. For VPN/host access use
the **easytier** skill. Fleet — each 8×RTX 4090, `root` SSH over the mesh:

| SSH alias | host | internal IP |
|---|---|---|
| mnbvcgpu1 | xsy-01 | 10.253.253.1 |
| mnbvcgpu2 | xsy-02 | 10.253.253.2 |
| mnbvcgpu3 | hgy-01 | 10.253.253.3 |

In the transfer commands below, `$TARGET` is the destination host's internal IP.

## Quick reference

| Task | Command (on the GPU host, in the repo, `PATH=/usr/local/bin:$PATH`) |
|---|---|
| Detect GPU → `.deploy.env` | `bash scripts/detect_gpu.sh` |
| Bring up stack (build path) | `bash scripts/deploy.sh` |
| Batch a PDF folder | `bash scripts/batch_process.sh <pdf-dir> out/<run>` |
| Smoke (subset) | `bash scripts/batch_process.sh <pdf-dir> out/smoke --limit 2` |
| Health | `curl localhost:8000/health ; curl localhost:8765/health` |
| Router eval / threshold | see **Router** below |

`batch_process.sh` writes `results.jsonl` (one row per PDF×parser), `markdown/`,
`quality_handoff_matrix.json`, and a `.tar.gz`. Bench PDFs live in
`packages/pdfsys-bench/{olmocr_bench_50,omnidocbench_100}/pdfs` (150 total).

## Bring up the stack — reuse images, don't rebuild

**Building `mineru.gpu` from scratch is ~90 min on a CN network. Don't.** If any
sibling host already has the images, stream them over the internal mesh (~50 MB/s),
which is far faster than a rebuild and needs no registry:

```bash
# on a host that HAS the images (e.g. hgy-01), push to the target ($TARGET = its internal IP):
docker save pdfsys-mineru:gpu pdfsys-quality:dev | zstd -T0 -3 \
  | ssh root@$TARGET 'zstd -d -T0 | docker load'
# also reuse its warm model cache + config (skips ~7 GB HF download):
rsync -a /root/.cache/huggingface/ root@$TARGET:/root/.cache/huggingface/
scp /root/mineru.json root@$TARGET:/root/mineru.json
```

Only fall back to `scripts/deploy.sh` (build + `download_models.sh`) when no sibling
has the images. On CN hosts always pass the pip + HF mirrors (`hf-mirror.com`,
aliyun PyPI) — see `docs/deployment/gpu-server.md` § CN networking.

Start containers with the correct mounts and a **pinned GPU** (the boxes are shared —
never grab `--gpus all`; pick an idle index from `nvidia-smi`):

```bash
docker run -d --name pdfsys-mineru --gpus '"device=2"' \
  --network pdfsys_default -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /root/mineru.json:/root/mineru.json:ro \
  pdfsys-mineru:gpu --host 0.0.0.0 --port 8000

docker run -d --name pdfsys-quality --network pdfsys_default -p 8765:8765 \
  -v /root/.cache/huggingface:/cache/huggingface:ro \
  -v $PWD/packages/pdfsys-bench/src/pdfsys_bench/_quality_server.py:/app/quality_server.py:ro \
  pdfsys-quality:dev
```

The `mineru.json` bind-mount fixes VLM path resolution; the `_quality_server.py`
bind-mount is **required when reusing an old quality image** — it carries the
ordinal-multiclass-head fix the baked image predates. Verify:
`curl -s localhost:8765/health` → `"model":"miracleyin/mnbvc-pdf-quality-scorer-modernbert"`.

`uv` may be missing on a fresh host: `python3 -m venv /root/.uvbox && /root/.uvbox/bin/pip install uv -i https://mirrors.aliyun.com/pypi/simple/ && ln -sf /root/.uvbox/bin/uv /usr/local/bin/uv`, and run with `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`.

## Long jobs MUST survive the SSH/VPN link

Batch runs take minutes to hours. The mesh VPN drops, and a dropped
`ssh 'nohup … &'` can **kill the job or silently launch it twice** (duplicate
runs clobber each other's output). Never background a job through the SSH session.
Detach it from the session with a transient systemd unit:

```bash
systemd-run --unit=pdfsys-batch --collect \
  --working-directory=$PWD \            # the repo checkout on this host
  --setenv=PATH=/usr/local/bin:/usr/bin:/bin \
  --setenv=UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  bash -c 'bash scripts/batch_process.sh packages/pdfsys-bench out/bench150 > /root/bench150.log 2>&1'
# poll from anywhere, reconnect-safe:
ssh <host> 'systemctl is-active pdfsys-batch; tail -5 /root/bench150.log'
```

Note: `pgrep extract_matrix` finds nothing — the process is `python3`; use `pgrep -f`.

## Router — tuned threshold and the cliff

The Stage-A router (`Router(ocr_threshold=…)`) decides mupdf (text-ok) vs OCR;
Stage-B splits pipeline/vlm by `layout_cache.json` `has_complex`. **Default is
0.05** (tuned on bench-150). Rules:

- Keep `ocr_threshold` **below 0.60** — the classifier's `ocr_prob` is bimodal
  (≈90 PDFs <0.05, a cluster at ≈0.60); crossing 0.60 dumps that cluster into
  mupdf and quality pass-rate (score ≥2.0) collapses from ~25% to ~15%.
- On a quality-only metric no router setting beats **always-vlm** — the router's
  value is GPU cost (it diverts confident born-digital PDFs to the free mupdf
  lane, ~60% fewer GPU calls), not top-line quality.

Evaluate a routing policy offline (no re-extraction) by joining the router's
per-PDF choice with the already-scored matrix: run `Router.classify()` per PDF,
look up the chosen parser's score in `quality_handoff_matrix_scored.json`, count
`≥ 2.0`. To sweep the threshold, re-route each PDF at each candidate value against
the same scores. (This session's scripts: `out/route_and_eval.py`, `out/router_threshold_sweep.json`.)

## Gotchas

| Symptom | Cause / fix |
|---|---|
| Build stalls / times out | CN network — reuse a sibling's image (above), or pass pip+HF mirrors |
| mineru OOM or steals a busy GPU | pin `--gpus '"device=N"'` to an idle index from `nvidia-smi` |
| CDI-only host: `--gpus` errors | use `--device nvidia.com/gpu=N` instead |
| HF download hangs on CN host | `HF_ENDPOINT=https://hf-mirror.com`, `NO_PROXY='*'` |
| quality serves wrong/old scores | bind-mount current `_quality_server.py` over the image's copy |
| batch job vanished after disconnect | it was backgrounded through SSH — relaunch with `systemd-run` |
| first VLM request ~150 s | vllm JIT cold start; steady state ~4 s/PDF |

## Common mistakes

- Grabbing `--gpus all` on a shared box (collides with other tenants).
- Rebuilding images when a sibling host already has them.
- Raising `ocr_threshold` toward 0.5+ "to be safe" — it lowers quality here; ≤0.60 hard rule.
- Trusting a reused quality image without the `_quality_server.py` mount.
