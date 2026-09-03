---
title: "PDFSystem: PB-Scale PDF Processing Pipeline"
emoji: 🚀
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: "PDF to Markdown pipeline with ML-powered routing"
---

# PDFSystem for MNBVC

<p align="center">
  <strong>PB-scale PDF → Pretraining Data Pipeline</strong><br>
  <em>FinePDFs-inspired architecture for Chinese-heavy, mixed-quality PDFs</em>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/roger1024/DocPipe">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces">
  </a>
  <a href="https://github.com/MIracleyin/pdfsystem_mnbvc">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Gradio-6.12.0-green" alt="Gradio">
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License">
</p>

---

## 🚀 Quick Links

| Platform | Link | Description |
|----------|------|-------------|
| **Live Demo** | [🤗 HF Spaces](https://huggingface.co/spaces/roger1024/DocPipe) | Upload PDF and try the pipeline instantly |
| **Source Code** | [GitHub](https://github.com/MIracleyin/pdfsystem_mnbvc) | Full source code and documentation |
| **Data Demo** | [🤗 pdfsys-page-v2-demo](https://huggingface.co/datasets/miracleyin/pdfsys-page-v2-demo) | The L2 output format on 25 real pages — `pages` / `images` / `page_images` / `pairs` |
| **MNBVC Format** | [🤗 example_mmdata_mnbvc](https://huggingface.co/datasets/miracleyin/example_mmdata_mnbvc) | Reference sample for the MNBVC multimodal block format (v2.1) |
| **Quality Scorer** | [🤗 ModernBERT fine-tune](https://huggingface.co/miracleyin/mnbvc-pdf-quality-scorer-modernbert) | OCR quality regression, 0–3 scale |
| **Parsers** | [pdfsys-parsers](https://github.com/MIracleyin/pdfsys-parsers) | Submodule at `external/parsers` — mupdf / pipeline / vlm backends + the vendored `pdfsys-types` contracts |

---

## ✨ Features

- **🧠 ML-Powered Routing**: XGBoost classifier (124 features) routes PDFs to optimal backend
- **⚡ Fast Path**: PyMuPDF extraction for text-ok documents (~10ms/page)
- **📊 Quality Scoring**: fine-tuned ModernBERT OCR quality assessment [0-3 scale], 8192-token context ([model on HF](https://huggingface.co/miracleyin/mnbvc-pdf-quality-scorer-modernbert))
- **🔍 Visual Debug**: Page preview with extracted bbox overlays
- **📦 Modular Design**: Stateless, backend-agnostic pipeline components

---

## 🎯 Current Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Stage-A Router** | ✅ Ready | XGBoost binary classifier with 124 PyMuPDF features |
| **MuPDF Parser** | ✅ Ready | Fast extraction for clean-text PDFs |
| **OCR Quality Scorer** | ✅ Ready | Fine-tuned ModernBERT regression (see docs/deployment/gpu-server.md) |
| **Stage-B Router** | ✅ Ready | LayoutDocument → PIPELINE / VLM / DEFERRED |
| **Layout Analyser** | ✅ Ready | DocLayout-YOLO + PP-DocLayoutV3 (dual backend) |
| **Pipeline Parser** | ✅ Ready | mineru pipeline mode via out-of-process `mineru-api` HTTP client |
| **VLM Parser** | ✅ Ready | mineru VLM mode (`mineru-api` HTTP, engine: transformers / mlx / vllm) for complex pages |
| **Unified CLI** | ✅ Ready | `pdfsys run -c config.yaml --stages ...` |
| **Annotation UI** | ✅ Ready | `pdfsys annotate` — PDF labeling + layout overlay |
| **L2 Dataset Format** | ✅ Ready | `pdfsys.page/v2` — one row per page, interleaved image-text ([spec](docs/superpowers/specs/2026-08-22-page-level-parquet-dataset-design.md), [sample](docs/schema/doc_dataset.v2.sample.md)) |
| **Split runs** | ✅ Ready | `--extract-backends` / `--pdf-list` / `--resume` — CPU and GPU lanes on machines that share no disk |
| **Standalone scoring** | ✅ Ready | `pdfsys score` — score a finished run's markdown against one remote scorer, without re-extracting |
| **L2 Packaging** | ✅ Ready | `pdfsys dataset` — both lanes reach L2: `--from-mineru` (pipeline/vlm sidecars), `--from-pdf-dir` (mupdf, re-extracted) |
| **Format Validator** | ✅ Ready | `pdfsys dataset-validate` — contract check before publishing |
| **MNBVC Export** | ✅ Ready | `pdfsys mnbvc-export` — → MNBVC multimodal block format ([mapping](docs/schema/mnbvc-mm-compat.md)) |
| **Published demo** | ✅ Live | [`pdfsys-page-v2-demo`](https://huggingface.co/datasets/miracleyin/pdfsys-page-v2-demo) — 25 pages, all three extractors, validator-clean |
| **CI** | ⚠️ Not running | `.github/workflows/ci.yml` exists but the workflow is not active. `uv run pytest` + `pdfsys release verify` locally are the real gate |

---

## 🏃 Quick Start

### Option 1: Online Demo (Fastest)

Visit [Hugging Face Spaces](https://huggingface.co/spaces/roger1024/DocPipe) and upload a PDF — no installation required.

### Option 2: Local Development

> `--recurse-submodules` matters: the parser backends live in
> [`pdfsys-parsers`](https://github.com/MIracleyin/pdfsys-parsers) and
> `pyproject.toml` lists `external/parsers/packages/*` as workspace members, so
> `uv sync` fails without them. Both repositories are public and clone
> anonymously.

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone (with submodules) and bootstrap
git clone --recurse-submodules https://github.com/MIracleyin/pdfsystem_mnbvc.git
cd pdfsystem_mnbvc
bash scripts/bootstrap.sh

# 3. Download router weights (257 KB, one-time)
python -m pdfsys_router.download_weights

# 4. Run interactive demo
python app.py
# Open http://localhost:7860
```

### Option 3: Batch Processing

```bash
python -m pdfsys_bench \
  --pdf-dir /path/to/pdfs \
  --out results.jsonl \
  --markdown-dir ./extracted
```

### Option 3b: Split the run across machines that share no disk

The CPU work and the GPU work do not have to happen on the same box, or at the
same time. `--extract-backends` says which backends *this* machine runs;
anything else is recorded as another machine's work
(`skip_reason=lane-filter`) with the path needed to hand it over.

```bash
# CPU box: extract what mupdf can, queue the rest. No GPU, no MinerU.
pdfsys run --pdf-dir /data/corpus --out-dir ./p1 \
           --stages router,extract --extract-backends mupdf \
           --markdown-dir markdown --resume

# The queue. Paths relative to the corpus root travel between machines.
jq -r 'select(.skip_reason != null) | .pdf_path' ./p1/results.jsonl \
  | sed 's|^/data/corpus/||' > gpu_lane.txt

# Ship the PDFs the list names, then the list itself.
rsync -a --partial --files-from=gpu_lane.txt /data/corpus/ gpu01:/mnt/lane/
scp gpu_lane.txt gpu01:/mnt/lane.txt

# GPU box: the same list, against wherever this machine mounted it.
# No layout stage — MinerU does its own internally and is handed only the PDF.
# --parser-output-dir is what makes the result packageable: mineru-api's own
# copy of the sidecars is garbage-collected, and no volume is mounted for it.
pdfsys run --pdf-list /mnt/lane.txt --path-root /mnt/lane --out-dir ./p2 \
           --stages router,extract --extract-backends pipeline \
           --parser-output-dir ./p2/mineru --markdown-dir markdown \
           --ocr-threshold 0.05 --resume

# One CUDA scorer on the GPU box serves both lanes. Nothing is re-extracted
# and no markdown ships: only the text crosses, clipped to the 40k the server
# truncates at anyway (~40 KB/doc).
CUDA_VISIBLE_DEVICES=0 python -m pdfsys_bench._quality_server \
  --host 0.0.0.0 --port 8765 --device cuda &

QUALITY_URL=http://gpu01:8765 pdfsys score \
  --results ./p1/results.jsonl --markdown-dir ./p1/markdown \
  --out ./p1/results.scored.jsonl --resume      # from the CPU box
QUALITY_URL=http://127.0.0.1:8765 pdfsys score \
  --results ./p2/results.jsonl --markdown-dir ./p2/markdown \
  --out ./p2/results.scored.jsonl --resume      # on the GPU box

# Package each lane into the same dataset directory, under its own --shard.
# The CPU lane packages BY LIST, not by scanning: the corpus root also holds
# the documents the GPU lane owns, and mupdf would re-extract those into empty
# pages carrying doc_ids the GPU shard already used. --meta is checked against
# that, so scanning the root here is an error rather than a broken shard.
jq -r 'select(.extract_backend=="mupdf" and .skip_reason==null and .error_class==null)
       | .pdf_path' ./p1/results.scored.jsonl > cpu_lane.txt

pdfsys dataset --from-pdf-list cpu_lane.txt --to ./dataset/v2 --shard cpu-00 \
               --meta ./p1/results.scored.jsonl
pdfsys dataset --from-mineru ./p2/mineru --to ./dataset/v2 --shard gpu-00 \
               --meta ./p2/results.scored.jsonl

pdfsys dataset-validate --shard ./dataset/v2

# For the VLM lane, layout IS load-bearing — only stage-B ever says "vlm":
#   --stages router,layout,extract --vlm --extract-backends pipeline,vlm
```

Use the same `--ocr-threshold` on both boxes. Phase 2 re-runs stage-A (asking
for `extract` pulls in `router`), so a different threshold can re-classify a
document as `mupdf` on the GPU box, where the lane filter then skips it — and
the CPU box had already handed it away. It would fall out of both lanes. That
shows up as a nonzero `lane-filter` count on the GPU box, which the run warns
about; expect zero.

`--resume` appends to `results.jsonl` and skips the documents already in it, so
a machine that dies at hour six restarts where it stopped — and does not
destroy the worklist the other machine is waiting on. `--limit` names the same
slice of the corpus on every invocation, so it composes with resume. Splitting
a list file (`split -n l/8 gpu_lane.txt`) is all fleet sharding needs.

### Option 4: Package a run into the L2 dataset

The pipeline's own output (`results.jsonl` + MinerU sidecars) is L1 telemetry.
Turning it into the publishable dataset is a separate, explicit step:

```bash
# One row per PAGE, keyed (doc_id, page_index). Figure crops in images/.
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --meta ./out/results.jsonl --pairs

# …or full-page rasters instead, with figures cut out by bbox on read.
# The two image modes are mutually exclusive — MinerU's crops are already
# sub-rectangles of a 200-dpi page render, so keeping both stores the same
# pixels twice.
pdfsys dataset --from-mineru ./out --to ./dataset/v2 \
               --images pages --pdf-dir ./data/pdfs --render-dpi 200

# The text-ok documents never reached MinerU, so they leave no sidecars.
# Package those straight from the PDFs. This re-runs mupdf (~10ms/page),
# because a run persists only merged markdown with no page boundaries.
# Whole-page rasters are the default here — mupdf has no crops to store.
#
# BY LIST, not by scanning ./data/pdfs: the corpus also holds the documents
# MinerU handled, and mupdf would re-extract those into pages carrying doc_ids
# the shard above already used. --meta is checked against exactly that, so
# scanning the whole corpus here is an error rather than a broken shard.
jq -r 'select(.extract_backend=="mupdf" and .skip_reason==null and .error_class==null)
       | .pdf_path' ./out/results.jsonl > mupdf_lane.txt

pdfsys dataset --from-pdf-list mupdf_lane.txt --to ./dataset/v2 --shard mupdf-00 \
               --meta ./out/results.jsonl

# Contract check. Run this before anything leaves the machine.
pdfsys dataset-validate --shard ./dataset/v2

# Re-emit in the MNBVC multimodal block format.
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet --dialect v2
```

The two lanes write separate *shards* — one parquet each — into the same
dataset directory. Sortedness is a per-file promise, because a reader
reassembles a document by scanning one file; the two lanes' doc_id ranges are
free to interleave. What the shards may not do is overlap: a document belongs
to exactly one lane, and `pdfsys dataset` refuses to reuse a shard name without
`--overwrite` rather than truncate the other lane's work.

Worked example of one real page:
[`docs/schema/doc_dataset.v2.sample.md`](docs/schema/doc_dataset.v2.sample.md).

### Option 5: Docker deployment (recommended for GPU boxes)

Three microservices behind HTTP: `mineru` (parsers), `quality`
(ModernBERT scorer), `cli` (orchestrator). CPU and GPU images, HF
weights mounted via volume. Full walk-through in
[`docs/deployment/gpu-server.md`](docs/deployment/gpu-server.md).

```bash
git clone --recurse-submodules https://github.com/MIracleyin/pdfsystem_mnbvc.git
cd pdfsystem_mnbvc

# Detect CPU vs GPU, download weights, build + start services, smoke.
bash scripts/detect_gpu.sh          # emits .deploy.env
bash scripts/download_models.sh     # ~6 GB HF cache
bash scripts/deploy.sh              # docker compose build + up + healthcheck
```

### Option 5: Parser-matrix annotation dataset

Run every PDF through every parser (mupdf + pipeline + vlm) and
produce one self-contained JSON of all candidate extractions —
designed for the downstream quality-scoring / annotation loop.

```bash
# Against a live docker-compose stack (Option 4)
bash scripts/batch_process.sh /data/pdfs out/annotation-set

# Directly against local subprocesses (Option 2)
uv run python scripts/extract_matrix.py \
  --pdf-dir /data/pdfs \
  --out out/annotation-set/results.jsonl \
  --markdown-dir out/annotation-set/markdown \
  --vlm-engine mlx-engine       # or vllm-engine on NVIDIA
uv run python scripts/emit_quality_handoff_matrix.py \
  out/annotation-set/results.jsonl
```

Produces `results.jsonl` (one row per PDF × parser),
`markdown/<sha>__<parser>.md` per successful extraction, and a
`quality_handoff_matrix.json` with all markdown inlined and grouped by
`file_id`. Reference: 150 PDFs × 3 parsers on 8× RTX 4090 → 9.2 min
(~14× vs Apple Silicon mlx-engine).

### Component versioning

This project is a **system release**: a tuple of (main repo commit, pinned
component commits). Each independently-versioned component lives in its own
git repo, mounted as a submodule under `external/`. The current pins live
in [`system_release.toml`](system_release.toml).

| Command | Purpose |
|---|---|
| `bash scripts/bootstrap.sh` | Init submodules + uv sync + verify pins (idempotent) |
| `uv run pdfsys release status` | Show current pins vs submodule HEADs |
| `uv run pdfsys release verify` | CI guard — exit non-zero if any pin drifted |
| `uv run pdfsys release lock` | Bump pins to match submodule HEADs (refuses if working tree dirty) |

For the full architectural rationale (Goodhart prevention, why parsers and the
quality scorer get independent release cadences), see
[`docs/superpowers/specs/2026-05-30-parsers-submodule-design.md`](docs/superpowers/specs/2026-05-30-parsers-submodule-design.md)
and [`docs/architecture/LAYERS.md`](docs/architecture/LAYERS.md#component-versioning).

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
   PDF Input  ───►  │  Stage-A Router │  XGBoost (124 features)
                    │  (Implemented)  │  ~10ms per PDF
                    └────────┬────────┘
                             │ ocr_prob
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      ┌─────────┐      ┌──────────┐      ┌─────────┐
      │  MUPDF  │      │ PIPELINE │      │   VLM   │
      │  (Fast) │      │  (OCR)   │      │(Complex)│
      └────┬────┘      └──────────┘      └─────────┘
           │
           ▼
   ┌─────────────────────────────────────┐
   │  ExtractedDoc: Markdown + Segments  │
   └─────────────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────────────┐
   │  Quality Scorer (ModernBERT-large)  │
   │  Score: [0, 3]                      │
   └─────────────────────────────────────┘
```

> **Heavy-ML isolation:** the PIPELINE and VLM parsers and the Quality Scorer never import `mineru`/`torch` in the host process — each spawns a dedicated subprocess (`mineru-api`, `_quality_server`) and talks to it over HTTP. See `ARCHITECTURE.md` § Key Design Decisions #6.

---

## 📦 Workspace Packages

| Package | Purpose | Dependencies |
|---------|---------|--------------|
| `pdfsys-core` | Shared types, schemas, layout cache | stdlib only |
| `pdfsys-router` | Stage-A/Stage-B routing decisions | pymupdf, xgboost, pandas, sklearn |
| `pdfsys-parser-mupdf` | Fast PyMuPDF extraction | pymupdf |
| `pdfsys-bench` | Evaluation harness + quality scorer | torch, transformers |
| `pdfsys-layout-analyser` | DocLayout-YOLO / PP-DocLayoutV3 detection | doclayout-yolo, transformers |
| `pdfsys-parser-pipeline` | mineru pipeline mode (out-of-process HTTP client) | httpx, mineru[pipeline] |
| `pdfsys-parser-vlm` | mineru VLM mode (out-of-process HTTP client) | httpx, mineru[vlm] (+mineru[mlx] on arm64-darwin) |
| `pdfsys-cli` | Unified CLI + YAML config + annotation UI | pyyaml |

---

## 📊 Benchmark Results

**OmniDocBench-100 Dataset:**

```
Backend Split:    mupdf=70    pipeline=30
Avg OCR Prob:     mupdf=0.034  pipeline=0.634
Extraction:       70 success   0 errors
Quality Score:    avg=1.71     min=0.39   max=2.73
Timing:           router=49ms  extract=7ms  quality=3.6s
```

---

## 🎨 Demo Interface

The Gradio demo provides:

- **📤 PDF Upload**: Drag-and-drop or click to upload
- **📈 Routing Info**: OCR probability, selected backend, page count
- **🖼️ Page Preview**: First page with colored bbox overlays
- **📝 Markdown Output**: Extracted text content
- **📋 Segment Table**: Block-level extraction details
- **🔧 Feature View**: Selected router features
- **📄 Raw JSON**: Complete pipeline output
- **⭐ Quality Score**: Optional ModernBERT scoring

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/PRD.md`](docs/PRD.md) | Product Requirements & Architecture Rationale |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Implementation Timeline & Milestones |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development Guidelines & Commit Conventions |
| [`demo/README.md`](demo/README.md) | Demo-specific Documentation |

---

## 💻 Development

### Data Structures

**Router Output:**
```python
@dataclass
class RouterDecision:
    backend: Backend          # MUPDF | PIPELINE | VLM | DEFERRED
    ocr_prob: float           # P(needs OCR) [0, 1]
    num_pages: int
    is_form: bool
    features: dict            # 124-dim feature vector
```

**Parser Output:**
```python
@dataclass(frozen=True)
class ExtractedDoc:
    sha256: str
    backend: Backend
    segments: tuple[Segment, ...]
    markdown: str
    stats: dict
```

### CLI Reference

```bash
# Download router weights
python -m pdfsys_router.download_weights

# Run benchmark
python -m pdfsys_bench \
  --pdf-dir PATH \
  --out results.jsonl \
  --no-quality          # Skip quality scoring
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">
  Built with ❤️ for the <a href="https://github.com/esbatmop/MNBVC">MNBVC</a> corpus project
</p>
