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

---

## ✨ Features

- **🧠 ML-Powered Routing**: XGBoost classifier (124 features) routes PDFs to optimal backend
- **⚡ Fast Path**: PyMuPDF extraction for text-ok documents (~10ms/page)
- **📊 Quality Scoring**: fine-tuned ModernBERT OCR quality assessment [0-3 scale], 8192-token context
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

---

## 🏃 Quick Start

### Option 1: Online Demo (Fastest)

Visit [Hugging Face Spaces](https://huggingface.co/spaces/roger1024/DocPipe) and upload a PDF — no installation required.

### Option 2: Local Development

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

### Option 4: Docker deployment (recommended for GPU boxes)

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
