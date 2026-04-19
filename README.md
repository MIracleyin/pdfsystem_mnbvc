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
- **📊 Quality Scoring**: ModernBERT-large OCR quality assessment [0-3 scale]
- **🔍 Visual Debug**: Page preview with extracted bbox overlays
- **📦 Modular Design**: Stateless, backend-agnostic pipeline components

---

## 🎯 Current Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Stage-A Router** | ✅ Ready | XGBoost binary classifier with 124 PyMuPDF features |
| **MuPDF Parser** | ✅ Ready | Fast extraction for clean-text PDFs |
| **OCR Quality Scorer** | ✅ Ready | ModernBERT-large regression model |
| **Stage-B Router** | ✅ Ready | LayoutDocument → PIPELINE / VLM / DEFERRED |
| **Layout Analyser** | ✅ Ready | DocLayout-YOLO + PP-DocLayoutV3 (dual backend) |
| **Pipeline Parser** | ✅ Ready | Region-level OCR via RapidOCR |
| **VLM Parser** | ✅ Ready | MinerU 2.5 Pro (magic-pdf) for complex pages |
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

# 2. Clone and setup
git clone https://github.com/MIracleyin/pdfsystem_mnbvc.git
cd pdfsystem_mnbvc
uv sync

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

---

## 📦 Workspace Packages

| Package | Purpose | Dependencies |
|---------|---------|--------------|
| `pdfsys-core` | Shared types, schemas, layout cache | stdlib only |
| `pdfsys-router` | Stage-A/Stage-B routing decisions | pymupdf, xgboost, pandas, sklearn |
| `pdfsys-parser-mupdf` | Fast PyMuPDF extraction | pymupdf |
| `pdfsys-bench` | Evaluation harness + quality scorer | torch, transformers |
| `pdfsys-layout-analyser` | DocLayout-YOLO / PP-DocLayoutV3 detection | doclayout-yolo, transformers |
| `pdfsys-parser-pipeline` | Region-level OCR via RapidOCR | rapidocr-onnxruntime |
| `pdfsys-parser-vlm` | MinerU 2.5 Pro VLM extraction | magic-pdf |
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
