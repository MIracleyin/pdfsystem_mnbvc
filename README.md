---
title: PDFSystem MNBVC Demo
emoji: 📄
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: FinePDFs-style PDF pipeline demo for MNBVC
---

# pdfsys-mnbvc

PB-scale PDF → pretraining-data pipeline for the [MNBVC](https://github.com/esbatmop/MNBVC) corpus project.
FinePDFs-inspired architecture adapted for Chinese-heavy, mixed-quality input.

> **Try it:** 
> - 🚀 **在线 Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/roger1024/DocPipe) - 直接上传 PDF 体验完整流程
> - 💻 **本地运行**: `python app.py` - 详见下方 [Quick start](#quick-start)
> 
> 部署到 Hugging Face Spaces 只需一键，YAML header 就是全部配置。详见 [`demo/README.md`](demo/README.md)

## Current status: MVP closed loop ✅

The first end-to-end path — **Router → MuPDF parser → OCR quality scorer** — is working on the OmniDocBench-100 evaluation set. PDFs that need OCR are routed to `PIPELINE` but not yet extracted (that backend is not implemented yet).

## Quick start

### 方式一：在线体验（最快）

直接访问 [Hugging Face Spaces Demo](https://huggingface.co/spaces/roger1024/DocPipe) 上传 PDF 即可体验，无需安装任何环境。

### 方式二：本地运行

```bash
# 1. Install uv (>= 0.4)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo and sync all workspace packages
git clone https://github.com/MIracleyin/pdfsystem_mnbvc.git
cd pdfsystem_mnbvc
uv sync

# 3. Fetch the XGBoost router weights (257 KB, one-time)
python -m pdfsys_router.download_weights

# 4. Run Gradio demo
python app.py
# 访问 http://localhost:7860

# 5. Or run the MVP closed loop on the bench dataset
python -m pdfsys_bench \
  --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
  --out out/bench_omnidoc100.jsonl \
  --markdown-dir out/bench_omnidoc100_md
```

> **Note:** The first run downloads the ModernBERT-large quality scorer
> (~800 MB) from HuggingFace Hub. Set `HF_HOME` to control where it's
> cached. If you don't need quality scoring, add `--no-quality` to skip it.

> **Note:** The bench dataset (omnidocbench_100) is NOT committed to the repo.
> You need to obtain it separately and place it under
> `packages/pdfsys-bench/omnidocbench_100/`.

## Architecture

```
          ┌──────────────┐
PDF  ──►  │ pdfsys-router│  stage A: XGBoost (124 PyMuPDF features)
          └──────┬───────┘
                 │
      text-ok ◄──┴──► needs-ocr
         │               │
         ▼               ▼
  parser-mupdf   pdfsys-layout-analyser  (runs once, caches LayoutDocument)
                         │
                         ▼
                  stage B decision
                         │
          no-complex ◄───┴───► complex (tables / formulas)
                 │                   │
                 ▼                   ▼
         parser-pipeline       parser-vlm
```

### What's implemented

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage-A router** | ✅ | XGBoost binary classifier, ported from FinePDFs. 124 features (4 doc-level + 15 page-level × 8 sampled pages). Routes to `MUPDF` (text-ok) or `PIPELINE` (needs-ocr). |
| **MuPDF parser** | ✅ | `page.get_text("blocks", sort=True)` → `ExtractedDoc` with normalized bbox and merged Markdown. Fast path for clean-text PDFs. |
| **OCR quality scorer** | ✅ | ModernBERT-large regression head (`HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn`). Scores extracted text on a [0, 3] scale. |
| **Bench CLI** | ✅ | `python -m pdfsys_bench` — drives the full loop, emits per-doc JSONL + summary JSON. |
| Stage-B router | ❌ | Pending layout-analyser and LayoutCache integration. |
| Layout analyser | ❌ | PP-DocLayoutV3 / docling-layout-heron runner — not started. |
| Pipeline parser | ❌ | Region-level OCR (RapidOCR / PaddleOCR) — not started. |
| VLM parser | ❌ | MinerU 2.5 / PaddleOCR-VL on complex regions — not started. |

### MVP benchmark results (OmniDocBench-100)

```
Backend split:  mupdf=70  pipeline=30
Avg ocr_prob:   mupdf=0.034  pipeline=0.634
Extracted:      70   Errors: 0
Quality:        avg=1.71  min=0.39  max=2.73
Per-doc time:   router=49ms  extract=7ms  quality=3.6s
```

## Workspace packages

| Package | Role | Dependencies |
|---------|------|-------------|
| `pdfsys-core` | Shared dataclasses, enums, layout cache, serde. No PDF/ML deps. | stdlib only |
| `pdfsys-router` | Stage-A XGBoost classifier + Stage-B layout decision (stub). | pymupdf, xgboost, pandas, numpy, scikit-learn |
| `pdfsys-layout-analyser` | Page layout model runner. Stub only. | — |
| `pdfsys-parser-mupdf` | Text-ok backend: PyMuPDF block extraction → Markdown. | pymupdf |
| `pdfsys-parser-pipeline` | OCR backend for simple layouts. Stub only. | — |
| `pdfsys-parser-vlm` | VLM backend for complex layouts. Stub only. | — |
| `pdfsys-bench` | Closed-loop evaluation harness + quality scorer. | torch, transformers, pdfsys-router, pdfsys-parser-mupdf |

### Package dependency graph

```
pdfsys-core  ◄── pdfsys-router
             ◄── pdfsys-parser-mupdf
             ◄── pdfsys-parser-pipeline  (stub)
             ◄── pdfsys-parser-vlm       (stub)
             ◄── pdfsys-layout-analyser  (stub)

pdfsys-router        ◄── pdfsys-bench
pdfsys-parser-mupdf  ◄── pdfsys-bench
```

`pdfsys-core` is the root dependency: every other package imports it, and it has zero external deps beyond the Python stdlib.

## Key data structures

### Router output (`RouterDecision`)

```python
@dataclass
class RouterDecision:
    backend: Backend          # MUPDF | PIPELINE | VLM | DEFERRED
    ocr_prob: float           # P(needs OCR) from XGBoost, [0, 1]
    num_pages: int
    is_form: bool
    garbled_text_ratio: float
    is_encrypted: bool
    needs_password: bool
    features: dict            # full 124-feature vector for debugging
    error: str | None
```

### Parser output (`ExtractedDoc`)

```python
@dataclass(frozen=True)
class ExtractedDoc:
    sha256: str
    backend: Backend
    segments: tuple[Segment, ...]   # ordered block-level units
    markdown: str                    # segments merged with \n\n
    stats: dict
```

Each `Segment` carries `page_index`, `RegionType` (TEXT/IMAGE/TABLE/FORMULA), `content` (Markdown / HTML / LaTeX), and a normalized `BBox` in [0, 1].

### Quality score

```python
@dataclass
class QualityScore:
    score: float        # [0, 3]: 0=garbage, 1=format issues, 2=minor, 3=clean
    num_chars: int
    num_tokens: int
    model: str
```

## Design principles

1. **Stateless processing.** No manifest, no central DB. Every PDF produces self-contained output. Following FinePDFs' datatrove-style design.
2. **Content-addressable caching.** LayoutCache shards by `sha256 + model_tag`. Bumping the model tag lazily invalidates old entries.
3. **Atomic writes.** All file outputs use `tmp + os.replace()` for crash safety.
4. **Normalized coordinates.** BBox is always `[0, 1]` with origin top-left; backends convert to pixels/points on demand.
5. **Backend-agnostic output.** All three parser backends emit the same `ExtractedDoc` / `Segment` schema, so downstream stages don't need to know which backend produced a document.

## CLI reference

### `python -m pdfsys_bench`

```
usage: pdfsys-bench [-h] --pdf-dir PDF_DIR --out OUT [--limit N]
                    [--no-quality] [--quality-model MODEL]
                    [--router-weights PATH] [--markdown-dir DIR]
                    [--ocr-threshold FLOAT]

Run the MVP pdfsys closed loop.

options:
  --pdf-dir PATH       Directory of PDFs to process (recursive).
  --out PATH           Output JSONL path (one line per PDF).
  --limit N            Cap the number of PDFs processed.
  --no-quality         Skip the ModernBERT quality scorer.
  --quality-model ID   HuggingFace model for quality scoring.
  --router-weights P   Path to xgb_classifier.ubj.
  --markdown-dir DIR   Dump per-PDF extracted markdown here.
  --ocr-threshold F    P(ocr) threshold (default: 0.5).
```

### `python -m pdfsys_router.download_weights`

Downloads the XGBoost router weights (~257 KB) from the FinePDFs Git LFS.

```bash
python -m pdfsys_router.download_weights          # first time
python -m pdfsys_router.download_weights --force   # re-download
```

## Output format

The JSONL output (`--out`) has one JSON object per PDF:

```json
{
  "pdf_path": "packages/pdfsys-bench/omnidocbench_100/pdfs/example.pdf",
  "sha256": "a53b50cb0d3d...",
  "backend": "mupdf",
  "ocr_prob": 0.003,
  "num_pages": 1,
  "is_form": false,
  "garbled_text_ratio": 0.0,
  "router_error": null,
  "extract_stats": {"page_count": 1, "pages_extracted": 1, "segment_count": 5, "char_count": 5734},
  "extract_error": null,
  "quality_score": 2.45,
  "quality_num_chars": 5734,
  "quality_num_tokens": 512,
  "quality_model": "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn",
  "markdown_chars": 5734,
  "wall_ms_router": 42.1,
  "wall_ms_extract": 6.3,
  "wall_ms_quality": 3421.0
}
```

A companion `.summary.json` file is also written with aggregate statistics.

## Demo 功能说明

在线 Demo 展示了完整的 MVP 流程，包含以下功能：

| 功能 | 描述 |
|------|------|
| **PDF 上传** | 支持拖拽或点击上传 PDF 文件 |
| **路由决策** | 实时显示 XGBoost 路由器的 OCR 概率和选择的 Backend |
| **页面预览** | 第一页渲染并叠加提取的文本块边界框（颜色标识 Backend） |
| **Markdown 输出** | PyMuPDF 提取的文本内容 |
| **Segments 表格** | 详细的块级提取信息（类型、坐标、字符数等） |
| **Router Features** | 精选的 124 维特征子集展示 |
| **Raw JSON** | 完整的 pipeline 输出数据 |
| **OCR 质量评分** | 可选的 ModernBERT 质量评分（默认关闭，约 3-5 秒） |

### Demo 技术栈
- **Frontend**: Gradio 6.12.0
- **Backend**: Python 3.11 + PyMuPDF + XGBoost
- **部署**: Hugging Face Spaces (CPU)

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/PRD.md`](docs/PRD.md) | 完整产品需求文档，包含资源预算和架构原理 |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 优先级排序的实现计划、工作量估算和验收标准 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 命名规范、一致性规则、提交格式 |
| [`demo/README.md`](demo/README.md) | Gradio Demo 详情 + Hugging Face Spaces 部署指南 |

## Collaborating with Cursor

This repo ships a full set of [Cursor project rules](https://docs.cursor.com/context/rules) under `.cursor/rules/`. They give the AI agent the same mental model senior contributors have — including the non-obvious bits (FinePDFs feature parity, `pdfsys-core` zero-dep rule, Gradio UI/logic separation) that a new collaborator would otherwise step on.

### Quick start

```bash
# One-shot bootstrap: checks python/uv, syncs workspace, downloads router weights.
bash scripts/setup_cursor.sh
```

Then open the repo in Cursor (≥ 0.50, which supports `.cursor/rules/*.mdc`). The always-on rules activate immediately; file-specific rules attach as you open matching files.

### Active rules

| Rule | Scope | What it enforces |
|------|-------|------------------|
| `00-project-context.mdc` | always | Project goals, tech stack, must-read docs, explicit non-goals. |
| `01-architecture-invariants.mdc` | always | 7 load-bearing invariants (zero-dep core, stateless processing, normalized bbox, etc.). |
| `02-commit-workflow.mdc` | always | Conventional commits with package-scoped names; pre-commit checklist. |
| `03-doc-sync.mdc` | always | Doc-sync mapping table: which code change forces which doc update. Cursor proactively scans after edits. |
| `10-python-standards.mdc` | `**/*.py` | Type hints, frozen dataclass, lazy imports for heavy deps. |
| `20-core-contracts.mdc` | `packages/pdfsys-core/**` | Zero external deps; no I/O; schema change ripple rules. |
| `21-router-parity.mdc` | `packages/pdfsys-router/**` | FinePDFs 124-feature parity is sacred; how to verify. |
| `22-parser-backends.mdc` | `packages/pdfsys-parser-*/**` | All three backends must emit identical `ExtractedDoc`. |
| `23-bench-scorer.mdc` | `packages/pdfsys-bench/**` | torch/transformers lazy load; bf16 default; loop never raises. |
| `30-gradio-demo.mdc` | `demo/**,app.py` | UI layer has no business logic; callbacks never raise; lazy singletons. |

### Recommended Cursor workflow

1. **Before touching `pdfsys-core`** — read `20-core-contracts.mdc`. The AI will refuse to add third-party deps here and surface schema-ripple questions.
2. **Before touching `feature_extractor.py`** — `21-router-parity.mdc` kicks in; the AI will suggest running the parity check before you commit.
3. **When building a new parser backend** — `22-parser-backends.mdc` walks through the 6-step addition procedure and refuses partial impls.
4. **When writing demo UI** — `30-gradio-demo.mdc` rejects `import pymupdf` in `demo/app.py` (belongs in `demo/pipeline.py`).

### Authoring new rules

Rules live in `.cursor/rules/*.mdc`. Format:

```yaml
---
description: Short description shown in the rule picker
globs: packages/<pkg>/**/*.py    # omit for always-on rules
alwaysApply: false                # true = always loaded
---

# Rule Title

- Bullet rule 1 (with ✅/❌ example)
- Bullet rule 2
```

Keep each rule under 100 lines, one concern per file. See existing rules for patterns.

## License

Apache-2.0
