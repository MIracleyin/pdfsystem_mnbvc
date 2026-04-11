# pdfsys-mnbvc

PB-scale PDF → pretraining-data pipeline for the [MNBVC](https://github.com/esbatmop/MNBVC) corpus project.
FinePDFs-inspired architecture adapted for Chinese-heavy, mixed-quality input.

## Current status: MVP closed loop ✅

The first end-to-end path — **Router → MuPDF parser → OCR quality scorer** — is working on the OmniDocBench-100 evaluation set. PDFs that need OCR are routed to `PIPELINE` but not yet extracted (that backend is not implemented yet).

## Quick start

```bash
# 1. Install uv (>= 0.4)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo and sync all workspace packages
git clone <this-repo-url>
cd pdfsystem_mnbvc
uv sync

# 3. Fetch the XGBoost router weights (257 KB, one-time)
python -m pdfsys_router.download_weights

# 4. Run the MVP closed loop on the bench dataset
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

## Docs

- `docs/PRD.md` — full PRD with resource budgets and roadmap.

## License

Apache-2.0
