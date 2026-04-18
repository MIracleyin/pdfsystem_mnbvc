# Architecture

PB-scale PDF → pretraining data. Dual-path: CPU text extraction (90%) + GPU OCR/VLM (10%).

```
PDF ──► Stage-A Router (XGBoost, CPU, ≤10ms)
            │
     text-ok │                    needs-ocr
            ▼                          ▼
     parser-mupdf              Layout Analyser (DocLayout-YOLO)
     (CPU, 10-30 PDF/s)              │
            │               Stage-B Decider
            │                    │          │
            │             simple │          │ complex (TABLE/FORMULA)
            │                    ▼          ▼
            │           parser-pipeline  parser-vlm
            │           (RapidOCR, CPU)  (MinerU, GPU)
            │                    │          │
            └────────────────────┴──────────┘
                                 │
                          Quality Scorer (ModernBERT)
                                 │
                          JSONL + Markdown output
```

## Packages

| Package | Layer | Role |
|---------|-------|------|
| `pdfsys-core` | **Foundation** | Types, enums, dataclasses, LayoutCache, serde. Zero external deps. |
| `pdfsys-router` | Processing | XGBoost PDF classifier (124 PyMuPDF features) + Stage-B decider |
| `pdfsys-layout-analyser` | Processing | DocLayout-YOLO region detection → LayoutDocument |
| `pdfsys-parser-mupdf` | Processing | Text-ok fast path: PyMuPDF blocks → Markdown |
| `pdfsys-parser-pipeline` | Processing | Region-level OCR: LayoutDocument → RapidOCR → Markdown |
| `pdfsys-parser-vlm` | Processing | Complex pages: MinerU 2.5 Pro end-to-end extraction |
| `pdfsys-bench` | Evaluation | Quality scorer (ModernBERT) + benchmark datasets |
| `pdfsys-cli` | Orchestration | YAML config + stage-aware pipeline runner |

## Layer Rules

See `docs/architecture/LAYERS.md` for the full dependency matrix and enforcement.

## Key Design Decisions

1. **Stateless processing** — no manifest, no central DB. Every PDF → self-contained output.
2. **Content-addressable cache** — LayoutCache keyed by `sha256 + model_tag`.
3. **Atomic writes** — `tmp + os.replace()` for crash safety.
4. **Backend-agnostic output** — all parsers emit the same `ExtractedDoc` / `Segment` schema.
5. **Lazy heavy deps** — torch, transformers, magic-pdf imported only when needed.

## Storage Layers (Production)

- **L0 (cold):** Raw PDFs, S3/OSS/MinIO, PB-scale, immutable
- **L1 (warm):** Intermediate Parquet/JSONL, disposable and rebuildable
- **L2 (hot):** Final Parquet dataset, partitioned by lang/source/quality
