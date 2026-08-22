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
            │         (mineru pipeline) (mineru vlm-<engine>)
            │         └── both: out-of-process mineru-api subprocess + HTTP ──┘
            │                    │          │
            └────────────────────┴──────────┘
                                 │
                          Quality Scorer (ModernBERT)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   L1  results.jsonl + dataset.parquet    L2  pdfsys dataset
       (run telemetry: routing probs,         (pdfsys.page/v2 — one row per
        stage timings, error class)            PAGE, keyed (doc_id, page_index))
                                                        │
                                              pdfsys dataset-validate
                                              (format contract; must pass
                                               before anything is published)
                                                        │
                                              pdfsys mnbvc-export
                                              (→ MNBVC mmDataBlock parquet)
```

The split at the bottom is the one that matters: **L1 is telemetry, L2 is the
product**. L1 answers "did the run behave?" — one flat row per PDF. L2 answers
"what do I train on?" — one row per page, with the image/text interleaving
encoded inline in the page text so the model-derived block structure stays a
droppable column.

## Packages

| Package | Layer | Role |
|---------|-------|------|
| `pdfsys-core` | **Foundation** | Types, enums, dataclasses, LayoutCache, serde. Zero external deps. |
| `pdfsys-router` | Processing | XGBoost PDF classifier (124 PyMuPDF features) + Stage-B decider |
| `pdfsys-layout-analyser` | Processing | DocLayout-YOLO region detection → LayoutDocument |
| `pdfsys-parser-mupdf` | Processing | Text-ok fast path: PyMuPDF blocks → Markdown |
| `pdfsys-parser-pipeline` | Processing | OCR pipeline: out-of-process HTTP client to `mineru-api` (pipeline mode) → Markdown |
| `pdfsys-parser-vlm` | Processing | Complex pages: out-of-process HTTP client to `mineru-api` (vlm-`<engine>`: transformers/mlx/vllm) |
| `pdfsys-bench` | Evaluation | Quality scorer (ModernBERT) + benchmark datasets |
| `pdfsys-cli` | Orchestration | YAML config + stage-aware pipeline runner |

## Layer Rules

See `docs/architecture/LAYERS.md` for the full dependency matrix and enforcement.

## Key Design Decisions

1. **Stateless processing** — no manifest, no central DB. Every PDF → self-contained output.
2. **Content-addressable cache** — LayoutCache keyed by `sha256 + model_tag`.
3. **Atomic writes** — `tmp + os.replace()` for crash safety.
4. **Backend-agnostic output** — all parsers emit the same `ExtractedDoc` / `Segment` schema.
5. **Lazy heavy deps** — torch, transformers, mineru imported only when needed.
6. **Out-of-process ML boundary** — `parser-pipeline`, `parser-vlm`, and the quality scorer never import mineru/torch in the host process. Each spawns a dedicated subprocess (`mineru-api` for parsers, `_quality_server` for the scorer) and talks to it over HTTP. This sidesteps the macOS spawn-pool / MPS-vs-MLX deadlocks hit during the mineru migration and is the natural seam for the planned component-versioning split. See `docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md §15`.

## Storage Layers (Production)

- **L0 (cold):** Raw PDFs, S3/OSS/MinIO, PB-scale, immutable
- **L1 (warm):** Intermediate Parquet/JSONL, disposable and rebuildable. `dataset.parquet` (`pdfsys_cli.parquet_writer`) is run telemetry — one flat row per PDF with routing probabilities, stage timings, error class.
- **L2 (hot):** Final Parquet dataset, partitioned by lang/source/quality. Format is `pdfsys.page/v2` (`pdfsys_cli.dataset_writer`): **one row per page**, keyed `(doc_id, page_index)` — an identity the PDF gives us, not one a layout model invents. The page's `text` carries the image interleaving inline as `![](img://<sha256>)`, so the model-derived `blocks` column is droppable enrichment. Image bytes live in content-addressed side tables (`images` for crops, optional `page_images` for full-page rasters). Plain-text, interleaved (OBELICS-shaped), image-text-pair and (page-image, page-text) views are all projections. See `docs/superpowers/specs/2026-08-22-page-level-parquet-dataset-design.md` and `docs/schema/doc_dataset.v2.json`.
