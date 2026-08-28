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
       (run telemetry: routing probs,         ├─ --from-mineru <run dir>
        stage timings, error class)           │    pipeline / vlm sidecars
                                              └─ --from-pdf-dir <pdf dir>
                                                   mupdf lane, re-extracted
                                                        │
                                              pdfsys.page/v2 — one row per
                                              PAGE, keyed (doc_id, page_index)
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

**Why L2 has two entrances.** The MinerU lanes leave `content_list.json` behind,
which is already a reading-order interleaved list — packaging it is a pure
re-encoding. The mupdf lane leaves nothing comparable: a run writes one merged
`markdown/<sha256>.md` with no page boundaries, and `segments_excerpt` in
`results.jsonl` is filled only on the VLM branch and truncated to 200
characters, so it is a visualisation artifact rather than a data source.
`--from-pdf-dir` therefore re-extracts from the PDF. At ~10 ms/page that is
cheaper than designing, writing and versioning a format to persist something
mupdf can recompute on demand. It also means the two entrances differ in what
pixels they can hold: MinerU crops figures, mupdf never rasterises anything, so
its only image mode is whole-page renders (`--images pages`, the default there).

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

### MinerU backends we use, and one we don't

Both parser packages pin `mineru>=3.1,<4.0` and speak to `mineru-api` over
HTTP with `backend=pipeline` or `backend=vlm-<engine>`. MinerU 3.3.1 added a
third family — `hybrid-engine` / `hybrid-http-client` — together with an
`effort` parameter (`medium` / `high`) whose release notes claim 35–220%
faster parsing at `medium`.

**`effort` does nothing for us.** MinerU's own API documents it as *"(Adapted
only for hybrid backend)"*, so sending it alongside `backend=pipeline` or
`backend=vlm-*` is a no-op. Reaching that speedup means adopting the hybrid
backend, which is a different accuracy profile and a benchmark run, not a
parameter to pass. Recorded here because the release notes read like a free
win and it is easy to draw that conclusion twice.

Two upstream caps are worth knowing rather than fighting: `mineru` requires
`transformers<5.0.0` (which is currently what keeps the ModernBERT scorer
away from the transformers 5 break), and `mineru[mlx]` requires
`mlx-vlm<0.4`. Both are upstream's judgement, not ours to duplicate.

**MinerU 4.0 removes the VLM backend** (`4.0.0a1`, replaced by an
ImagePayloadCache/doclib architecture). `pdfsys-parser-vlm` is built on it, so
the `<4.0` upper bound is one of the few load-bearing ones in this repo —
see `docs/ROADMAP.md §3.4` for when an upper bound earns its keep.

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
