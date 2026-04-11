# pdfsys-mnbvc

PB-scale PDF → pretraining-data pipeline for the MNBVC corpus project.
FinePDFs-inspired architecture adapted for Chinese-heavy, mixed-quality input.

## Architecture

Two-stage routing, cascaded:

```
          ┌──────────────┐
PDF  ─►   │ pdfsys-router│  stage A (cheap classifier)
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

The `LayoutDocument` produced by `pdfsys-layout-analyser` is cached to disk
and consumed by **both** the stage-B decision in `pdfsys-router` **and** the
downstream parser backend — layout inference runs at most once per PDF.

## Workspace packages

| Package | Role |
|---|---|
| `pdfsys-core` | Shared dataclasses (`PdfRecord`, `LayoutDocument`), manifest IO, layout cache. No PDF/ML deps. |
| `pdfsys-router` | Two-stage router. Stage A text-ok/needs-ocr; Stage B pipeline/vlm from cached layout. |
| `pdfsys-layout-analyser` | Page layout model runner (PP-DocLayoutV3 / docling-layout-heron). Runs once, writes cache. |
| `pdfsys-parser-mupdf` | Text-ok backend. PyMuPDF + reading order → Markdown. |
| `pdfsys-parser-pipeline` | Needs-ocr + simple layout backend. Region-level OCR (RapidOCR / PaddleOCR-classic). |
| `pdfsys-parser-vlm` | Needs-ocr + complex layout backend. MinerU 2.5 / PaddleOCR-VL on complex regions. |
| `pdfsys-bench` | Cross-backend throughput / latency / F1 evaluation. |

## Setup (macOS)

```bash
# Requires uv >= 0.4
uv sync
```

Running a single PDF through the pipeline, and orchestration above the
extraction core (ingest / dedup / quality / tokenize) are not implemented
yet — see `docs/PRD.md` for the full design.

## Docs

- `docs/PRD.md` — full PRD with resource budgets and roadmap.

## License

Apache-2.0
