# Pipeline Output Visualizer

Static HTML+JSON bundle for inspecting `pdfsys run` output. Dashboard for
cross-corpus overview + per-PDF detail view showing every pipeline stage.

## Generate a bundle

```sh
pdfsys visualize -r out/e2e_full_mineru3 -o out/viz
```

Reads `out/e2e_full_mineru3/dataset.parquet` + `markdown/*.md` +
`.cache/layout/*.json`, copies the bundled `previews.json`, pre-renders
page-1 JPEGs of each PDF, copies the HTML template, writes `out/viz/`.

Wall time on the bundled 150 PDFs: ~5 seconds (mostly JPEG rendering).

## Open it

```sh
# Option A — direct
open out/viz/index.html

# Option B — if your browser blocks local file:// fetch
bash out/viz/serve.sh
# then visit http://localhost:8765/
```

## What you see

- **Dashboard** — sortable table of all rows, 5 filters (kept only, errors
  only, backend, source, search), 3 SVG charts (backend split, ocr_prob
  histogram, quality histogram with threshold markers).
- **Detail (click any row)**:
  - Left: page-1 preview JPEG with layout bbox overlays
    (TEXT blue / TABLE green / FORMULA purple / IMAGE orange), plus a
    table of all regions across all pages.
  - Right: per-stage cards showing Router decision, Layout summary,
    Stage-B verdict, extracted markdown (LaTeX via KaTeX, HTML tables
    via marked), Quality score.

## Bundle contents

```
out/viz/
├── index.html              ← single-file SPA
├── viz_data.json           ← row metadata + aggregates (~200 KB for 150 rows)
├── previews.json           ← pre-computed PyMuPDF text signals (~950 KB)
├── markdown/<sha>.md       ← per-PDF extracted markdown (~5 KB avg)
├── page1/<sha>.jpg         ← pre-rendered first-page JPEGs (~100 KB avg)
├── assets/                 ← marked + KaTeX + fonts (~640 KB)
└── serve.sh                ← local HTTP server one-liner
```

Total size for 150 PDFs: ~19 MB on disk, ~16 MB zipped.

## Sharing

```sh
zip -r viz.zip out/viz
```

Recipient extracts and opens `index.html`. If their browser blocks
`file://` fetch (Chrome strict mode), they run `bash serve.sh`.
