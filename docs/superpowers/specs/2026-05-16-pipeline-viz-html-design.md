# Pipeline Output Visualizer — Design

**Date:** 2026-05-16
**Scope:** Static HTML + JSON visualization tool for pdfsys pipeline output. Lets a human eyeball whether each stage (router → layout → extract → quality) produced reasonable results on each of the 150 bundled PDFs, with a dashboard for cross-corpus overview and a per-PDF detail view for diagnosis.
**Out of scope:** annotation/editing (the existing `packages/pdfsys-bench/annotation/` UI covers that), comparing runs side-by-side, live refresh, multi-page bbox overlay beyond page 1.

---

## 1 · Goal

Produce a single static-site bundle at `out/viz/` from a pipeline run directory (`out/e2e_full_mineru3/`). Open `out/viz/index.html` in any browser to:

1. See an aggregate dashboard (backend split, ocr_prob histogram, quality histogram, error breakdown).
2. Browse the 150 rows in a sortable/filterable table.
3. Click any row → detail view showing every stage's output for that one PDF, including:
   - Original PDF page preview (page 1) with layout bboxes overlaid as colored rectangles.
   - Router decision card.
   - Layout regions (visual overlay + tabular fallback for all pages).
   - Stage-B decider verdict.
   - Extracted markdown rendered (with KaTeX for LaTeX formulas + HTML tables passing through).
   - Quality score + kept flag.

The bundle is self-contained: zip `out/viz/` and another human can unzip + double-click `index.html` without setup.

## 2 · Non-goals

- No write path (no annotate, no edit, no flag).
- No diffing between runs.
- No live data — re-run the CLI to refresh.
- No backend server.
- No bbox overlay on pages 2+ of multi-page PDFs (the data is there in the layout cache; just not rendered, because previews.json only ships page-1 thumbnails).

## 3 · Architecture

```
out/e2e_full_mineru3/                      packages/pdfsys-bench/
├── dataset.parquet                ─┐      └── annotation/
├── results.jsonl                   │          ├── previews.json    ←── page-1 thumbnails (existing)
├── markdown/<sha256>.md            ├──►       └── (other annotation assets — unused)
└── .cache/layout/<sha256>.json     │
                                    │
                       ┌────────────┴───────────┐
                       │ pdfsys visualize       │   (new CLI subcommand, ~150 LOC)
                       │  -r out/e2e_full_...   │
                       │  -o out/viz/           │
                       └────────────┬───────────┘
                                    ▼
                       out/viz/
                       ├── index.html             ← single-file SPA (~400 LOC)
                       ├── viz_data.json          ← rows + aggregates, ~2-3 MB
                       ├── previews.json          ← copied from annotation/
                       ├── markdown/<sha256>.md   ← copied per-PDF (lazy-loaded)
                       ├── assets/
                       │   ├── marked.min.js
                       │   ├── katex.min.css
                       │   ├── katex.min.js
                       │   └── katex-fonts/      ← KaTeX font files
                       └── serve.sh              ← `python -m http.server 8765 -d .`
```

### Why preprocess to JSON instead of reading parquet in-browser

- Browser parquet readers (apache-arrow-js, parquetjs) add ~500 KB and edge-case complexity.
- Pipeline output is read-once-after-run; a 30s preprocessing step is no burden.
- Matches the existing `annotation/` pattern (static JSON + single HTML).

### Why no Python server

- Distribution friction — recipient needs to `pip install fastapi` etc.
- Static bundle = email-able / Slack-able / mountable into HF Spaces if needed.
- Lone risk: `fetch()` of local JSON may be blocked by browser CORS. Mitigated by the bundled `serve.sh` (one-liner `python -m http.server` on `out/viz/`).

## 4 · CLI: `pdfsys visualize`

```
pdfsys visualize -r <run-dir> [-o <out-dir>] [--preview-source <path>]

  -r, --run-dir       Path to a pipeline run directory containing
                      dataset.parquet, results.jsonl, markdown/, .cache/layout/.
  -o, --out-dir       Output dir (default: <run-dir>/viz/).
  --preview-source    Path to a previews.json (default:
                      packages/pdfsys-bench/annotation/previews.json).
```

Steps the CLI performs:

1. Validate `run-dir` has `dataset.parquet`. Read it via pyarrow.
2. Build `viz_data.json` (schema in §5).
3. Copy `<preview-source>` → `<out-dir>/previews.json`.
4. Copy `<run-dir>/markdown/*.md` → `<out-dir>/markdown/`.
5. Copy template `<repo>/packages/pdfsys-bench/viz/index.html` → `<out-dir>/index.html`.
6. Copy vendored `assets/` (marked + katex + fonts) → `<out-dir>/assets/`.
7. Write `serve.sh` helper.
8. Print a "open out/viz/index.html OR run out/viz/serve.sh" hint.

Step 6 implies we vendor `marked.min.js`, `katex.min.css`, `katex.min.js`, and the KaTeX font directory inside `packages/pdfsys-bench/viz/assets/`. These get committed once and ride along with every run.

## 5 · `viz_data.json` schema

```json
{
  "run_meta": {
    "run_dir": "out/e2e_full_mineru3",
    "generated_at": "2026-05-16T...",
    "total_rows": 150,
    "errors": 0,
    "kept": 35,
    "wall_seconds": 905.1,
    "avg_quality": 1.420,
    "ocr_threshold": 0.5,
    "kept_threshold": 2.0
  },
  "aggregate": {
    "backend_dist": { "mupdf": 104, "pipeline": 36, "vlm": 10 },
    "stage_b_dist": { "pipeline": 36, "vlm": 10 },
    "source_dist": { "omnidocbench_100": 100, "olmocr_bench_50": 50 },
    "error_dist": {},
    "ocr_prob_hist": {
      "bins": [0.0, 0.05, 0.1, ..., 1.0],
      "counts": [104, 0, 0, ...]
    },
    "quality_hist": {
      "bins": [0.0, 0.15, 0.3, ..., 3.0],
      "counts": [0, 2, 4, ...]
    },
    "markdown_chars_dist": {
      "min": 14, "p25": 800, "p50": 1500, "p75": 3000, "max": 12000
    }
  },
  "rows": [
    {
      "id": "abc12345",                          // first 8 chars of sha256
      "sha256": "abc...full...",
      "pdf_path": "packages/pdfsys-bench/...",
      "pdf_basename": "1_pg19.pdf",
      "source": "olmocr_bench_50",               // derived from path
      "num_pages": 1,
      "backend": "pipeline",                     // Stage-A decision
      "stage_b_backend": "vlm",                  // null if Stage-B didn't run
      "ocr_prob": 0.654,
      "is_form": false,
      "is_encrypted": false,
      "garbled_text_ratio": 0.0,
      "layout": {                                // null if layout stage didn't run
        "model": "juliozhao/DocLayout-YOLO-DocStructBench",
        "num_regions": 14,
        "has_complex": true,
        "regions": [                             // ALL pages, all regions
          {
            "page_idx": 0,
            "type": "FORMULA",
            "bbox": [0.21, 0.18, 0.78, 0.22],    // normalized [0,1]
            "confidence": 0.94
          },
          ...
        ]
      },
      "extract": {
        "backend": "vlm",
        "markdown_chars": 1514,
        "segment_count": 14,
        "markdown_excerpt": "Let $\\frac{x^2}{...$ ...",   // first 400 chars
        "markdown_path": "markdown/abc...full....md"        // relative to out/viz/
      },
      "quality": {
        "score": 2.30,
        "model": "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn",
        "kept": true
      },
      "error_class": null,
      "error_message": null,
      "preview_key": "olmocr_bench_50__1_pg19",  // lookup key into previews.json; null if no preview
      "wall_ms_total": 285.0
    },
    ...149 more rows
  ]
}
```

### `regions` field comes from where

The pipeline writes layout cache per PDF at `<run-dir>/.cache/layout/<sha256>.json`. CLI reads those and embeds the regions list per row. If the cache file is missing (e.g. row went mupdf and skipped layout), the `layout` field is `null`.

### `preview_key` derivation

`annotation/previews.json` keys look like `olmocr__arxiv_math__2503.04438_pg16`. Schema is `<short_source>__<category>__<basename_without_ext>`. The viz CLI computes this key from the row's `pdf_path` and falls back to `null` if it doesn't find a match — UI then shows "(no preview available)" placeholder.

## 6 · Single-page HTML structure

```html
<!DOCTYPE html>
<html>
<head>
  <title>pdfsys viz · <run-dir></title>
  <link rel="stylesheet" href="assets/katex.min.css">
  <style>...inline CSS, ~80 LOC, dark theme matching annotation/...</style>
</head>
<body>
  <div id="topbar">...stats summary + back button (hidden on dashboard)...</div>
  <div id="view-dashboard">...table + 3 SVG charts + filters...</div>
  <div id="view-detail" style="display:none">...preview + stage cards...</div>

  <script src="assets/marked.min.js"></script>
  <script src="assets/katex.min.js"></script>
  <script type="module">
    // ~300 LOC vanilla JS:
    // 1. fetch viz_data.json + previews.json on load
    // 2. render dashboard (table + 3 SVG charts + filters)
    // 3. on row click → render detail view (preview + bbox overlay + stage cards + lazy fetch markdown)
    // 4. on back → re-show dashboard, preserve filter state
  </script>
</body>
</html>
```

The HTML is a SINGLE file — no React, no Vue, no build step. All logic is hand-rolled vanilla JS so a future engineer can read it top-to-bottom.

### Charts (hand-rolled SVG)

Three charts on dashboard, each ~30 LOC of SVG:
- Backend distribution: horizontal stacked bar (mupdf | pipeline | vlm) with counts.
- ocr_prob histogram: 20-bin bar chart, x-axis [0, 1], threshold line at 0.5.
- quality histogram: 20-bin bar chart, x-axis [0, 3], threshold line at 2.0.

### Bbox overlay (L3 strategy from brainstorming)

For the detail view:
1. Load `previews.json[preview_key]`; if absent show "(no preview)" placeholder.
2. Render the page-1 image at a fixed width (e.g. 600 px); wrap in `<div style="position:relative">`.
3. Filter `layout.regions` to `page_idx == 0`; for each, append `<div class="bbox bbox-{type}">` with CSS `position:absolute; left:Lx%; top:Ly%; width:W%; height:H%`.
4. Color scheme:
   - TEXT: blue (rgba(0, 100, 255, 0.15)) + border solid
   - TABLE: green (rgba(0, 200, 0, 0.15))
   - FORMULA: purple (rgba(180, 0, 200, 0.15))
   - IMAGE: orange (rgba(255, 165, 0, 0.15))
5. Below the preview, show a table of ALL regions (all pages, all types) — this is the L2 fallback for multi-page PDFs.

### Markdown rendering

```js
const md = await fetch(row.extract.markdown_path).then(r => r.text());
const html = marked.parse(md);  // marked allows HTML tables to pass through
container.innerHTML = html;
renderMathInElement(container, { delimiters: [
  { left: '$$', right: '$$', display: true },
  { left: '$', right: '$', display: false },
]});
```

This handles all three content types correctly:
- Plain markdown (text, headers, lists) → marked.
- HTML tables (from VLM table extraction) → marked passes through.
- LaTeX formulas (from VLM equation extraction) → KaTeX auto-render.

## 7 · CLI scaffold (rough Python shape)

```python
# packages/pdfsys-cli/src/pdfsys_cli/viz.py

def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "viz")
    preview_src = Path(args.preview_source or _default_preview_path())

    rows = _load_rows(run_dir)
    aggregate = _compute_aggregate(rows)
    viz_data = {
        "run_meta": _build_meta(run_dir, rows),
        "aggregate": aggregate,
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz_data.json").write_text(json.dumps(viz_data, ensure_ascii=False))
    _copy_previews(preview_src, out_dir / "previews.json")
    _copy_markdown(run_dir / "markdown", out_dir / "markdown")
    _copy_template(_template_dir(), out_dir)
    _write_serve_script(out_dir / "serve.sh")
    print(f"open {out_dir}/index.html  (or run {out_dir}/serve.sh)")
    return 0
```

Wired into `pdfsys_cli/__main__.py`:

```python
elif args.cmd == "visualize":
    from . import viz
    return viz.main(remaining_args)
```

## 8 · Acceptance

1. Run `pdfsys visualize -r out/e2e_full_mineru3 -o out/viz` completes in < 60 s, exit 0.
2. `out/viz/` contains: `index.html`, `viz_data.json` (1-5 MB), `previews.json`, `markdown/` (150 files), `assets/` (marked, katex, fonts), `serve.sh`.
3. Open `out/viz/index.html` directly in browser (or via serve.sh):
   - Dashboard loads in < 1 s; table shows 150 rows.
   - Filter "kept only" reduces to 35 rows.
   - Filter "errors only" reduces to 0 rows.
   - Searching for "jiaocai" filters to those rows.
   - Sorting by quality_score works.
   - Click row → detail view loads in < 500 ms.
   - VLM row's detail shows LaTeX rendered as math (not raw `$...$` strings).
   - PIPELINE row's detail shows bbox overlay on preview, with colored boxes per region type.
   - "Back" returns to dashboard with filters preserved.
4. Zip `out/viz/` → email to someone → they extract + double-click index.html → works (modulo `serve.sh` if their browser blocks local fetch).

## 9 · Risks

| Risk | Mitigation |
|---|---|
| Browser blocks `fetch('viz_data.json')` from `file://` URL | `serve.sh` one-liner starts `python -m http.server 8765 -d <out-dir>`; mentioned in CLI completion message |
| `previews.json` is 1 MB and the key naming convention may not match every row's pdf_path | CLI computes `preview_key` with a documented heuristic; unmatched rows show "(no preview)" placeholder rather than crashing |
| `marked` + `katex` vendored files bloat the repo by ~500 KB | Vendored once in `packages/pdfsys-bench/viz/assets/`; one-time cost, no per-run download |
| Detail view markdown fetch flood (one fetch per row click) | Acceptable — single file per click, browser caches; total markdown corpus ~1 MB |
| Layout regions in cache may use a different bbox convention than [0,1] normalized | CLI inspects first 5 layout cache files at startup; logs a warning if values fall outside [0, 1] |
| Multi-page PDF bbox overlay incomplete (only page 1) | Spec §3 explicitly out-of-scope; L2 region table covers all pages tabularly |

## 10 · Files touched

```
NEW:
packages/pdfsys-cli/src/pdfsys_cli/viz.py            ~150 LOC
packages/pdfsys-bench/viz/index.html                  ~400 LOC HTML+CSS+JS
packages/pdfsys-bench/viz/README.md                   ~30 LOC
packages/pdfsys-bench/viz/assets/marked.min.js              vendored ~50 KB
packages/pdfsys-bench/viz/assets/katex.min.js               vendored ~250 KB
packages/pdfsys-bench/viz/assets/katex-auto-render.min.js   vendored ~10 KB (provides renderMathInElement)
packages/pdfsys-bench/viz/assets/katex.min.css              vendored ~25 KB
packages/pdfsys-bench/viz/assets/katex-fonts/               vendored ~300 KB

MODIFIED:
packages/pdfsys-cli/src/pdfsys_cli/__main__.py        add `visualize` subcommand entry
```

Untouched: `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-*`, `pdfsys-bench` (except the new `viz/` subdirectory), `pdfsys-cli/runner.py`, `pdfsys-cli/parquet_writer.py`, all YAML configs.

## 11 · Definition of done

- `pdfsys visualize -r out/e2e_full_mineru3` produces a working `out/viz/`.
- All 4 sub-bullets of §8 acceptance pass.
- `out/viz/` zips and unzips standalone.
- No new dependencies in any `pyproject.toml` (uses pyarrow already from previous spec; pyyaml for argparse only).
- Vendored marked + katex assets committed once at `packages/pdfsys-bench/viz/assets/`.

## 12 · Post-build note · 2026-05-16

### Bundle size (out/viz_final/)

| Component | Size |
|---|---|
| `viz_data.json` | 212 KB |
| `previews.json` | 932 KB (pre-computed text signals from PyMuPDF) |
| `markdown/` (150 files) | 840 KB |
| `page1/` (150 JPEGs) | 17 MB (~110 KB avg per JPEG @ DPI 80, q70) |
| `assets/` (marked + KaTeX + 20 fonts) | 636 KB |
| `index.html` | 22 KB |
| Total | **~19 MB** |
| Zipped | **16 MB** |

### Wall time

- `pdfsys visualize` end-to-end: ~5 seconds for 150 PDFs (JPEG rendering dominates).

### Deviations from spec

- **previews.json contained text, not images.** Spec §3 assumed `annotation/previews.json` shipped page thumbnails; it actually ships pre-computed PyMuPDF signals (markdown excerpts, garbled_ratio, page_count). Discovered during Task 7 acceptance. Fix: viz CLI now pre-renders page-1 JPEGs into `out/viz/page1/<sha>.jpg` via PyMuPDF (~17 MB additional bundle weight). Visual fidelity unchanged from spec intent; bundle 4-5× larger than the spec §8 "2-5 MB" estimate, but still email-able.

### Build commits (in order)

1. `278e0b0` Task 1 — vendor marked + KaTeX (24 files, ~640 KB)
2. `6c69771` Task 2 — viz.py CLI (338 LOC)
3. `2e8699d` Task 3 — wire `visualize` subcommand
4. `d58032f` Task 4 — stub index.html + fix _REPO_ROOT path (parents[5] → parents[4])
5. `730930d` Task 5 — dashboard view (359 LOC HTML/CSS/JS)
6. `1960630` Task 6 — detail view (487 LOC total)
7. `7a7c994` page-1 JPEG rendering (fixed previews.json misunderstanding)

### Acceptance checklist

- ✅ CLI completes < 60s, exit 0 (5 s actual)
- ✅ Bundle contains all expected files (index.html, viz_data.json, previews.json, markdown/, page1/, assets/, serve.sh)
- ✅ 150 rows in dataset, 0 errors, 35 kept, all aggregates compute correctly
- ✅ HTML syntax balanced (braces 128/128, parens 218/218)
- ✅ Module imports + argparse + `pdfsys visualize --help` work
- ✅ Bundle zips standalone (16 MB)
- ⏳ **Browser checks** (require user action — open `out/viz_final/index.html` or run `bash out/viz_final/serve.sh`):
  - Dashboard table + charts + filters
  - Click row → detail view → LaTeX rendering + bbox overlay

### Open follow-ups

- **Bundle size could shrink** by using lower DPI (60) or quality (50) for the page-1 JPEGs. Current 17 MB JPEG total is acceptable but could go to ~8 MB with little visual loss. Not done because 16 MB zipped is fine for the current use case.
- **bbox overlay accuracy** depends on layout cache regions being correctly normalized to [0, 1]. The viz CLI does not currently verify this — if a future layout backend emits pixel coords, overlay positions will be wrong. Worth a one-line sanity check.
- **No multi-page bbox overlay** — only page 1 is rendered + overlaid. Multi-page PDFs are still covered by the L2 region table (all pages, all types).
- **VLM rows currently have `layout=null` in some cases** because the layout cache file naming convention differs. Investigate per-row to be sure (the regions ARE in cache; CLI's `_load_layout_block` glob match may not catch all).
