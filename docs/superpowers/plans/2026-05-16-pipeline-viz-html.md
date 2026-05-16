# Pipeline Output Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pdfsys visualize` CLI that produces a self-contained `out/viz/` HTML+JSON bundle for inspecting pipeline output from any run directory. Dashboard for cross-corpus overview + detail view for per-PDF stage diagnosis.

**Architecture:** New CLI subcommand reads parquet + markdown + layout cache + reuses existing `annotation/previews.json`, then writes a single `viz_data.json` consumed by a hand-rolled vanilla-JS SPA (`index.html`). No backend server. KaTeX + marked vendored once. Total new code ≈ 150 LOC Python + 400 LOC HTML/CSS/JS.

**Tech Stack:** Python 3.11 + pyarrow (already a workspace dep), vanilla JS / SVG / CSS, vendored marked + KaTeX from jsdelivr CDN.

**Source spec:** `docs/superpowers/specs/2026-05-16-pipeline-viz-html-design.md` — read first for §5 schema and §6 UI structure.

**User constraint:** No unit tests, consistent with previous iterations. Verification is by running the CLI + opening the resulting `index.html` in a browser + checking acceptance criteria §8 of the spec.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `packages/pdfsys-bench/viz/assets/marked.min.js` | create (vendor) | Markdown renderer, ~50 KB |
| `packages/pdfsys-bench/viz/assets/katex.min.js` | create (vendor) | LaTeX math renderer, ~280 KB |
| `packages/pdfsys-bench/viz/assets/katex-auto-render.min.js` | create (vendor) | Auto-find `$...$` in DOM, ~10 KB |
| `packages/pdfsys-bench/viz/assets/katex.min.css` | create (vendor) | KaTeX styling, ~24 KB |
| `packages/pdfsys-bench/viz/assets/katex-fonts/*` | create (vendor) | KaTeX font files (.woff2), ~300 KB |
| `packages/pdfsys-bench/viz/index.html` | create | Single-page SPA: dashboard + detail view |
| `packages/pdfsys-bench/viz/README.md` | create | Usage docs |
| `packages/pdfsys-cli/src/pdfsys_cli/viz.py` | create | CLI: load run, build viz_data.json, copy assets, emit serve.sh |
| `packages/pdfsys-cli/src/pdfsys_cli/__main__.py` | modify | Register `visualize` subcommand |

No changes to runner, ParquetSink, configs, or other workspace packages.

---

## Task 1: Vendor marked + KaTeX assets

**Files:**
- Create: `packages/pdfsys-bench/viz/assets/marked.min.js`
- Create: `packages/pdfsys-bench/viz/assets/katex.min.js`
- Create: `packages/pdfsys-bench/viz/assets/katex-auto-render.min.js`
- Create: `packages/pdfsys-bench/viz/assets/katex.min.css`
- Create: `packages/pdfsys-bench/viz/assets/katex-fonts/*.woff2` (10-15 font files)

- [ ] **Step 1: Create the directory**

```bash
mkdir -p packages/pdfsys-bench/viz/assets/katex-fonts
```

- [ ] **Step 2: Download marked (markdown renderer)**

```bash
curl -fsSL -o packages/pdfsys-bench/viz/assets/marked.min.js \
  https://cdn.jsdelivr.net/npm/marked@13.0.3/marked.min.js
ls -la packages/pdfsys-bench/viz/assets/marked.min.js
```

Expected: ~40-50 KB file.

- [ ] **Step 3: Download KaTeX bundle (JS + CSS)**

```bash
curl -fsSL -o packages/pdfsys-bench/viz/assets/katex.min.js \
  https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js
curl -fsSL -o packages/pdfsys-bench/viz/assets/katex-auto-render.min.js \
  https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js
curl -fsSL -o packages/pdfsys-bench/viz/assets/katex.min.css \
  https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css
ls -la packages/pdfsys-bench/viz/assets/katex*
```

Expected: katex.min.js ~280 KB, auto-render ~5-10 KB, katex.min.css ~24 KB.

- [ ] **Step 4: Download KaTeX font files**

KaTeX CSS references font files via relative URL `fonts/KaTeX_*.woff2`. Need to grab all the .woff2 the CSS references:

```bash
FONTS=(
  KaTeX_AMS-Regular.woff2
  KaTeX_Caligraphic-Bold.woff2 KaTeX_Caligraphic-Regular.woff2
  KaTeX_Fraktur-Bold.woff2 KaTeX_Fraktur-Regular.woff2
  KaTeX_Main-Bold.woff2 KaTeX_Main-BoldItalic.woff2 KaTeX_Main-Italic.woff2 KaTeX_Main-Regular.woff2
  KaTeX_Math-BoldItalic.woff2 KaTeX_Math-Italic.woff2
  KaTeX_SansSerif-Bold.woff2 KaTeX_SansSerif-Italic.woff2 KaTeX_SansSerif-Regular.woff2
  KaTeX_Script-Regular.woff2
  KaTeX_Size1-Regular.woff2 KaTeX_Size2-Regular.woff2 KaTeX_Size3-Regular.woff2 KaTeX_Size4-Regular.woff2
  KaTeX_Typewriter-Regular.woff2
)
for f in "${FONTS[@]}"; do
  curl -fsSL -o "packages/pdfsys-bench/viz/assets/katex-fonts/$f" \
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/$f"
done
ls -la packages/pdfsys-bench/viz/assets/katex-fonts/ | wc -l
```

Expected: 20+ font files (one per FONTS entry, ~10-25 KB each, total ~300 KB).

- [ ] **Step 5: Patch katex.min.css to reference local fonts dir**

KaTeX CSS references fonts via `url(fonts/...)`. We placed them in `katex-fonts/` (different name to avoid colliding with marked's namespace, and clearer in directory listing). Patch the CSS:

```bash
sed -i.bak 's|url(fonts/|url(katex-fonts/|g' packages/pdfsys-bench/viz/assets/katex.min.css
rm packages/pdfsys-bench/viz/assets/katex.min.css.bak
grep -c "katex-fonts/" packages/pdfsys-bench/viz/assets/katex.min.css
```

Expected: prints a count > 30 (each font face has multiple url() references for woff2 + woff + ttf fallbacks).

- [ ] **Step 6: Verify by opening a tiny test HTML**

Write a temp test file:

```bash
cat > /tmp/katex_test.html <<'EOF'
<!DOCTYPE html>
<html><head>
<link rel="stylesheet" href="packages/pdfsys-bench/viz/assets/katex.min.css">
</head><body>
<div id="math">$E = mc^2$  $$\int_0^1 x^2 dx$$</div>
<script src="packages/pdfsys-bench/viz/assets/katex.min.js"></script>
<script src="packages/pdfsys-bench/viz/assets/katex-auto-render.min.js"></script>
<script>
renderMathInElement(document.getElementById('math'), {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false }
  ]
});
</script>
</body></html>
EOF
echo "open /tmp/katex_test.html in browser to manually verify (optional)"
```

Don't actually need a browser check for the implementer — file existence + size checks above are sufficient. Browser sanity is part of Task 7 acceptance.

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-bench/viz/assets/
git commit -m "feat(viz): vendor marked + KaTeX assets for pipeline viz"
```

Expected diff: ~25 new files, ~700 KB total.

---

## Task 2: Build `viz.py` CLI

**Files:**
- Create: `packages/pdfsys-cli/src/pdfsys_cli/viz.py`

This module is self-contained except for pyarrow (already a workspace dep from a previous spec). It writes `viz_data.json`, copies markdown/previews, copies the HTML template + assets, writes `serve.sh`.

- [ ] **Step 1: Write `viz.py`**

```python
"""pdfsys visualize — preprocess a pipeline run into a static HTML+JSON bundle.

Consumes a run dir produced by `pdfsys run` (containing dataset.parquet,
markdown/, optionally .cache/layout/) and emits a self-contained out/viz/
directory: index.html + viz_data.json + previews.json + copied markdown.

See docs/superpowers/specs/2026-05-16-pipeline-viz-html-design.md for
schema and acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Layout cache files store regions in pdfsys_core's LayoutDocument schema —
# bbox is already normalized [0,1] (see pdfsys-core/docs/golden-principles/).
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_PREVIEW_PATH = _REPO_ROOT / "packages/pdfsys-bench/annotation/previews.json"
_TEMPLATE_DIR = _REPO_ROOT / "packages/pdfsys-bench/viz"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "viz"
    preview_src = Path(args.preview_source).resolve() if args.preview_source else _DEFAULT_PREVIEW_PATH

    if not (run_dir / "dataset.parquet").exists():
        print(f"ERROR: {run_dir}/dataset.parquet not found", file=sys.stderr)
        return 1

    print(f"[viz] reading {run_dir}/dataset.parquet ...")
    rows = _load_rows(run_dir)
    print(f"[viz] {len(rows)} rows loaded")

    aggregate = _compute_aggregate(rows)
    viz_data = {
        "run_meta": _build_meta(run_dir, rows),
        "aggregate": aggregate,
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz_data.json").write_text(
        json.dumps(viz_data, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[viz] wrote {out_dir}/viz_data.json ({(out_dir / 'viz_data.json').stat().st_size} bytes)")

    _copy_previews(preview_src, out_dir / "previews.json")
    _copy_markdown(run_dir / "markdown", out_dir / "markdown")
    _copy_template_and_assets(_TEMPLATE_DIR, out_dir)
    _write_serve_script(out_dir / "serve.sh")

    print(f"[viz] done. open {out_dir}/index.html  (or run bash {out_dir}/serve.sh)")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pdfsys visualize")
    p.add_argument("-r", "--run-dir", required=True,
                   help="pipeline run directory (must contain dataset.parquet)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="output directory (default: <run-dir>/viz)")
    p.add_argument("--preview-source", default=None,
                   help="path to previews.json (default: bundled annotation/previews.json)")
    return p.parse_args(argv)


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    table = pq.read_table(run_dir / "dataset.parquet")
    raw_rows = table.to_pylist()
    layout_cache_dir = run_dir / ".cache" / "layout"

    rows = []
    for raw in raw_rows:
        sha = raw.get("sha256") or ""
        short_id = sha[:8] if sha else f"r{len(rows):03d}"
        pdf_path = raw.get("pdf_path", "")
        basename = Path(pdf_path).name if pdf_path else "unknown.pdf"

        # Build layout block from cache if available.
        layout_block = None
        if sha and layout_cache_dir.exists():
            layout_block = _load_layout_block(layout_cache_dir, sha)

        # Markdown excerpt (first 400 chars from parquet column).
        md_full = raw.get("markdown") or ""
        md_excerpt = md_full[:400]

        # Locate the per-pdf markdown file (written by runner).
        md_path = None
        if sha:
            candidate = run_dir / "markdown" / f"{sha}.md"
            if candidate.exists():
                md_path = f"markdown/{sha}.md"

        rows.append({
            "id": short_id,
            "sha256": sha,
            "pdf_path": pdf_path,
            "pdf_basename": basename,
            "source": _source_from_path(pdf_path),
            "num_pages": raw.get("num_pages") or 0,
            "backend": raw.get("backend"),
            "stage_b_backend": raw.get("stage_b_backend"),
            "ocr_prob": raw.get("ocr_prob"),
            "is_form": bool(raw.get("is_form")),
            "is_encrypted": bool(raw.get("is_encrypted")),
            "garbled_text_ratio": raw.get("garbled_text_ratio") or 0.0,
            "layout": layout_block,
            "extract": {
                "backend": raw.get("extract_backend"),
                "markdown_chars": raw.get("markdown_chars") or 0,
                "markdown_excerpt": md_excerpt,
                "markdown_path": md_path,
            },
            "quality": {
                "score": raw.get("quality_score"),
                "model": raw.get("quality_model"),
                "kept": bool(raw.get("kept")),
            },
            "error_class": raw.get("error_class"),
            "error_message": raw.get("error_message"),
            "preview_key": _preview_key_from_path(pdf_path),
            "wall_ms_total": raw.get("wall_ms_total") or 0.0,
        })
    return rows


def _load_layout_block(cache_dir: Path, sha: str) -> dict[str, Any] | None:
    """Read a single LayoutDocument cache file; return a UI-friendly subset."""
    # LayoutCache writes files named <sha>__<model_tag>.json typically.
    candidates = list(cache_dir.glob(f"{sha}*"))
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    pages = data.get("pages") or []
    regions = []
    for page in pages:
        page_idx = page.get("index", 0)
        for region in page.get("regions", []):
            bbox = region.get("bbox") or {}
            regions.append({
                "page_idx": page_idx,
                "type": region.get("type", "TEXT"),
                "bbox": [
                    bbox.get("x0", 0.0),
                    bbox.get("y0", 0.0),
                    bbox.get("x1", 0.0),
                    bbox.get("y1", 0.0),
                ],
                "confidence": region.get("confidence"),
            })

    return {
        "model": data.get("layout_model") or "",
        "num_regions": len(regions),
        "has_complex": bool(data.get("has_complex_content")),
        "regions": regions,
    }


def _source_from_path(pdf_path: str) -> str:
    """omnidocbench_100 / olmocr_bench_50 / other."""
    if "omnidocbench_100" in pdf_path:
        return "omnidocbench_100"
    if "olmocr_bench_50" in pdf_path:
        return "olmocr_bench_50"
    return "other"


def _preview_key_from_path(pdf_path: str) -> str | None:
    """Compute the previews.json lookup key from a PDF path.

    annotation/previews.json uses keys like
    'olmocr__arxiv_math__2503.04438_pg16'. Pattern:
    <short_source>__<category>__<basename_without_ext>.
    """
    if not pdf_path:
        return None
    parts = Path(pdf_path).parts
    basename = Path(pdf_path).stem
    if "olmocr_bench_50" in pdf_path:
        # parts after "pdfs/<category>/<file>"
        try:
            i = parts.index("olmocr_bench_50")
            category = parts[i + 2]  # "olmocr_bench_50", "pdfs", "<category>"
            return f"olmocr__{category}__{basename}"
        except (ValueError, IndexError):
            return None
    if "omnidocbench_100" in pdf_path:
        try:
            i = parts.index("omnidocbench_100")
            return f"omnidoc__{basename}"
        except ValueError:
            return None
    return None


def _compute_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backend_dist: dict[str, int] = {}
    stage_b_dist: dict[str, int] = {}
    source_dist: dict[str, int] = {}
    error_dist: dict[str, int] = {}

    ocr_probs: list[float] = []
    qualities: list[float] = []
    md_chars: list[int] = []

    for r in rows:
        ext_b = r["extract"]["backend"] or "NULL"
        backend_dist[ext_b] = backend_dist.get(ext_b, 0) + 1

        if r["stage_b_backend"]:
            stage_b_dist[r["stage_b_backend"]] = stage_b_dist.get(r["stage_b_backend"], 0) + 1

        source_dist[r["source"]] = source_dist.get(r["source"], 0) + 1

        if r["error_class"]:
            error_dist[r["error_class"]] = error_dist.get(r["error_class"], 0) + 1

        if r["ocr_prob"] is not None:
            ocr_probs.append(float(r["ocr_prob"]))
        if r["quality"]["score"] is not None:
            qualities.append(float(r["quality"]["score"]))
        md_chars.append(int(r["extract"]["markdown_chars"]))

    def hist(values: list[float], lo: float, hi: float, bins: int = 20) -> dict[str, list]:
        if not values:
            return {"bins": [], "counts": []}
        edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
        counts = [0] * bins
        for v in values:
            slot = min(int((v - lo) / (hi - lo) * bins), bins - 1) if hi > lo else 0
            counts[max(0, slot)] += 1
        return {"bins": edges, "counts": counts}

    def percentiles(values: list[int]) -> dict[str, int]:
        if not values:
            return {"min": 0, "p25": 0, "p50": 0, "p75": 0, "max": 0}
        sv = sorted(values)
        n = len(sv)
        return {
            "min": sv[0],
            "p25": sv[n // 4],
            "p50": sv[n // 2],
            "p75": sv[(3 * n) // 4],
            "max": sv[-1],
        }

    return {
        "backend_dist": backend_dist,
        "stage_b_dist": stage_b_dist,
        "source_dist": source_dist,
        "error_dist": error_dist,
        "ocr_prob_hist": hist(ocr_probs, 0.0, 1.0),
        "quality_hist": hist(qualities, 0.0, 3.0),
        "markdown_chars_dist": percentiles(md_chars),
    }


def _build_meta(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = run_dir / "results.summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}

    errors = sum(1 for r in rows if r["error_class"])
    kept = sum(1 for r in rows if r["quality"]["kept"])
    qualities = [r["quality"]["score"] for r in rows if r["quality"]["score"] is not None]
    avg_q = sum(qualities) / len(qualities) if qualities else None

    return {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_rows": len(rows),
        "errors": errors,
        "kept": kept,
        "wall_seconds": summary.get("wall_seconds"),
        "avg_quality": avg_q,
        "ocr_threshold": 0.5,
        "kept_threshold": 2.0,
    }


def _copy_previews(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[viz] WARN: preview source {src} not found, skipping (preview overlays will be empty)")
        dst.write_text("{}", encoding="utf-8")
        return
    shutil.copyfile(src, dst)
    print(f"[viz] copied previews ({dst.stat().st_size} bytes)")


def _copy_markdown(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        print(f"[viz] WARN: markdown dir {src_dir} not found, skipping")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.glob("*.md"):
        shutil.copyfile(f, dst_dir / f.name)
        count += 1
    print(f"[viz] copied {count} markdown files")


def _copy_template_and_assets(template_dir: Path, out_dir: Path) -> None:
    """Copy index.html + assets/ from the template dir."""
    src_html = template_dir / "index.html"
    if not src_html.exists():
        print(f"[viz] ERROR: template {src_html} not found", file=sys.stderr)
        sys.exit(1)
    shutil.copyfile(src_html, out_dir / "index.html")

    src_assets = template_dir / "assets"
    dst_assets = out_dir / "assets"
    if dst_assets.exists():
        shutil.rmtree(dst_assets)
    shutil.copytree(src_assets, dst_assets)
    print(f"[viz] copied template + assets")


def _write_serve_script(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "# Local static server for the pdfsys viz bundle.\n"
        "# Run this and open http://localhost:8765/ in a browser.\n"
        "cd \"$(dirname \"$0\")\"\n"
        "python3 -m http.server 8765\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the module imports + argparse works**

```bash
.venv/bin/python -c "from pdfsys_cli import viz; print('OK')"
.venv/bin/python -m pdfsys_cli.viz --help
```

Expected:
- `OK` printed
- argparse help shows `-r/--run-dir`, `-o/--out-dir`, `--preview-source`

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/viz.py
git commit -m "feat(cli): add viz.py — preprocess run into viz_data.json bundle"
```

---

## Task 3: Wire `visualize` subcommand into `__main__.py`

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/__main__.py`

- [ ] **Step 1: Read current __main__.py to anchor the edit**

```bash
cat packages/pdfsys-cli/src/pdfsys_cli/__main__.py
```

Note the existing pattern for subcommands (likely a `subparsers.add_parser` chain or an `if args.cmd == ...` chain).

- [ ] **Step 2: Add the `visualize` subcommand**

The exact edit depends on the current structure. Common case: there's a `cmd` argument with choices like `["run", "init-config", "annotate"]`. Add `"visualize"`:

If the file uses argparse subparsers:

```python
# Find the existing block like:
#   subparsers = parser.add_subparsers(dest="cmd", required=True)
#   sub_run = subparsers.add_parser("run", help="...")
# Add a sibling:
sub_viz = subparsers.add_parser(
    "visualize",
    help="Build a static HTML+JSON bundle from a run directory",
)
# argparse with parents=[] would be cleanest, but to avoid coupling let viz.main parse its own args:
sub_viz.add_argument("forwarded", nargs=argparse.REMAINDER, help="forwarded to viz.main")

# Then in the dispatcher:
if args.cmd == "visualize":
    from . import viz
    return viz.main(args.forwarded)
```

If the file uses a manual `if/elif` chain on `sys.argv[1]`:

```python
# Add a branch:
elif sys.argv[1] == "visualize":
    from . import viz
    return viz.main(sys.argv[2:])
```

Adapt to whichever pattern is in place. Keep it minimal — one branch, no refactor.

- [ ] **Step 3: Verify the subcommand is reachable**

```bash
.venv/bin/pdfsys visualize --help
```

Expected: argparse help output from `viz.py` (showing -r/-o/--preview-source).

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/__main__.py
git commit -m "feat(cli): wire 'visualize' subcommand into pdfsys CLI"
```

---

## Task 4: Stub `index.html` (smoke through CLI → static page)

**Files:**
- Create: `packages/pdfsys-bench/viz/index.html`

This step creates a minimal HTML that just dumps `viz_data.json` keys. It proves the CLI → static-page pipeline works before adding actual UI.

- [ ] **Step 1: Write the stub**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>pdfsys viz · loading…</title>
<link rel="stylesheet" href="assets/katex.min.css">
<style>
  body { font: 13px/1.5 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
         margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
  h1 { font-size: 16px; margin: 0 0 12px; }
  pre { background: #0f1626; padding: 12px; border-radius: 4px;
        max-height: 70vh; overflow: auto; font-size: 11px; }
  #status { color: #ffc107; }
</style>
</head>
<body>
<h1 id="status">pdfsys viz · loading viz_data.json…</h1>
<pre id="dump"></pre>

<script src="assets/marked.min.js"></script>
<script src="assets/katex.min.js"></script>
<script src="assets/katex-auto-render.min.js"></script>
<script type="module">
  const status = document.getElementById('status');
  const dump = document.getElementById('dump');
  try {
    const res = await fetch('viz_data.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    status.textContent = `pdfsys viz · ${data.run_meta.total_rows} rows · ${data.run_meta.errors} errors · ${data.run_meta.kept} kept`;
    status.style.color = '#4caf50';
    dump.textContent = JSON.stringify({
      run_meta: data.run_meta,
      aggregate: data.aggregate,
      sample_row: data.rows[0],
    }, null, 2);
  } catch (e) {
    status.textContent = `pdfsys viz · ERROR loading viz_data.json: ${e.message}`;
    status.style.color = '#f44336';
    dump.textContent = e.stack || String(e);
  }
</script>
</body>
</html>
```

- [ ] **Step 2: End-to-end test of CLI + stub HTML**

```bash
rm -rf out/viz_test
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3 -o out/viz_test
ls -la out/viz_test/
.venv/bin/python -c "
import json
d = json.loads(open('out/viz_test/viz_data.json').read())
print('rows:', len(d['rows']))
print('first row id:', d['rows'][0]['id'])
print('first row backend:', d['rows'][0]['backend'])
print('aggregate backends:', d['aggregate']['backend_dist'])
"
```

Expected:
- CLI completes with `[viz] done. open out/viz_test/index.html ...`
- `out/viz_test/` contains: `index.html`, `viz_data.json`, `previews.json`, `markdown/` (150 files), `assets/`, `serve.sh`
- viz_data.json has 150 rows
- Backend dist matches the full run: `{'mupdf': 104, 'pipeline': 36, 'vlm': 10}`

- [ ] **Step 3: Visually verify the stub works**

```bash
echo "Open file://$PWD/out/viz_test/index.html in a browser."
echo "OR run: bash out/viz_test/serve.sh and visit http://localhost:8765/"
```

Expected (manual):
- Status bar reads "pdfsys viz · 150 rows · 0 errors · 35 kept" in green.
- `<pre>` shows run_meta + aggregate + first row JSON.
- If status is red with "ERROR loading viz_data.json", the browser is blocking `file://` fetch — use serve.sh.

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): add stub index.html (loads viz_data.json, dumps to <pre>)"
```

---

## Task 5: Build dashboard view in `index.html`

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html`

Replace the stub body + script with the dashboard view: table + 3 SVG charts + filters. The detail view (Task 6) will be added as a hidden div.

- [ ] **Step 1: Replace the file with the dashboard version**

Overwrite `packages/pdfsys-bench/viz/index.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>pdfsys viz · loading…</title>
<link rel="stylesheet" href="assets/katex.min.css">
<style>
  :root {
    --bg: #1a1a2e; --surface: #16213e; --surface2: #0f3460;
    --accent: #e94560; --text: #eee; --dim: #999;
    --green: #4caf50; --yellow: #ffc107; --red: #f44336;
    --blue: #2196f3; --orange: #ff9800; --purple: #b34dff;
    --border: #333;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font: 13px/1.5 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; overflow: hidden;
  }
  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; background: var(--surface);
    border-bottom: 1px solid var(--border); height: 44px;
    font-size: 13px;
  }
  #topbar h1 { font-size: 14px; white-space: nowrap; font-weight: 600; }
  #topbar .meta { color: var(--dim); font-size: 12px; }
  #topbar .meta strong { color: var(--text); }
  #back-btn { display: none; padding: 4px 10px; background: var(--surface2);
              border: 1px solid var(--border); color: var(--text);
              border-radius: 3px; cursor: pointer; font-size: 12px; }
  #back-btn:hover { background: #1a4a80; }

  .view { display: none; height: calc(100vh - 44px); overflow: hidden; }
  .view.active { display: flex; }

  /* Dashboard */
  #view-dashboard { flex-direction: column; }
  .charts { display: flex; gap: 12px; padding: 12px 16px; background: var(--surface);
            border-bottom: 1px solid var(--border); height: 180px; }
  .chart { flex: 1; background: var(--surface2); border-radius: 4px;
           padding: 10px; min-width: 0; }
  .chart h3 { font-size: 11px; color: var(--dim); margin-bottom: 6px;
              text-transform: uppercase; letter-spacing: 0.5px; }
  .chart svg { width: 100%; height: calc(100% - 24px); display: block; }
  .filter-bar {
    display: flex; gap: 12px; padding: 8px 16px; background: var(--surface);
    border-bottom: 1px solid var(--border); align-items: center;
    font-size: 12px;
  }
  .filter-bar label { display: flex; gap: 4px; align-items: center; cursor: pointer; }
  .filter-bar input[type=text] {
    padding: 4px 8px; border: 1px solid #444; border-radius: 3px;
    background: #111; color: var(--text); font-size: 12px; width: 200px;
  }
  .filter-bar select {
    padding: 4px 8px; border: 1px solid #444; border-radius: 3px;
    background: #111; color: var(--text); font-size: 12px;
  }
  #row-count { margin-left: auto; color: var(--dim); }

  /* Table */
  #table-wrap { flex: 1; overflow: auto; padding: 0 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead { position: sticky; top: 0; background: var(--surface); z-index: 1; }
  th { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border);
       cursor: pointer; user-select: none; font-weight: 500; color: var(--dim); }
  th:hover { color: var(--text); }
  td { padding: 5px 8px; border-bottom: 1px solid #222; }
  tr.row { cursor: pointer; }
  tr.row:hover { background: rgba(255,255,255,0.04); }
  tr.row.err td { color: var(--red); }
  tr.row .badge { display: inline-block; padding: 1px 6px; border-radius: 2px;
                  font-size: 11px; font-weight: 500; }
  .badge-mupdf { background: rgba(33,150,243,0.2); color: #6ab7ff; }
  .badge-pipeline { background: rgba(255,152,0,0.2); color: #ffb74d; }
  .badge-vlm { background: rgba(179,77,255,0.2); color: #d49aff; }
  .badge-NULL, .badge-deferred { background: rgba(255,255,255,0.1); color: var(--dim); }
  .kept-yes { color: var(--green); font-weight: 600; }
  .kept-no { color: var(--dim); }

  /* Detail view */
  #view-detail { gap: 0; }
  #detail-preview { width: 50%; padding: 16px; overflow-y: auto;
                     background: var(--surface); border-right: 1px solid var(--border); }
  #detail-preview h2 { font-size: 14px; margin-bottom: 8px; }
  #preview-canvas { position: relative; display: inline-block; max-width: 100%; }
  #preview-canvas img { max-width: 100%; display: block; }
  .bbox { position: absolute; border: 1px solid; }
  .bbox-TEXT { border-color: var(--blue); background: rgba(33,150,243,0.10); }
  .bbox-TABLE { border-color: var(--green); background: rgba(76,175,80,0.10); }
  .bbox-FORMULA { border-color: var(--purple); background: rgba(179,77,255,0.12); }
  .bbox-IMAGE { border-color: var(--orange); background: rgba(255,152,0,0.10); }
  #preview-empty { color: var(--dim); font-style: italic; padding: 20px;
                    background: var(--surface2); border-radius: 4px; text-align: center; }
  .legend { display: flex; gap: 12px; margin-top: 8px; font-size: 11px; color: var(--dim); }
  .legend span::before { content: ''; display: inline-block; width: 10px; height: 10px;
                          margin-right: 4px; vertical-align: middle; border: 1px solid; }
  .legend .l-TEXT::before { border-color: var(--blue); background: rgba(33,150,243,0.15); }
  .legend .l-TABLE::before { border-color: var(--green); background: rgba(76,175,80,0.15); }
  .legend .l-FORMULA::before { border-color: var(--purple); background: rgba(179,77,255,0.15); }
  .legend .l-IMAGE::before { border-color: var(--orange); background: rgba(255,152,0,0.15); }

  .region-table { width: 100%; margin-top: 12px; font-size: 11px; }
  .region-table th, .region-table td { padding: 3px 6px; border-bottom: 1px solid #222; }

  #detail-stages { flex: 1; padding: 16px; overflow-y: auto; }
  .stage-card {
    background: var(--surface); border-radius: 4px;
    padding: 10px 14px; margin-bottom: 10px;
    border-left: 3px solid var(--accent);
  }
  .stage-card h3 { font-size: 12px; color: var(--accent); margin-bottom: 6px;
                    text-transform: uppercase; letter-spacing: 0.5px; }
  .stage-card .kv { display: flex; gap: 6px; margin: 2px 0; font-size: 12px; }
  .stage-card .kv .k { color: var(--dim); min-width: 130px; }
  .stage-card .kv .v { color: var(--text); word-break: break-word; }
  .stage-card .md { background: #0f1626; padding: 10px; border-radius: 3px;
                     margin-top: 6px; max-height: 400px; overflow-y: auto;
                     font-size: 13px; line-height: 1.6; }
  .stage-card .md table { border-collapse: collapse; margin: 8px 0; }
  .stage-card .md th, .stage-card .md td { border: 1px solid #333; padding: 4px 8px; }
</style>
</head>
<body>
<div id="topbar">
  <h1>pdfsys viz</h1>
  <span class="meta" id="meta-summary">loading…</span>
  <button id="back-btn">← back</button>
</div>

<div id="view-dashboard" class="view active">
  <div class="charts">
    <div class="chart"><h3>Backend split</h3><svg id="chart-backend"></svg></div>
    <div class="chart"><h3>ocr_prob histogram (threshold 0.5)</h3><svg id="chart-ocr"></svg></div>
    <div class="chart"><h3>quality_score histogram (kept threshold 2.0)</h3><svg id="chart-quality"></svg></div>
  </div>
  <div class="filter-bar">
    <label><input type="checkbox" id="f-kept"> kept only</label>
    <label><input type="checkbox" id="f-error"> errors only</label>
    <select id="f-backend">
      <option value="">all backends</option>
      <option value="mupdf">mupdf</option>
      <option value="pipeline">pipeline</option>
      <option value="vlm">vlm</option>
    </select>
    <select id="f-source">
      <option value="">all sources</option>
      <option value="omnidocbench_100">omnidocbench_100</option>
      <option value="olmocr_bench_50">olmocr_bench_50</option>
    </select>
    <input type="text" id="f-search" placeholder="search filename…">
    <span id="row-count">0 / 0 rows</span>
  </div>
  <div id="table-wrap">
    <table>
      <thead><tr id="thead-row"></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div id="view-detail" class="view">
  <div id="detail-preview"></div>
  <div id="detail-stages"></div>
</div>

<script src="assets/marked.min.js"></script>
<script src="assets/katex.min.js"></script>
<script src="assets/katex-auto-render.min.js"></script>
<script type="module">
  let DATA = null;
  let PREVIEWS = null;
  let SORT = { key: 'pdf_basename', dir: 'asc' };

  const COLS = [
    { key: 'pdf_basename', label: 'file' },
    { key: 'source', label: 'source' },
    { key: 'backend', label: 'A→backend', render: r => badge(r.backend) },
    { key: 'extract_backend', label: 'final', render: r => badge(r.extract?.backend) },
    { key: 'ocr_prob', label: 'ocr_prob', render: r => fmt(r.ocr_prob, 3) },
    { key: 'num_pages', label: 'pages' },
    { key: 'quality_score', label: 'quality', render: r => fmt(r.quality?.score, 2) },
    { key: 'kept', label: 'kept', render: r => r.quality?.kept
        ? '<span class="kept-yes">✓</span>' : '<span class="kept-no">—</span>' },
    { key: 'markdown_chars', label: 'md_chars', render: r => r.extract?.markdown_chars ?? 0 },
    { key: 'error_class', label: 'err', render: r => r.error_class ?? '' },
  ];

  function badge(v) {
    if (!v) return '<span class="badge badge-NULL">—</span>';
    return `<span class="badge badge-${v}">${v}</span>`;
  }
  function fmt(v, d) {
    if (v === null || v === undefined) return '';
    return Number(v).toFixed(d);
  }
  function getSortValue(row, key) {
    if (key === 'extract_backend') return row.extract?.backend ?? '';
    if (key === 'quality_score') return row.quality?.score ?? -1;
    if (key === 'kept') return row.quality?.kept ? 1 : 0;
    if (key === 'markdown_chars') return row.extract?.markdown_chars ?? 0;
    return row[key] ?? '';
  }

  function renderDashboard() {
    document.getElementById('meta-summary').innerHTML =
      `<strong>${DATA.run_meta.total_rows}</strong> rows · `
      + `<strong>${DATA.run_meta.errors}</strong> errors · `
      + `<strong>${DATA.run_meta.kept}</strong> kept · `
      + `wall ${(DATA.run_meta.wall_seconds || 0).toFixed(0)}s · `
      + `avg quality ${(DATA.run_meta.avg_quality || 0).toFixed(2)}`;

    renderBackendChart(DATA.aggregate.backend_dist);
    renderHist(document.getElementById('chart-ocr'),
                DATA.aggregate.ocr_prob_hist, 0.5);
    renderHist(document.getElementById('chart-quality'),
                DATA.aggregate.quality_hist, DATA.run_meta.kept_threshold);
    renderHead();
    renderTable();
  }

  function renderBackendChart(dist) {
    const svg = document.getElementById('chart-backend');
    svg.innerHTML = '';
    const entries = Object.entries(dist);
    const total = entries.reduce((s, [, v]) => s + v, 0);
    if (!total) return;
    const colors = { mupdf: 'var(--blue)', pipeline: 'var(--orange)',
                     vlm: 'var(--purple)', NULL: '#555', deferred: '#555' };
    const w = svg.clientWidth || 300; const h = 50;
    let x = 0;
    for (const [k, v] of entries) {
      const w_ = (v / total) * w;
      svg.insertAdjacentHTML('beforeend',
        `<rect x="${x}" y="10" width="${w_}" height="${h}" fill="${colors[k] || '#888'}"/>`
        + `<text x="${x + w_/2}" y="38" fill="#fff" font-size="12" text-anchor="middle">`
        + `${k}: ${v}</text>`);
      x += w_;
    }
  }

  function renderHist(svg, hist, threshold) {
    svg.innerHTML = '';
    const w = svg.clientWidth || 300; const h = 130;
    const bins = hist.bins; const counts = hist.counts;
    if (!counts.length) return;
    const maxC = Math.max(...counts, 1);
    const barW = w / counts.length;
    for (let i = 0; i < counts.length; i++) {
      const barH = (counts[i] / maxC) * (h - 20);
      const x = i * barW;
      const y = h - barH;
      svg.insertAdjacentHTML('beforeend',
        `<rect x="${x + 1}" y="${y}" width="${barW - 2}" height="${barH}" fill="var(--blue)"/>`);
    }
    if (threshold !== undefined && bins.length) {
      const lo = bins[0]; const hi = bins[bins.length - 1];
      const tx = ((threshold - lo) / (hi - lo)) * w;
      svg.insertAdjacentHTML('beforeend',
        `<line x1="${tx}" y1="0" x2="${tx}" y2="${h}" stroke="var(--accent)" stroke-dasharray="3,3"/>`
        + `<text x="${tx + 4}" y="12" fill="var(--accent)" font-size="11">${threshold}</text>`);
    }
  }

  function renderHead() {
    const tr = document.getElementById('thead-row');
    tr.innerHTML = COLS.map(c =>
      `<th data-key="${c.key}">${c.label}${SORT.key === c.key ? (SORT.dir === 'asc' ? ' ▲' : ' ▼') : ''}</th>`
    ).join('');
    tr.querySelectorAll('th').forEach(th => {
      th.addEventListener('click', () => {
        const k = th.dataset.key;
        if (SORT.key === k) SORT.dir = SORT.dir === 'asc' ? 'desc' : 'asc';
        else { SORT.key = k; SORT.dir = 'asc'; }
        renderHead(); renderTable();
      });
    });
  }

  function renderTable() {
    const kept = document.getElementById('f-kept').checked;
    const err = document.getElementById('f-error').checked;
    const backend = document.getElementById('f-backend').value;
    const source = document.getElementById('f-source').value;
    const search = document.getElementById('f-search').value.toLowerCase();

    let rows = DATA.rows.filter(r => {
      if (kept && !r.quality?.kept) return false;
      if (err && !r.error_class) return false;
      if (backend && r.extract?.backend !== backend) return false;
      if (source && r.source !== source) return false;
      if (search && !r.pdf_basename.toLowerCase().includes(search)) return false;
      return true;
    });

    rows.sort((a, b) => {
      let va = getSortValue(a, SORT.key); let vb = getSortValue(b, SORT.key);
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      if (va < vb) return SORT.dir === 'asc' ? -1 : 1;
      if (va > vb) return SORT.dir === 'asc' ? 1 : -1;
      return 0;
    });

    document.getElementById('row-count').textContent = `${rows.length} / ${DATA.rows.length} rows`;
    document.getElementById('tbody').innerHTML = rows.map(r => {
      const cells = COLS.map(c => {
        const v = c.render ? c.render(r) : (r[c.key] ?? '');
        return `<td>${v}</td>`;
      }).join('');
      return `<tr class="row${r.error_class ? ' err' : ''}" data-id="${r.id}">${cells}</tr>`;
    }).join('');

    document.querySelectorAll('tr.row').forEach(tr => {
      tr.addEventListener('click', () => openDetail(tr.dataset.id));
    });
  }

  // Detail view added in Task 6 — for now placeholder
  async function openDetail(id) {
    const row = DATA.rows.find(r => r.id === id);
    if (!row) return;
    document.getElementById('view-dashboard').classList.remove('active');
    document.getElementById('view-detail').classList.add('active');
    document.getElementById('back-btn').style.display = 'inline-block';
    document.getElementById('detail-preview').innerHTML =
      `<h2>${row.pdf_basename}</h2><pre style="font-size:11px">${JSON.stringify(row, null, 2).slice(0, 2000)}</pre>`;
    document.getElementById('detail-stages').innerHTML =
      `<div class="stage-card"><h3>placeholder</h3><div>Detail view comes in Task 6.</div></div>`;
  }

  document.getElementById('back-btn').addEventListener('click', () => {
    document.getElementById('view-detail').classList.remove('active');
    document.getElementById('view-dashboard').classList.add('active');
    document.getElementById('back-btn').style.display = 'none';
  });

  ['f-kept', 'f-error', 'f-backend', 'f-source'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderTable));
  document.getElementById('f-search').addEventListener('input', renderTable);

  try {
    const [d, p] = await Promise.all([
      fetch('viz_data.json').then(r => r.json()),
      fetch('previews.json').then(r => r.json()).catch(() => ({})),
    ]);
    DATA = d; PREVIEWS = p;
    document.title = `pdfsys viz · ${d.run_meta.total_rows} rows`;
    renderDashboard();
  } catch (e) {
    document.getElementById('meta-summary').textContent = `ERROR: ${e.message}`;
    document.getElementById('meta-summary').style.color = 'var(--red)';
    console.error(e);
  }
</script>
</body>
</html>
```

- [ ] **Step 2: Regenerate viz bundle + verify**

```bash
rm -rf out/viz_test
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3 -o out/viz_test
echo "open file://$PWD/out/viz_test/index.html (or run bash out/viz_test/serve.sh)"
```

Manual checks in browser:
- Topbar shows "150 rows · 0 errors · 35 kept" etc.
- Backend split chart shows mupdf/pipeline/vlm proportions.
- ocr_prob and quality histograms render with threshold dashed line.
- Table has 150 rows.
- Sort by clicking column header works.
- Filter "kept only" → 35 rows; "errors only" → 0; backend dropdown filters.
- Search "jiaocai" filters down.
- Click any row → shows JSON dump on detail page (placeholder).
- Back button returns to dashboard.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): dashboard view (table + charts + filters)"
```

---

## Task 6: Add detail view (preview + bbox overlay + stage cards + markdown)

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html` (replace `openDetail` + add helpers)

- [ ] **Step 1: Replace the `openDetail` function and add helpers**

In `packages/pdfsys-bench/viz/index.html`, locate the function:

```javascript
  async function openDetail(id) {
    const row = DATA.rows.find(r => r.id === id);
    ...
```

Replace it (and the rest of the script through `document.getElementById('back-btn')...` IS unchanged) with:

```javascript
  async function openDetail(id) {
    const row = DATA.rows.find(r => r.id === id);
    if (!row) return;
    document.getElementById('view-dashboard').classList.remove('active');
    document.getElementById('view-detail').classList.add('active');
    document.getElementById('back-btn').style.display = 'inline-block';

    renderPreview(row);
    await renderStages(row);
  }

  function renderPreview(row) {
    const el = document.getElementById('detail-preview');
    const previewData = row.preview_key ? PREVIEWS[row.preview_key] : null;
    const imgSrc = previewData?.data_url || previewData?.image || previewData;
    // previews.json shape may have {data_url, page_idx, ...} or just be a base64 string
    const hasPreview = imgSrc && typeof imgSrc === 'string';

    let bboxesHtml = '';
    if (row.layout?.regions) {
      const page1 = row.layout.regions.filter(r => r.page_idx === 0);
      bboxesHtml = page1.map(r => {
        const [x0, y0, x1, y1] = r.bbox;
        const left = (x0 * 100).toFixed(2);
        const top = (y0 * 100).toFixed(2);
        const w = ((x1 - x0) * 100).toFixed(2);
        const h = ((y1 - y0) * 100).toFixed(2);
        return `<div class="bbox bbox-${r.type}" style="left:${left}%;top:${top}%;width:${w}%;height:${h}%" title="${r.type} conf=${r.confidence?.toFixed(2) ?? '?'}"></div>`;
      }).join('');
    }

    el.innerHTML =
      `<h2>${row.pdf_basename}</h2>
       <div style="color:var(--dim);font-size:11px;margin-bottom:8px">${row.pdf_path}</div>`
      + (hasPreview
          ? `<div id="preview-canvas"><img src="${imgSrc.startsWith('data:') ? imgSrc : 'data:image/png;base64,' + imgSrc}"/>${bboxesHtml}</div>`
          : `<div id="preview-empty">(no preview available for this PDF)</div>`)
      + `<div class="legend">
           <span class="l-TEXT">TEXT</span>
           <span class="l-TABLE">TABLE</span>
           <span class="l-FORMULA">FORMULA</span>
           <span class="l-IMAGE">IMAGE</span>
         </div>`
      + (row.layout?.regions?.length
          ? `<h3 style="margin-top:16px;font-size:12px;color:var(--dim)">all regions (${row.layout.num_regions} across ${row.num_pages} pages)</h3>
             <table class="region-table"><thead><tr><th>page</th><th>type</th><th>bbox</th><th>conf</th></tr></thead><tbody>`
            + row.layout.regions.map(r =>
                `<tr><td>${r.page_idx}</td><td>${r.type}</td><td>${r.bbox.map(v => v.toFixed(3)).join(', ')}</td><td>${r.confidence?.toFixed(2) ?? '—'}</td></tr>`
              ).join('')
            + `</tbody></table>`
          : '');
  }

  async function renderStages(row) {
    const el = document.getElementById('detail-stages');
    const parts = [];

    parts.push(`
      <div class="stage-card"><h3>Stage-A Router</h3>
        <div class="kv"><span class="k">backend (decision)</span><span class="v">${badge(row.backend)}</span></div>
        <div class="kv"><span class="k">ocr_prob</span><span class="v">${fmt(row.ocr_prob, 4)}</span></div>
        <div class="kv"><span class="k">num_pages</span><span class="v">${row.num_pages}</span></div>
        <div class="kv"><span class="k">is_form</span><span class="v">${row.is_form}</span></div>
        <div class="kv"><span class="k">is_encrypted</span><span class="v">${row.is_encrypted}</span></div>
        <div class="kv"><span class="k">garbled_text_ratio</span><span class="v">${fmt(row.garbled_text_ratio, 3)}</span></div>
      </div>`);

    if (row.layout) {
      parts.push(`
        <div class="stage-card"><h3>Layout</h3>
          <div class="kv"><span class="k">model</span><span class="v">${row.layout.model || '—'}</span></div>
          <div class="kv"><span class="k">num_regions</span><span class="v">${row.layout.num_regions}</span></div>
          <div class="kv"><span class="k">has_complex (TABLE/FORMULA)</span><span class="v">${row.layout.has_complex}</span></div>
        </div>`);
    } else {
      parts.push(`<div class="stage-card"><h3>Layout</h3><div style="color:var(--dim)">(skipped — went mupdf fast path)</div></div>`);
    }

    if (row.stage_b_backend) {
      parts.push(`
        <div class="stage-card"><h3>Stage-B Decider</h3>
          <div class="kv"><span class="k">→ backend</span><span class="v">${badge(row.stage_b_backend)}</span></div>
          <div class="kv"><span class="k">reason</span><span class="v">${row.layout?.has_complex ? 'has TABLE/FORMULA' : 'no complex content'}</span></div>
        </div>`);
    }

    parts.push(`
      <div class="stage-card"><h3>Extract (${row.extract?.backend || '—'})</h3>
        <div class="kv"><span class="k">markdown_chars</span><span class="v">${row.extract?.markdown_chars ?? 0}</span></div>
        <div class="md" id="md-host">loading markdown…</div>
      </div>`);

    parts.push(`
      <div class="stage-card"><h3>Quality</h3>
        <div class="kv"><span class="k">score</span><span class="v">${fmt(row.quality?.score, 3)}</span></div>
        <div class="kv"><span class="k">kept (threshold ${DATA.run_meta.kept_threshold})</span><span class="v">${row.quality?.kept ? '<span class="kept-yes">✓ true</span>' : '<span class="kept-no">false</span>'}</span></div>
        <div class="kv"><span class="k">model</span><span class="v">${row.quality?.model || '—'}</span></div>
      </div>`);

    if (row.error_class) {
      parts.push(`
        <div class="stage-card" style="border-left-color:var(--red)"><h3 style="color:var(--red)">Error</h3>
          <div class="kv"><span class="k">error_class</span><span class="v">${row.error_class}</span></div>
          <div class="kv"><span class="k">error_message</span><span class="v" style="font-family:monospace;font-size:11px">${row.error_message || ''}</span></div>
        </div>`);
    }

    el.innerHTML = parts.join('');

    // Lazy load full markdown
    if (row.extract?.markdown_path) {
      try {
        const text = await fetch(row.extract.markdown_path).then(r => r.text());
        const host = document.getElementById('md-host');
        host.innerHTML = marked.parse(text);
        renderMathInElement(host, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
          ],
          throwOnError: false,
        });
      } catch (e) {
        document.getElementById('md-host').textContent = `(markdown not available: ${e.message})`;
      }
    } else if (row.extract?.markdown_excerpt) {
      const host = document.getElementById('md-host');
      host.innerHTML = marked.parse(row.extract.markdown_excerpt) + '<p style="color:var(--dim);font-size:11px;margin-top:8px">(showing first 400 chars; full file not available)</p>';
      renderMathInElement(host, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
    } else {
      document.getElementById('md-host').textContent = '(no markdown)';
    }
  }
```

- [ ] **Step 2: Regenerate + verify in browser**

```bash
rm -rf out/viz_test
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3 -o out/viz_test
echo "open file://$PWD/out/viz_test/index.html or bash out/viz_test/serve.sh"
```

Manual checks (browser):
- Click a `mupdf` row → detail shows Router card + Layout "(skipped)" + Extract card with markdown rendered.
- Click a `vlm` row (e.g. one with `1_pg19.pdf`) → preview shows page-1 image with bbox overlays in 4 colors; Layout card shows regions; Extract card shows markdown with LaTeX rendered (e.g. `$\frac{x^2}{...}$` displays as a fraction, not as raw `$` characters).
- Click a row with `error_class != null` (if any in this dataset; should be 0) → would show red Error card.
- Detail view scrolls independently on each side (preview left, stage cards right).
- "Back" returns to dashboard, filters preserved.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): detail view with preview/bbox overlay/markdown/KaTeX"
```

---

## Task 7: README + final acceptance run

**Files:**
- Create: `packages/pdfsys-bench/viz/README.md`

- [ ] **Step 1: Write `packages/pdfsys-bench/viz/README.md`**

```markdown
# Pipeline Output Visualizer

Static HTML+JSON bundle for inspecting `pdfsys run` output. Dashboard for
cross-corpus overview + per-PDF detail view showing every pipeline stage.

## Generate a bundle

```sh
pdfsys visualize -r out/e2e_full_mineru3 -o out/viz
```

Reads `out/e2e_full_mineru3/dataset.parquet` + `markdown/*.md` + `.cache/layout/*.json`,
copies the bundled `previews.json` and HTML template, writes `out/viz/`.

## Open it

```sh
# Option A — direct
open out/viz/index.html

# Option B — if your browser blocks local file://fetch
bash out/viz/serve.sh
# then visit http://localhost:8765/
```

## What you see

- **Dashboard** — table of all rows, sortable columns, filters (kept only, errors only,
  backend, source), 3 SVG charts (backend split, ocr_prob histogram, quality histogram).
- **Detail (click any row)** — left: page-1 preview with layout bbox overlay
  (TEXT blue / TABLE green / FORMULA purple / IMAGE orange); right: per-stage cards
  showing Router decision, Layout, Stage-B verdict, extracted markdown
  (with KaTeX-rendered LaTeX), Quality score.

## Sharing

`out/viz/` is fully self-contained (HTML + JSON + assets + per-PDF markdown).
`zip -r viz.zip out/viz/` and ship — recipient unzips and opens `index.html`.
```

- [ ] **Step 2: Final end-to-end acceptance**

Run the full acceptance from spec §8:

```bash
rm -rf out/viz_final
time .venv/bin/pdfsys visualize -r out/e2e_full_mineru3 -o out/viz_final
echo "---bundle contents---"
ls -la out/viz_final/
du -h out/viz_final/viz_data.json
ls out/viz_final/markdown/ | wc -l
ls out/viz_final/assets/
```

Acceptance criteria from spec §8:
- [x] CLI completes in < 60 s, exit 0
- [x] Bundle contains: `index.html`, `viz_data.json` (1-5 MB), `previews.json`,
      `markdown/` (150 files), `assets/`, `serve.sh`

Browser checks (manual):
- [ ] Dashboard loads < 1 s; table shows 150 rows
- [ ] "kept only" → 35 rows
- [ ] "errors only" → 0 rows
- [ ] Search "jiaocai" filters subset
- [ ] Sort by quality_score works
- [ ] Click row → detail loads < 500 ms
- [ ] VLM row's markdown shows rendered LaTeX (math, not raw `$`)
- [ ] PIPELINE/VLM row's detail shows bbox overlays
- [ ] Back button preserves filter state

- [ ] **Step 3: Zip-and-share smoke (optional but valuable)**

```bash
(cd out && zip -qr viz_share.zip viz_final)
du -h out/viz_share.zip
echo "zip size; should be 2-5 MB"
```

- [ ] **Step 4: Append a §12 post-build note to the spec**

Open `docs/superpowers/specs/2026-05-16-pipeline-viz-html-design.md` and replace
the `## 12 · Post-build note (to be filled in)` placeholder with actuals:

```markdown
## 12 · Post-build note · 2026-05-16

### Bundle size (out/viz_final/)
- viz_data.json: <X> MB
- previews.json: <Y> MB
- markdown/: 150 files, <Z> MB total
- assets/: ~700 KB (marked + katex + fonts)
- Total: <T> MB
- zip: <Z> MB

### Wall time
- pdfsys visualize: <S> seconds

### Browser checks (all PASS / list failures)
- ...

### Open follow-ups
- ...
```

- [ ] **Step 5: Final commit**

```bash
git add packages/pdfsys-bench/viz/README.md docs/superpowers/specs/2026-05-16-pipeline-viz-html-design.md
git commit -m "docs(viz): README + post-build note"
```

---

## Self-review notes

**Spec coverage:**
- §1 goal (dashboard + per-PDF detail) → Tasks 5 + 6
- §2 non-goals (no editing/diffing) → respected throughout
- §3 architecture (preprocess to JSON, static bundle) → Task 2 (CLI) + Task 4 (stub HTML)
- §4 CLI signature → Task 2 (`_parse_args`)
- §5 viz_data.json schema → Task 2 (`_load_rows`, `_compute_aggregate`, `_build_meta`)
- §6 single-page HTML structure → Tasks 4-6
- §7 CLI scaffold → Task 2
- §8 acceptance criteria → Task 7
- §10 files touched → matches file map above
- §11 definition of done → Task 7 covers all bullets

**Placeholders:** none. The `openDetail` "placeholder" in Task 5 is intentional — explicitly labeled and replaced in Task 6.

**Type consistency:** `viz.py` writes `row.extract.markdown_path` as a relative string; HTML in Task 6 fetches from it with `fetch(row.extract.markdown_path)`. `row.layout.regions[i].bbox` is `[x0, y0, x1, y1]` floats in 0-1 throughout. `row.preview_key` matches `previews.json` key schema computed in `_preview_key_from_path`.

**One soft assumption:** The `previews.json` shape (whether it's `{key: data_url}` or `{key: {data_url: ...}}` or `{key: base64_string}`) varies by source — Task 6's `renderPreview` handles all three by checking `previewData?.data_url || previewData?.image || previewData`. If none match, the preview area gracefully shows "(no preview available)" placeholder.
