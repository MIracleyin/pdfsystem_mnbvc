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
# This file lives at packages/pdfsys-cli/src/pdfsys_cli/viz.py;
# parents[4] climbs up to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
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

        layout_block = None
        if sha and layout_cache_dir.exists():
            layout_block = _load_layout_block(layout_cache_dir, sha)

        md_full = raw.get("markdown") or ""
        md_excerpt = md_full[:400]

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
    if "omnidocbench_100" in pdf_path:
        return "omnidocbench_100"
    if "olmocr_bench_50" in pdf_path:
        return "olmocr_bench_50"
    return "other"


def _preview_key_from_path(pdf_path: str) -> str | None:
    if not pdf_path:
        return None
    parts = Path(pdf_path).parts
    basename = Path(pdf_path).stem
    if "olmocr_bench_50" in pdf_path:
        try:
            i = parts.index("olmocr_bench_50")
            category = parts[i + 2]
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
