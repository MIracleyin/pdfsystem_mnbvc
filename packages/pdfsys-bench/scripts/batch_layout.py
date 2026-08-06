#!/usr/bin/env python3
"""Batch layout analysis on all bench PDFs using PP-DocLayoutV3.

Runs layout detection on every PDF, saves results incrementally to
annotation/layout_cache.json. Skips already-processed PDFs.

Usage:
    uv run python3 scripts/batch_layout.py
    uv run python3 scripts/batch_layout.py --backend yolo   # use YOLO instead
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
META_PATH = BENCH_ROOT / "annotation" / "metadata.json"
CACHE_PATH = BENCH_ROOT / "annotation" / "layout_cache.json"


def layout_to_dict(layout) -> dict:
    """Convert LayoutDocument to JSON-serializable dict."""
    pages = []
    for lp in layout.pages:
        regions = []
        for r in lp.regions:
            regions.append({
                "region_id": r.region_id,
                "type": r.type.value,
                "bbox": [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1],
                "confidence": round(r.confidence, 3),
                "reading_order": r.reading_order,
            })
        pages.append({
            "index": lp.index,
            "width_pt": lp.page_width_pt,
            "height_pt": lp.page_height_pt,
            "regions": regions,
        })
    return {
        "sha256": layout.sha256,
        "layout_model": layout.layout_model,
        "has_complex": layout.has_complex_content,
        "page_count": layout.page_count,
        "pages": pages,
    }


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    tmp.replace(CACHE_PATH)


def main():
    backend = "pp-doclayoutv3"
    if "--backend" in sys.argv:
        idx = sys.argv.index("--backend")
        if idx + 1 < len(sys.argv):
            backend = sys.argv[idx + 1]

    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    # Load existing cache
    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            print(f"Resuming: {len(cache)} cached results")
        except Exception:
            cache = {}

    # Lazy-load analyser
    print(f"Loading {backend} model...")
    from pdfsys_layout_analyser import LayoutAnalyser

    if backend == "pp-doclayoutv3":
        analyser = LayoutAnalyser(
            model_path="PaddlePaddle/PP-DocLayoutV3_safetensors",
            backend="pp-doclayoutv3",
        )
    else:
        analyser = LayoutAnalyser()

    print("Model loaded. Starting batch analysis...\n")

    total = len(meta["pdfs"])
    processed = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, pdf in enumerate(meta["pdfs"], 1):
        pid = pdf["id"]
        cache_key = f"{backend}::{pid}"

        if cache_key in cache and not cache[cache_key].get("error"):
            skipped += 1
            continue

        pdf_path = BENCH_ROOT / pdf["rel_path"]
        if not pdf_path.exists():
            print(f"[{i}/{total}] SKIP (not found): {pdf['filename']}")
            continue

        print(f"[{i}/{total}] {pdf['filename']}", end=" ", flush=True)
        t0 = time.time()

        try:
            layout = analyser.analyse(str(pdf_path), sha256=pdf.get("sha256"))
            result = layout_to_dict(layout)
            result["elapsed_ms"] = int((time.time() - t0) * 1000)

            # Summary stats
            type_counts: dict[str, int] = {}
            for pg in result["pages"]:
                for r in pg["regions"]:
                    type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

            result["type_counts"] = type_counts
            cache[cache_key] = result
            processed += 1

            elapsed = int((time.time() - t0) * 1000)
            types_str = " ".join(f"{t}={n}" for t, n in sorted(type_counts.items()))
            print(f"({elapsed}ms) {types_str}")

        except Exception as e:
            cache[cache_key] = {"error": str(e), "traceback": traceback.format_exc()}
            errors += 1
            print(f"ERROR: {e}")

        # Save incrementally every 5 files
        if processed % 5 == 0:
            save_cache(cache)

    save_cache(cache)

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total:.0f}s: processed={processed}, skipped={skipped}, errors={errors}")
    print(f"Cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
