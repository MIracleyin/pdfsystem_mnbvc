#!/usr/bin/env python3
"""Run mupdf text extraction on every bench PDF to generate quick-reference previews.

For each PDF in annotation/metadata.json, this script:
  - Extracts per-page text via PyMuPDF
  - Computes lightweight stats (total chars, pages with text, garbled ratio)
  - Runs the mupdf backend parser to get the final Markdown output

Outputs:
  annotation/previews.json — {pdf_id: {stats..., per_page_text, markdown}}

Must be run in an environment with pymupdf + pdfsys-parser-mupdf installed.
Recommended:
  uv run --package pdfsys-parser-mupdf python3 scripts/build_mupdf_previews.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
META_PATH = BENCH_ROOT / "annotation" / "metadata.json"
OUT_PATH = BENCH_ROOT / "annotation" / "previews.json"

# U+FFFD = Unicode replacement character; indicates decoding failure.
REPLACEMENT = "\ufffd"


def extract_one(pdf_path: Path) -> dict:
    """Extract per-page text + parser markdown + stats for a single PDF."""
    import pymupdf  # noqa: WPS433

    result: dict = {
        "error": None,
        "page_count": 0,
        "pages_with_text": 0,
        "total_chars": 0,
        "garbled_ratio": 0.0,
        "per_page_text": [],
        "markdown": "",
        "markdown_chars": 0,
        "elapsed_ms": 0,
    }
    t0 = time.time()

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        result["error"] = f"open failed: {e}"
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    try:
        result["page_count"] = len(doc)
        per_page_text: list[str] = []
        total_chars = 0
        garbled_chars = 0
        pages_with_text = 0

        for page in doc:
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            per_page_text.append(text)
            total_chars += len(text)
            garbled_chars += text.count(REPLACEMENT)
            if text.strip():
                pages_with_text += 1

        result["per_page_text"] = per_page_text
        result["total_chars"] = total_chars
        result["pages_with_text"] = pages_with_text
        result["garbled_ratio"] = (garbled_chars / total_chars) if total_chars else 0.0
    finally:
        doc.close()

    # Run the full parser to get the final markdown, if available.
    try:
        from pdfsys_parser_mupdf import extract_doc  # noqa: WPS433

        extracted = extract_doc(pdf_path)
        result["markdown"] = extracted.markdown
        result["markdown_chars"] = len(extracted.markdown)
    except Exception as e:
        # Not fatal — per_page_text is still usable
        result["markdown_error"] = str(e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


def main():
    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found. Run build_annotation_meta.py first.", file=sys.stderr)
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    # Allow incremental: if previews.json exists, skip already-processed PDFs
    existing: dict = {}
    if OUT_PATH.exists():
        try:
            with open(OUT_PATH) as f:
                existing = json.load(f)
            print(f"Resuming: {len(existing)} existing previews found")
        except Exception:
            existing = {}

    out: dict = dict(existing)
    total = len(meta["pdfs"])
    processed = 0
    skipped = 0

    for i, p in enumerate(meta["pdfs"], 1):
        pdf_id = p["id"]
        if pdf_id in out and not out[pdf_id].get("error"):
            skipped += 1
            continue

        pdf_path = BENCH_ROOT / p["rel_path"]
        print(f"[{i}/{total}] {p['filename']}", end=" ", flush=True)

        try:
            out[pdf_id] = extract_one(pdf_path)
        except Exception:
            out[pdf_id] = {"error": traceback.format_exc()}

        r = out[pdf_id]
        if r.get("error"):
            print(f"ERROR: {r['error'][:80]}")
        else:
            print(
                f"pages={r['page_count']} text_pages={r['pages_with_text']} "
                f"chars={r['total_chars']} garbled={r['garbled_ratio']:.3f} "
                f"md_chars={r['markdown_chars']} ({r['elapsed_ms']}ms)"
            )
        processed += 1

        # Save incrementally every 10 files
        if processed % 10 == 0:
            _save(out)

    _save(out)
    print(f"\nDone: processed {processed}, skipped (already done) {skipped}")
    print(f"Wrote {OUT_PATH}")


def _save(out: dict):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tmp.replace(OUT_PATH)


if __name__ == "__main__":
    main()
