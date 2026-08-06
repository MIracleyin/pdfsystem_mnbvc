#!/usr/bin/env python3
"""Build a unified metadata JSON for all bench PDFs (olmocr + omnidocbench).

Usage:
    python3 scripts/build_annotation_meta.py

Outputs:
    annotation/metadata.json
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
OLMOCR_DIR = BENCH_ROOT / "olmocr_bench_50"
OMNIDOC_DIR = BENCH_ROOT / "omnidocbench_100"
OUT_PATH = BENCH_ROOT / "annotation" / "metadata.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_page_count(path: Path) -> int | None:
    """Try to get page count; attempts pymupdf, fitz, then regex fallback."""
    for mod_name in ("pymupdf", "fitz"):
        try:
            mod = __import__(mod_name)
            with mod.open(str(path)) as doc:
                return len(doc)
        except Exception:
            continue
    # Fallback: count /Type /Page in raw PDF (rough but zero-dep)
    try:
        import re
        raw = path.read_bytes()
        # Count page objects — not 100% accurate but good enough
        return len(re.findall(rb"/Type\s*/Page(?!\w)", raw))
    except Exception:
        return None


def collect_olmocr() -> list[dict]:
    """Collect PDFs from olmocr_bench_50/pdfs/<category>/<file>.pdf"""
    pdf_root = OLMOCR_DIR / "pdfs"
    if not pdf_root.exists():
        return []

    # Load test counts per PDF
    tests_per_pdf: dict[str, int] = {}
    tests_path = OLMOCR_DIR / "subset_tests.jsonl"
    if tests_path.exists():
        with open(tests_path) as f:
            for line in f:
                row = json.loads(line)
                key = row.get("pdf", "")
                tests_per_pdf[key] = tests_per_pdf.get(key, 0) + 1

    records = []
    for category_dir in sorted(pdf_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for pdf_path in sorted(category_dir.glob("*.pdf")):
            rel = pdf_path.relative_to(BENCH_ROOT)
            olmocr_key = f"{category}/{pdf_path.name}"
            records.append({
                "id": f"olmocr__{category}__{pdf_path.stem}",
                "filename": pdf_path.name,
                "rel_path": str(rel),
                "source": "olmocr_bench_50",
                "category": category,
                "sha256": sha256_file(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "page_count": file_page_count(pdf_path),
                "olmocr_test_count": tests_per_pdf.get(olmocr_key, 0),
                # Annotation fields (to be filled by annotator)
                "label": None,          # text_ok | need_ocr | ambiguous | unusable | hold | keep
                "ocr_reasons": [],      # subset of predefined reasons
                "n_ocr_pages": None,    # int or null
                "reason_short": "",     # free text
                "custom_tags": [],      # free-form user tags
                "annotator": None,
                "annotated_at": None,
            })
    return records


def collect_omnidoc() -> list[dict]:
    """Collect PDFs from omnidocbench_100/pdfs/<file>.pdf with upstream metadata."""
    pdf_root = OMNIDOC_DIR / "pdfs"
    if not pdf_root.exists():
        return []

    # Build lookup from omnidocbench annotations
    annot_lookup: dict[str, dict] = {}
    annot_path = OMNIDOC_DIR / "subset_100.json"
    if annot_path.exists():
        with open(annot_path) as f:
            data = json.load(f)
        for item in data:
            # page_info.image_path → stem matches PDF filename stem
            page_info = item.get("page_info", {})
            img_path = page_info.get("image_path", "")
            img_stem = Path(img_path).stem
            # Attributes are nested under page_attribute
            attr = page_info.get("page_attribute", {})
            annot_lookup[img_stem] = {
                "data_source": attr.get("data_source"),
                "language": attr.get("language"),
                "layout": attr.get("layout"),
                "special_issues": attr.get("special_issue", []),
                "num_regions": len(item.get("layout_dets", [])),
            }

    records = []
    for pdf_path in sorted(pdf_root.glob("*.pdf")):
        rel = pdf_path.relative_to(BENCH_ROOT)
        stem = pdf_path.stem
        meta = annot_lookup.get(stem, {})
        records.append({
            "id": f"omnidoc__{stem}",
            "filename": pdf_path.name,
            "rel_path": str(rel),
            "source": "omnidocbench_100",
            "category": meta.get("data_source", "unknown"),
            "sha256": sha256_file(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "page_count": file_page_count(pdf_path),
            "omnidoc_language": meta.get("language"),
            "omnidoc_layout": meta.get("layout"),
            "omnidoc_special_issues": meta.get("special_issues", []),
            "omnidoc_num_regions": meta.get("num_regions"),
            # Annotation fields
            "label": None,
            "ocr_reasons": [],
            "n_ocr_pages": None,
            "reason_short": "",
            "custom_tags": [],
            "annotator": None,
            "annotated_at": None,
        })
    return records


def main():
    print("Collecting olmocr_bench_50 PDFs...")
    olmocr = collect_olmocr()
    print(f"  Found {len(olmocr)} PDFs")

    print("Collecting omnidocbench_100 PDFs...")
    omnidoc = collect_omnidoc()
    print(f"  Found {len(omnidoc)} PDFs")

    all_records = olmocr + omnidoc
    output = {
        "version": 1,
        "total": len(all_records),
        "sources": {
            "olmocr_bench_50": len(olmocr),
            "omnidocbench_100": len(omnidoc),
        },
        "label_schema": {
            "labels": ["text_ok", "need_ocr", "ambiguous", "unusable", "discard", "hold", "keep"],
            "ocr_reasons": [
                "scan_no_ocr_layer",
                "three_plus_column",
                "has_tables",
                "image_with_text",
                "complex_formula",
                "handwriting",
                "pages_gt_10",
                "all_pages_empty",
            ],
        },
        "pdfs": all_records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUT_PATH} ({len(all_records)} PDFs)")


if __name__ == "__main__":
    main()
