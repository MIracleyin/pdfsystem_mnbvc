#!/usr/bin/env python3
"""Enrich metadata.json with auto-detected page attributes.

Reads previews.json + layout_cache.json + omnidoc upstream metadata to
pre-fill page-level attributes for efficient annotation.

Auto-detected attributes:
  - language:  from CJK/Latin character ratio in extracted text
  - scan_type: from text coverage (born_digital / scanned / mixed)
  - data_source: from category metadata + heuristics
  - layout: from layout analysis column detection + omnidoc upstream
  - special_issues: from layout regions + text stats + omnidoc upstream
  - ocr_suggestion: derived from all signals above

Usage:
    python3 scripts/enrich_metadata.py
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
META_PATH = BENCH_ROOT / "annotation" / "metadata.json"
PREVIEW_PATH = BENCH_ROOT / "annotation" / "previews.json"
LAYOUT_CACHE_PATH = BENCH_ROOT / "annotation" / "layout_cache.json"


# ── Language detection ──────────────────────────────────────────────

def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 10:
        return "unknown"
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    total = cjk + latin
    if total < 5:
        return "unknown"
    ratio = cjk / total
    if ratio > 0.6:
        return "simplified_chinese"
    elif ratio < 0.2:
        return "english"
    else:
        return "en_ch_mixed"


# ── Scan type detection ─────────────────────────────────────────────

def detect_scan_type(preview: dict | None) -> str:
    if not preview or preview.get("error"):
        return "unknown"
    pages = preview.get("page_count", 0)
    text_pages = preview.get("pages_with_text", 0)
    if pages == 0:
        return "unknown"
    coverage = text_pages / pages
    if coverage == 0:
        return "scanned"
    elif coverage >= 0.8:
        return "born_digital"
    else:
        return "mixed"


# ── Data source mapping ─────────────────────────────────────────────

OLMOCR_CATEGORY_MAP = {
    "arxiv_math": "academic_literature",
    "headers_footers": "academic_literature",
    "long_tiny_text": "book",
    "multi_column": "academic_literature",
    "old_scans": "book",
    "tables": "research_report",
    "forms": "form",
}


def detect_data_source(pdf: dict) -> str:
    if pdf.get("source") == "omnidocbench_100":
        cat = pdf.get("category", "unknown")
        if cat != "unknown":
            return cat
    cat = pdf.get("category", "")
    return OLMOCR_CATEGORY_MAP.get(cat, "other")


# ── Layout detection from layout_cache ───────────────────────────────

def detect_layout_from_regions(layout_data: dict | None) -> str | None:
    """Infer column layout from text region bounding boxes.

    Heuristic: on each page, cluster text regions by x-center.
    If most pages have 1 cluster → single_column, 2 → double, etc.
    """
    if not layout_data or layout_data.get("error"):
        return None

    pages = layout_data.get("pages", [])
    if not pages:
        return None

    col_counts: list[int] = []
    for pg in pages:
        text_regions = [r for r in pg.get("regions", []) if r["type"] == "text"]
        if len(text_regions) < 2:
            col_counts.append(1)
            continue

        # Cluster x-centers using simple gap detection
        x_centers = sorted(
            (r["bbox"][0] + r["bbox"][2]) / 2 for r in text_regions
        )
        clusters = 1
        prev = x_centers[0]
        for xc in x_centers[1:]:
            # If gap > 15% of page width, new column
            if xc - prev > 0.15:
                clusters += 1
                prev = xc
            else:
                prev = (prev + xc) / 2  # running average

        col_counts.append(min(clusters, 4))

    if not col_counts:
        return None

    # Majority vote across pages
    from collections import Counter
    most_common = Counter(col_counts).most_common(1)[0][0]
    return {1: "single_column", 2: "double_column", 3: "three_column"}.get(
        most_common, "multi_column"
    )


def detect_layout(pdf: dict, layout_data: dict | None) -> str | None:
    """Detect layout from layout analysis first, then omnidoc, then category."""
    # From layout analysis
    from_layout = detect_layout_from_regions(layout_data)
    if from_layout:
        return from_layout
    # OmniDoc upstream
    layout = pdf.get("omnidoc_layout")
    if layout:
        return layout
    # OlmOCR category hint
    cat = pdf.get("category", "")
    if cat == "multi_column":
        return "double_column"
    return None


# ── Special issues detection (with layout) ───────────────────────────

def detect_special_issues(
    pdf: dict, preview: dict | None, layout_data: dict | None
) -> list[str]:
    issues: list[str] = []

    # From omnidoc upstream
    omnidoc_issues = pdf.get("omnidoc_special_issues", [])
    if omnidoc_issues:
        issue_map = {
            "colorful_backgroud": "colorful_background",
            "fuzzy_scan": "fuzzy_scan",
            "watermark": "watermark",
            "table_full_line": "has_table",
            "table_wireless_line": "has_table",
            "table_fewer_line": "has_table",
            "table_omission_line": "has_table",
            "table_span": "has_table",
            "table_horizontal": "has_table",
            "table_veritical": "has_table",
            "table_with_formula": "has_formula",
            "table_with_img": "has_table",
        }
        for iss in omnidoc_issues:
            mapped = issue_map.get(iss)
            if mapped and mapped not in issues:
                issues.append(mapped)
            if iss == "table_with_formula" and "has_formula" not in issues:
                issues.append("has_formula")

    # From layout analysis regions
    if layout_data and not layout_data.get("error"):
        type_counts = layout_data.get("type_counts", {})
        if type_counts.get("table", 0) > 0 and "has_table" not in issues:
            issues.append("has_table")
        if type_counts.get("formula", 0) > 0 and "has_formula" not in issues:
            issues.append("has_formula")
        # Images co-existing with text are likely figures with captions.
        if (
            type_counts.get("image", 0) > 0
            and "has_figure_with_text" not in issues
            and type_counts.get("text", 0) > 0
        ):
            issues.append("has_figure_with_text")

    # From preview text stats
    if preview and not preview.get("error"):
        garbled = preview.get("garbled_ratio", 0)
        if garbled > 0.05 and "garbled_text" not in issues:
            issues.append("garbled_text")
        pages = preview.get("page_count", 0)
        text_pages = preview.get("pages_with_text", 0)
        if (
            pages > 0
            and 0 < text_pages < pages
            and "partial_text_pages" not in issues
        ):
            issues.append("partial_text_pages")

    return issues


# ── OCR suggestion ───────────────────────────────────────────────────

def suggest_ocr(
    scan_type: str,
    preview: dict | None,
    special_issues: list[str],
    layout_data: dict | None,
) -> tuple[str, str]:
    reasons: list[str] = []

    if scan_type == "scanned":
        return "need_ocr", "scanned document, no text layer"

    if preview and not preview.get("error"):
        garbled = preview.get("garbled_ratio", 0)
        total = preview.get("total_chars", 0)
        pages = preview.get("page_count", 0)

        if total == 0 and pages > 0:
            return "need_ocr", "no text extracted"
        if garbled > 0.1:
            return "need_ocr", f"high garbled ratio ({garbled:.0%})"
        if total < 50 and pages > 0:
            reasons.append(f"very few chars ({total})")

    if scan_type == "mixed":
        reasons.append("mixed scan type")

    # Layout-derived signals
    if layout_data and not layout_data.get("error"):
        tc = layout_data.get("type_counts", {})
        if tc.get("table", 0) >= 3:
            reasons.append(f"heavy table content ({tc['table']} regions)")
        if tc.get("formula", 0) >= 2:
            reasons.append(f"formulas detected ({tc['formula']} regions)")
        if layout_data.get("has_complex"):
            reasons.append("complex content (table/formula)")

    if scan_type == "born_digital" and preview and not preview.get("error"):
        garbled = preview.get("garbled_ratio", 0)
        total = preview.get("total_chars", 0)
        if garbled < 0.01 and total > 200:
            if not reasons:
                return "text_ok", "born digital, good text extraction"
            # Has some issues but text is OK — add positive signal
            reasons.insert(0, "text layer OK but")

    if reasons:
        return "need_ocr", "; ".join(reasons)

    return "ambiguous", "insufficient signals"


# ── Main enrichment ──────────────────────────────────────────────────

def enrich(meta: dict, previews: dict, layout_cache: dict) -> dict:
    meta["version"] = 2
    meta["label_schema"] = {
        "data_sources": [
            "academic_literature", "book", "colorful_textbook", "exam_paper",
            "magazine", "newspaper", "note", "research_report", "PPT2PDF",
            "slides", "form", "other",
        ],
        "languages": [
            "simplified_chinese", "english", "en_ch_mixed", "other", "unknown",
        ],
        "layouts": [
            "single_column", "double_column", "three_column",
            "1andmore_column", "multi_column", "other_layout",
        ],
        "scan_types": ["born_digital", "scanned", "mixed", "unknown"],
        "special_issues": [
            "watermark", "colorful_background", "fuzzy_scan",
            "has_table", "has_formula", "has_figure_with_text",
            "handwriting", "vertical_text", "rotated_text",
            "stamp_seal", "header_footer_heavy",
            "garbled_text", "partial_text_pages",
        ],
        "ocr_decisions": ["text_ok", "need_ocr"],
    }

    for pdf in meta["pdfs"]:
        pid = pdf["id"]
        preview = previews.get(pid)

        # Find layout data — try pp-doclayoutv3 first, then yolo
        layout_data = None
        for backend in ("pp-doclayoutv3", "yolo"):
            key = f"{backend}::{pid}"
            if key in layout_cache:
                ld = layout_cache[key]
                if not ld.get("error"):
                    layout_data = ld
                    break

        # Text for language detection
        full_text = ""
        if preview and not preview.get("error"):
            per_page = preview.get("per_page_text", [])
            full_text = "\n".join(per_page)

        # Pre-fill signals
        pdf["pre_signals"] = {}
        if preview and not preview.get("error"):
            pc = preview.get("page_count", 0)
            pdf["pre_signals"] = {
                "has_text_layer": preview.get("total_chars", 0) > 0,
                "text_coverage": round(preview["pages_with_text"] / pc, 2) if pc > 0 else 0,
                "total_chars": preview.get("total_chars", 0),
                "garbled_ratio": round(preview.get("garbled_ratio", 0), 4),
                "md_chars": preview.get("markdown_chars", 0),
                "chars_per_page": round(preview["total_chars"] / pc) if pc > 0 else 0,
            }

        # Layout signals
        if layout_data:
            tc = layout_data.get("type_counts", {})
            pdf["pre_signals"]["layout_model"] = layout_data.get("layout_model", "")
            pdf["pre_signals"]["has_complex"] = layout_data.get("has_complex", False)
            pdf["pre_signals"]["region_counts"] = tc
            pdf["pre_signals"]["total_regions"] = sum(tc.values())

        # Auto-detect attributes (only if not confirmed)
        if not pdf.get("confirmed"):
            scan_type = detect_scan_type(preview)

            language = detect_language(full_text)
            omnidoc_lang = pdf.get("omnidoc_language")
            if omnidoc_lang:
                lang_map = {
                    "text_simplified_chinese": "simplified_chinese",
                    "text_english": "english",
                    "text_en_ch_mixed": "en_ch_mixed",
                }
                mapped = lang_map.get(omnidoc_lang, "unknown")
                if mapped != "unknown":
                    language = mapped

            data_source = detect_data_source(pdf)
            layout = detect_layout(pdf, layout_data)
            special_issues = detect_special_issues(pdf, preview, layout_data)
            ocr_decision, ocr_reason = suggest_ocr(
                scan_type, preview, special_issues, layout_data
            )

            pdf["attr_data_source"] = data_source
            pdf["attr_language"] = language
            pdf["attr_layout"] = layout
            pdf["attr_scan_type"] = scan_type
            pdf["attr_special_issues"] = special_issues
            pdf["ocr_suggestion"] = ocr_decision
            pdf["ocr_suggestion_reason"] = ocr_reason

            if "ocr_decision" not in pdf:
                pdf["ocr_decision"] = None
            if "confirmed" not in pdf:
                pdf["confirmed"] = False
            if "annotator" not in pdf:
                pdf["annotator"] = None
            if "annotated_at" not in pdf:
                pdf["annotated_at"] = None

    return meta


def main():
    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found.")
        return

    previews: dict = {}
    if PREVIEW_PATH.exists():
        with open(PREVIEW_PATH) as f:
            previews = json.load(f)
        print(f"Loaded {len(previews)} previews")

    layout_cache: dict = {}
    if LAYOUT_CACHE_PATH.exists():
        with open(LAYOUT_CACHE_PATH) as f:
            layout_cache = json.load(f)
        print(f"Loaded {len(layout_cache)} layout results")
    else:
        print("WARNING: layout_cache.json not found. Run batch_layout.py first.")

    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Loaded metadata: {meta['total']} PDFs")

    enriched = enrich(meta, previews, layout_cache)

    # Stats
    stats: dict[str, int] = {}
    lang_stats: dict[str, int] = {}
    scan_stats: dict[str, int] = {}
    layout_stats: dict[str, int] = {}
    issue_stats: dict[str, int] = {}
    for pdf in enriched["pdfs"]:
        sug = pdf.get("ocr_suggestion", "ambiguous")
        stats[sug] = stats.get(sug, 0) + 1
        lang = pdf.get("attr_language", "unknown")
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
        scan = pdf.get("attr_scan_type", "unknown")
        scan_stats[scan] = scan_stats.get(scan, 0) + 1
        layout = pdf.get("attr_layout") or "unknown"
        layout_stats[layout] = layout_stats.get(layout, 0) + 1
        for iss in pdf.get("attr_special_issues", []):
            issue_stats[iss] = issue_stats.get(iss, 0) + 1

    print(f"\nOCR suggestions:  {stats}")
    print(f"Languages:        {lang_stats}")
    print(f"Scan types:       {scan_stats}")
    print(f"Layouts:          {layout_stats}")
    print(f"Special issues:   {issue_stats}")

    filled = sum(1 for p in enriched["pdfs"] if p.get("attr_layout"))
    print(f"\nLayout filled: {filled}/{len(enriched['pdfs'])}")

    tmp = META_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    tmp.replace(META_PATH)
    print(f"Wrote enriched {META_PATH}")


if __name__ == "__main__":
    main()
