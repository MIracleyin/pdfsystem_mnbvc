"""PyMuPDF-only feature extractor for the Stage-A router classifier.

Ported verbatim (modulo stylistic cleanup and removal of datatrove imports)
from FinePDFs' ``blocks/predictor/ocr_predictor.py``:

    https://github.com/huggingface/finepdfs/blob/main/blocks/predictor/ocr_predictor.py

The goal is bit-exact feature compatibility with the upstream XGBoost
``xgb.ubj`` weights. If you touch anything in here, run the parity harness
in ``pdfsys-bench`` against FinePDFs' reference output first.

The extractor samples up to ``num_pages_to_sample`` pages at random, then
computes:

* 4 doc-level features: ``num_pages_successfully_sampled``,
  ``garbled_text_ratio``, ``is_form``, ``creator_or_producer_is_known_scanner``.
* 15 page-level features × 8 sampled pages = 120 features.

:func:`flatten_per_page_features` produces the flat 124-feature dict the
XGBoost model expects, in the exact column order of ``feature_names_in_``.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pymupdf


# Keep this list in sync with FinePDFs upstream. These strings are
# lowercased substring-matched against PDF metadata creator/producer to
# flag scanner-origin PDFs which almost always need OCR.
KNOWN_SCANNER_STRINGS: tuple[str, ...] = (
    "scanner",
    "scan",
    "epson",
    "hp scanjet",
    "canon",
    "fujitsu",
    "kodak",
    "brother",
    "xerox",
    "lexmark",
    "kmc",
    "kofax",
    "ricoh",
    "iris",
    "capturedocument",
    "paperport",
    "readiris",
    "simpleocr",
)

# Strip-merge tuning constants — used to coalesce image slices that some
# PDFs explode into dozens of thin rectangles, so we don't overcount.
JUNK_IMAGE_THRESHOLD_RATIO = 0.5
JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD = 3
MERGE_MAX_OFFSET = 5
MERGE_MAX_GAP = 2


def flatten_per_page_features(
    feature_dict_sample: dict[str, Any],
    sample_to_k_page_features: int = 8,
) -> dict[str, Any]:
    """Flatten a nested feature dict into the flat schema XGBoost expects.

    The XGBoost model was trained on a 124-column DataFrame whose columns
    are, in order:

        num_pages_successfully_sampled
        garbled_text_ratio
        is_form
        creator_or_producer_is_known_scanner
        page_level_unique_font_counts_page1
        ...
        page_level_vector_graphics_obj_count_page8

    If fewer than 8 pages were actually sampled, pages are resampled with
    replacement to pad the vector — this matches the upstream behavior.
    Seed numpy before calling this function if you need determinism.
    """
    flattened: dict[str, Any] = {}

    doc_level_features = (
        "num_pages_successfully_sampled",
        "num_unique_image_xrefs",
        "num_junk_image_xrefs",
        "garbled_text_ratio",
        "is_form",
        "creator_or_producer_is_known_scanner",
        "class",
    )

    used_keys: set[str] = set()

    for key in doc_level_features:
        if key in feature_dict_sample:
            flattened[key] = feature_dict_sample[key]
            used_keys.add(key)

    page_level_features = (
        "page_level_unique_font_counts",
        "page_level_char_counts",
        "page_level_text_box_counts",
        "page_level_avg_text_box_lengths",
        "page_level_text_area_ratios",
        "page_level_hidden_char_counts",
        "page_level_hidden_text_box_counts",
        "page_level_hidden_avg_text_box_lengths",
        "page_level_hidden_text_area_ratios",
        "page_level_image_counts",
        "page_level_non_junk_image_counts",
        "page_level_bitmap_proportions",
        "page_level_max_merged_strip_areas",
        "page_level_drawing_strokes_count",
        "page_level_vector_graphics_obj_count",
    )

    num_pages = len(feature_dict_sample["page_level_unique_font_counts"])
    page_indices = list(range(num_pages))
    # If we don't have enough pages, resample random pages. Upstream uses
    # np.random.choice here, so seed numpy if determinism matters.
    if num_pages < sample_to_k_page_features:
        extra = np.random.choice(
            num_pages, sample_to_k_page_features - num_pages, replace=True
        ).tolist()
        page_indices += extra

    for key in page_level_features:
        list_data = feature_dict_sample.get(key)
        if list_data is None:
            continue
        for page_idx, ind in enumerate(page_indices):
            flattened[f"{key}_page{page_idx + 1}"] = list_data[ind]
        used_keys.add(key)

    return flattened


class PDFFeatureExtractor:
    """PyMuPDF feature extraction. Pure — no I/O, no network, no state."""

    def __init__(self, num_pages_to_sample: int = 8, num_chunks: int = 1) -> None:
        if not isinstance(num_pages_to_sample, int):
            raise ValueError("num_pages_to_sample must be an integer.")
        self.num_pages_to_sample = num_pages_to_sample
        self.num_chunks = num_chunks

    # --------------------------------------------------------------- sampling

    def _get_sampled_page_indices(self, doc: pymupdf.Document) -> list[list[int]]:
        total_pages = len(doc)
        if total_pages == 0 or self.num_pages_to_sample <= 0:
            return []

        available = list(range(total_pages))
        sampled: list[list[int]] = []

        if self.num_chunks == -1:
            num_chunks = len(available) // self.num_pages_to_sample + 1
        else:
            num_chunks = self.num_chunks

        for _ in range(num_chunks):
            if not available:
                break
            chunk_size = min(self.num_pages_to_sample, len(available))
            chunk = random.sample(available, chunk_size)
            for idx in chunk:
                available.remove(idx)
            sampled.append(sorted(chunk))

        return sampled

    # ----------------------------------------------------------- doc-level

    def _get_garbled_text_per_page(
        self, doc: pymupdf.Document
    ) -> tuple[list[int], list[int]]:
        all_text: list[int] = []
        garbled_text: list[int] = []
        replacement = chr(0xFFFD)
        for page in doc:
            text = page.get_text(
                "text",
                flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP,
            )
            all_text.append(len(text))
            garbled_text.append(text.count(replacement))
        return all_text, garbled_text

    def _check_creator_producer_scanner(self, doc: pymupdf.Document) -> bool:
        metadata = doc.metadata or {}
        creator = (metadata.get("creator") or "").lower()
        producer = (metadata.get("producer") or "").lower()
        for keyword in KNOWN_SCANNER_STRINGS:
            if keyword in creator or keyword in producer:
                return True
        return False

    def _extract_document_level_stats_from_sampled_pages(
        self, doc: pymupdf.Document, sampled_page_indices: list[int]
    ) -> dict[str, Any]:
        """Identify junk images (same xref repeated on most sampled pages)."""
        stats: dict[str, Any] = {"junk_image_xrefs_list": []}

        if not sampled_page_indices:
            return stats

        all_instances: list[int] = []
        per_page: dict[int, set[int]] = {}
        for page_idx in sampled_page_indices:
            try:
                page = doc.load_page(page_idx)
                unique_xrefs: set[int] = set()
                for img_def in page.get_images(full=False):
                    xref = img_def[0]
                    if xref == 0:
                        continue
                    unique_xrefs.add(xref)
                    all_instances.append(xref)
                per_page[page_idx] = unique_xrefs
            except Exception:
                per_page[page_idx] = set()

        if not all_instances:
            return stats

        stats["num_unique_image_xrefs"] = len(set(all_instances))

        xref_page_counts: Counter[int] = Counter()
        for page_xrefs in per_page.values():
            xref_page_counts.update(page_xrefs)

        num_sampled = len(sampled_page_indices)
        # Upstream overrides the ratio check and requires an xref to be on
        # every sampled page to be flagged as junk — matches FinePDFs.
        min_threshold = num_sampled

        junk_list: list[int] = []
        if num_sampled >= JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD:
            for xref, count in xref_page_counts.items():
                if count >= min_threshold:
                    junk_list.append(xref)

        stats["num_junk_image_xrefs"] = len(junk_list)
        stats["junk_image_xrefs_list"] = junk_list
        return stats

    # ------------------------------------------------------------- imaging

    def _heuristic_merge_image_strips_on_page(
        self,
        single_page_image_list: list[list[Any]],
        page_width: float,
        page_height: float,
    ) -> list[list[Any]]:
        if not single_page_image_list:
            return []

        deduped: list[list[Any]] = []
        seen: set[tuple[float, float, float, float]] = set()
        for img_data in single_page_image_list:
            key = (img_data[0], img_data[1], img_data[2], img_data[3])
            if key not in seen:
                seen.add(key)
                deduped.append(img_data)
        if not deduped:
            return []

        deduped.sort(key=lambda img: (img[1], img[0]))
        merged: list[list[Any]] = [deduped[0]]

        for img in deduped[1:]:
            x0, y0, x1, y1, imgid = img
            last = merged[-1]
            lx0, ly0, lx1, ly1, _ = last

            cur_w = abs(x1 - x0)
            cur_h = abs(y1 - y0)
            full_w = page_width > 0 and cur_w >= page_width * 0.9
            full_h = page_height > 0 and cur_h >= page_height * 0.9

            can_merge = False
            if full_w:
                if (
                    abs(lx0 - x0) <= MERGE_MAX_OFFSET
                    and abs(lx1 - x1) <= MERGE_MAX_OFFSET
                    and abs(y0 - ly1) <= MERGE_MAX_GAP
                ):
                    can_merge = True
            if not can_merge and full_h:
                if (
                    abs(ly0 - y0) <= MERGE_MAX_OFFSET
                    and abs(ly1 - y1) <= MERGE_MAX_OFFSET
                    and abs(x0 - lx1) <= MERGE_MAX_GAP
                ):
                    can_merge = True

            if can_merge:
                merged[-1] = [
                    min(x0, lx0),
                    min(y0, ly0),
                    max(x1, lx1),
                    max(y1, ly1),
                    imgid,
                ]
            else:
                merged.append(img)

        return merged

    # ---------------------------------------------------------------- main

    def compute_features_per_chunk(
        self, doc: pymupdf.Document, sampled_page_indices: list[int]
    ) -> dict[str, Any]:
        features: dict[str, Any] = {
            "is_form": False,
            "creator_or_producer_is_known_scanner": False,
            "garbled_text_ratio": 0,
            "page_level_unique_font_counts": [],
            "page_level_char_counts": [],
            "page_level_text_box_counts": [],
            "page_level_avg_text_box_lengths": [],
            "page_level_text_area_ratios": [],
            "page_level_hidden_char_counts": [],
            "page_level_hidden_text_box_counts": [],
            "page_level_hidden_avg_text_box_lengths": [],
            "page_level_hidden_text_area_ratios": [],
            "page_level_image_counts": [],
            "page_level_non_junk_image_counts": [],
            "page_level_bitmap_proportions": [],
            "page_level_max_merged_strip_areas": [],
            "page_level_drawing_strokes_count": [],
            "page_level_vector_graphics_obj_count": [],
            "num_pages_successfully_sampled": 0,
            "num_pages_requested_for_sampling": 0,
            "sampled_page_indices": [],
        }

        features["num_pages_requested_for_sampling"] = len(sampled_page_indices)
        if not sampled_page_indices:
            return features

        doc_stats = self._extract_document_level_stats_from_sampled_pages(
            doc, sampled_page_indices
        )
        junk_xrefs: set[int] = set(doc_stats.get("junk_image_xrefs_list", []))

        features["is_form"] = bool(doc.is_form_pdf) if doc.is_form_pdf is not None else False
        features["creator_or_producer_is_known_scanner"] = self._check_creator_producer_scanner(doc)

        # Garbled text: U+FFFD replacement character / total chars. Computed
        # over ALL pages, but the rate reported to XGBoost is restricted to
        # the sampled pages (upstream semantics).
        all_text, garbled_text = self._get_garbled_text_per_page(doc)
        all_sum = sum(all_text)
        garb_sum = sum(garbled_text)
        features["global_garbled_text_ratio"] = 0 if all_sum == 0 else garb_sum / all_sum

        sampled_garb = sum(garbled_text[i] for i in sampled_page_indices)
        sampled_all = sum(all_text[i] for i in sampled_page_indices)
        features["garbled_text_ratio"] = 0 if sampled_all == 0 else sampled_garb / sampled_all

        for page_idx in sampled_page_indices:
            try:
                page = doc.load_page(page_idx)
            except Exception:
                continue

            features["sampled_page_indices"].append(page_idx)
            features["num_pages_successfully_sampled"] += 1

            page_rect = page.rect
            page_area = float(page_rect.width * page_rect.height) or 1.0

            # --- Fonts ---
            fonts: set[str] = set()
            try:
                for fi in page.get_fonts(full=True):
                    if len(fi) > 3 and fi[3]:
                        fonts.add(fi[3])
            except Exception:
                pass
            features["page_level_unique_font_counts"].append(len(fonts))

            # --- Visible vs hidden text via texttrace ---
            char_count = 0
            text_area = 0.0
            text_boxes = 0
            hidden_chars = 0
            hidden_area = 0.0
            hidden_boxes = 0
            try:
                for tr in page.get_texttrace():
                    n = len(tr.get("chars", []))
                    bbox = tr.get("bbox", (0, 0, 0, 0))
                    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if tr.get("type") == 3 or tr.get("opacity", 1.0) == 0:
                        hidden_chars += n
                        hidden_area += box_area
                        hidden_boxes += 1
                    else:
                        char_count += n
                        text_area += box_area
                        text_boxes += 1
            except Exception:
                pass

            features["page_level_char_counts"].append(char_count)
            features["page_level_text_box_counts"].append(text_boxes)
            features["page_level_avg_text_box_lengths"].append(
                text_area / text_boxes if text_boxes else 0.0
            )
            features["page_level_text_area_ratios"].append(text_area / page_area)
            features["page_level_hidden_char_counts"].append(hidden_chars)
            features["page_level_hidden_text_box_counts"].append(hidden_boxes)
            features["page_level_hidden_avg_text_box_lengths"].append(
                hidden_area / hidden_boxes if hidden_boxes else 0.0
            )
            features["page_level_hidden_text_area_ratios"].append(hidden_area / page_area)

            # --- Images ---
            total_imgs = 0
            non_junk_imgs = 0
            non_junk_rects: list[list[Any]] = []
            try:
                for img_def in page.get_images(full=False):
                    xref = img_def[0]
                    if xref == 0:
                        continue
                    rects = page.get_image_rects(xref, transform=False)
                    total_imgs += len(rects)
                    if xref not in junk_xrefs:
                        non_junk_imgs += len(rects)
                        for r in rects:
                            if r.is_empty or r.is_infinite:
                                continue
                            non_junk_rects.append([r.x0, r.y0, r.x1, r.y1, xref])
            except Exception:
                pass

            features["page_level_image_counts"].append(total_imgs)
            features["page_level_non_junk_image_counts"].append(non_junk_imgs)

            merged = self._heuristic_merge_image_strips_on_page(
                non_junk_rects, page_rect.width, page_rect.height
            )
            strip_areas = [abs(b[2] - b[0]) * abs(b[3] - b[1]) for b in merged]
            if strip_areas:
                features["page_level_max_merged_strip_areas"].append(max(strip_areas) / page_area)
                features["page_level_bitmap_proportions"].append(sum(strip_areas) / page_area)
            else:
                features["page_level_max_merged_strip_areas"].append(0.0)
                features["page_level_bitmap_proportions"].append(0.0)

            # --- Drawings / vector graphics ---
            stroke_count = 0
            vector_objs = 0
            try:
                drawings = page.get_cdrawings()
                vector_objs = len(drawings)
                for path in drawings:
                    for item in path.get("items", []):
                        if item[0] in ("l", "c", "q"):
                            stroke_count += 1
                    if path.get("rect") or path.get("quad"):
                        if path.get("stroke_opacity", 1) > 0 and path.get("color"):
                            stroke_count += 1
            except Exception:
                pass
            features["page_level_drawing_strokes_count"].append(stroke_count)
            features["page_level_vector_graphics_obj_count"].append(vector_objs)

        return features

    def extract_all_features(self, doc: pymupdf.Document) -> list[dict[str, Any]]:
        chunks = self._get_sampled_page_indices(doc)
        return [self.compute_features_per_chunk(doc, c) for c in chunks]
