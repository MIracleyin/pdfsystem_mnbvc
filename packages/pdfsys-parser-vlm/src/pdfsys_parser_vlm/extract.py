"""VLM extraction backend using MinerU (magic-pdf).

Handles the "needs-ocr AND has complex content" branch — pages with
tables, formulas, or heavy mixed layouts that benefit from a vision-
language model approach.

MinerU 2.5 Pro (1.2B) processes full pages end-to-end: layout detection,
OCR, table/formula recognition, reading-order reconstruction. This module
wraps MinerU's pipeline API and maps its structured output back to the
shared :class:`pdfsys_core.ExtractedDoc` / :class:`pdfsys_core.Segment`
contracts.

Heavy dependencies (``magic_pdf``, ``torch``) are imported lazily.

Usage::

    from pdfsys_parser_vlm import VlmParser

    parser = VlmParser()
    doc = parser.extract("complex.pdf", sha256="abc...")

Or with a pre-built LayoutDocument (only process complex pages)::

    doc = parser.extract_complex_pages("complex.pdf", layout, sha256="abc...")
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    BBox,
    ExtractedDoc,
    LayoutDocument,
    RegionType,
    Segment,
    VlmConfig,
    merge_segments_to_markdown,
)

# MinerU content type → our RegionType mapping.
_MINERU_TYPE_MAP: dict[str, RegionType] = {
    "text": RegionType.TEXT,
    "title": RegionType.TEXT,
    "interline_equation": RegionType.FORMULA,
    "inline_equation": RegionType.FORMULA,
    "equation": RegionType.FORMULA,
    "table": RegionType.TABLE,
    "image": RegionType.IMAGE,
    "figure": RegionType.IMAGE,
    "figure_caption": RegionType.TEXT,
    "table_caption": RegionType.TEXT,
    "table_footnote": RegionType.TEXT,
    "header": RegionType.TEXT,
    "footer": RegionType.TEXT,
    "reference": RegionType.TEXT,
}


class VlmParser:
    """MinerU-based VLM extraction parser.

    Wraps MinerU (magic-pdf) to process complex PDF pages and produce
    :class:`pdfsys_core.ExtractedDoc` output.
    """

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()
        self._mineru_ready = False

    # ------------------------------------------------------------------ api

    def extract(
        self,
        pdf_path: str | Path,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Process an entire PDF through MinerU and return ExtractedDoc."""
        path = Path(pdf_path)
        sha = sha256 or _sha256_of_file(path)

        with path.open("rb") as f:
            pdf_bytes = f.read()

        return self._run_mineru(pdf_bytes, sha)

    def extract_bytes(
        self,
        pdf_bytes: bytes,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Same as :meth:`extract`, but from an in-memory buffer."""
        sha = sha256 or hashlib.sha256(pdf_bytes).hexdigest()
        return self._run_mineru(pdf_bytes, sha)

    def extract_complex_pages(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Process only pages flagged as having complex content.

        Reads the :class:`LayoutDocument` to identify which pages contain
        tables or formulas, then runs MinerU on the full PDF but only
        keeps segments from those complex pages. Simple pages are skipped
        (they should be handled by the pipeline parser).
        """
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)

        # Identify pages with complex content.
        complex_pages: set[int] = set()
        for lp in layout.pages:
            for region in lp.regions:
                if region.type in (RegionType.TABLE, RegionType.FORMULA):
                    complex_pages.add(lp.index)
                    break

        if not complex_pages:
            # No complex pages — return empty doc.
            return ExtractedDoc(
                sha256=sha,
                backend=Backend.VLM,
                segments=(),
                markdown="",
                stats={"complex_pages": 0, "reason": "no_complex_content"},
            )

        with path.open("rb") as f:
            pdf_bytes = f.read()

        full_doc = self._run_mineru(pdf_bytes, sha)

        # Filter segments to only complex pages.
        filtered = tuple(
            s for s in full_doc.segments if s.page_index in complex_pages
        )
        markdown = merge_segments_to_markdown(filtered)

        stats = dict(full_doc.stats)
        stats["complex_pages"] = len(complex_pages)
        stats["complex_page_indices"] = sorted(complex_pages)

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=filtered,
            markdown=markdown,
            stats=stats,
        )

    # --------------------------------------------------------------- internal

    def _run_mineru(self, pdf_bytes: bytes, sha256: str) -> ExtractedDoc:
        """Run MinerU pipeline on raw PDF bytes."""
        content_list, md_content, stats = self._invoke_magic_pdf(pdf_bytes)

        segments = self._content_list_to_segments(content_list)
        seg_tuple = tuple(segments)

        # Prefer our own markdown assembly for consistency, but fall back to
        # MinerU's if our assembly is empty (e.g. image-only pages).
        markdown = merge_segments_to_markdown(seg_tuple)
        if not markdown.strip() and md_content:
            markdown = md_content.strip() + "\n"

        stats["char_count"] = len(markdown)
        stats["segment_count"] = len(seg_tuple)
        stats["vlm_model"] = self.config.model

        return ExtractedDoc(
            sha256=sha256,
            backend=Backend.VLM,
            segments=seg_tuple,
            markdown=markdown,
            stats=stats,
        )

    def _invoke_magic_pdf(
        self, pdf_bytes: bytes
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """Call MinerU's pipeline API and return (content_list, markdown, stats).

        Supports both the newer ``mineru`` API and the older ``magic_pdf``
        API, trying the newer one first.
        """
        # Try newer MinerU API first (mineru >= 2.0).
        try:
            return self._invoke_mineru_v2(pdf_bytes)
        except ImportError:
            pass

        # Fall back to magic_pdf API.
        return self._invoke_magic_pdf_v1(pdf_bytes)

    def _invoke_mineru_v2(
        self, pdf_bytes: bytes
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """MinerU >= 2.0 API (``magic_pdf.data.dataset``)."""
        from magic_pdf.data.data_reader_writer import (  # noqa: PLC0415
            FileBasedDataWriter,
        )
        from magic_pdf.data.dataset import PymuDocDataset  # noqa: PLC0415
        from magic_pdf.model.doc_analyze_by_custom_model import (  # noqa: PLC0415
            doc_analyze,
        )

        with tempfile.TemporaryDirectory(prefix="pdfsys_vlm_") as tmpdir:
            image_dir = os.path.join(tmpdir, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(image_dir)

            ds = PymuDocDataset(pdf_bytes)

            # Always use OCR mode for complex content.
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

            md_content = pipe_result.get_markdown(image_dir)
            content_list = pipe_result.get_content_list(image_dir)

        stats: dict[str, Any] = {
            "api": "mineru_v2",
            "page_count": len(content_list) if isinstance(content_list, list) else 0,
        }

        # Flatten if content_list is per-page.
        flat_list: list[dict[str, Any]] = []
        if content_list and isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    flat_list.append(item)
                elif isinstance(item, list):
                    flat_list.extend(item)

        return flat_list, md_content, stats

    def _invoke_magic_pdf_v1(
        self, pdf_bytes: bytes
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """Older magic_pdf API (< 2.0)."""
        from magic_pdf.pipe.OCRPipe import OCRPipe  # noqa: PLC0415
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="pdfsys_vlm_") as tmpdir:
            image_dir = os.path.join(tmpdir, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_writer = DiskReaderWriter(image_dir)

            pipe = OCRPipe(pdf_bytes, [], image_writer)
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()

            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
            content_list = pipe.pipe_mk_uni_format(image_dir, drop_mode="none")

        stats: dict[str, Any] = {"api": "magic_pdf_v1"}

        flat_list: list[dict[str, Any]] = []
        if content_list and isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    flat_list.append(item)
                elif isinstance(item, list):
                    flat_list.extend(item)

        return flat_list, md_content, stats

    def _content_list_to_segments(
        self, content_list: list[dict[str, Any]]
    ) -> list[Segment]:
        """Map MinerU's content_list items to our Segment format."""
        segments: list[Segment] = []

        for item in content_list:
            item_type = item.get("type", "text")
            region_type = _MINERU_TYPE_MAP.get(item_type, RegionType.TEXT)

            # Extract text content.
            content = ""
            if region_type == RegionType.IMAGE:
                content = item.get("img_path", "") or "[image]"
            elif region_type == RegionType.TABLE:
                # MinerU may output table as HTML or LaTeX.
                content = (
                    item.get("html", "")
                    or item.get("latex", "")
                    or item.get("text", "")
                    or item.get("md", "")
                )
            elif region_type == RegionType.FORMULA:
                content = (
                    item.get("latex", "")
                    or item.get("text", "")
                    or item.get("md", "")
                )
            else:
                content = item.get("text", "") or item.get("md", "")

            if not content:
                continue

            # Parse bbox if available.
            bbox = None
            raw_bbox = item.get("bbox")
            page_w = item.get("page_width", 0)
            page_h = item.get("page_height", 0)
            if raw_bbox and len(raw_bbox) == 4 and page_w > 0 and page_h > 0:
                nx0 = max(0.0, min(1.0, raw_bbox[0] / page_w))
                ny0 = max(0.0, min(1.0, raw_bbox[1] / page_h))
                nx1 = max(0.0, min(1.0, raw_bbox[2] / page_w))
                ny1 = max(0.0, min(1.0, raw_bbox[3] / page_h))
                if nx1 > nx0 and ny1 > ny0:
                    try:
                        bbox = BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
                    except ValueError:
                        bbox = None

            page_index = item.get("page_idx", item.get("page_index", 0))

            segments.append(
                Segment(
                    index=len(segments),
                    backend=Backend.VLM,
                    page_index=int(page_index),
                    type=region_type,
                    content=content.strip(),
                    bbox=bbox,
                    source_region_id=None,
                )
            )

        return segments


# ---------------------------------------------------------------- convenience

def extract_doc(
    pdf_path: str | Path,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: create a VLM parser, extract, return."""
    parser = VlmParser(config=config)
    return parser.extract(pdf_path, sha256=sha256)


def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: extract only complex pages identified by layout."""
    parser = VlmParser(config=config)
    return parser.extract_complex_pages(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
