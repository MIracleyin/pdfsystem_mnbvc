"""PyMuPDF-based text extraction for the mupdf (text-ok) backend.

This is the simplest of the three parser backends. It assumes the PDF
already has a clean text layer and just needs unwrapping into Markdown —
which is why the router routes here only when the XGBoost classifier says
``ocr_prob < threshold``.

We use ``page.get_text("blocks")`` which returns paragraph-shaped blocks
with coordinates already in reading order (PyMuPDF's internal sorting).
Each block becomes one :class:`pdfsys_core.Segment` of type
:attr:`pdfsys_core.RegionType.TEXT`, with its bbox normalized to ``[0, 1]``.
Empty and image-only blocks are dropped.

No layout-model dependency, no GPU, no OCR — this is the text-ok fast
path, and stays that way.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pymupdf

from pdfsys_core import (
    Backend,
    BBox,
    ExtractedDoc,
    RegionType,
    Segment,
    merge_segments_to_markdown,
)


# PyMuPDF block tuple layout: (x0, y0, x1, y1, text, block_no, block_type).
# block_type 0 = text, 1 = image.
_TEXT_BLOCK_TYPE = 0


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_text(text: str) -> str:
    """Trim trailing whitespace and collapse PyMuPDF's soft linebreaks.

    PyMuPDF returns block text with intra-paragraph newlines. For Markdown
    emission we keep paragraphs on one line; actual paragraph breaks come
    from the block boundaries themselves.
    """
    if not text:
        return ""
    # Strip and replace single newlines with spaces while preserving
    # double-newlines (rare, but occasionally emitted for list items).
    paragraphs = [p.strip() for p in text.split("\n\n")]
    joined = "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())
    return joined.strip()


def _block_bbox(
    block: tuple[Any, ...],
    page_width_pt: float,
    page_height_pt: float,
) -> BBox | None:
    """Normalize a PyMuPDF block bbox to ``[0, 1]`` or return None on overflow."""
    x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
    if page_width_pt <= 0 or page_height_pt <= 0:
        return None

    def clamp(v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    nx0 = clamp(x0 / page_width_pt)
    ny0 = clamp(y0 / page_height_pt)
    nx1 = clamp(x1 / page_width_pt)
    ny1 = clamp(y1 / page_height_pt)
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    try:
        return BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
    except ValueError:
        return None


def extract_doc(pdf_path: str | Path) -> ExtractedDoc:
    """Run the mupdf backend on a single PDF file and return its ExtractedDoc."""
    path = Path(pdf_path)
    sha256 = _sha256_of_file(path)
    doc = pymupdf.open(str(path))
    try:
        return _extract(doc, sha256)
    finally:
        doc.close()


def extract_doc_bytes(pdf_bytes: bytes, sha256: str | None = None) -> ExtractedDoc:
    """Run the mupdf backend on an in-memory PDF buffer."""
    sha = sha256 or _sha256_of_bytes(pdf_bytes)
    doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        return _extract(doc, sha)
    finally:
        doc.close()


def _extract(doc: pymupdf.Document, sha256: str) -> ExtractedDoc:
    segments: list[Segment] = []
    pages_extracted = 0
    pages_skipped = 0

    for page_index, page in enumerate(doc):
        page_width_pt = float(page.rect.width)
        page_height_pt = float(page.rect.height)

        try:
            blocks = page.get_text(
                "blocks",
                flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP,
                sort=True,
            )
        except Exception:
            pages_skipped += 1
            continue

        pages_extracted += 1
        for block in blocks:
            # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) < 7:
                continue
            if block[6] != _TEXT_BLOCK_TYPE:
                # image block — mupdf backend doesn't emit IMAGE segments by
                # design; image-heavy PDFs should have been routed elsewhere.
                continue
            text = _normalize_text(block[4] or "")
            if not text:
                continue
            bbox = _block_bbox(block, page_width_pt, page_height_pt)
            segments.append(
                Segment(
                    index=len(segments),
                    backend=Backend.MUPDF,
                    page_index=page_index,
                    type=RegionType.TEXT,
                    content=text,
                    bbox=bbox,
                    source_region_id=None,
                )
            )

    seg_tuple = tuple(segments)
    markdown = merge_segments_to_markdown(seg_tuple)

    stats: dict[str, Any] = {
        "page_count": len(doc),
        "pages_extracted": pages_extracted,
        "pages_skipped": pages_skipped,
        "segment_count": len(seg_tuple),
        "char_count": len(markdown),
    }

    return ExtractedDoc(
        sha256=sha256,
        backend=Backend.MUPDF,
        segments=seg_tuple,
        markdown=markdown,
        stats=stats,
    )
