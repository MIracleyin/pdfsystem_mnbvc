"""Region-level OCR pipeline backend.

Consumes a :class:`pdfsys_core.LayoutDocument` (written by the layout
analyser), renders each region's bounding box from the PDF at the
configured DPI, runs line-level OCR (RapidOCR by default), and assembles
the results into an :class:`pdfsys_core.ExtractedDoc`.

This is the "needs-ocr AND no complex content" path — simple scanned
documents, image-based PDFs without tables/formulas. Complex pages go to
the VLM parser instead.

Heavy dependencies (``pymupdf``, ``rapidocr_onnxruntime``, ``PIL``) are
imported lazily.

Usage::

    from pdfsys_parser_pipeline import extract_doc_from_layout

    layout = cache.load(sha256, model_tag)
    doc = extract_doc_from_layout("file.pdf", layout)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    ExtractedDoc,
    LayoutDocument,
    PipelineConfig,
    RegionType,
    Segment,
    merge_segments_to_markdown,
)

from .ocr_engine import OcrEngine, create_ocr_engine


class PipelineParser:
    """Region-level OCR parser. Stateless after init (model loaded lazily)."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._ocr_engine = ocr_engine

    def _ensure_engine(self) -> OcrEngine:
        if self._ocr_engine is None:
            self._ocr_engine = create_ocr_engine(
                engine_name=self.config.ocr_engine,
                languages=self.config.languages,
            )
        return self._ocr_engine

    def extract(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Extract text from all regions defined in *layout* using OCR."""
        import pymupdf  # noqa: PLC0415

        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)
        engine = self._ensure_engine()

        doc = pymupdf.open(str(path))
        try:
            return self._extract(doc, layout, sha, engine)
        finally:
            doc.close()

    def extract_bytes(
        self,
        pdf_bytes: bytes,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Same as :meth:`extract`, but from an in-memory buffer."""
        import io

        import pymupdf  # noqa: PLC0415

        sha = sha256 or layout.sha256 or hashlib.sha256(pdf_bytes).hexdigest()
        engine = self._ensure_engine()

        doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        try:
            return self._extract(doc, layout, sha, engine)
        finally:
            doc.close()

    def _extract(
        self,
        doc: Any,
        layout: LayoutDocument,
        sha256: str,
        engine: OcrEngine,
    ) -> ExtractedDoc:
        import pymupdf  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        segments: list[Segment] = []
        pages_extracted = 0
        pages_skipped = 0
        dpi = self.config.render_dpi
        zoom = dpi / 72.0

        for layout_page in layout.pages:
            try:
                page = doc.load_page(layout_page.index)
            except Exception:
                pages_skipped += 1
                continue

            pages_extracted += 1
            page_width_pt = layout_page.page_width_pt
            page_height_pt = layout_page.page_height_pt

            for region in layout_page.regions:
                # Convert normalized bbox to PDF points, then render that
                # clip area at the configured DPI.
                pt_x0, pt_y0, pt_x1, pt_y1 = region.bbox.to_points(
                    page_width_pt, page_height_pt
                )
                clip = pymupdf.Rect(pt_x0, pt_y0, pt_x1, pt_y1)
                mat = pymupdf.Matrix(zoom, zoom)

                try:
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    img = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples
                    )
                except Exception:
                    continue

                # Skip very small regions (likely artifacts).
                if pix.width < 10 or pix.height < 10:
                    continue

                # For IMAGE regions, record a placeholder — we don't OCR images.
                if region.type == RegionType.IMAGE:
                    segments.append(
                        Segment(
                            index=len(segments),
                            backend=Backend.PIPELINE,
                            page_index=layout_page.index,
                            type=RegionType.IMAGE,
                            content="[image]",
                            bbox=region.bbox,
                            source_region_id=region.region_id,
                        )
                    )
                    continue

                # Run OCR on the rendered region.
                text = engine.recognize(img)
                text = text.strip()
                if not text:
                    continue

                segments.append(
                    Segment(
                        index=len(segments),
                        backend=Backend.PIPELINE,
                        page_index=layout_page.index,
                        type=region.type,
                        content=text,
                        bbox=region.bbox,
                        source_region_id=region.region_id,
                    )
                )

        seg_tuple = tuple(segments)
        markdown = merge_segments_to_markdown(seg_tuple)

        stats: dict[str, Any] = {
            "page_count": len(layout.pages),
            "pages_extracted": pages_extracted,
            "pages_skipped": pages_skipped,
            "segment_count": len(seg_tuple),
            "char_count": len(markdown),
            "ocr_engine": self.config.ocr_engine,
            "render_dpi": dpi,
        }

        return ExtractedDoc(
            sha256=sha256,
            backend=Backend.PIPELINE,
            segments=seg_tuple,
            markdown=markdown,
            stats=stats,
        )


# ---------------------------------------------------------------- convenience

def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: PipelineConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: create a parser, extract, return."""
    parser = PipelineParser(config=config)
    return parser.extract(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
