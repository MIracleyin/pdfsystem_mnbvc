"""VLM extraction backend — region-based via mineru 3.x MinerUClient.

Replaces the prior do_parse-based implementation (2026-05-14). Layout
analyser (DocLayout-YOLO) is the single source of truth for which
regions exist on each page; mineru's job narrows to "recognize this
cropped region as type X".

Pipeline:
    1. Filter to complex pages (any TABLE / FORMULA region).
    2. For each complex page:
       a. Render the page to PIL via PyMuPDF (DPI 200).
       b. Crop each region's bbox from the page image.
       c. Build (crops, types) lists and call
          MinerUClient.batch_content_extract(crops, types).
       d. Each result becomes a Segment.
    3. merge_segments_to_markdown handles reading order.

Heavy dependencies (mineru, mineru_vl_utils, pymupdf, PIL) are imported
lazily inside the methods that need them.

Verified API contract: see out/mineru_region_spike_notes.md (2026-05-16).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    ExtractedDoc,
    LayoutDocument,
    RegionType,
    Segment,
    VlmConfig,
    merge_segments_to_markdown,
)

# pdfsys RegionType → mineru type for batch_content_extract.
# Only 4 entries — RegionType enum has exactly these values.
_REGIONTYPE_TO_MINERU: dict[RegionType, str] = {
    RegionType.TEXT: "text",
    RegionType.IMAGE: "image",
    RegionType.TABLE: "table",
    RegionType.FORMULA: "equation",
}

# Minimum crop dimension (px). mineru VL rejects tiny images.
_MIN_CROP_PX = 10

# DPI for PDF page → image rendering. Matches pdfsys-layout-analyser default.
_DEFAULT_RENDER_DPI = 200


class VlmParser:
    """Region-based VLM extraction parser using mineru 3.x."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()
        self._client = None
        self._render_dpi = _DEFAULT_RENDER_DPI

    # ------------------------------------------------------------------ api

    def extract(
        self, pdf_path: str | Path, sha256: str | None = None
    ) -> ExtractedDoc:
        """Region-based VlmParser requires a LayoutDocument."""
        raise NotImplementedError(
            "Region-based VlmParser requires a LayoutDocument. "
            "Use extract_complex_pages(pdf_path, layout)."
        )

    def extract_bytes(
        self, pdf_bytes: bytes, sha256: str | None = None
    ) -> ExtractedDoc:
        """Region-based VlmParser requires a LayoutDocument."""
        raise NotImplementedError(
            "Region-based VlmParser requires a LayoutDocument. "
            "Use extract_complex_pages(pdf_path, layout)."
        )

    def extract_complex_pages(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Extract every region on every page that has TABLE or FORMULA content.

        The runner only calls this method when Stage-B routes the PDF to
        VLM (which happens iff layout.has_complex_content is true). Within
        such pages, ALL region types (TEXT included) are extracted via
        mineru VL — keeping the lane single-engine.
        """
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)

        complex_pages = {
            lp.index
            for lp in layout.pages
            if any(r.type in (RegionType.TABLE, RegionType.FORMULA) for r in lp.regions)
        }

        if not complex_pages:
            return ExtractedDoc(
                sha256=sha,
                backend=Backend.VLM,
                segments=(),
                markdown="",
                stats={
                    "vlm_engine": "mineru-3.x region-based",
                    "complex_pages": 0,
                    "reason": "no_complex_content",
                },
            )

        segments, page_failures, region_failures, region_type_counts = (
            self._run_vlm_per_region(path, layout, complex_pages)
        )

        seg_tuple = tuple(segments)
        markdown = merge_segments_to_markdown(seg_tuple)

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=seg_tuple,
            markdown=markdown,
            stats={
                "vlm_engine": "mineru-3.x region-based",
                "vlm_model": self.config.model,
                "complex_pages": len(complex_pages),
                "complex_page_indices": sorted(complex_pages),
                "region_count": len(seg_tuple),
                "region_failures": region_failures,
                "region_type_counts": region_type_counts,
                "page_failures": page_failures,
                "char_count": len(markdown),
                "segment_count": len(seg_tuple),
            },
        )

    # --------------------------------------------------------------- internal

    def _ensure_client(self) -> Any:
        """Lazily construct MinerUClient via mineru's ModelSingleton.

        Direct MinerUClient(backend='transformers') raises ValueError
        because model_path is required. ModelSingleton handles the
        auto-download + dtype + device-map plumbing; it also caches the
        client process-wide so do_parse and our direct path share weights.
        """
        if self._client is None:
            from mineru.backend.vlm.vlm_analyze import ModelSingleton  # noqa: PLC0415

            self._client = ModelSingleton().get_model(
                backend="transformers",
                model_path=None,
                server_url=None,
            )
        return self._client

    def _run_vlm_per_region(
        self,
        pdf_path: Path,
        layout: LayoutDocument,
        complex_pages: set[int],
    ) -> tuple[list[Segment], list[dict[str, Any]], int, dict[str, int]]:
        """Crop layout regions from complex pages, batch-extract via mineru.

        Returns (segments, page_failures, region_failures, region_type_counts).
        """
        import pymupdf  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        client = self._ensure_client()

        segments: list[Segment] = []
        page_failures: list[dict[str, Any]] = []
        region_failures = 0
        region_type_counts: dict[str, int] = {}

        doc = pymupdf.open(str(pdf_path))
        try:
            for layout_page in layout.pages:
                if layout_page.index not in complex_pages:
                    continue
                if layout_page.index >= len(doc):
                    page_failures.append({
                        "page": layout_page.index,
                        "error": "page index out of range",
                    })
                    continue

                page = doc[layout_page.index]
                pix = page.get_pixmap(dpi=self._render_dpi)
                page_img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples
                )
                W, H = page_img.size

                crops: list[Any] = []
                types: list[str] = []
                region_refs: list[Any] = []

                for region in layout_page.regions:
                    mineru_type = _REGIONTYPE_TO_MINERU.get(region.type, "text")
                    bbox = region.bbox
                    box = (
                        int(bbox.x0 * W),
                        int(bbox.y0 * H),
                        int(bbox.x1 * W),
                        int(bbox.y1 * H),
                    )
                    crop = page_img.crop(box)
                    if crop.size[0] < _MIN_CROP_PX or crop.size[1] < _MIN_CROP_PX:
                        region_failures += 1
                        continue
                    crops.append(crop)
                    types.append(mineru_type)
                    region_refs.append(region)

                if not crops:
                    continue

                try:
                    results = client.batch_content_extract(crops, types)
                except Exception as e:  # noqa: BLE001 — one page fails, others continue
                    page_failures.append({
                        "page": layout_page.index,
                        "error": f"{type(e).__name__}: {e}"[:200],
                    })
                    continue

                for region, result in zip(region_refs, results):
                    # ExtractStr is a subclass of str — see spike notes.
                    text = str(result).strip() if result is not None else ""
                    if not text:
                        region_failures += 1
                        # Still emit the segment with empty content so the
                        # page region list stays complete in viz.
                    region_type_counts[region.type.value] = (
                        region_type_counts.get(region.type.value, 0) + 1
                    )
                    segments.append(
                        Segment(
                            index=len(segments),
                            backend=Backend.VLM,
                            page_index=layout_page.index,
                            type=region.type,
                            content=text,
                            bbox=region.bbox,
                            source_region_id=None,
                        )
                    )
        finally:
            doc.close()

        return segments, page_failures, region_failures, region_type_counts


# ---------------------------------------------------------------- convenience

def extract_doc(
    pdf_path: str | Path,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """Region-based VlmParser cannot extract without a LayoutDocument; raises."""
    parser = VlmParser(config=config)
    return parser.extract(pdf_path, sha256=sha256)


def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: extract complex pages via region-based mineru."""
    parser = VlmParser(config=config)
    return parser.extract_complex_pages(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
