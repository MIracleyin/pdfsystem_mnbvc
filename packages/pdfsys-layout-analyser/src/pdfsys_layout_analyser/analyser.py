"""Document layout detection using DocLayout-YOLO.

Renders each PDF page to an image via PyMuPDF, runs DocLayout-YOLO (a
YOLO-based detector trained on DocStructBench), and maps detections to
:class:`pdfsys_core.LayoutDocument`.

The default model is ``juliozhao/DocLayout-YOLO-DocStructBench`` from
HuggingFace Hub. This can be overridden via the ``model_path`` parameter
to use PP-DocLayoutV3 or any compatible YOLO checkpoint.

Heavy dependencies (``doclayout_yolo``, ``torch``) are imported lazily so
that merely importing :mod:`pdfsys_layout_analyser` does not pull them in.

DocLayout-YOLO class labels::

    0: title         → TEXT
    1: plain text    → TEXT
    2: abandon       → (skipped)
    3: figure        → IMAGE
    4: figure_caption → TEXT
    5: table         → TABLE
    6: table_caption → TEXT
    7: table_footnote → TEXT
    8: isolate_formula → FORMULA
    9: formula_caption → TEXT
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pdfsys_core import (
    BBox,
    LayoutConfig,
    LayoutDocument,
    LayoutPage,
    LayoutRegion,
    RegionType,
    make_region_id,
)

# DocLayout-YOLO class name → RegionType mapping.
_CLASS_MAP: dict[str, RegionType] = {
    "title": RegionType.TEXT,
    "plain text": RegionType.TEXT,
    "Text": RegionType.TEXT,
    "Title": RegionType.TEXT,
    "figure_caption": RegionType.TEXT,
    "table_caption": RegionType.TEXT,
    "table_footnote": RegionType.TEXT,
    "formula_caption": RegionType.TEXT,
    "Figure caption": RegionType.TEXT,
    "Table caption": RegionType.TEXT,
    "Table footnote": RegionType.TEXT,
    "Formula caption": RegionType.TEXT,
    "figure": RegionType.IMAGE,
    "Figure": RegionType.IMAGE,
    "table": RegionType.TABLE,
    "Table": RegionType.TABLE,
    "isolate_formula": RegionType.FORMULA,
    "Isolate formula": RegionType.FORMULA,
}

# Classes to skip entirely.
_SKIP_CLASSES = {"abandon", "Abandon"}

DEFAULT_MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DEFAULT_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


class LayoutAnalyser:
    """Runs layout detection on every page of a PDF.

    Usage::

        analyser = LayoutAnalyser()
        layout = analyser.analyse("doc.pdf", sha256="abc123...")
        cache.save(layout)
    """

    def __init__(
        self,
        config: LayoutConfig | None = None,
        model_path: str | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self.config = config or LayoutConfig(
            model_name="doclayout-yolo-docstructbench",
            model_version="1.0",
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model_path = model_path or DEFAULT_MODEL_REPO
        self._model: Any = None

    # ------------------------------------------------------------------ lazy

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from doclayout_yolo import YOLOv10  # noqa: PLC0415

        model_path = self._model_path

        # If it looks like a HuggingFace repo ID (contains /), download the
        # model file first via huggingface_hub.
        if "/" in model_path and not Path(model_path).exists():
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            model_path = hf_hub_download(
                repo_id=model_path,
                filename=DEFAULT_MODEL_FILENAME,
            )

        self._model = YOLOv10(model_path)

    # ------------------------------------------------------------------ api

    def analyse(self, pdf_path: str | Path, sha256: str | None = None) -> LayoutDocument:
        """Analyse all pages of *pdf_path* and return a :class:`LayoutDocument`.

        If *sha256* is not provided it is computed from the file contents.
        """
        import pymupdf  # noqa: PLC0415

        self._ensure_model()

        path = Path(pdf_path)
        if sha256 is None:
            sha256 = _sha256_of_file(path)

        doc = pymupdf.open(str(path))
        try:
            pages: list[LayoutPage] = []
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                layout_page = self._analyse_page(page, page_idx)
                pages.append(layout_page)

            return LayoutDocument(
                sha256=sha256,
                layout_model=self.config.model_tag,
                pages=tuple(pages),
            )
        finally:
            doc.close()

    def analyse_bytes(
        self, pdf_bytes: bytes, sha256: str | None = None
    ) -> LayoutDocument:
        """Same as :meth:`analyse`, but from an in-memory buffer."""
        import io

        import pymupdf  # noqa: PLC0415

        self._ensure_model()

        sha = sha256 or hashlib.sha256(pdf_bytes).hexdigest()
        doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        try:
            pages: list[LayoutPage] = []
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                layout_page = self._analyse_page(page, page_idx)
                pages.append(layout_page)

            return LayoutDocument(
                sha256=sha,
                layout_model=self.config.model_tag,
                pages=tuple(pages),
            )
        finally:
            doc.close()

    # --------------------------------------------------------------- internal

    def _render_page_to_pil(self, page: Any) -> Any:
        """Render a PyMuPDF page object to a PIL Image."""
        import pymupdf  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        dpi = self.config.render_dpi
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _analyse_page(self, page: Any, page_index: int) -> LayoutPage:
        """Detect layout regions on a single page."""
        page_width_pt = float(page.rect.width)
        page_height_pt = float(page.rect.height)

        img = self._render_page_to_pil(page)
        img_w, img_h = img.size

        # Run DocLayout-YOLO detection.
        results = self._model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        raw_regions: list[LayoutRegion] = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names.get(cls_id, "")

                    if cls_name in _SKIP_CLASSES:
                        continue

                    region_type = _CLASS_MAP.get(cls_name, RegionType.TEXT)
                    conf = float(box.conf.item())

                    # Normalize bbox to [0, 1].
                    x0, y0, x1, y1 = box.xyxy[0].tolist()
                    nx0 = max(0.0, min(1.0, x0 / img_w))
                    ny0 = max(0.0, min(1.0, y0 / img_h))
                    nx1 = max(0.0, min(1.0, x1 / img_w))
                    ny1 = max(0.0, min(1.0, y1 / img_h))

                    if nx1 <= nx0 or ny1 <= ny0:
                        continue

                    raw_regions.append(
                        LayoutRegion(
                            region_id="",  # placeholder, reassigned below
                            type=region_type,
                            bbox=BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1),
                            confidence=conf,
                            reading_order=0,
                        )
                    )

        # Sort by reading order: top-to-bottom, left-to-right.
        raw_regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))

        # Reassign stable region ids and reading order after sort.
        final_regions: list[LayoutRegion] = []
        for i, r in enumerate(raw_regions):
            final_regions.append(
                LayoutRegion(
                    region_id=make_region_id(page_index, i),
                    type=r.type,
                    bbox=r.bbox,
                    confidence=r.confidence,
                    reading_order=i,
                )
            )

        return LayoutPage(
            index=page_index,
            page_width_pt=page_width_pt,
            page_height_pt=page_height_pt,
            regions=tuple(final_regions),
        )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
