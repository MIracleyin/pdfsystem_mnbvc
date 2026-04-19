"""Document layout detection with pluggable model backends.

Supports two backends:

1. **DocLayout-YOLO** (default) — ``juliozhao/DocLayout-YOLO-DocStructBench``
   YOLO-based detector via ``doclayout_yolo``. Fast, lightweight.

2. **PP-DocLayoutV3** — ``PaddlePaddle/PP-DocLayoutV3_safetensors``
   RT-DETR Transformer via HuggingFace ``transformers``. More accurate,
   supports reading-order prediction natively.

Select backend via ``model_path``:
- Any path/repo containing ``DocLayout-YOLO`` → YOLO backend
- ``PaddlePaddle/PP-DocLayoutV3*`` → transformers backend
- Or set ``backend="yolo"`` / ``backend="pp-doclayoutv3"`` explicitly.

Heavy dependencies imported lazily.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from pdfsys_core import (
    BBox,
    LayoutConfig,
    LayoutDocument,
    LayoutPage,
    LayoutRegion,
    RegionType,
    make_region_id,
)

# ---------------------------------------------------------------- label maps

# DocLayout-YOLO class name → RegionType
_YOLO_CLASS_MAP: dict[str, RegionType] = {
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
_YOLO_SKIP = {"abandon", "Abandon"}

# PP-DocLayoutV3 label → RegionType
_PPV3_CLASS_MAP: dict[str, RegionType] = {
    "paragraph": RegionType.TEXT,
    "text": RegionType.TEXT,
    "title": RegionType.TEXT,
    "header": RegionType.TEXT,
    "footer": RegionType.TEXT,
    "caption": RegionType.TEXT,
    "figure_caption": RegionType.TEXT,
    "table_caption": RegionType.TEXT,
    "reference": RegionType.TEXT,
    "abstract": RegionType.TEXT,
    "content": RegionType.TEXT,
    "toc": RegionType.TEXT,
    "figure": RegionType.IMAGE,
    "image": RegionType.IMAGE,
    "table": RegionType.TABLE,
    "formula": RegionType.FORMULA,
    "equation": RegionType.FORMULA,
    "seal": RegionType.IMAGE,
}
_PPV3_SKIP: set[str] = set()

DEFAULT_YOLO_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DEFAULT_YOLO_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DEFAULT_PPV3_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


# ---------------------------------------------------------------- backend protocol

class DetectionResult:
    """One detected region from a model backend."""

    __slots__ = ("label", "confidence", "x0", "y0", "x1", "y1")

    def __init__(self, label: str, confidence: float,
                 x0: float, y0: float, x1: float, y1: float) -> None:
        self.label = label
        self.confidence = confidence
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class _YoloBackend:
    """DocLayout-YOLO backend."""

    def __init__(self, model_path: str, conf: float, iou: float) -> None:
        from doclayout_yolo import YOLOv10  # noqa: PLC0415

        resolved = model_path
        if "/" in model_path and not Path(model_path).exists():
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            resolved = hf_hub_download(repo_id=model_path, filename=DEFAULT_YOLO_FILENAME)

        self._model = YOLOv10(resolved)
        self._conf = conf
        self._iou = iou
        self.class_map = _YOLO_CLASS_MAP
        self.skip_classes = _YOLO_SKIP
        self.name = "doclayout-yolo-docstructbench"
        self.version = "1.0"

    def detect(self, image: Any) -> list[DetectionResult]:
        """Run detection, return normalized [0,1] boxes."""
        results = self._model.predict(image, conf=self._conf, iou=self._iou, verbose=False)
        if not results or len(results) == 0:
            return []

        img_w, img_h = image.size
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[DetectionResult] = []
        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names.get(cls_id, "")
            if cls_name in self.skip_classes:
                continue
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            detections.append(DetectionResult(
                label=cls_name,
                confidence=float(box.conf.item()),
                x0=x0 / img_w, y0=y0 / img_h,
                x1=x1 / img_w, y1=y1 / img_h,
            ))
        return detections


class _PPv3Backend:
    """PP-DocLayoutV3 backend via HuggingFace transformers."""

    def __init__(self, model_path: str, conf: float) -> None:
        from transformers import (  # noqa: PLC0415
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )

        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model = AutoModelForObjectDetection.from_pretrained(model_path)
        self._model.eval()
        self._conf = conf
        self._id2label = self._model.config.id2label
        self.class_map = _PPV3_CLASS_MAP
        self.skip_classes = _PPV3_SKIP
        self.name = "pp-doclayoutv3"
        self.version = "1.0"

    def detect(self, image: Any) -> list[DetectionResult]:
        """Run detection, return normalized [0,1] boxes."""
        import torch  # noqa: PLC0415

        inputs = self._processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)

        img_w, img_h = image.size
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=[(img_h, img_w)], threshold=self._conf
        )

        if not results:
            return []

        detections: list[DetectionResult] = []
        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            label_name = self._id2label.get(label_id.item(), f"class_{label_id.item()}")
            if label_name.lower() in self.skip_classes:
                continue
            x0, y0, x1, y1 = box.tolist()
            detections.append(DetectionResult(
                label=label_name.lower(),
                confidence=float(score.item()),
                x0=x0 / img_w, y0=y0 / img_h,
                x1=x1 / img_w, y1=y1 / img_h,
            ))
        return detections


# ---------------------------------------------------------------- main class

def _guess_backend(model_path: str) -> str:
    """Guess which backend to use based on the model path."""
    lower = model_path.lower()
    if "pp-doclayout" in lower or "paddlepaddle" in lower:
        return "pp-doclayoutv3"
    return "yolo"


class LayoutAnalyser:
    """Runs layout detection on every page of a PDF.

    Usage::

        # DocLayout-YOLO (default)
        analyser = LayoutAnalyser()

        # PP-DocLayoutV3
        analyser = LayoutAnalyser(model_path="PaddlePaddle/PP-DocLayoutV3_safetensors")

        # Explicit backend
        analyser = LayoutAnalyser(backend="pp-doclayoutv3")

        layout = analyser.analyse("doc.pdf")
        cache.save(layout)
    """

    def __init__(
        self,
        config: LayoutConfig | None = None,
        model_path: str | None = None,
        backend: str | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self._model_path = model_path or DEFAULT_YOLO_REPO
        self._backend_name = backend or _guess_backend(self._model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._backend: _YoloBackend | _PPv3Backend | None = None

        if config:
            self.config = config
        else:
            name = (self._backend_name if self._backend_name == "pp-doclayoutv3"
                    else "doclayout-yolo-docstructbench")
            self.config = LayoutConfig(model_name=name, model_version="1.0")

    # ------------------------------------------------------------------ lazy

    def _ensure_backend(self) -> _YoloBackend | _PPv3Backend:
        if self._backend is not None:
            return self._backend

        if self._backend_name == "pp-doclayoutv3":
            model = self._model_path
            if model == DEFAULT_YOLO_REPO:
                model = DEFAULT_PPV3_REPO
            self._backend = _PPv3Backend(model, self.conf_threshold)
        else:
            self._backend = _YoloBackend(
                self._model_path, self.conf_threshold, self.iou_threshold
            )

        # Update config to reflect actual model.
        self.config = LayoutConfig(
            model_name=self._backend.name,
            model_version=self._backend.version,
            render_dpi=self.config.render_dpi,
        )
        return self._backend

    # ------------------------------------------------------------------ api

    def analyse(self, pdf_path: str | Path, sha256: str | None = None) -> LayoutDocument:
        """Analyse all pages of *pdf_path* and return a :class:`LayoutDocument`."""
        import pymupdf  # noqa: PLC0415

        self._ensure_backend()
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

    def analyse_bytes(self, pdf_bytes: bytes, sha256: str | None = None) -> LayoutDocument:
        """Same as :meth:`analyse`, but from an in-memory buffer."""
        import io
        import pymupdf  # noqa: PLC0415

        self._ensure_backend()
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
        import pymupdf  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        dpi = self.config.render_dpi
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _analyse_page(self, page: Any, page_index: int) -> LayoutPage:
        page_width_pt = float(page.rect.width)
        page_height_pt = float(page.rect.height)

        img = self._render_page_to_pil(page)
        backend = self._ensure_backend()
        detections = backend.detect(img)

        raw_regions: list[LayoutRegion] = []
        for det in detections:
            region_type = backend.class_map.get(det.label, RegionType.TEXT)
            nx0 = max(0.0, min(1.0, det.x0))
            ny0 = max(0.0, min(1.0, det.y0))
            nx1 = max(0.0, min(1.0, det.x1))
            ny1 = max(0.0, min(1.0, det.y1))
            if nx1 <= nx0 or ny1 <= ny0:
                continue
            raw_regions.append(LayoutRegion(
                region_id="",
                type=region_type,
                bbox=BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1),
                confidence=det.confidence,
                reading_order=0,
            ))

        # Sort top-to-bottom, left-to-right.
        raw_regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))

        final_regions = [
            LayoutRegion(
                region_id=make_region_id(page_index, i),
                type=r.type, bbox=r.bbox,
                confidence=r.confidence, reading_order=i,
            )
            for i, r in enumerate(raw_regions)
        ]

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
