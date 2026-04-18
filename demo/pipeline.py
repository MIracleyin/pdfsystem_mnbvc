"""End-to-end wiring used by the Gradio demo.

Wraps the three production-path components in one callable:

    Router (Stage-A XGBoost)
        └─► Backend.MUPDF  → pdfsys_parser_mupdf.extract_doc
        └─► anything else  → not extracted (Pipeline/VLM/Deferred are
                             still stubs in this repo; we surface the
                             router decision and stop).

Kept deliberately Gradio-free so the same code is unit-testable and
reusable from notebooks. ``app.py`` only imports :func:`run_pipeline`
and :func:`render_first_page_with_bboxes`.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw


# ------------------------------------------------------------------ singletons

_ROUTER: Any = None
_SCORER: Any = None


def _ensure_router_weights() -> None:
    """Make sure the XGBoost weights are on disk. No-op if already present."""
    from pdfsys_router.download_weights import download, target_path

    if not target_path().is_file():
        download()


def get_router(ocr_threshold: float = 0.5):
    """Lazy-load the singleton Router. Weights download on first call."""
    global _ROUTER
    _ensure_router_weights()
    from pdfsys_router import Router

    if _ROUTER is None or abs(_ROUTER.ocr_threshold - ocr_threshold) > 1e-9:
        _ROUTER = Router(ocr_threshold=ocr_threshold)
    return _ROUTER


def get_scorer():
    """Lazy-load the singleton ModernBERT quality scorer (~800 MB download)."""
    global _SCORER
    if _SCORER is None:
        from pdfsys_bench.quality import OcrQualityScorer

        _SCORER = OcrQualityScorer()
    return _SCORER


# ------------------------------------------------------------------ data class


@dataclass(slots=True)
class PipelineResult:
    """Everything the UI needs in one flat object."""

    # Router
    backend: str
    ocr_prob: float
    num_pages: int
    is_form: bool
    garbled_text_ratio: float
    is_encrypted: bool
    needs_password: bool
    router_error: str | None
    router_features: dict[str, Any] = field(default_factory=dict)

    # Extract (only when backend == mupdf)
    sha256: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    extract_stats: dict[str, Any] = field(default_factory=dict)
    extract_error: str | None = None

    # Quality
    quality_score: float | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    quality_error: str | None = None

    # Wall times (ms)
    wall_ms_router: float = 0.0
    wall_ms_extract: float = 0.0
    wall_ms_quality: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-friendly dict for the raw output tab."""
        return {
            "backend": self.backend,
            "ocr_prob": self.ocr_prob,
            "num_pages": self.num_pages,
            "is_form": self.is_form,
            "garbled_text_ratio": self.garbled_text_ratio,
            "is_encrypted": self.is_encrypted,
            "needs_password": self.needs_password,
            "router_error": self.router_error,
            "sha256": self.sha256,
            "num_segments": len(self.segments),
            "markdown_chars": len(self.markdown),
            "extract_stats": self.extract_stats,
            "extract_error": self.extract_error,
            "quality_score": self.quality_score,
            "quality_num_tokens": self.quality_num_tokens,
            "quality_model": self.quality_model,
            "quality_error": self.quality_error,
            "wall_ms_router": round(self.wall_ms_router, 1),
            "wall_ms_extract": round(self.wall_ms_extract, 1),
            "wall_ms_quality": round(self.wall_ms_quality, 1),
        }


# -------------------------------------------------------------------- helpers


def _segment_to_row(seg: Any) -> dict[str, Any]:
    """Flatten a :class:`pdfsys_core.Segment` for the UI table."""
    bbox = seg.bbox
    bbox_tuple = None if bbox is None else (
        round(bbox.x0, 4),
        round(bbox.y0, 4),
        round(bbox.x1, 4),
        round(bbox.y1, 4),
    )
    return {
        "index": seg.index,
        "page": seg.page_index,
        "type": seg.type.value,
        "bbox_norm": bbox_tuple,
        "chars": len(seg.content),
        "preview": seg.content[:120].replace("\n", " "),
    }


# ------------------------------------------------------------------ core entry


def run_pipeline(
    pdf_path: str | Path,
    *,
    run_quality: bool = False,
    ocr_threshold: float = 0.5,
) -> PipelineResult:
    """Route the PDF, extract if text-ok, optionally score quality.

    Never raises on malformed input — all failure modes surface via the
    ``*_error`` fields so the UI can present them uniformly.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # -- Stage-A router -------------------------------------------------------
    router = get_router(ocr_threshold=ocr_threshold)
    t0 = time.perf_counter()
    decision = router.classify(pdf_path)
    t1 = time.perf_counter()

    result = PipelineResult(
        backend=decision.backend.value,
        ocr_prob=float(decision.ocr_prob) if decision.ocr_prob == decision.ocr_prob else float("nan"),
        num_pages=decision.num_pages,
        is_form=decision.is_form,
        garbled_text_ratio=decision.garbled_text_ratio,
        is_encrypted=decision.is_encrypted,
        needs_password=decision.needs_password,
        router_error=decision.error,
        router_features=dict(decision.features or {}),
        wall_ms_router=(t1 - t0) * 1000.0,
    )

    # -- MuPDF extraction (only for text-ok path) -----------------------------
    from pdfsys_core import Backend
    from pdfsys_parser_mupdf import extract_doc

    if decision.backend == Backend.MUPDF and decision.error is None:
        try:
            t2 = time.perf_counter()
            extracted = extract_doc(pdf_path)
            t3 = time.perf_counter()
            result.sha256 = extracted.sha256
            result.segments = [_segment_to_row(s) for s in extracted.segments]
            result.markdown = extracted.markdown
            result.extract_stats = dict(extracted.stats)
            result.wall_ms_extract = (t3 - t2) * 1000.0
        except Exception as e:  # noqa: BLE001 — surface to UI
            result.extract_error = f"{type(e).__name__}: {e}"

    # -- Quality scoring (optional, heavy) ------------------------------------
    if run_quality and result.markdown:
        try:
            scorer = get_scorer()
            t4 = time.perf_counter()
            q = scorer.score(result.markdown)
            t5 = time.perf_counter()
            result.quality_score = q.score
            result.quality_num_tokens = q.num_tokens
            result.quality_model = q.model
            result.wall_ms_quality = (t5 - t4) * 1000.0
        except Exception as e:  # noqa: BLE001
            result.quality_error = f"{type(e).__name__}: {e}"

    return result


# ----------------------------------------------------------------- rendering


_BACKEND_COLOR = {
    "mupdf": (39, 174, 96),      # green — text-ok fast path
    "pipeline": (243, 156, 18),  # orange — OCR pipeline (stub)
    "vlm": (155, 89, 182),       # purple — VLM (stub)
    "deferred": (127, 140, 141), # gray — held back
}


def render_first_page_with_bboxes(
    pdf_path: str | Path,
    result: PipelineResult,
    page_index: int = 0,
    target_max_side: int = 1100,
) -> Image.Image | None:
    """Render ``page_index`` of the PDF and overlay MuPDF segment bboxes.

    Falls back to ``None`` on any failure (corrupted / encrypted / etc.).
    """
    pdf_path = Path(pdf_path)
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return None

    try:
        if len(doc) == 0 or page_index >= len(doc):
            return None
        page = doc[page_index]
        rect = page.rect
        # Scale so the longest side ~= target_max_side (for UI readability).
        zoom = max(1.0, target_max_side / max(rect.width, rect.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:
        return None
    finally:
        doc.close()

    # Overlay segment bboxes for the selected page only.
    color = _BACKEND_COLOR.get(result.backend, (52, 152, 219))
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    drawn = 0
    for seg in result.segments:
        if seg["page"] != page_index or seg["bbox_norm"] is None:
            continue
        x0, y0, x1, y1 = seg["bbox_norm"]
        box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        # Semi-transparent fill + solid outline.
        draw.rectangle(box, fill=(*color, 28), outline=(*color, 220), width=2)
        # Small index badge.
        label = str(seg["index"])
        tx, ty = box[0] + 2, box[1] + 2
        draw.rectangle((tx, ty, tx + 6 + 7 * len(label), ty + 16), fill=(*color, 220))
        draw.text((tx + 3, ty + 1), label, fill=(255, 255, 255))
        drawn += 1

    return img


def pick_curated_features(features: dict[str, Any]) -> list[list[Any]]:
    """Select a small, meaningful subset of the 124-feature vector for display.

    The full vector goes into the raw JSON tab; this is the "at a glance"
    view. Ordered by importance / interpretability, not by XGBoost column
    order.
    """
    keys_in_order = [
        "num_pages_successfully_sampled",
        "garbled_text_ratio",
        "is_form",
        "creator_or_producer_is_known_scanner",
        "num_unique_image_xrefs",
        "num_junk_image_xrefs",
        "page_level_char_counts_page1",
        "page_level_unique_font_counts_page1",
        "page_level_text_area_ratios_page1",
        "page_level_image_counts_page1",
        "page_level_bitmap_proportions_page1",
        "page_level_vector_graphics_obj_count_page1",
        "page_level_hidden_char_counts_page1",
    ]
    rows: list[list[Any]] = []
    for k in keys_in_order:
        if k in features:
            v = features[k]
            if isinstance(v, float):
                v = round(v, 4)
            rows.append([k, v])
    return rows
