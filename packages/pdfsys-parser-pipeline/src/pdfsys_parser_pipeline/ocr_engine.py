"""Pluggable line-level OCR engine wrapper.

Provides a uniform interface over RapidOCR (default) and PaddleOCR-classic.
Heavy deps are imported lazily so that importing this module is free.

Usage::

    engine = create_ocr_engine("rapidocr")
    text = engine.recognize(pil_image)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OcrEngine(Protocol):
    """Minimal protocol that any OCR engine must satisfy."""

    def recognize(self, image: Any) -> str:
        """Run OCR on a PIL Image or numpy array. Return the extracted text."""
        ...


class RapidOcrEngine:
    """RapidOCR (ONNX Runtime) wrapper.

    Uses ``rapidocr_onnxruntime`` which ships its own lightweight PaddleOCR
    ONNX models — no PaddlePaddle required, pure CPU inference.
    """

    def __init__(self, languages: tuple[str, ...] = ("ch", "en")) -> None:
        self._languages = languages
        self._engine: Any = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

        self._engine = RapidOCR()

    def recognize(self, image: Any) -> str:
        """Run RapidOCR on an image. Returns concatenated text lines."""
        import numpy as np  # noqa: PLC0415

        self._ensure_loaded()
        assert self._engine is not None

        # RapidOCR accepts numpy arrays or PIL Images.
        # Ensure we pass a numpy array for consistency.
        if hasattr(image, "convert"):
            # PIL Image → numpy
            image = np.array(image)

        result, _ = self._engine(image)
        if result is None:
            return ""

        # result is a list of [bbox, text, confidence] triples.
        lines: list[str] = []
        for item in result:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())

        return "\n".join(lines)


class PaddleOcrEngine:
    """PaddleOCR-classic wrapper (requires ``paddleocr`` + ``paddlepaddle``).

    Only instantiate this if PaddlePaddle is installed; otherwise use
    :class:`RapidOcrEngine` as the default.
    """

    def __init__(
        self,
        languages: tuple[str, ...] = ("ch",),
        use_gpu: bool = False,
    ) -> None:
        self._languages = languages
        self._use_gpu = use_gpu
        self._engine: Any = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from paddleocr import PaddleOCR  # noqa: PLC0415

        self._engine = PaddleOCR(
            lang=self._languages[0] if self._languages else "ch",
            use_gpu=self._use_gpu,
            show_log=False,
        )

    def recognize(self, image: Any) -> str:
        """Run PaddleOCR on an image. Returns concatenated text lines."""
        import numpy as np  # noqa: PLC0415

        self._ensure_loaded()
        assert self._engine is not None

        if hasattr(image, "convert"):
            image = np.array(image)

        result = self._engine.ocr(image, cls=True)
        if result is None or not result:
            return ""

        lines: list[str] = []
        for page_result in result:
            if page_result is None:
                continue
            for item in page_result:
                if len(item) >= 2 and item[1] and len(item[1]) >= 1:
                    lines.append(str(item[1][0]).strip())

        return "\n".join(lines)


def create_ocr_engine(
    engine_name: str = "rapidocr",
    languages: tuple[str, ...] = ("ch", "en"),
    use_gpu: bool = False,
) -> OcrEngine:
    """Factory: create an OCR engine by name.

    Supported names: ``"rapidocr"`` (default), ``"paddleocr"``.
    """
    if engine_name == "rapidocr":
        return RapidOcrEngine(languages=languages)
    elif engine_name == "paddleocr":
        return PaddleOcrEngine(languages=languages, use_gpu=use_gpu)
    else:
        raise ValueError(
            f"Unknown OCR engine {engine_name!r}. "
            "Supported: 'rapidocr', 'paddleocr'."
        )
