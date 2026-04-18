"""pdfsys-parser-pipeline — OCR-pipeline backend.

Handles the "needs-ocr AND no complex content" branch. Reads the cached
LayoutDocument produced by pdfsys-layout-analyser, renders each region via
PyMuPDF, runs line-level OCR (RapidOCR / PaddleOCR-classic, selectable via
config), and assembles the Markdown output. CPU-friendly.
"""

from __future__ import annotations

from .extract import PipelineParser, extract_doc_from_layout
from .ocr_engine import OcrEngine, RapidOcrEngine, create_ocr_engine

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "PipelineParser",
    "extract_doc_from_layout",
    "OcrEngine",
    "RapidOcrEngine",
    "create_ocr_engine",
]
