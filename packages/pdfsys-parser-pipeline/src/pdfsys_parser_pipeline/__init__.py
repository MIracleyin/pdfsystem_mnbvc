"""pdfsys-parser-pipeline — OCR-pipeline backend.

Handles the "needs-ocr AND no complex content" branch. Reads the cached
LayoutDocument produced by pdfsys-layout-analyser, renders each region via
PyMuPDF, runs line-level OCR (RapidOCR / PaddleOCR-classic, selectable via
config), and assembles the Markdown output. CPU-friendly.
"""

__version__ = "0.0.1"
