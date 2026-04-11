"""pdfsys-parser-mupdf — text-ok extraction backend.

Consumes PDFs classified as text-ok by pdfsys-router. Uses PyMuPDF for
block extraction (``page.get_text("blocks", sort=True)``) and emits
Markdown. Does NOT depend on pdfsys-layout-analyser.
"""

from __future__ import annotations

from .extract import extract_doc, extract_doc_bytes

__version__ = "0.0.1"

__all__ = ["__version__", "extract_doc", "extract_doc_bytes"]
