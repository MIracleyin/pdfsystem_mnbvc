"""pdfsys-parser-mupdf — text-ok extraction backend.

Consumes PDFs classified as text-ok by pdfsys-router. Uses PyMuPDF for
block extraction, simple two-column reading order, and emits Markdown.
Does NOT depend on pdfsys-layout-analyser.
"""

__version__ = "0.0.1"
