"""pdfsys-parser-vlm — VLM extraction backend.

Handles the "needs-ocr AND complex content" branch (tables, formulas,
heavy mixed layouts). Uses MinerU 2.5 Pro (magic-pdf) for end-to-end
processing of complex document pages. GPU path.

Supports both the newer ``mineru`` (>= 2.0) and older ``magic_pdf`` APIs.
"""

from __future__ import annotations

from .extract import VlmParser, extract_doc, extract_doc_from_layout

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "VlmParser",
    "extract_doc",
    "extract_doc_from_layout",
]
