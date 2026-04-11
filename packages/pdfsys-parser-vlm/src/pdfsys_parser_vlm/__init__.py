"""pdfsys-parser-vlm — VLM extraction backend.

Handles the "needs-ocr AND complex content" branch (tables, formulas,
heavy mixed layouts). Reads the cached LayoutDocument and only routes
complex regions through a VLM (MinerU 2.5 / PaddleOCR-VL). GPU path.
"""

__version__ = "0.0.1"
