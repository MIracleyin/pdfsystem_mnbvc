"""pdfsys-layout-analyser — page layout detection.

Runs exactly once per PDF on the needs-ocr branch; the produced
LayoutDocument is cached and read by both pdfsys-router (stage-B decision)
and the downstream parser backends (pipeline / vlm).

Default model: DocLayout-YOLO-DocStructBench (HuggingFace).
Alternative: PP-DocLayoutV3 via PaddleX (pass ``model_path`` to
:class:`LayoutAnalyser`).
"""

from __future__ import annotations

from .analyser import LayoutAnalyser

__version__ = "0.0.1"

__all__ = ["__version__", "LayoutAnalyser"]
