"""pdfsys-bench — evaluation harness and MVP closed-loop runner.

Runs a PDF directory through router → parser → OCR-quality scorer and
writes one JSONL row per PDF. This is the minimal end-to-end harness; a
richer benchmark (throughput, F1 against gold Markdown, cross-backend
comparison) will layer on top of it.
"""

from __future__ import annotations

from .loop import LoopResult, run_loop
from .quality import OcrQualityScorer, QualityScore

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "LoopResult",
    "run_loop",
    "OcrQualityScorer",
    "QualityScore",
]
