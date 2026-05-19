"""pdfsys-bench — evaluation harness and MVP closed-loop runner.

Runs a PDF directory through router → parser → OCR-quality scorer and
writes one JSONL row per PDF. This is the minimal end-to-end harness; a
richer benchmark (throughput, F1 against gold Markdown, cross-backend
comparison) will layer on top of it.
"""

from __future__ import annotations

from .cascade import (
    CascadeAttempt,
    CascadeResult,
    CascadeStage,
    run_cascade,
)
from .loop import LoopResult, run_loop
from .quality import OcrQualityScorer, QualityScore
from .quality_rules import HardCheckResult, check_extracted_text

__version__ = "0.0.1"

__all__ = [
    "CascadeAttempt",
    "CascadeResult",
    "CascadeStage",
    "HardCheckResult",
    "LoopResult",
    "OcrQualityScorer",
    "QualityScore",
    "__version__",
    "check_extracted_text",
    "run_cascade",
    "run_loop",
]
