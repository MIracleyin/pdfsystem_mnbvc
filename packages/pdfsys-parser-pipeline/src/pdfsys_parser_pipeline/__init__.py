"""pdfsys-parser-pipeline — mineru pipeline-mode wrapper.

Thin shim over ``mineru.cli.common.do_parse(backend="pipeline")``. The
old RapidOCR / region-OCR pipeline was deleted in the mineru migration
(2026-05-22, spec: docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md).
"""

from __future__ import annotations

from .extract import PipelineParser

__version__ = "0.1.0"

__all__ = [
    "PipelineParser",
    "__version__",
]
