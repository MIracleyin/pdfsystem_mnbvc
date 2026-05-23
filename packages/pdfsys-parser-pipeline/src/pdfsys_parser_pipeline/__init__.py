"""pdfsys-parser-pipeline — out-of-process mineru pipeline-mode parser.

Talks to a ``mineru-api`` subprocess over HTTP. The bench / CLI client
never imports mineru, so mineru's torch + multiprocessing + Metal
machinery cannot collide with the client's import surface. Spec:
``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

from .extract import PipelineParser

__version__ = "0.2.0"

__all__ = [
    "PipelineParser",
    "__version__",
]
