"""pdfsys-parser-vlm — out-of-process mineru VLM-mode parser.

Talks to a ``mineru-api`` subprocess over HTTP. The bench / CLI client
never imports mineru, so MLX + torch + Metal cannot collide with the
client's import surface. Spec:
``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

from .extract import VlmParser

__version__ = "0.2.0"

__all__ = [
    "VlmParser",
    "__version__",
]
