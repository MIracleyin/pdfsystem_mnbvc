"""pdfsys-parser-vlm — mineru VLM-mode wrapper.

Thin shim over ``mineru.cli.common.do_parse(backend="vlm-<engine>")``.
The old region-based ModelSingleton path was deleted in the mineru
migration (2026-05-22).
"""

from __future__ import annotations

from . import _macos_workaround  # noqa: F401  — must run before mineru is used
from .extract import VlmParser

__version__ = "0.1.0"

__all__ = [
    "VlmParser",
    "__version__",
]
