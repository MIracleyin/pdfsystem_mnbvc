"""Output schema shared by every parser backend.

Canonical definitions live in :mod:`pdfsys_types.extract`; this module
re-exports them so existing ``from pdfsys_core import X`` and
``isinstance(x, pdfsys_core.X)`` continue to work after the parsers were
extracted to an external submodule.
"""

from __future__ import annotations

from pdfsys_types.extract import ExtractedDoc, Segment, merge_segments_to_markdown

__all__ = ["ExtractedDoc", "Segment", "merge_segments_to_markdown"]
