"""Core enum types and the immutable PdfRecord metadata object.

Canonical definitions live in :mod:`pdfsys_types.types`; this module
re-exports them so existing ``from pdfsys_core import X`` and
``isinstance(x, pdfsys_core.X)`` continue to work after the parsers were
extracted to an external submodule.
"""

from __future__ import annotations

from pdfsys_types.types import Backend, PdfRecord, RegionType

__all__ = ["Backend", "PdfRecord", "RegionType"]
