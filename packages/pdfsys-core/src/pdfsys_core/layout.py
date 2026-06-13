"""LayoutDocument schema — the contract between layout-analyser and every parser.

Canonical definitions live in :mod:`pdfsys_types.layout`; this module
re-exports them so existing ``from pdfsys_core import X`` and
``isinstance(x, pdfsys_core.X)`` continue to work after the parsers were
extracted to an external submodule.
"""

from __future__ import annotations

from pdfsys_types.layout import (
    BBox,
    LayoutDocument,
    LayoutPage,
    LayoutRegion,
    make_region_id,
)

__all__ = [
    "BBox",
    "LayoutRegion",
    "LayoutPage",
    "LayoutDocument",
    "make_region_id",
]
