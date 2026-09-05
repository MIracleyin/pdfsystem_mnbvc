"""Runtime configuration dataclasses shared across every pdfsys stage.

Canonical definitions live in :mod:`pdfsys_types.config`; this module
re-exports them so existing ``from pdfsys_core import X`` and
``isinstance(x, pdfsys_core.X)`` continue to work after the parsers were
extracted to an external submodule.
"""

from __future__ import annotations

from pdfsys_types.config import (
    LayoutConfig,
    MupdfConfig,
    PathsConfig,
    PdfsysConfig,
    PipelineConfig,
    RouterConfig,
    RuntimeConfig,
    VlmConfig,
)

__all__ = [
    "LayoutConfig",
    "MupdfConfig",
    "PathsConfig",
    "PdfsysConfig",
    "PipelineConfig",
    "RouterConfig",
    "RuntimeConfig",
    "VlmConfig",
]
