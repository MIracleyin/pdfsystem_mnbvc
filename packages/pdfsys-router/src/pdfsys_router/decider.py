"""Stage-B decider: pipeline vs vlm, driven by LayoutDocument.has_complex_content.

After Stage-A routes a PDF to "needs-ocr", the layout analyser runs and
writes a :class:`pdfsys_core.LayoutDocument` to cache. Stage-B reads that
cached layout and decides:

* No complex content (no TABLE/FORMULA regions) → ``Backend.PIPELINE``
* Has complex content AND VLM lane enabled     → ``Backend.VLM``
* Has complex content AND VLM lane disabled     → ``Backend.DEFERRED``

This module is called by :class:`classifier.Router` when the layout cache
is populated. It does not do any I/O itself — just reads the pre-computed
:attr:`LayoutDocument.has_complex_content` flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdfsys_core import (
    Backend,
    LayoutCache,
    LayoutDocument,
    RouterConfig,
)


@dataclass(slots=True)
class StageBDecision:
    """Result of the Stage-B decision."""

    backend: Backend
    has_complex_content: bool
    layout_model: str
    num_pages: int
    num_regions: int
    num_complex_regions: int


def decide(
    layout: LayoutDocument,
    config: RouterConfig | None = None,
) -> StageBDecision:
    """Given a LayoutDocument, decide which backend to use.

    Parameters
    ----------
    layout:
        The layout document produced by the layout analyser.
    config:
        Router configuration. Only ``vlm_enabled`` is consulted.
    """
    cfg = config or RouterConfig()

    from pdfsys_core import RegionType  # noqa: PLC0415

    complex_types = (RegionType.TABLE, RegionType.FORMULA)
    total_regions = 0
    complex_count = 0
    for page in layout.pages:
        for region in page.regions:
            total_regions += 1
            if region.type in complex_types:
                complex_count += 1

    has_complex = layout.has_complex_content

    if has_complex:
        backend = Backend.VLM if cfg.vlm_enabled else Backend.DEFERRED
    else:
        backend = Backend.PIPELINE

    return StageBDecision(
        backend=backend,
        has_complex_content=has_complex,
        layout_model=layout.layout_model,
        num_pages=layout.page_count,
        num_regions=total_regions,
        num_complex_regions=complex_count,
    )


def decide_from_cache(
    sha256: str,
    layout_model_tag: str,
    cache: LayoutCache,
    config: RouterConfig | None = None,
) -> StageBDecision | None:
    """Convenience: load from cache and decide. Returns None if not cached."""
    if not cache.exists(sha256, layout_model_tag):
        return None
    layout = cache.load(sha256, layout_model_tag)
    return decide(layout, config=config)
