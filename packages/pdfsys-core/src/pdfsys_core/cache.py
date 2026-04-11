"""LayoutCache — on-disk cache keyed by (sha256, layout_model_version).

Ensures layout-analyser runs at most once per PDF; both router (for complex-
content decisions) and parser-pipeline / parser-vlm read from this cache.

Stub only — real implementation lands later.
"""
