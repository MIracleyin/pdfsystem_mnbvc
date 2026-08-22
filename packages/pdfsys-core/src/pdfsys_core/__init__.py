"""pdfsys-core — shared data contracts for the pdfsys pipeline.

This package holds only pure-Python dataclasses, enums, configuration, and
the content-addressable LayoutCache. It MUST NOT depend on any PDF, OCR,
or ML library — its only imports are from the Python stdlib.

Public surface:

* Enums         — :class:`RegionType`, :class:`Backend`
* Metadata      — :class:`PdfRecord`
* Layout schema — :class:`BBox`, :class:`LayoutRegion`, :class:`LayoutPage`,
                  :class:`LayoutDocument`, :func:`make_region_id`
* Parser output — :class:`Segment`, :class:`ExtractedDoc`,
                  :func:`merge_segments_to_markdown`
* Dataset (L2)  — :class:`PageRecord`, :class:`Block`, :class:`ImageBlob`,
                  :class:`ImageTextPair` and the projections in
                  :mod:`pdfsys_core.dataset`
* Cache         — :class:`LayoutCache`
* Config        — :class:`PdfsysConfig` and its sub-configs
* Serde         — :func:`to_dict`, :func:`from_dict`
"""

from __future__ import annotations

from .cache import LayoutCache
from .config import (
    LayoutConfig,
    MupdfConfig,
    PathsConfig,
    PdfsysConfig,
    PipelineConfig,
    RouterConfig,
    RuntimeConfig,
    VlmConfig,
)
from .dataset import (
    DATASET_SCHEMA_VERSION,
    FURNITURE_TYPES,
    IMAGE_REF_RE,
    Block,
    BlockType,
    ImageBlob,
    ImageTextPair,
    PageRecord,
    blocks_from_content_list,
    blocks_from_segments,
    image_id_for,
    image_ref,
    iter_pairs,
    link_mentions,
    probe_image,
    render_markdown,
    split_pages,
    strip_image_refs,
    to_interleaved,
)
from .extract import ExtractedDoc, Segment, merge_segments_to_markdown
from .layout import BBox, LayoutDocument, LayoutPage, LayoutRegion, make_region_id
from .serde import from_dict, to_dict
from .types import Backend, PdfRecord, RegionType

__version__ = "0.0.1"

__all__ = [
    # version
    "__version__",
    # enums
    "Backend",
    "RegionType",
    # metadata
    "PdfRecord",
    # layout
    "BBox",
    "LayoutRegion",
    "LayoutPage",
    "LayoutDocument",
    "make_region_id",
    # extract
    "Segment",
    "ExtractedDoc",
    "merge_segments_to_markdown",
    # dataset (L2 publish format)
    "DATASET_SCHEMA_VERSION",
    "BlockType",
    "FURNITURE_TYPES",
    "IMAGE_REF_RE",
    "Block",
    "ImageBlob",
    "PageRecord",
    "ImageTextPair",
    "image_id_for",
    "image_ref",
    "strip_image_refs",
    "probe_image",
    "blocks_from_content_list",
    "blocks_from_segments",
    "link_mentions",
    "render_markdown",
    "split_pages",
    "to_interleaved",
    "iter_pairs",
    # cache
    "LayoutCache",
    # config
    "PdfsysConfig",
    "PathsConfig",
    "RouterConfig",
    "LayoutConfig",
    "MupdfConfig",
    "PipelineConfig",
    "VlmConfig",
    "RuntimeConfig",
    # serde
    "to_dict",
    "from_dict",
]
