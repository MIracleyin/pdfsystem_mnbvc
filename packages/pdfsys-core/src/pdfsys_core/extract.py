"""Output schema shared by every parser backend.

All three backends (mupdf, pipeline, vlm) emit an :class:`ExtractedDoc`.
The structure is deliberately identical across backends so that downstream
stages (dedup, quality, tokenize) do not care which backend produced a doc.

A :class:`Segment` represents one block-level semantic unit:

* ``TEXT``    — ``content`` is Markdown (paragraph-level).
* ``TABLE``   — ``content`` is HTML (use conversion helpers to/from OTSL).
* ``FORMULA`` — ``content`` is a LaTeX string (display formula).
* ``IMAGE``   — ``content`` is a path (or a data URI) to the extracted image.

The mupdf backend fills ``bbox`` directly from PyMuPDF's native block
extraction (``page.get_text("blocks")``), which already returns
paragraph-shaped blocks with coordinates. The pipeline/vlm backends copy
``bbox`` from the upstream :class:`LayoutRegion` they consumed.

:class:`ExtractedDoc` also carries a merged ``markdown`` string — the
segments joined in reading order with blank-line separators. Simple
downstream consumers read ``markdown`` and ignore the segment list;
richer consumers (PII localization, region-level quality scoring) walk
``segments`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .layout import BBox
from .types import Backend, RegionType


@dataclass(frozen=True, slots=True)
class Segment:
    """A block-level extracted unit. See module docstring for ``content`` semantics."""

    index: int
    backend: Backend
    page_index: int
    type: RegionType
    content: str
    bbox: BBox | None = None
    # region_id of the upstream LayoutRegion this segment derives from.
    # None for the mupdf backend (which does not consume a LayoutDocument).
    source_region_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDoc:
    """Per-PDF extraction output. Written once per (sha256, backend) pair."""

    sha256: str
    backend: Backend
    segments: tuple[Segment, ...]
    markdown: str
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def char_count(self) -> int:
        return len(self.markdown)


def merge_segments_to_markdown(segments: tuple[Segment, ...]) -> str:
    """Join segment contents in order with blank-line separators.

    Backends call this after assembling their segment list to produce the
    top-level :attr:`ExtractedDoc.markdown` field. Keeping the merge logic
    here guarantees all three backends produce identically-shaped output.
    """
    return "\n\n".join(seg.content for seg in segments if seg.content).strip() + "\n"
