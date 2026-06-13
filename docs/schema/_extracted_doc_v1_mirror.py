# AUTO-GENERATED — DO NOT EDIT BY HAND.
# Regenerate via: python docs/schema/generate_dataclass.py
# Source of truth: docs/schema/extracted_doc.v1.json
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Backend(StrEnum):
    MUPDF = 'mupdf'
    PIPELINE = 'pipeline'
    VLM = 'vlm'
    DEFERRED = 'deferred'


class RegionType(StrEnum):
    TEXT = 'text'
    IMAGE = 'image'
    TABLE = 'table'
    FORMULA = 'formula'


@dataclass(frozen=True, slots=True)
class BBox:
    """Normalized bounding box. All coordinates are in [0.0, 1.0]; origin is top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x0", self.x0), ("y0", self.y0),
            ("x1", self.x1), ("y1", self.y1),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"BBox.{name}={value!r} outside [0, 1]")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(
                f"BBox has non-positive size: x0={self.x0} x1={self.x1} y0={self.y0} y1={self.y1}"
            )


@dataclass(frozen=True, slots=True)
class Segment:
    """A block-level extracted unit."""

    index: int
    backend: Backend
    page_index: int
    type: RegionType
    content: str
    bbox: BBox | None = None
    source_region_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDoc:
    """Per-PDF extraction output produced by any backend."""

    sha256: str
    backend: Backend
    segments: tuple[Segment, ...]
    markdown: str
    stats: dict[str, Any] = field(default_factory=dict)
