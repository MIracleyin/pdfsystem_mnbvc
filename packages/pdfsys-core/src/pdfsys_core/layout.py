"""LayoutDocument schema — the contract between layout-analyser and every parser.

The layout analyser runs at most once per PDF and writes a LayoutDocument to
LayoutCache. Three independent consumers then read the same object:

* ``pdfsys_router`` stage-B — for the ``has_complex_content`` decision.
* ``pdfsys_parser_pipeline`` — renders each region at its own DPI and runs OCR.
* ``pdfsys_parser_vlm`` — hands complex regions to a VLM, renders simple ones.

Because layout is produced once and consumed many times, the schema here is
the single most load-bearing piece in pdfsys-core. Everything is frozen and
fully immutable; coordinates are normalized so downstream consumers can
re-render at any DPI without protocol renegotiation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import RegionType


def make_region_id(page_index: int, reading_order: int) -> str:
    """Stable, human-readable region id. Format: ``p{page}_r{order}``."""
    return f"p{page_index}_r{reading_order}"


@dataclass(frozen=True, slots=True)
class BBox:
    """Normalized bounding box.

    All four coordinates live in ``[0.0, 1.0]``; origin is top-left and y
    points downward (matching every CV / layout model in the wild). Parser
    backends convert to pixels or PDF points on demand using the page's
    ``page_width_pt`` / ``page_height_pt`` from :class:`LayoutPage`.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"BBox.{name}={value!r} outside [0, 1]")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(
                f"BBox has non-positive size: x0={self.x0} x1={self.x1} y0={self.y0} y1={self.y1}"
            )

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixels(self, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """Convert to integer pixel coordinates at a given render resolution."""
        return (
            round(self.x0 * width_px),
            round(self.y0 * height_px),
            round(self.x1 * width_px),
            round(self.y1 * height_px),
        )

    def to_points(
        self, page_width_pt: float, page_height_pt: float
    ) -> tuple[float, float, float, float]:
        """Convert to PDF points (1/72 inch), matching PyMuPDF's coordinate space."""
        return (
            self.x0 * page_width_pt,
            self.y0 * page_height_pt,
            self.x1 * page_width_pt,
            self.y1 * page_height_pt,
        )


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    """A single block on a page, as detected by the layout model.

    ``reading_order`` comes directly from the layout model — no geometric
    post-processing. Regions inside :class:`LayoutPage` are pre-sorted by it.
    """

    region_id: str
    type: RegionType
    bbox: BBox
    confidence: float
    reading_order: int


@dataclass(frozen=True, slots=True)
class LayoutPage:
    """A single page's layout.

    ``page_width_pt`` and ``page_height_pt`` are retained so downstream
    consumers can map normalized bboxes back to PDF points for PyMuPDF
    operations (e.g. cropping an image for OCR).
    """

    index: int
    page_width_pt: float
    page_height_pt: float
    regions: tuple[LayoutRegion, ...]


@dataclass(frozen=True, slots=True)
class LayoutDocument:
    """Full layout for one PDF, written to LayoutCache exactly once per
    (sha256, layout_model) pair.

    ``layout_model`` acts as a schema version tag (e.g. ``pp-doclayoutv3@1.0``).
    Bumping it invalidates old cache entries lazily — new cache files are
    written under the new tag; old files remain until pruned.
    """

    sha256: str
    layout_model: str
    pages: tuple[LayoutPage, ...]

    @property
    def has_complex_content(self) -> bool:
        """True if any region on any page is TABLE or FORMULA.

        This is the sole predicate used by pdfsys-router stage-B to pick
        between parser-pipeline and parser-vlm (/ DEFERRED). Computed on
        demand via short-circuit ``any``; the cost is negligible and
        avoiding caching keeps the dataclass frozen + slotted.
        """
        complex_types = (RegionType.TABLE, RegionType.FORMULA)
        return any(region.type in complex_types for page in self.pages for region in page.regions)

    @property
    def page_count(self) -> int:
        return len(self.pages)
