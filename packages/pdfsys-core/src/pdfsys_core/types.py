"""Core enum types and the immutable PdfRecord metadata object.

This module has zero dependencies beyond the stdlib. It is imported by
every other module in pdfsys_core, so keep it free of I/O and domain logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RegionType(str, Enum):
    """Region classification used by layout-analyser and parser backends.

    Deliberately coarse: we extract text, not reconstruct document structure.
    Headers, titles, lists, captions and footnotes all collapse into TEXT.
    Inline formulas stay inside TEXT (as LaTeX spans); only display-level
    formulas get their own FORMULA region.
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"


class Backend(str, Enum):
    """Which extraction backend produced (or will produce) a given document."""

    MUPDF = "mupdf"
    PIPELINE = "pipeline"
    VLM = "vlm"
    # Complex pages held back when no VLM lane is currently available.
    # DEFERRED records are re-scanned and re-dispatched in a later batch.
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class PdfRecord:
    """Immutable metadata describing a single source PDF.

    PdfRecord is intentionally state-free: it carries no stage status, no
    backend selection, no timestamps beyond discovery. Per-stage outputs
    live in their own files on disk (content-addressable by ``sha256``);
    resumability is "does the output file exist?".

    ``provenance`` is a free-form string slot for upstream ingest pipelines
    to stamp origin metadata (e.g. source batch id, crawl host, policy tier).
    Upstream may store arbitrary JSON in there; pdfsys-core does not parse it.
    """

    sha256: str
    source_uri: str
    size_bytes: int
    mime: str = "application/pdf"
    discovered_at: float = 0.0  # unix epoch seconds; 0 means unknown
    provenance: str = ""  # opaque to pdfsys-core; upstream may dump JSON here
