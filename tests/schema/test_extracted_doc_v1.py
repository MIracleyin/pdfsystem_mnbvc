"""Contract tests: ExtractedDoc v1 JSON schema round-trip validation.

Validates that:
1. The frozen schema file exists at docs/schema/extracted_doc.v1.json.
2. A concrete ExtractedDoc instance serialises to a JSON dict that passes
   schema validation.
3. The JSON dict round-trips back to an equal ExtractedDoc via serde helpers.

Fixtures cover two edge-case combinations:
- A pipeline-style Segment with bbox set and source_region_id populated
  (the pipeline/vlm backends copy bbox + region_id from the upstream
  LayoutRegion they consumed).
- A degenerate Segment with bbox=None: bbox is rare-but-legal-None when
  block coordinates are degenerate. source_region_id=None marks the
  mupdf backend (which does not consume a LayoutDocument).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from pdfsys_core.extract import ExtractedDoc, Segment
from pdfsys_core.layout import BBox
from pdfsys_core.serde import from_dict, to_dict
from pdfsys_core.types import Backend, RegionType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_SCHEMA_PATH = _REPO_ROOT / "docs" / "schema" / "extracted_doc.v1.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_doc() -> ExtractedDoc:
    """Construct a representative ExtractedDoc covering both bbox states."""
    seg_with_bbox = Segment(
        index=0,
        backend=Backend.PIPELINE,
        page_index=0,
        type=RegionType.TEXT,
        content="Hello, world.",
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        source_region_id="p0_r0",
    )
    # mupdf backend: source_region_id is always None (no LayoutDocument
    # consumed). bbox is normally populated from PyMuPDF block coords; this
    # fixture exercises the rare bbox=None edge case (degenerate block).
    seg_no_bbox = Segment(
        index=1,
        backend=Backend.MUPDF,
        page_index=1,
        type=RegionType.TABLE,
        content="<table><tr><td>A</td></tr></table>",
        bbox=None,
        source_region_id=None,
    )
    return ExtractedDoc(
        sha256="abc123deadbeef",
        backend=Backend.PIPELINE,
        segments=(seg_with_bbox, seg_no_bbox),
        markdown="Hello, world.\n\n<table><tr><td>A</td></tr></table>\n",
        stats={"page_count": 2, "ocr_char_count": 42},
    )


def _make_valid_doc_blob() -> dict[str, Any]:
    """Convenience: a JSON-compatible dict that validates against the schema."""
    return to_dict(_make_valid_doc())


@pytest.fixture()
def sample_doc() -> ExtractedDoc:
    return _make_valid_doc()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_schema() -> dict:
    assert _SCHEMA_PATH.exists(), (
        f"Schema file not found: {_SCHEMA_PATH}\n"
        "Run Task 0.1 to generate docs/schema/extracted_doc.v1.json first."
    )
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _segment_subschema(schema: dict) -> dict:
    """Build a standalone schema for the Segment $def, preserving $defs refs."""
    return {
        "$schema": schema.get("$schema", ""),
        "$defs": schema.get("$defs", {}),
        **schema["$defs"]["Segment"],
    }


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

def test_schema_file_exists() -> None:
    """Schema file must exist at the canonical path."""
    assert _SCHEMA_PATH.exists(), f"Missing schema: {_SCHEMA_PATH}"


def test_serialised_doc_validates_against_schema(sample_doc: ExtractedDoc) -> None:
    """to_dict output must satisfy the JSON schema."""
    schema = _load_schema()
    blob = to_dict(sample_doc)
    # jsonschema raises jsonschema.ValidationError on failure.
    jsonschema.validate(instance=blob, schema=schema)


def test_round_trip_preserves_equality(sample_doc: ExtractedDoc) -> None:
    """from_dict(to_dict(doc)) must equal the original doc."""
    blob = to_dict(sample_doc)
    restored = from_dict(ExtractedDoc, blob)
    assert restored == sample_doc


def test_segment_with_bbox_validates(sample_doc: ExtractedDoc) -> None:
    """A segment carrying a BBox must validate under the Segment subschema."""
    schema = _load_schema()
    seg_blob = to_dict(sample_doc.segments[0])
    jsonschema.validate(instance=seg_blob, schema=_segment_subschema(schema))


def test_segment_without_bbox_validates(sample_doc: ExtractedDoc) -> None:
    """A degenerate-bbox mupdf segment (bbox=None, source_region_id=None) must validate.

    bbox=None covers the rare case where PyMuPDF returns degenerate block
    coords; source_region_id=None is the mupdf-backend marker (no upstream
    LayoutDocument was consumed).
    """
    schema = _load_schema()
    seg_blob = to_dict(sample_doc.segments[1])
    jsonschema.validate(instance=seg_blob, schema=_segment_subschema(schema))


def test_stats_accepts_arbitrary_keys(sample_doc: ExtractedDoc) -> None:
    """stats must accept any string-keyed structure (additionalProperties: true)."""
    schema = _load_schema()
    blob = to_dict(sample_doc)
    blob["stats"]["unexpected_new_key"] = [1, 2, 3]
    # Should still validate — stats is deliberately open.
    jsonschema.validate(instance=blob, schema=schema)


# ---------------------------------------------------------------------------
# Negative tests — schema must reject malformed instances
# ---------------------------------------------------------------------------

def test_additional_properties_rejected_on_top_level() -> None:
    """ExtractedDoc schema must reject extra top-level keys."""
    schema = _load_schema()
    blob = _make_valid_doc_blob()
    blob["unknown_field"] = "should_fail"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=blob, schema=schema)


def test_invalid_backend_rejected() -> None:
    """Schema rejects an unknown Backend enum value at the doc level."""
    blob = _make_valid_doc_blob()
    blob["backend"] = "unknown"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=blob, schema=_load_schema())


def test_invalid_region_type_rejected() -> None:
    """Schema rejects an unknown RegionType enum value on a Segment."""
    blob = _make_valid_doc_blob()
    blob["segments"][0]["type"] = "header"  # not in RegionType
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=blob, schema=_load_schema())


def test_bbox_out_of_range_rejected() -> None:
    """Schema rejects BBox coordinates outside [0.0, 1.0]."""
    blob = _make_valid_doc_blob()
    # segments[0] is the one with a bbox set in _make_valid_doc().
    blob["segments"][0]["bbox"]["x0"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=blob, schema=_load_schema())
