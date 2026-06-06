"""Contract tests: ExtractedDoc v1 JSON schema round-trip validation.

Validates that:
1. The frozen schema file exists at docs/schema/extracted_doc.v1.json.
2. A concrete ExtractedDoc instance serialises to a JSON dict that passes
   schema validation.
3. The JSON dict round-trips back to an equal ExtractedDoc via serde helpers.

Fixtures cover two edge-case combinations:
- A Segment with bbox set (BBox object present).
- A Segment with bbox=None and source_region_id=None (mupdf-style).
"""

from __future__ import annotations

import json
from pathlib import Path

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

@pytest.fixture()
def sample_doc() -> ExtractedDoc:
    """A minimal but representative ExtractedDoc with two segments."""
    seg_with_bbox = Segment(
        index=0,
        backend=Backend.PIPELINE,
        page_index=0,
        type=RegionType.TEXT,
        content="Hello, world.",
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        source_region_id="p0_r0",
    )
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_schema() -> dict:
    assert _SCHEMA_PATH.exists(), (
        f"Schema file not found: {_SCHEMA_PATH}\n"
        "Run Task 0.1 to generate docs/schema/extracted_doc.v1.json first."
    )
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
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
    """The segment that carries a BBox must itself validate under Segment schema."""
    schema = _load_schema()
    seg_blob = to_dict(sample_doc.segments[0])
    segment_schema = {
        "$schema": schema.get("$schema", ""),
        "$defs": schema.get("$defs", {}),
        **schema["$defs"]["Segment"],
    }
    jsonschema.validate(instance=seg_blob, schema=segment_schema)


def test_segment_without_bbox_validates(sample_doc: ExtractedDoc) -> None:
    """The mupdf-style segment (bbox=None, source_region_id=None) must validate."""
    schema = _load_schema()
    seg_blob = to_dict(sample_doc.segments[1])
    segment_schema = {
        "$schema": schema.get("$schema", ""),
        "$defs": schema.get("$defs", {}),
        **schema["$defs"]["Segment"],
    }
    jsonschema.validate(instance=seg_blob, schema=segment_schema)


def test_stats_accepts_arbitrary_keys(sample_doc: ExtractedDoc) -> None:
    """stats must accept any string-keyed structure (additionalProperties: true)."""
    schema = _load_schema()
    blob = to_dict(sample_doc)
    blob["stats"]["unexpected_new_key"] = [1, 2, 3]
    # Should still validate — stats is deliberately open.
    jsonschema.validate(instance=blob, schema=schema)


def test_additional_properties_rejected_on_top_level(sample_doc: ExtractedDoc) -> None:
    """ExtractedDoc schema must reject extra top-level keys."""
    schema = _load_schema()
    blob = to_dict(sample_doc)
    blob["unknown_field"] = "should_fail"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=blob, schema=schema)
