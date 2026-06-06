"""Codegen stability and self-containment tests for the _extracted_doc_v1_mirror module.

Tests:
1. test_mirror_matches_checked_in_file — running generate() produces the same bytes
   as the checked-in mirror. If this fails, re-run:
       python docs/schema/generate_dataclass.py
2. test_mirror_module_is_importable_and_self_contained — the mirror defines all 5
   required names and does not import from pdfsys_core.
3. test_mirror_dataclasses_have_same_field_set_as_originals — mirrors must match
   the original dataclass fields to catch schema-vs-original drift.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_MIRROR_PATH = _REPO_ROOT / "docs" / "schema" / "_extracted_doc_v1_mirror.py"
_GENERATOR_PATH = _REPO_ROOT / "docs" / "schema" / "generate_dataclass.py"


def _load_generate_module():
    """Dynamically import docs/schema/generate_dataclass.py without it being on sys.path."""
    spec = importlib.util.spec_from_file_location("generate_dataclass", _GENERATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mirror_module():
    """Dynamically import docs/schema/_extracted_doc_v1_mirror.py."""
    mod_name = "_extracted_doc_v1_mirror"
    # Re-use cached module if already loaded to avoid double-registration.
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _MIRROR_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass __module__ lookups resolve correctly.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: golden-file stability
# ---------------------------------------------------------------------------


def test_mirror_matches_checked_in_file() -> None:
    """generate() output must be byte-identical to the checked-in mirror.

    If this fails, regenerate:
        python docs/schema/generate_dataclass.py
    """
    assert _GENERATOR_PATH.exists(), f"Generator not found: {_GENERATOR_PATH}"
    assert _MIRROR_PATH.exists(), (
        f"Mirror file not found: {_MIRROR_PATH}\n"
        "Run: python docs/schema/generate_dataclass.py"
    )

    gen_mod = _load_generate_module()
    generated_content = gen_mod.generate()

    checked_in_content = _MIRROR_PATH.read_text(encoding="utf-8")

    assert generated_content == checked_in_content, (
        "Checked-in mirror is stale. Regenerate via:\n"
        "    python docs/schema/generate_dataclass.py\n\n"
        f"First diff: generated has {len(generated_content)} chars, "
        f"checked-in has {len(checked_in_content)} chars."
    )


# ---------------------------------------------------------------------------
# Test 2: importable and self-contained
# ---------------------------------------------------------------------------


def test_mirror_module_is_importable_and_self_contained() -> None:
    """Mirror module must define all 5 required names and have zero pdfsys_core imports."""
    assert _MIRROR_PATH.exists(), f"Mirror file not found: {_MIRROR_PATH}"

    mirror = _load_mirror_module()

    required_names = ["Backend", "RegionType", "BBox", "Segment", "ExtractedDoc"]
    for name in required_names:
        assert hasattr(mirror, name), (
            f"Mirror module is missing required name: {name!r}"
        )

    source = _MIRROR_PATH.read_text(encoding="utf-8")
    assert "pdfsys_core" not in source, (
        "Mirror file must not import from pdfsys_core — it is meant to be standalone.\n"
        "Found 'pdfsys_core' in the file source."
    )
    assert "from pdfsys_core" not in source
    assert "import pdfsys_core" not in source


# ---------------------------------------------------------------------------
# Test 3: field sets match originals (drift detection)
# ---------------------------------------------------------------------------


def test_mirror_dataclasses_have_same_field_set_as_originals() -> None:
    """Mirror dataclass fields must match the originals in pdfsys_core.

    Detects drift between the JSON schema (codegen source) and the actual
    pdfsys_core dataclasses. If this fails, the JSON schema needs updating.
    """
    from pdfsys_core.extract import ExtractedDoc as OrigExtractedDoc
    from pdfsys_core.extract import Segment as OrigSegment
    from pdfsys_core.layout import BBox as OrigBBox

    assert _MIRROR_PATH.exists(), f"Mirror file not found: {_MIRROR_PATH}"
    mirror = _load_mirror_module()

    for cls_name, orig_cls, mirror_cls in [
        ("BBox", OrigBBox, mirror.BBox),
        ("Segment", OrigSegment, mirror.Segment),
        ("ExtractedDoc", OrigExtractedDoc, mirror.ExtractedDoc),
    ]:
        orig_field_names = {f.name for f in dataclasses.fields(orig_cls)}
        mirror_field_names = {f.name for f in dataclasses.fields(mirror_cls)}

        assert orig_field_names == mirror_field_names, (
            f"{cls_name} field mismatch between original and mirror.\n"
            f"  Original fields:  {sorted(orig_field_names)}\n"
            f"  Mirror fields:    {sorted(mirror_field_names)}\n"
            "The JSON schema may need updating: python docs/schema/generate_dataclass.py"
        )
