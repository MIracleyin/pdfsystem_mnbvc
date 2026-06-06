"""Codegen: reads extracted_doc.v1.json and writes a self-contained dataclass mirror.

Usage:
    python docs/schema/generate_dataclass.py

The generated file (_extracted_doc_v1_mirror.py) is checked in and must be
regenerated whenever extracted_doc.v1.json changes.

Expose a ``generate() -> str`` function so the golden test can call it without
subprocess overhead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "extracted_doc.v1.json"
_OUTPUT_PATH = Path(__file__).parent / "_extracted_doc_v1_mirror.py"

# ---------------------------------------------------------------------------
# Header template
# ---------------------------------------------------------------------------

_HEADER = """\
# AUTO-GENERATED — DO NOT EDIT BY HAND.
# Regenerate via: python docs/schema/generate_dataclass.py
# Source of truth: docs/schema/extracted_doc.v1.json
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _indent(text: str, n: int = 4) -> str:
    prefix = " " * n
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _emit_str_enum(name: str, values: list[str]) -> str:
    """Emit a StrEnum class."""
    lines = [f"class {name}(StrEnum):"]
    for v in values:
        member = v.upper()
        lines.append(f"    {member} = {v!r}")
    return "\n".join(lines) + "\n"


def _resolve_type(prop: dict, defs: dict, nullable: bool = False) -> str:
    """Map a JSON schema property definition to a Python type annotation string."""
    # Handle oneOf with null pattern (nullable field)
    if "oneOf" in prop:
        non_null = [s for s in prop["oneOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            inner = _resolve_type(non_null[0], defs, nullable=False)
            return f"{inner} | None"
        return "Any"

    if "$ref" in prop:
        ref_name = prop["$ref"].split("/")[-1]
        return ref_name

    json_type = prop.get("type", "Any")

    if json_type == "string":
        return "str"
    if json_type == "integer":
        return "int"
    if json_type == "number":
        return "float"
    if json_type == "boolean":
        return "bool"
    if json_type == "array":
        items = prop.get("items", {})
        item_type = _resolve_type(items, defs)
        return f"tuple[{item_type}, ...]"
    if json_type == "object":
        return "dict[str, Any]"
    return "Any"


def _emit_bbox(defn: dict) -> str:
    """Emit BBox as a frozen+slots dataclass with __post_init__ validation."""
    props = defn["properties"]

    lines = [
        "@dataclass(frozen=True, slots=True)",
        "class BBox:",
        '    """Normalized bounding box. All coordinates are in [0.0, 1.0]; origin is top-left."""',
        "",
    ]

    for field_name, field_schema in props.items():
        py_type = _resolve_type(field_schema, {})
        lines.append(f"    {field_name}: {py_type}")

    lines += [
        "",
        "    def __post_init__(self) -> None:",
        "        for name, value in (",
        '            ("x0", self.x0), ("y0", self.y0),',
        '            ("x1", self.x1), ("y1", self.y1),',
        "        ):",
        "            if not (0.0 <= value <= 1.0):",
        '                raise ValueError(f"BBox.{name}={value!r} outside [0, 1]")',
        "        if self.x1 < self.x0 or self.y1 < self.y0:",
        "            raise ValueError(",
        "                f\"BBox has non-positive size: x0={self.x0} x1={self.x1}"
        " y0={self.y0} y1={self.y1}\"",
        "            )",
    ]

    return "\n".join(lines) + "\n"


def _emit_segment(defn: dict, defs: dict) -> str:
    """Emit Segment as a frozen+slots dataclass."""
    props = defn["properties"]

    lines = [
        "@dataclass(frozen=True, slots=True)",
        "class Segment:",
        '    """A block-level extracted unit."""',
        "",
    ]

    required_fields = []
    optional_fields = []

    for field_name, field_schema in props.items():
        py_type = _resolve_type(field_schema, defs)
        is_optional = "| None" in py_type
        if is_optional:
            optional_fields.append((field_name, py_type))
        else:
            required_fields.append((field_name, py_type))

    for field_name, py_type in required_fields:
        lines.append(f"    {field_name}: {py_type}")

    for field_name, py_type in optional_fields:
        lines.append(f"    {field_name}: {py_type} = None")

    return "\n".join(lines) + "\n"


def _emit_extracted_doc(defn: dict, defs: dict) -> str:
    """Emit ExtractedDoc as a frozen+slots dataclass."""
    props = defn["properties"]

    lines = [
        "@dataclass(frozen=True, slots=True)",
        "class ExtractedDoc:",
        '    """Per-PDF extraction output produced by any backend."""',
        "",
    ]

    # Classify fields: required non-defaulted first, then fields with defaults
    required_fields = []
    defaulted_fields = []

    for field_name, field_schema in props.items():
        py_type = _resolve_type(field_schema, defs)
        is_dict = py_type == "dict[str, Any]"
        is_optional = "| None" in py_type
        has_default = is_dict or is_optional

        if has_default:
            defaulted_fields.append((field_name, py_type, is_dict))
        else:
            required_fields.append((field_name, py_type))

    for field_name, py_type in required_fields:
        lines.append(f"    {field_name}: {py_type}")

    for field_name, py_type, is_dict in defaulted_fields:
        if is_dict:
            lines.append(f"    {field_name}: {py_type} = field(default_factory=dict)")
        else:
            lines.append(f"    {field_name}: {py_type} = None")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------


def generate() -> str:
    """Read the JSON schema and return the full content of the mirror file."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    defs = schema["$defs"]

    parts: list[str] = [_HEADER]

    # Enums first, sorted alphabetically by class name
    enum_names = sorted(
        name
        for name, defn in defs.items()
        if defn.get("type") == "string" and "enum" in defn
    )
    for name in enum_names:
        defn = defs[name]
        parts.append(_emit_str_enum(name, defn["enum"]))
        parts.append("")

    # Dataclasses in dependency order: BBox → Segment → ExtractedDoc
    parts.append(_emit_bbox(defs["BBox"]))
    parts.append("")
    parts.append(_emit_segment(defs["Segment"], defs))
    parts.append("")
    parts.append(_emit_extracted_doc(defs["ExtractedDoc"], defs))

    content = "\n".join(parts)
    # Ensure exactly one trailing newline
    content = content.rstrip("\n") + "\n"
    return content


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        content = generate()
        _OUTPUT_PATH.write_text(content, encoding="utf-8")
        print(f"wrote {_OUTPUT_PATH} ({len(content.encode('utf-8'))} bytes)")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
