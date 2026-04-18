"""Architecture boundary test — enforces the layer rules in docs/architecture/LAYERS.md.

Scans all Python source files in packages/*/src/ and validates that each
package only imports from allowed packages. Uses stdlib ``ast`` for parsing.

Run: ``uv run pytest tests/architecture/test_boundary.py -v``
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGES_DIR = PROJECT_ROOT / "packages"
KNOWN_VIOLATIONS_PATH = Path(__file__).parent / "known-violations.json"

# Layer rules: package → list of pdfsys packages it may import.
# "pdfsys-core" is always allowed (implicit for all packages).
LAYER_RULES: dict[str, list[str]] = {
    "pdfsys-core": [],  # stdlib only — no pdfsys imports
    "pdfsys-router": ["pdfsys_core"],
    "pdfsys-layout-analyser": ["pdfsys_core"],
    "pdfsys-parser-mupdf": ["pdfsys_core"],
    "pdfsys-parser-pipeline": ["pdfsys_core"],
    "pdfsys-parser-vlm": ["pdfsys_core"],
    "pdfsys-bench": ["pdfsys_core", "pdfsys_router", "pdfsys_parser_mupdf",
                      "pdfsys_layout_analyser", "pdfsys_parser_pipeline",
                      "pdfsys_parser_vlm"],
    "pdfsys-cli": ["pdfsys_core", "pdfsys_router", "pdfsys_parser_mupdf",
                    "pdfsys_layout_analyser", "pdfsys_parser_pipeline",
                    "pdfsys_parser_vlm", "pdfsys_bench"],
}

# All pdfsys package import names.
ALL_PDFSYS_MODULES = {
    "pdfsys_core", "pdfsys_router", "pdfsys_layout_analyser",
    "pdfsys_parser_mupdf", "pdfsys_parser_pipeline", "pdfsys_parser_vlm",
    "pdfsys_bench", "pdfsys_cli",
}


def _pkg_name_from_path(file_path: Path) -> str | None:
    """Map a source file path to its package name (kebab-case)."""
    try:
        rel = file_path.relative_to(PACKAGES_DIR)
        return rel.parts[0]  # e.g., "pdfsys-core"
    except (ValueError, IndexError):
        return None


def _scan_imports(file_path: Path) -> list[dict[str, str]]:
    """Parse a Python file and return any pdfsys import violations."""
    pkg_name = _pkg_name_from_path(file_path)
    if pkg_name is None or pkg_name not in LAYER_RULES:
        return []

    allowed = set(LAYER_RULES[pkg_name])
    source = file_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[dict[str, str]] = []

    for node in ast.walk(tree):
        targets: list[str] = []

        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module]

        for target in targets:
            # Extract the top-level pdfsys module name.
            top_module = target.split(".")[0]
            if top_module not in ALL_PDFSYS_MODULES:
                continue  # external or stdlib — not our concern

            # Check if this import is self-referential (same package).
            # Convert package name to import name for comparison.
            self_module = pkg_name.replace("-", "_")
            if top_module == self_module:
                continue  # importing within own package is fine

            if top_module not in allowed:
                violations.append({
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "line": str(getattr(node, "lineno", 0)),
                    "imports": target,
                    "from_pkg": pkg_name,
                    "to_module": top_module,
                })

    return violations


def _collect_all_violations() -> list[dict[str, str]]:
    """Scan all source files and collect violations."""
    all_violations: list[dict[str, str]] = []
    for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
        src_dir = pkg_dir / "src"
        if not src_dir.is_dir():
            continue
        for py_file in sorted(src_dir.rglob("*.py")):
            all_violations.extend(_scan_imports(py_file))
    return all_violations


def _load_known_violations() -> list[dict[str, str]]:
    if KNOWN_VIOLATIONS_PATH.exists():
        return json.loads(KNOWN_VIOLATIONS_PATH.read_text())
    return []


def test_no_new_architecture_violations():
    """Fail if any import violates the layer rules (unless in known-violations.json)."""
    known = _load_known_violations()
    known_set = {(v["file"], v["imports"]) for v in known}

    all_violations = _collect_all_violations()
    new_violations = [
        v for v in all_violations if (v["file"], v["imports"]) not in known_set
    ]

    assert not new_violations, "\n".join(
        f"VIOLATION: {v['file']}:{v['line']} imports {v['imports']} — "
        f"{v['from_pkg']} cannot import {v['to_module']}. "
        f"See docs/architecture/LAYERS.md"
        for v in new_violations
    )


def test_violation_count_only_shrinks():
    """Ratchet: total violation count must never increase."""
    known = _load_known_violations()
    all_violations = _collect_all_violations()

    assert len(all_violations) <= len(known), (
        f"Violation count increased: {len(all_violations)} > baseline {len(known)}. "
        "Fix violations to reduce the count — never add new ones."
    )


def test_core_has_no_external_imports():
    """pdfsys-core must only use stdlib imports — no external packages."""
    # Well-known stdlib top-level modules (Python 3.11+). Not exhaustive,
    # but covers what pdfsys-core actually uses.
    STDLIB = {
        "__future__", "abc", "ast", "asyncio", "base64", "collections",
        "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
        "enum", "functools", "hashlib", "importlib", "inspect", "io",
        "itertools", "json", "logging", "math", "os", "pathlib",
        "re", "shutil", "socket", "string", "struct", "sys", "tempfile",
        "textwrap", "threading", "time", "traceback", "types", "typing",
        "unittest", "urllib", "uuid", "warnings", "weakref",
    }

    core_src = PACKAGES_DIR / "pdfsys-core" / "src"
    violations: list[str] = []

    for py_file in sorted(core_src.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in STDLIB and top != "pdfsys_core":
                        violations.append(
                            f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno} "
                            f"imports {alias.name} (not stdlib)"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top not in STDLIB and top != "pdfsys_core":
                    # Relative imports (node.level > 0) are intra-package.
                    if node.level == 0:
                        violations.append(
                            f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno} "
                            f"imports {node.module} (not stdlib)"
                        )

    assert not violations, (
        "pdfsys-core must have zero external dependencies.\n"
        + "\n".join(violations)
        + "\nSee docs/golden-principles/ZERO_DEP_CORE.md"
    )
