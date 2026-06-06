"""Tests for the ``pdfsys release status`` command (Task 1.2).

Focuses on ``render_status()`` (pure function, no subprocess/I/O) and
``_resolve_component_heads`` (skips in-tree components).

The ``cmd_status`` end-to-end test writes a minimal in-tree-only fixture to
``tmp_path`` so that no real git subprocess is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pdfsys_cli.release import (
    _resolve_component_heads,
    cmd_status,
    load_release,
    render_status,
)

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# render_status — pure rendering
# ---------------------------------------------------------------------------


def test_render_status_pin_matches() -> None:
    """All non-in-tree components match pinned commit → status: up-to-date."""
    sr = load_release(_FIXTURES / "two_components.toml")

    # parsers commit from fixture: abcdef1234567890abcdef1234567890abcdef12
    heads = {"parsers": "abcdef1234567890abcdef1234567890abcdef12"}

    output = render_status(sr, heads)

    # System header lines.
    assert "system.version" in output
    assert "0.4.0" in output
    assert "released_at" in output
    assert "2026-05-30" in output

    # Non-in-tree component shows abbreviated SHA and up-to-date.
    assert "components.parsers" in output
    assert "abcdef1…" in output  # abbreviated pinned SHA (U+2026 ellipsis)
    assert "up-to-date" in output

    # In-tree component: shows in-tree label, no HEAD line.
    assert "components.quality-scorer" in output
    assert "in-tree" in output
    # The HEAD line must NOT appear for the in-tree component.
    lines = output.splitlines()
    in_tree_section_started = False
    for line in lines:
        if "components.quality-scorer" in line:
            in_tree_section_started = True
        if in_tree_section_started and "external/" in line:
            pytest.fail("HEAD line must not appear for in-tree component")


def test_render_status_pin_drifted() -> None:
    """Non-in-tree component HEAD differs from pinned → status: DRIFTED."""
    sr = load_release(_FIXTURES / "two_components.toml")

    drifted_sha = "deadbeef1234567890abcdef1234567890deadbe"
    heads = {"parsers": drifted_sha}

    output = render_status(sr, heads)

    assert "DRIFTED" in output
    # up-to-date must not appear for the drifted component.
    assert "up-to-date" not in output
    # The actual HEAD SHA (abbreviated) must appear.
    assert "deadbee…" in output  # abbreviated head SHA


def test_render_status_missing_path() -> None:
    """heads[name] = None (path not found or git error) → shows MISSING or ERROR."""
    sr = load_release(_FIXTURES / "two_components.toml")

    heads: dict[str, str | None] = {"parsers": None}

    output = render_status(sr, heads)

    # Must show some indication of the problem.
    assert "MISSING" in output or "ERROR" in output


# ---------------------------------------------------------------------------
# _resolve_component_heads — subprocess helper
# ---------------------------------------------------------------------------


def test_resolve_component_heads_skips_in_tree() -> None:
    """In-tree components must NOT appear in the returned heads dict."""
    sr = load_release(_FIXTURES / "two_components.toml")

    heads = _resolve_component_heads(sr)

    # quality-scorer is in-tree — must be absent from the result dict.
    assert "quality-scorer" not in heads

    # parsers is NOT in-tree — must appear (value may be None if path missing).
    assert "parsers" in heads


# ---------------------------------------------------------------------------
# cmd_status — light end-to-end (in-tree only, no subprocess)
# ---------------------------------------------------------------------------


def test_cmd_status_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """cmd_status returns 0 for a minimal in-tree-only config."""
    toml_content = """\
[system]
version = "1.0.0"
released_at = "2026-06-01"
released_by = "ci"
notes = ""

[components.mylib]
repo = "in-tree:packages/mylib"
path = "packages/mylib"
commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
tag = "in-tree-0.1.0"
"""
    config_path = tmp_path / "system_release.toml"
    config_path.write_text(toml_content)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_status(args)

    assert rc == 0

    captured = capsys.readouterr()
    assert "1.0.0" in captured.out
    assert "in-tree" in captured.out
