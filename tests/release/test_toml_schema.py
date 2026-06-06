"""Tests for SystemRelease TOML parser and dataclass."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfsys_cli.release import load_release, parse_release

_FIXTURES = Path(__file__).parent / "fixtures"


def test_load_release_valid_fixture() -> None:
    """Parse valid.toml; assert fields and is_in_tree property."""
    sr = load_release(_FIXTURES / "valid.toml")

    assert sr.version == "0.4.0"
    assert sr.released_at == "2026-05-30"
    assert sr.released_by == "miracleyin"
    assert sr.notes == "Initial parsers pin."

    # Two components present.
    assert set(sr.components.keys()) == {"parsers", "quality-scorer"}

    parsers = sr.components["parsers"]
    assert parsers.name == "parsers"
    assert parsers.repo == "git@github.com:MIracleyin/pdfsys-parsers.git"
    assert parsers.commit == "0000000000000000000000000000000000000000"
    assert parsers.tag == "v0.1.0"
    assert parsers.schema_version == "extracted_doc.v1"
    assert not parsers.is_in_tree

    scorer = sr.components["quality-scorer"]
    assert scorer.name == "quality-scorer"
    assert scorer.repo == "in-tree:packages/pdfsys-quality"
    assert scorer.schema_version is None
    assert scorer.is_in_tree

    # Runtime values round-tripped.
    assert sr.runtime.values["mineru"] == "3.x"
    assert sr.runtime.values["python"] == "3.11+"


def test_short_sha_rejected() -> None:
    """Abbreviated SHA must be rejected."""
    with pytest.raises(ValueError, match=r"40.*hex|SHA"):
        load_release(_FIXTURES / "short_sha.toml")


def test_missing_required_key_rejected() -> None:
    """Missing [system].version must be rejected."""
    with pytest.raises(ValueError, match=r"version|required"):
        load_release(_FIXTURES / "missing_required_key.toml")


def test_t_publish_rejected() -> None:
    """t_publish in [system] must be rejected with a clear message."""
    with pytest.raises(ValueError, match=r"t_publish"):
        load_release(_FIXTURES / "t_publish_present.toml")


def test_empty_repo_rejected() -> None:
    """Empty repo string must be rejected."""
    with pytest.raises(ValueError, match=r"repo"):
        load_release(_FIXTURES / "empty_repo.toml")


def test_parse_release_from_bytes() -> None:
    """parse_release accepts raw bytes; result matches load_release on same content."""
    toml_bytes = (
        b"[system]\n"
        b'version = "1.0.0"\n'
        b'released_at = "2026-01-01"\n'
        b'released_by = "ci"\n'
        b'notes = ""\n'
        b"\n"
        b"[components.core]\n"
        b'repo = "https://github.com/example/core.git"\n'
        b'path = "external/core"\n'
        b'commit = "aabbccddeeff00112233445566778899aabbccdd"\n'
        b'tag = "v1.0.0"\n'
        b"\n"
        b"[runtime]\n"
        b'python = "3.11+"\n'
    )
    sr = parse_release(toml_bytes)
    assert sr.version == "1.0.0"
    assert sr.released_by == "ci"
    assert "core" in sr.components
    assert sr.components["core"].commit == "aabbccddeeff00112233445566778899aabbccdd"
    assert not sr.components["core"].is_in_tree
    assert sr.runtime.values["python"] == "3.11+"
