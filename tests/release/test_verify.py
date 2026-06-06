"""Tests for the ``pdfsys release verify`` command (Task 1.4).

Uses a tmp-dir fixture that sets up a fake two-component repo:
- One external component backed by a real (temporary) git repo.
- One in-tree component.

All tests use plain pytest functions and ``capsys`` for I/O capture.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from pdfsys_cli.release import (
    STATUS_DRIFTED,
    STATUS_MISSING,
    cmd_verify,
)

# ---------------------------------------------------------------------------
# Shared helpers (parallel to test_lock.py without import coupling)
# ---------------------------------------------------------------------------

_OLD_SHA = "a" * 40
_IN_TREE_SHA = "b" * 40


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command, returning stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_fake_repo(tmp_path: Path) -> tuple[Path, str]:
    """Set up a fake main repo with one external submodule and one in-tree component.

    Returns:
        (repo_root, external_head_sha) — the repo root Path and the HEAD SHA
        of the external component's git repo.
    """
    repo = tmp_path / "fake_repo"
    repo.mkdir()

    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "test"], repo)

    sub = repo / "external" / "parsers"
    sub.mkdir(parents=True)
    _git(["init", "-b", "main"], sub)
    _git(["config", "user.email", "test@example.com"], sub)
    _git(["config", "user.name", "test"], sub)

    (sub / "README.md").write_text("parsers stub\n")
    _git(["add", "README.md"], sub)
    _git(["commit", "-m", "init"], sub)

    sub_head = _git(["rev-parse", "HEAD"], sub)

    toml_content = f"""\
[system]
version = "0.4.0"
released_at = "2026-05-30"
released_by = "ci"
notes = "Test fixture."

[components.parsers]
repo = "git@github.com:example/parsers.git"
path = "external/parsers"
commit = "{sub_head}"
tag = "v0.1.0"

[components.quality-scorer]
repo = "in-tree:packages/pdfsys-quality"
path = "packages/pdfsys-quality"
commit = "{_IN_TREE_SHA}"
tag = "in-tree-0.1.0"

[runtime]
python = "3.11+"
"""
    config_path = repo / "system_release.toml"
    config_path.write_text(toml_content)

    return repo, sub_head


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_verify_pass_when_all_match(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """verify returns 0 when external HEAD matches pinned commit."""
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_verify(args)

    assert rc == 0, "cmd_verify must return 0 when all components match"

    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "up-to-date" in captured.out
    # The summary line itself must report the correct non-in-tree count.
    assert "1 component(s) up-to-date" in captured.out
    # in-tree component name appears in the full status block on stdout
    assert "quality-scorer" in captured.out
    # No failure summary on stderr
    assert captured.err == ""


def test_verify_fail_on_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """verify returns 1 when pinned commit differs from actual HEAD."""
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Overwrite the pin with a wrong SHA so it drifts.
    content = config_path.read_text().replace(_sub_head, _OLD_SHA)
    config_path.write_text(content)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_verify(args)

    assert rc == 1, "cmd_verify must return 1 on DRIFTED component"

    captured = capsys.readouterr()
    # Summary goes to stderr on FAIL
    assert "FAIL" in captured.err
    assert "parsers" in captured.err
    assert STATUS_DRIFTED in captured.err
    # Full status block must be on stdout
    assert "components.parsers" in captured.out


def test_verify_fail_on_missing_path(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """verify returns 1 when external component path does not exist."""
    # Build a config pointing at a non-existent path (no git repo created there).
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    config_path = repo / "system_release.toml"
    config_path.write_text(
        f"""\
[system]
version = "0.1.0"
released_at = "2026-06-01"
released_by = "ci"
notes = ""

[components.parsers]
repo = "git@github.com:example/parsers.git"
path = "external/parsers"
commit = "{_OLD_SHA}"
tag = "v0.1.0"
"""
    )

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_verify(args)

    assert rc == 1, "cmd_verify must return 1 when component path is missing"

    captured = capsys.readouterr()
    assert "FAIL" in captured.err
    assert "parsers" in captured.err
    # STATUS_MISSING constant substring must appear
    assert STATUS_MISSING in captured.err


def test_verify_in_tree_only_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """verify returns 0 when config has only in-tree components."""
    config_path = tmp_path / "system_release.toml"
    config_path.write_text(
        f"""\
[system]
version = "1.0.0"
released_at = "2026-06-01"
released_by = "ci"
notes = ""

[components.mylib]
repo = "in-tree:packages/mylib"
path = "packages/mylib"
commit = "{'a' * 40}"
tag = "in-tree-0.1.0"
"""
    )

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_verify(args)

    assert rc == 0, "cmd_verify must return 0 when all components are in-tree"

    captured = capsys.readouterr()
    assert "PASS" in captured.out
    # The summary mentions in-tree count
    assert "in-tree" in captured.out
    # The summary line itself must report the correct in-tree count.
    assert "(1 in-tree)" in captured.out
    assert captured.err == ""


def test_verify_summary_to_stderr_on_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """On FAIL the summary goes to stderr; full block goes to stdout."""
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Drift the pin.
    content = config_path.read_text().replace(_sub_head, _OLD_SHA)
    config_path.write_text(content)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_verify(args)

    assert rc == 1

    captured = capsys.readouterr()
    # FAIL summary must be on stderr, NOT stdout
    assert "FAIL" in captured.err
    assert "FAIL" not in captured.out
    # Full status block must be on stdout (not empty)
    assert len(captured.out.strip()) > 0
    assert "components.parsers" in captured.out
