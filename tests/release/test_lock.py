"""Tests for the ``pdfsys release lock`` command (Task 1.3).

Uses a tmp-dir fixture that sets up a fake two-component repo:
- One external component backed by a real (temporary) git repo.
- One in-tree component.

The main repo directory is also ``git init``-ed so the dirty-check has
a real git working tree to query.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from pdfsys_cli.release import cmd_lock

# ---------------------------------------------------------------------------
# Shared helpers
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

    # Initialise the main repo.
    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "test"], repo)

    # Create the external submodule as a separate git repo.
    sub = repo / "external" / "parsers"
    sub.mkdir(parents=True)
    _git(["init", "-b", "main"], sub)
    _git(["config", "user.email", "test@example.com"], sub)
    _git(["config", "user.name", "test"], sub)

    # Give the submodule at least one commit so rev-parse HEAD works.
    (sub / "README.md").write_text("parsers stub\n")
    _git(["add", "README.md"], sub)
    _git(["commit", "-m", "init"], sub)

    # Capture the submodule HEAD.
    sub_head = _git(["rev-parse", "HEAD"], sub)

    # Write system_release.toml pinned to the OLD sha (differs from sub_head).
    toml_content = f"""\
[system]
version = "0.4.0"
released_at = "2026-05-30"
released_by = "ci"
notes = "Test fixture."

[components.parsers]
repo = "git@github.com:example/parsers.git"
path = "external/parsers"
commit = "{_OLD_SHA}"
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

    # Ignore the nested git repo so it doesn't appear as an untracked entry
    # in the main repo's `git status --porcelain` output.
    (repo / ".gitignore").write_text("external/\n")

    # Commit everything in the main repo so the working tree is clean.
    _git(["add", ".gitignore", "system_release.toml"], repo)
    _git(["commit", "-m", "init"], repo)

    return repo, sub_head


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lock_updates_changed_component(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """lock writes the new HEAD SHA into the TOML when component has drifted."""
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0, "cmd_lock must return 0 on success"

    # TOML must now contain the new SHA.
    updated_text = config_path.read_text()
    assert sub_head in updated_text, "new HEAD SHA must be written to TOML"
    assert _OLD_SHA not in updated_text, "old pinned SHA must be replaced"

    # stdout must contain the diff lines.
    captured = capsys.readouterr()
    assert "Wrote system_release.toml. Diff:" in captured.out
    assert "components.parsers.commit" in captured.out
    assert sub_head[:7] in captured.out
    assert _OLD_SHA[:7] in captured.out


def test_lock_refuses_dirty_tree(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """lock returns 1 and leaves TOML unchanged when working tree is dirty."""
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Make the working tree dirty by adding an untracked file.
    (repo / "dirty.txt").write_text("I am dirty\n")

    original_text = config_path.read_text()

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 1, "cmd_lock must return 1 for dirty tree"

    # TOML must be byte-for-byte identical.
    assert config_path.read_text() == original_text, "TOML must not be modified"

    # Stderr must mention uncommitted changes.
    captured = capsys.readouterr()
    assert "uncommitted changes" in captured.err or "dirty" in captured.err.lower()


def test_lock_no_op_when_up_to_date(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """lock returns 0 with an already-up-to-date message when pin matches HEAD."""
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Pre-update the TOML pin to match the current HEAD.
    content = config_path.read_text().replace(_OLD_SHA, sub_head)
    config_path.write_text(content)
    # Re-commit so the tree is clean again.
    _git(["add", "system_release.toml"], repo)
    _git(["commit", "-m", "pre-pin"], repo)

    original_text = config_path.read_text()

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0
    assert config_path.read_text() == original_text, "TOML must not change when already up-to-date"

    captured = capsys.readouterr()
    assert "already up-to-date" in captured.out


def test_lock_skips_in_tree_component(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """lock never updates the pin of an in-tree component and prints a warning."""
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Pre-pin parsers so it won't change (only in-tree component is "different").
    content = config_path.read_text().replace(_OLD_SHA, sub_head)
    config_path.write_text(content)
    _git(["add", "system_release.toml"], repo)
    _git(["commit", "-m", "pre-pin"], repo)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0

    # In-tree commit field must remain unchanged.
    updated_text = config_path.read_text()
    assert _IN_TREE_SHA in updated_text, "in-tree SHA must not be modified by lock"

    # A warning about the in-tree component must appear.
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "quality-scorer" in captured.out


def test_lock_preserves_comments(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """lock uses tomlkit so inline and block comments survive the round-trip."""
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Insert a comment above the parsers section.
    original = config_path.read_text()
    commented = original.replace(
        "[components.parsers]",
        "# This comment must survive the lock round-trip\n[components.parsers]",
    )
    config_path.write_text(commented)
    _git(["add", "system_release.toml"], repo)
    _git(["commit", "-m", "add-comment"], repo)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0

    after = config_path.read_text()
    assert "# This comment must survive the lock round-trip" in after, (
        "tomlkit must preserve TOML comments during lock"
    )
    assert sub_head in after, "new SHA must still be written despite comment"
