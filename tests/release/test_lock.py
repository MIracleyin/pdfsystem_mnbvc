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

    # git treats a nested .git dir as an unregistered submodule and reports it
    # as untracked unless excluded. Real registered submodules (post-Task 2.4)
    # wouldn't need this, but the fake-repo scaffolding here uses raw `git init`
    # for the external component, so the parent repo sees `external/` as a
    # stray untracked path. Adding it to .gitignore keeps the working tree
    # clean for the dirty-check pre-flight in cmd_lock.
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


def test_lock_refuses_dirty_component_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """lock returns 1 when a non-in-tree component's working tree is dirty.

    The semantic intent of the dirty check is "don't pin a component's HEAD
    when its working tree has uncommitted changes that wouldn't be reflected
    in the pin." Dirtying the external component's own git tree must trip
    the check; the error must name the offending component(s).
    """
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Dirty the EXTERNAL component's working tree (not the main repo).
    sub = repo / "external" / "parsers"
    (sub / "wip.txt").write_text("uncommitted work in submodule\n")

    original_text = config_path.read_text()

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 1, "cmd_lock must return 1 when a component tree is dirty"

    # TOML must be byte-for-byte identical.
    assert config_path.read_text() == original_text, "TOML must not be modified"

    # Stderr must mention the offending component by name.
    captured = capsys.readouterr()
    assert "uncommitted changes" in captured.err
    assert "parsers" in captured.err, (
        "error must name the dirty component so the user knows where to look"
    )


def test_lock_allows_dirty_main_repo_outside_components(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """lock proceeds when the main repo is dirty *outside* component paths.

    Pins concrete behavior: lock owns ``system_release.toml`` and must not
    gatekeep unrelated WIP in the main repo. This is the Task 2.5 workflow —
    the user edits ``repo``/``path``/``tag`` in ``system_release.toml`` and
    then runs ``lock`` to auto-pin ``commit``; the edits make the main repo
    dirty but the component trees are clean.
    """
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Dirty the main repo with an unrelated untracked file outside any
    # component path. (The fixture's only external component lives at
    # external/parsers; dirty.txt is at the repo root.)
    (repo / "dirty.txt").write_text("unrelated WIP at the repo root\n")

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0, (
        "cmd_lock must succeed when only the main repo is dirty and all "
        "component trees are clean"
    )

    # Pin must have been updated to the live submodule HEAD.
    assert sub_head in config_path.read_text()


def test_lock_succeeds_when_system_release_toml_is_dirty(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """lock proceeds when ``system_release.toml`` itself has unstaged edits.

    This is THE Task 2.5 scenario: the user manually edits ``repo``/``path``/
    ``tag`` on a component, then runs ``lock`` to auto-pin ``commit`` in the
    same working state. The previous whole-repo dirty check made this
    impossible; the per-component check must allow it.
    """
    repo, sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"

    # Edit system_release.toml without committing — simulate the Task 2.5
    # manual edit step (here we just bump the human-readable tag).
    content = config_path.read_text()
    content = content.replace('tag = "v0.1.0"', 'tag = "v0.2.0"')
    config_path.write_text(content)

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 0, "lock must proceed when only system_release.toml is dirty"

    # Both edits must be present: the manual `tag` bump AND the auto-pinned
    # `commit` rewrite to the submodule HEAD.
    after = config_path.read_text()
    assert 'tag = "v0.2.0"' in after, "manual tag edit must be preserved"
    assert sub_head in after, "lock must still write the new HEAD SHA"


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

    # A warning about the in-tree component must appear on stderr (so it
    # doesn't pollute structured stdout consumers).
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "quality-scorer" in captured.err


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


def test_lock_treats_non_git_component_path_as_dirty(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A component path that is not a git repository must be treated as dirty.

    ``git status --porcelain`` exits 128 with empty stdout outside a git repo.
    Earlier ``_working_tree_dirty`` implementations returned ``False`` (not
    dirty) in that case, letting ``_write_lock`` proceed on an untracked
    directory. This regression test pins the fail-safe behaviour: a non-git
    component path → return 1, TOML untouched.

    After the per-component dirty-check refactor this test exercises the
    fail-safe per *component* path (not the main repo root). In-tree
    components are skipped from the check, so this fixture must declare an
    external component whose ``path`` exists but isn't a git repo.
    """
    # Neither the main repo nor the component path is a git repository.
    # If the main repo were git-init-ed, `git -C external/parsers status`
    # would walk up to the parent repo and report clean — defeating the
    # test. Both have to be raw directories for `git status` to exit 128.
    repo = tmp_path / "not_a_git_repo"
    repo.mkdir()

    comp_path = repo / "external" / "parsers"
    comp_path.mkdir(parents=True)
    (comp_path / "README.md").write_text("not a git repo\n")

    config_path = repo / "system_release.toml"
    config_path.write_text(
        f"""\
[system]
version = "0.1.0"
released_at = "2026-06-06"
released_by = "ci"
notes = ""

[components.parsers]
repo = "git@github.com:example/parsers.git"
path = "external/parsers"
commit = "{'a' * 40}"
tag = "v0.1.0"
"""
    )

    original_text = config_path.read_text()

    args = argparse.Namespace(config=str(config_path))
    rc = cmd_lock(args)

    assert rc == 1, "non-git component path must be treated as dirty (fail-safe)"
    assert config_path.read_text() == original_text, "TOML must not be modified"

    captured = capsys.readouterr()
    assert "uncommitted changes" in captured.err
    assert "parsers" in captured.err, (
        "error must name the offending component"
    )


# ---------------------------------------------------------------------------
# tag relabelling
#
# `lock` used to rewrite `commit` and leave `tag` alone, so a pin could
# advertise the release it had just moved off: `release status` printed
# "0795144… (tag v0.2.0)" for a commit that was v0.3.0.
# ---------------------------------------------------------------------------


def test_lock_relabels_the_tag_when_the_new_head_is_tagged(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    repo, sub_head = _make_fake_repo(tmp_path)
    _git(["tag", "-a", "v0.3.0", "-m", "release"], repo / "external" / "parsers")
    config_path = repo / "system_release.toml"

    assert cmd_lock(argparse.Namespace(config=str(config_path))) == 0

    updated = config_path.read_text()
    assert 'tag = "v0.3.0"' in updated
    assert 'tag = "v0.1.0"' not in updated
    assert sub_head in updated, "the commit still has to move too"
    assert "components.parsers.tag: v0.1.0 → v0.3.0" in capsys.readouterr().out


def test_an_untagged_head_is_described_rather_than_left_stale(tmp_path: Path) -> None:
    """Past a tag, `v0.3.0-1-g<sha>` is true; keeping `v0.1.0` is not."""
    repo, _ = _make_fake_repo(tmp_path)
    sub = repo / "external" / "parsers"
    _git(["tag", "-a", "v0.3.0", "-m", "release"], sub)
    (sub / "extra.md").write_text("more\n")
    _git(["add", "extra.md"], sub)
    _git(["commit", "-m", "past the tag"], sub)

    assert cmd_lock(argparse.Namespace(config=str(repo / "system_release.toml"))) == 0

    updated = (repo / "system_release.toml").read_text()
    assert 'tag = "v0.3.0-1-g' in updated
    assert 'tag = "v0.1.0"' not in updated


def test_a_repo_with_no_tags_keeps_its_handwritten_label(tmp_path: Path) -> None:
    """`tag` is a human release label, not necessarily a git tag —
    `in-tree-0.1.0` is one. Overwriting it with a bare SHA would destroy
    information to fix a smaller problem, so an untaggable repo is left alone."""
    repo, sub_head = _make_fake_repo(tmp_path)   # fixture creates no tags

    assert cmd_lock(argparse.Namespace(config=str(repo / "system_release.toml"))) == 0

    updated = (repo / "system_release.toml").read_text()
    assert 'tag = "v0.1.0"' in updated, "label survives"
    assert sub_head in updated, "commit still moves"


def test_a_current_pin_keeps_its_label_even_once_a_tag_appears(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Only a moving commit relabels. Otherwise `lock` would stop being a
    no-op and would overwrite labels a maintainer chose deliberately."""
    repo, _sub_head = _make_fake_repo(tmp_path)
    config_path = repo / "system_release.toml"
    cmd_lock(argparse.Namespace(config=str(config_path)))       # pin now current
    _git(["tag", "-a", "v9.9.9", "-m", "later"], repo / "external" / "parsers")
    capsys.readouterr()

    assert cmd_lock(argparse.Namespace(config=str(config_path))) == 0

    assert "already up-to-date" in capsys.readouterr().out
    assert 'tag = "v0.1.0"' in config_path.read_text()
