"""SystemRelease TOML parser and dataclass.

Parses ``system_release.toml`` — the bill-of-materials that pins each
independently-versioned component (parsers, scorer, …) to a 40-char commit
SHA + human-readable tag.

See docs/superpowers/specs/2026-05-30-parsers-submodule-design.md §5 for
the authoritative schema.

Usage::

    from pdfsys_cli.release import load_release, parse_release

    sr = load_release("system_release.toml")
    for name, comp in sr.components.items():
        print(name, comp.commit, comp.is_in_tree)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit

# 40-char lowercase hex SHA — uppercase letters are rejected by design so CI
# catches copy-paste errors from git log --abbrev or GitHub UI.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_T_PUBLISH_MSG = (
    "t_publish is a gate-profile field; system_release.toml is not a gate profile"
)

# Keys allowed inside a [components.<name>] table. Anything else is rejected
# so typos surface immediately instead of being silently ignored.
_KNOWN_COMPONENT_KEYS = frozenset(
    {"repo", "path", "commit", "tag", "schema_version"}
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Component:
    """A single pinned component entry from ``[components.<name>]``."""

    name: str
    """The dict key from [components.<name>] — e.g. ``parsers``, ``quality-scorer``."""
    repo: str
    """Either a real URL or ``in-tree:<path>``."""
    path: str
    """Where the component lives in the working tree."""
    commit: str
    """40-char lowercase hex SHA."""
    tag: str
    """Human-readable tag."""
    schema_version: str | None = None
    """Optional wire-format version the component speaks."""

    @property
    def is_in_tree(self) -> bool:
        """True when ``repo`` is an in-tree pin (``"in-tree:<path>"`` prefix)."""
        return self.repo.startswith("in-tree:")


@dataclass(frozen=True, slots=True)
class Runtime:
    """Informational runtime pins (e.g. mineru, python). Keys are not validated."""

    values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SystemRelease:
    """Parsed representation of ``system_release.toml``."""

    version: str
    """[system].version — SemVer string, stored as-is."""
    released_at: str
    """[system].released_at — ISO date string, not parsed to a date object."""
    released_by: str
    """[system].released_by — release author handle."""
    notes: str
    """[system].notes — free-form; may be empty string."""
    components: dict[str, Component]
    """Components keyed by their TOML name (e.g. ``"parsers"``)."""
    runtime: Runtime
    """Informational runtime pins."""


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _reject_t_publish(table: dict, location: str) -> None:
    """Reject a t_publish field anywhere — it belongs in gate profiles, not here."""
    if "t_publish" in table:
        raise ValueError(f"{_T_PUBLISH_MSG} (found in {location})")


def _require_str(value: object, location: str, field_name: str) -> str:
    """Return ``value`` as ``str`` or raise ``ValueError`` with location context.

    Guards against TOML values typed as int/float/bool/list/etc. silently
    landing in declared-string dataclass fields.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{location}.{field_name} must be a string, got {type(value).__name__}"
        )
    return value


def _parse_component(name: str, raw: dict) -> Component:
    location = f"[components.{name}]"

    for required_key in ("repo", "path", "commit", "tag"):
        if required_key not in raw:
            raise ValueError(
                f"{location} is missing required key '{required_key}'"
            )

    unknown = set(raw) - _KNOWN_COMPONENT_KEYS
    if unknown:
        raise ValueError(
            f"{location} has unknown keys: {sorted(unknown)}"
        )

    repo = _require_str(raw["repo"], location, "repo")
    if not repo:
        raise ValueError(f"{location}.repo must not be empty")
    if repo.startswith("in-tree:") and not repo[len("in-tree:"):]:
        raise ValueError(
            f"{location}.repo starts with 'in-tree:' but the path after it is empty"
        )

    path = _require_str(raw["path"], location, "path")
    commit = _require_str(raw["commit"], location, "commit")
    if not _SHA_RE.match(commit):
        raise ValueError(
            f"{location}.commit must be a 40-char lowercase hex SHA; "
            f"got {commit!r} ({len(commit)} chars)"
        )
    tag = _require_str(raw["tag"], location, "tag")

    schema_version: str | None = None
    if "schema_version" in raw:
        schema_version = _require_str(
            raw["schema_version"], location, "schema_version"
        )

    return Component(
        name=name,
        repo=repo,
        path=path,
        commit=commit,
        tag=tag,
        schema_version=schema_version,
    )


def _parse_system_release(data: dict) -> SystemRelease:
    """Convert a raw TOML dict into a validated :class:`SystemRelease`.

    The ``t_publish`` scan is exhaustive: root, [system], every
    [components.<name>], and [runtime] are all checked so a stray gate-profile
    field cannot slip in unnoticed.
    """

    # ---- t_publish guard at root ----
    _reject_t_publish(data, "root")

    # ---- [system] table ----
    if "system" not in data:
        raise ValueError("Missing required table [system]")

    system = data["system"]
    _reject_t_publish(system, "[system]")

    for key in ("version", "released_at", "released_by"):
        if key not in system:
            raise ValueError(
                f"[system] is missing required key '{key}'"
            )

    version = _require_str(system["version"], "[system]", "version")
    released_at = _require_str(system["released_at"], "[system]", "released_at")
    released_by = _require_str(system["released_by"], "[system]", "released_by")
    notes = _require_str(system.get("notes", ""), "[system]", "notes")

    # ---- [components] table ----
    if "components" not in data:
        raise ValueError("Missing required table [components]")

    raw_components: dict = data["components"]
    if not raw_components:
        raise ValueError("[components] table must contain at least one component")

    components: dict[str, Component] = {}
    for comp_name, comp_raw in raw_components.items():
        _reject_t_publish(comp_raw, f"[components.{comp_name}]")
        components[comp_name] = _parse_component(comp_name, comp_raw)

    # ---- [runtime] table (optional, informational) ----
    raw_runtime: dict = data.get("runtime", {})
    _reject_t_publish(raw_runtime, "[runtime]")
    runtime = Runtime(values=dict(raw_runtime))

    return SystemRelease(
        version=version,
        released_at=released_at,
        released_by=released_by,
        notes=notes,
        components=components,
        runtime=runtime,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_release(toml_bytes: bytes) -> SystemRelease:
    """Parse a ``system_release.toml`` from raw bytes.

    Args:
        toml_bytes: UTF-8 encoded TOML content.

    Returns:
        A validated :class:`SystemRelease` instance.

    Raises:
        ValueError: If any validation rule is violated.
        tomllib.TOMLDecodeError: If the bytes are not valid TOML.
    """
    data = tomllib.loads(toml_bytes.decode("utf-8"))
    return _parse_system_release(data)


def load_release(toml_path: str | Path) -> SystemRelease:
    """Parse a ``system_release.toml`` from a file path.

    Thin wrapper over :func:`parse_release` so the two entry points share
    a single validation kernel.

    Args:
        toml_path: Path to the TOML file.

    Returns:
        A validated :class:`SystemRelease` instance.

    Raises:
        ValueError: If any validation rule is violated.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        FileNotFoundError: If the file does not exist.
    """
    return parse_release(Path(toml_path).read_bytes())


# ---------------------------------------------------------------------------
# CLI status command
# ---------------------------------------------------------------------------

# Column width for left-aligned keys (the part before the colon).
_KEY_WIDTH = 28

# Status labels — kept as module-level constants so tests can import them
# rather than hard-coding substrings.
STATUS_UP_TO_DATE = "up-to-date"
STATUS_DRIFTED = "DRIFTED (pinned ≠ HEAD)"
STATUS_IN_TREE = "in-tree"
STATUS_MISSING = "MISSING (HEAD unreadable)"


def _fmt_row(prefix: str, key: str, value: str) -> str:
    """Return a formatted ``prefix key : value`` line with a trailing newline."""
    return f"{prefix} {key:<{_KEY_WIDTH}}: {value}\n"


def _abbrev_sha(sha: str) -> str:
    """Abbreviate a 40-char SHA to 7 chars + U+2026 ellipsis."""
    return f"{sha[:7]}…"


def render_status(
    release: SystemRelease,
    heads: dict[str, str | None],
) -> str:
    """Render the ``pdfsys release status`` output as a string.

    Pure function — no I/O, no subprocess calls.

    Args:
        release: Parsed :class:`SystemRelease`.
        heads:   Map from component name to the live HEAD SHA (or ``None``
                 when the path is missing/unreadable). In-tree components
                 should be absent from this dict (they are skipped by
                 :func:`_resolve_component_heads`).

    Returns:
        The full formatted status block (newline-terminated).
    """
    lines: list[str] = []

    # System header.
    lines.append(_fmt_row("✓", "system.version", release.version))
    lines.append(_fmt_row("✓", "released_at", release.released_at))

    for name, comp in release.components.items():
        section_header = f"─ components.{name}\n"
        lines.append(section_header)

        pinned_abbrev = _abbrev_sha(comp.commit)
        tag_note = f"  (tag {comp.tag})"

        lines.append(
            _fmt_row(" ", "pinned commit", f"{pinned_abbrev}{tag_note}")
        )

        if comp.is_in_tree:
            lines.append(_fmt_row(" ", "status", STATUS_IN_TREE))
        else:
            head_sha: str | None = heads.get(name)

            if head_sha is None:
                lines.append(_fmt_row(" ", "status", STATUS_MISSING))
            else:
                head_abbrev = _abbrev_sha(head_sha)
                external_key = f"external/{comp.path.split('/')[-1]} HEAD"
                lines.append(_fmt_row(" ", external_key, head_abbrev))
                if head_sha == comp.commit:
                    lines.append(_fmt_row(" ", "status", STATUS_UP_TO_DATE))
                else:
                    lines.append(_fmt_row(" ", "status", STATUS_DRIFTED))

    return "".join(lines)


def _resolve_component_heads(
    release: SystemRelease,
    base_dir: Path,
) -> dict[str, str | None]:
    """Resolve the live HEAD SHA for each non-in-tree component.

    Shells out to ``git -C <path> rev-parse HEAD`` for every component
    whose ``is_in_tree`` property is ``False``.  In-tree components are
    omitted from the returned dict entirely.

    Component paths are resolved relative to ``base_dir`` (typically the
    directory containing ``system_release.toml``), not the current working
    directory.  This lets ``pdfsys release status --config /abs/path/...``
    work from anywhere.

    Args:
        release:  Parsed :class:`SystemRelease`.
        base_dir: Directory that component paths are resolved against.

    Returns:
        Dict mapping component name to 40-char SHA string, or ``None`` when
        the path does not exist or the git command fails.
    """
    result: dict[str, str | None] = {}

    for name, comp in release.components.items():
        if comp.is_in_tree:
            continue

        path = (base_dir / comp.path).resolve()
        if not path.exists():
            result[name] = None
            continue

        try:
            proc = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # FileNotFoundError fires when git itself is not on PATH.
            result[name] = None
            continue

        if proc.returncode != 0:
            result[name] = None
        else:
            result[name] = proc.stdout.strip()

    return result


def cmd_status(args: object) -> int:
    """Print pin vs HEAD diff for each component.  Exit 0.

    Args:
        args: argparse Namespace with a ``config`` attribute (path to
              ``system_release.toml``).

    Returns:
        Always 0.
    """
    config_path: str = getattr(args, "config", "system_release.toml")
    release = load_release(config_path)
    base_dir = Path(config_path).resolve().parent
    heads = _resolve_component_heads(release, base_dir)
    sys.stdout.write(render_status(release, heads))
    return 0


# ---------------------------------------------------------------------------
# CLI lock command
# ---------------------------------------------------------------------------


def _working_tree_dirty(base_dir: Path) -> bool:
    """Return True if the git working tree at ``base_dir`` has uncommitted changes.

    Uses ``git -C <base_dir> status --porcelain``.  Any non-empty output means
    the tree is dirty.  Errors (no git binary, timeout, or git returning
    non-zero — e.g. exit 128 when ``base_dir`` is not a git repo) are treated
    as dirty so we never overwrite the lock file from an unknown state.

    Args:
        base_dir: Directory passed to ``git -C``.  Typically the directory that
                  contains ``system_release.toml``.

    Returns:
        ``True`` when the working tree is dirty, when git cannot be run, or
        when ``base_dir`` is not a git repository.  ``False`` only when git
        exits 0 with empty stdout.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(base_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # No git binary or timed out — treat as dirty (fail-safe).
        return True
    # Non-zero exit (e.g. 128 when base_dir is not a git repo) is also dirty.
    return bool(proc.stdout.strip()) or proc.returncode != 0


def _abbrev_lock(sha: str) -> str:
    """Return the first 7 chars of a SHA.

    Distinct from :func:`_abbrev_sha` because the lock diff output uses a
    compact ``a1b2c3d → e4f5g6h`` format where the trailing U+2026 ellipsis
    in ``_abbrev_sha`` would add visual noise.  The two call sites are kept
    separate so each can evolve its formatting independently.

    Both ``old`` and ``new`` SHAs at the call site originate from validated
    :class:`Component` instances or :func:`_resolve_component_heads` after a
    ``None`` check, so the type is strictly ``str``.
    """
    return sha[:7]


def _compute_lock_changes(
    release: SystemRelease,
    heads: dict[str, str | None],
) -> list[tuple[str, str, str]]:
    """Compute which components need their pinned commit updated.

    Pure function — no I/O.

    In-tree components are skipped entirely.  External components where the
    resolved HEAD is ``None`` (path not found / git error) are also skipped
    — we don't want to overwrite a known-good pin with ``None``.

    Args:
        release: Parsed :class:`SystemRelease`.
        heads:   Map from component name to resolved HEAD SHA (or ``None``).

    Returns:
        List of ``(name, old_sha, new_sha)`` tuples for components whose HEAD
        differs from the current pin.  Empty list when nothing has changed.
    """
    changes: list[tuple[str, str, str]] = []
    for name, comp in release.components.items():
        if comp.is_in_tree:
            continue
        new_sha = heads.get(name)
        if new_sha is None:
            continue
        if new_sha != comp.commit:
            changes.append((name, comp.commit, new_sha))
    return changes


def _write_lock(
    config_path: Path,
    changes: list[tuple[str, str, str]],
) -> None:
    """Apply ``changes`` to ``config_path`` in-place, preserving comments.

    Uses :mod:`tomlkit` for a comment-preserving round-trip and an atomic
    write (write-to-temp + :func:`os.replace`) so a mid-write interrupt
    (KeyboardInterrupt, ENOSPC, EPERM) cannot leave ``system_release.toml``
    corrupt or zero-byte.

    Args:
        config_path: Absolute path to ``system_release.toml``.
        changes:     List of ``(name, old_sha, new_sha)`` produced by
                     :func:`_compute_lock_changes`.
    """
    doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    for name, _old, new_sha in changes:
        doc["components"][name]["commit"] = new_sha  # type: ignore[index]

    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
        os.replace(tmp, config_path)
    except BaseException:
        # Clean up the partial temp file on any failure path, including
        # KeyboardInterrupt — then re-raise so the caller can decide.
        tmp.unlink(missing_ok=True)
        raise


def _print_in_tree_warnings(release: SystemRelease) -> None:
    """Print a WARNING line for every in-tree component (to stderr).

    This reminds the user that in-tree components are not automatically
    updated by ``lock`` and must be pinned manually at release time.
    Warnings go to stderr so they don't pollute structured stdout consumers.

    Args:
        release: Parsed :class:`SystemRelease`.
    """
    for name, comp in release.components.items():
        if comp.is_in_tree:
            print(f"WARNING: components.{name} is still in-tree", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI verify command
# ---------------------------------------------------------------------------


def _classify_failures(
    release: SystemRelease,
    heads: dict[str, str | None],
) -> list[tuple[str, str]]:
    """Return components that are DRIFTED or MISSING.

    Pure function — no I/O.

    In-tree components are always treated as PASS and are skipped entirely.

    Args:
        release: Parsed :class:`SystemRelease`.
        heads:   Map from component name to resolved HEAD SHA (or ``None``).

    Returns:
        List of ``(name, status_label)`` tuples for components whose status is
        :data:`STATUS_DRIFTED` or :data:`STATUS_MISSING`.  Empty when all
        non-in-tree components are up-to-date.
    """
    # TODO: tag verification (future)
    failures: list[tuple[str, str]] = []
    for name, comp in release.components.items():
        if comp.is_in_tree:
            continue
        head_sha = heads.get(name)
        if head_sha is None:
            failures.append((name, STATUS_MISSING))
        elif head_sha != comp.commit:
            failures.append((name, STATUS_DRIFTED))
    return failures


def _print_verify_summary_fail(failures: list[tuple[str, str]]) -> None:
    """Print the FAIL summary line to stderr.

    Args:
        failures: Non-empty list of ``(name, status_label)`` tuples as
                  returned by :func:`_classify_failures`.
    """
    parts = ", ".join(f"{name}: {label}" for name, label in failures)
    print(f"verify: FAIL — {parts}", file=sys.stderr)


def cmd_verify(args: object) -> int:
    """Verify pinned commits match resolved HEADs. CI gate.

    Exits 0 if every non-in-tree component is up-to-date.
    Exits 1 if any component is DRIFTED or MISSING.

    Args:
        args: argparse Namespace with a ``config`` attribute (path to
              ``system_release.toml``).

    Returns:
        0 on PASS, 1 on FAIL.
    """
    config_path = Path(getattr(args, "config", "system_release.toml")).resolve()
    base_dir = config_path.parent
    release = load_release(config_path)
    heads = _resolve_component_heads(release, base_dir)

    failures = _classify_failures(release, heads)

    if failures:
        _print_verify_summary_fail(failures)
        sys.stdout.write(render_status(release, heads))
        return 1

    n_pass = sum(1 for c in release.components.values() if not c.is_in_tree)
    n_in_tree = sum(1 for c in release.components.values() if c.is_in_tree)
    print(f"verify: PASS — {n_pass} component(s) up-to-date ({n_in_tree} in-tree)")
    sys.stdout.write(render_status(release, heads))
    return 0


# ---------------------------------------------------------------------------
# CLI lock command
# ---------------------------------------------------------------------------


def cmd_lock(args: object) -> int:
    """Update ``system_release.toml`` in place to reflect current submodule HEADs.

    Steps:

    1. Dirty-check the working tree.  Refuse to write if uncommitted changes
       exist (prevents overwriting a known-good pin in an ambiguous state).
    2. Read the current release pin.
    3. Resolve live HEAD SHAs for all non-in-tree components.
    4. Compute which components have drifted from their pin.
    5. Write the updated TOML (comments preserved via ``tomlkit``).
    6. Print a human-readable diff and in-tree warnings.

    Args:
        args: argparse Namespace with a ``config`` attribute (path to
              ``system_release.toml``).

    Returns:
        0 on success (including no-op), 1 if the working tree is dirty.
    """
    config_path = Path(getattr(args, "config", "system_release.toml")).resolve()
    base_dir = config_path.parent

    # Read the release first — it tells us which component paths exist and
    # which are in-tree. Pure / read-only / fast, so safe to run before the
    # dirty gate.
    release = load_release(config_path)

    # 1. dirty check — per non-in-tree component, not the whole main repo.
    #
    # The semantic intent is "don't pin a component's HEAD when its working
    # tree has uncommitted changes that wouldn't be reflected in the pin."
    # That maps to `git -C <component.path> status --porcelain`, not to a
    # check on the main repo root.
    #
    # Checking the main repo root broke the in-tree → external migration
    # workflow: editing `repo`/`path`/`tag` in system_release.toml has to
    # happen *before* `lock` rewrites `commit` (otherwise the path resolves
    # to the wrong git repo), but those edits made the main repo dirty and
    # blocked lock from running at all.
    #
    # In-tree components are skipped — they're not auto-pinned anyway
    # (`_resolve_component_heads` / `_compute_lock_changes` both gate on
    # `is_in_tree`), so refusing on their tree state would be inconsistent.
    # Dirtiness in `system_release.toml` itself or elsewhere outside
    # component paths is by design irrelevant: lock owns the file and the
    # user may have unrelated WIP.
    dirty_components: list[str] = []
    for name, comp in release.components.items():
        if comp.is_in_tree:
            continue
        comp_path = (base_dir / comp.path).resolve()
        if _working_tree_dirty(comp_path):
            dirty_components.append(name)
    if dirty_components:
        names = ", ".join(dirty_components)
        print(
            "error: refusing to write system_release.toml — working tree "
            f"has uncommitted changes in component(s): {names}; "
            "commit or stash changes before locking",
            file=sys.stderr,
        )
        return 1

    # 3. resolve HEADs
    heads = _resolve_component_heads(release, base_dir)

    # 4. compute diff
    changes = _compute_lock_changes(release, heads)

    if not changes:
        print(f"{config_path.name} already up-to-date.")
        _print_in_tree_warnings(release)
        return 0

    # 5. write (atomic; surface disk errors as a clean exit-1, not a traceback)
    try:
        _write_lock(config_path, changes)
    except OSError as exc:
        print(
            f"error: failed to write {config_path.name}: {exc}",
            file=sys.stderr,
        )
        return 1

    # 6. report
    print(f"Wrote {config_path.name}. Diff:")
    for name, old, new in changes:
        print(
            f"  components.{name}.commit: {_abbrev_lock(old)} → {_abbrev_lock(new)}"
        )
    _print_in_tree_warnings(release)
    return 0
