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

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

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
