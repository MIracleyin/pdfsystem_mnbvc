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
    if "t_publish" in table:
        raise ValueError(f"{_T_PUBLISH_MSG} (found in {location})")


def _parse_component(name: str, raw: dict) -> Component:
    _reject_t_publish(raw, f"[components.{name}]")

    for required_key in ("repo", "path", "commit", "tag"):
        if required_key not in raw:
            raise ValueError(
                f"[components.{name}] is missing required key '{required_key}'"
            )

    repo: str = raw["repo"]
    if not repo:
        raise ValueError(
            f"[components.{name}].repo must not be empty"
        )
    if repo.startswith("in-tree:") and not repo[len("in-tree:"):]:
        raise ValueError(
            f"[components.{name}].repo starts with 'in-tree:' but the path after it is empty"
        )

    commit: str = raw["commit"]
    if not _SHA_RE.match(commit):
        raise ValueError(
            f"[components.{name}].commit must be a 40-char lowercase hex SHA; "
            f"got {commit!r} ({len(commit)} chars)"
        )

    return Component(
        name=name,
        repo=repo,
        path=raw["path"],
        commit=commit,
        tag=raw["tag"],
        schema_version=raw.get("schema_version"),
    )


def _parse_system_release(data: dict) -> SystemRelease:
    """Convert a raw TOML dict into a validated :class:`SystemRelease`."""

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
    runtime = Runtime(values=dict(raw_runtime))

    return SystemRelease(
        version=system["version"],
        released_at=system["released_at"],
        released_by=system["released_by"],
        notes=system.get("notes", ""),
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
    data = tomllib.loads(toml_bytes.decode())
    return _parse_system_release(data)


def load_release(toml_path: str | Path) -> SystemRelease:
    """Parse a ``system_release.toml`` from a file path.

    Args:
        toml_path: Path to the TOML file.

    Returns:
        A validated :class:`SystemRelease` instance.

    Raises:
        ValueError: If any validation rule is violated.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(toml_path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return _parse_system_release(data)
