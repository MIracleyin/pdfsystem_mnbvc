"""Release-gate engine — turns bench JSONL into a release manifest.

This module sits DOWNSTREAM of the bench loop. It must not import from
``pdfsys_bench.loop`` or ``pdfsys_bench.cascade``; it consumes their
JSONL output only.

See ``docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCORER_VERSION = "release-gate-v0.1"


@dataclass(slots=True, frozen=True)
class ThresholdProfile:
    """Decoded threshold profile. Loaded from a TOML file.

    The ``identifier`` (``<name>@<version>``) is written into every
    manifest row's ``threshold_profile`` field so downstream consumers
    can trace decisions back to a specific policy.
    """

    name: str
    version: str
    created_at: str
    description: str
    t_publish: float
    t_reject: float
    grade_boundaries: dict[str, float]
    disabled_blockers: frozenset[str] = field(default_factory=frozenset)

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"


_REQUIRED_TOP_KEYS = ("name", "version", "created_at", "description")
_REQUIRED_GRADE_KEYS = ("excellent", "good", "fair")


def load_profile(path: str | Path) -> ThresholdProfile:
    """Parse a TOML profile and validate it.

    Raises:
        FileNotFoundError: if ``path`` doesn't exist.
        ValueError: on missing keys, non-monotonic grade boundaries, or
                    ``t_reject >= t_publish``.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    for key in _REQUIRED_TOP_KEYS:
        if key not in raw:
            raise ValueError(f"profile missing required key: {key!r}")

    grades_raw = raw.get("grade_boundaries", {})
    for key in _REQUIRED_GRADE_KEYS:
        if key not in grades_raw:
            raise ValueError(f"profile missing grade_boundaries.{key}")

    decision_raw = raw.get("decision", {})
    if "t_publish" not in decision_raw or "t_reject" not in decision_raw:
        raise ValueError("profile missing decision.t_publish or decision.t_reject")

    t_publish = float(decision_raw["t_publish"])
    t_reject = float(decision_raw["t_reject"])
    if not (t_reject < t_publish):
        raise ValueError(
            f"profile requires t_reject < t_publish; "
            f"got t_reject={t_reject}, t_publish={t_publish}"
        )

    grades = {k: float(grades_raw[k]) for k in _REQUIRED_GRADE_KEYS}
    # Strict descending: excellent > good > fair
    if not (grades["excellent"] > grades["good"] > grades["fair"]):
        raise ValueError(
            f"grade boundaries must be strictly monotonic descending; got {grades}"
        )

    blockers_raw = raw.get("blockers", {})
    disabled = frozenset(blockers_raw.get("disable", []))

    return ThresholdProfile(
        name=str(raw["name"]),
        version=str(raw["version"]),
        created_at=str(raw["created_at"]),
        description=str(raw["description"]),
        t_publish=t_publish,
        t_reject=t_reject,
        grade_boundaries=grades,
        disabled_blockers=disabled,
    )
