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
from types import MappingProxyType

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
    grade_boundaries: MappingProxyType[str, float]
    disabled_blockers: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.grade_boundaries, MappingProxyType):
            object.__setattr__(
                self, "grade_boundaries", MappingProxyType(dict(self.grade_boundaries))
            )

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
    extra = set(grades_raw) - set(_REQUIRED_GRADE_KEYS)
    if extra:
        raise ValueError(f"grade_boundaries has unexpected keys: {sorted(extra)}")
    # Strict descending: excellent > good > fair
    if not (grades["excellent"] > grades["good"] > grades["fair"]):
        raise ValueError(
            f"grade boundaries must be strictly monotonic descending; got {grades}"
        )

    blockers_raw = raw.get("blockers", {})
    disabled = frozenset(blockers_raw.get("disable", []))
    if not all(isinstance(x, str) for x in disabled):
        raise ValueError(
            f"blockers.disable must be a list of strings; got {sorted(disabled, key=str)!r}"
        )

    return ThresholdProfile(
        name=str(raw["name"]),
        version=str(raw["version"]),
        created_at=str(raw["created_at"]),
        description=str(raw["description"]),
        t_publish=t_publish,
        t_reject=t_reject,
        grade_boundaries=MappingProxyType(grades),
        disabled_blockers=disabled,
    )


DECISION_PUBLISH = "publish"
DECISION_REVIEW = "review"
DECISION_REJECT = "reject"


def grade_for_score(score: float | None, profile: ThresholdProfile) -> str | None:
    """Map a numeric score to a grade label via the profile boundaries.

    Returns ``None`` if ``score`` is ``None`` (e.g. the bench was run
    with ``--no-quality``).
    """
    if score is None:
        return None
    g = profile.grade_boundaries
    if score >= g["excellent"]:
        return "excellent"
    if score >= g["good"]:
        return "good"
    if score >= g["fair"]:
        return "fair"
    return "poor"


def _final_blockers(row: dict) -> dict[str, bool]:
    """Pull Layer-1 blockers off the last cascade attempt (if any).

    Non-cascade rows have no ``cascade_attempts``; treat them as empty.
    """
    attempts = row.get("cascade_attempts") or []
    if not attempts:
        return {}
    last = attempts[-1]
    return dict(last.get("blockers") or {})


def decide(
    row: dict,
    profile: ThresholdProfile,
) -> tuple[str, str | None, list[str]]:
    """Apply the release-gate rules to one bench row.

    Returns:
        ``(decision, grade, reasons)`` where ``decision`` is one of
        ``"publish"``, ``"review"``, ``"reject"``; ``grade`` is the
        score-derived label (or ``None`` when score is missing); and
        ``reasons`` is a human-readable trail.

    Order:
        1. Any non-disabled Layer-1 blocker → reject (vetoes score).
        2. score is None → review (with explanatory reason).
        3. score >= t_publish → publish.
        4. score < t_reject → reject.
        5. else → review (grey band).
    """
    reasons: list[str] = []
    grade = grade_for_score(row.get("quality_score"), profile)

    blockers = _final_blockers(row)
    triggered = sorted(
        name
        for name, hit in blockers.items()
        if hit and name not in profile.disabled_blockers
    )
    if triggered:
        reasons.append(f"Layer-1 blockers triggered: {triggered}")
        return DECISION_REJECT, grade, reasons

    score = row.get("quality_score")
    if score is None:
        reasons.append("doc_quality_score missing — routed to review")
        return DECISION_REVIEW, grade, reasons

    if score >= profile.t_publish:
        reasons.append(
            f"doc_quality_score={score:.2f} >= t_publish={profile.t_publish:.2f}"
        )
        reasons.append("no Layer-1 blockers triggered")
        return DECISION_PUBLISH, grade, reasons

    if score < profile.t_reject:
        reasons.append(
            f"doc_quality_score={score:.2f} < t_reject={profile.t_reject:.2f}"
        )
        return DECISION_REJECT, grade, reasons

    reasons.append(
        f"doc_quality_score={score:.2f} in grey band "
        f"[{profile.t_reject:.2f}, {profile.t_publish:.2f}) — needs review"
    )
    return DECISION_REVIEW, grade, reasons
