"""Release-gate engine — turns bench JSONL into a release manifest.

This module sits DOWNSTREAM of the bench loop. It must not import from
``pdfsys_bench.loop`` or ``pdfsys_bench.cascade``; it consumes their
JSONL output only.

See ``docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable
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


# v1 reserves these schema keys for future Layer-3 / page-level work.
# Always emitted as null so consumers can rely on their presence.
_NULL_RESERVED_FIELDS = (
    "page_quality_p05",
    "page_quality_min",
    "bad_page_ratio",
    "visual_alignment_score",
    "consensus_score",
)

# LLM review fields — populated later by llm_review.py if it runs.
_NULL_LLM_FIELDS = (
    "quality_score_llm",
    "quality_reason_llm",
    "quality_model_llm",
)


def build_manifest_row(row: dict, profile: ThresholdProfile) -> dict:
    """Build one manifest row from one bench-loop row."""
    decision, grade, reasons = decide(row, profile)
    blockers = _final_blockers(row)
    score = row.get("quality_score")

    manifest: dict = {
        "doc_id": row.get("sha256"),
        "decision": decision,
        "doc_quality_score": score,
        "doc_quality_grade": grade,
        "blockers": blockers,
        "reasons": reasons,
        "cascade_final_stage": row.get("cascade_final_stage"),
    }
    for k in _NULL_RESERVED_FIELDS:
        manifest[k] = None
    manifest["scorer_version"] = SCORER_VERSION
    manifest["threshold_profile"] = profile.identifier
    for k in _NULL_LLM_FIELDS:
        manifest[k] = None
    return manifest


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def run_gate(
    bench_jsonl: str | Path,
    out_path: str | Path,
    profile_path: str | Path,
) -> dict:
    """Read bench JSONL, apply the gate, write the release manifest.

    Returns a summary dict with ``num_rows`` and ``by_decision``.
    """
    bench_jsonl = Path(bench_jsonl)
    out_path = Path(out_path)
    profile = load_profile(profile_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    num_rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(bench_jsonl):
            manifest = build_manifest_row(row, profile)
            out.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            counts[manifest["decision"]] += 1
            num_rows += 1

    return {
        "num_rows": num_rows,
        "by_decision": dict(counts),
        "out_path": str(out_path),
        "profile": profile.identifier,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys_bench.release_gate",
        description="Apply Layer-4 release decisions to a bench JSONL.",
    )
    p.add_argument("--bench-jsonl", required=True, type=Path,
                   help="Input bench JSONL from `python -m pdfsys_bench`")
    p.add_argument("--out", required=True, type=Path,
                   help="Output release manifest JSONL path")
    p.add_argument("--profile", required=True, type=Path,
                   help="Path to a TOML threshold profile")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_gate(args.bench_jsonl, args.out, args.profile)
    print(f"[release-gate] profile      = {summary['profile']}")
    print(f"[release-gate] num_rows     = {summary['num_rows']}")
    print(f"[release-gate] by_decision  = {summary['by_decision']}")
    print(f"[release-gate] manifest at  = {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
