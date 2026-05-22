"""Release-gate decision engine tests (Layer 4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pdfsys_bench.release_gate import ThresholdProfile, decide, grade_for_score, load_profile


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "profile.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_load_profile_happy_path(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = "fixture"

        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5

        [decision]
        t_publish = 2.0
        t_reject = 0.5

        [blockers]
        disable = []
    """)
    profile = load_profile(p)
    assert isinstance(profile, ThresholdProfile)
    assert profile.name == "test"
    assert profile.version == "1.0.0"
    assert profile.t_publish == 2.0
    assert profile.t_reject == 0.5
    assert profile.grade_boundaries == {"excellent": 2.5, "good": 1.5, "fair": 0.5}
    assert profile.disabled_blockers == frozenset()
    assert profile.identifier == "test@1.0.0"


def test_load_profile_rejects_missing_name(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match="name"):
        load_profile(p)


def test_load_profile_rejects_t_reject_above_t_publish(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "bad"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 1.0
        t_reject = 2.0
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match=r"t_reject.*<.*t_publish"):
        load_profile(p)


def test_load_profile_rejects_non_monotonic_grades(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "bad"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 1.5
        good = 2.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match=r"grade.*monotonic"):
        load_profile(p)


def test_load_profile_parses_disabled_blockers(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = ["repetition_loop", "too_short"]
    """)
    profile = load_profile(p)
    assert profile.disabled_blockers == frozenset({"repetition_loop", "too_short"})


def test_load_profile_grade_boundaries_is_immutable(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    profile = load_profile(p)
    with pytest.raises(TypeError):
        profile.grade_boundaries["excellent"] = 99.0  # type: ignore[index]


def test_load_profile_rejects_non_string_disable_entries(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = ["ok", 42]
    """)
    with pytest.raises(ValueError, match=r"blockers\.disable"):
        load_profile(p)


def test_load_profile_rejects_unknown_grade_keys(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        superb = 3.0
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match="unexpected keys"):
        load_profile(p)


def _profile(t_publish: float = 2.0, t_reject: float = 0.5) -> ThresholdProfile:
    return ThresholdProfile(
        name="t", version="1.0.0", created_at="2026-05-22", description="",
        t_publish=t_publish, t_reject=t_reject,
        grade_boundaries={"excellent": 2.5, "good": 1.5, "fair": 0.5},
        disabled_blockers=frozenset(),
    )


_DEFAULT_ATTEMPT_SENTINEL: object = object()


def _bench_row(
    *,
    quality_score: float | None = 2.3,
    cascade_attempts: list[dict] | None = _DEFAULT_ATTEMPT_SENTINEL,  # type: ignore[assignment]
) -> dict:
    if cascade_attempts is _DEFAULT_ATTEMPT_SENTINEL:
        cascade_attempts = [{
            "stage": "mupdf", "decision": "publish",
            "blockers": {"empty_output": False, "too_short": False,
                         "high_replacement_chars": False, "high_garbage_chars": False,
                         "repetition_loop": False},
            "metrics": {}, "error": None, "wall_ms": 1.0,
        }]
    return {
        "sha256": "abc123",
        "quality_score": quality_score,
        "cascade_decision": "publish",
        "cascade_final_stage": "mupdf",
        "cascade_attempts": cascade_attempts,
    }


def test_decide_publish_when_score_above_t_publish_and_no_blockers() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=2.3), _profile())
    assert decision == "publish"
    assert grade == "good"
    assert any("t_publish" in r for r in reasons)


def test_decide_reject_when_blocker_triggered() -> None:
    bad = _bench_row(
        quality_score=2.8,  # high score, but blocker vetoes
        cascade_attempts=[{
            "stage": "vlm",
            "decision": "escalate",
            "blockers": {"empty_output": False, "too_short": False,
                         "high_replacement_chars": False, "high_garbage_chars": False,
                         "repetition_loop": True},
            "metrics": {},
            "error": None,
            "wall_ms": 1.0,
        }],
    )
    decision, _grade, reasons = decide(bad, _profile())
    assert decision == "reject"
    assert any("repetition_loop" in r for r in reasons)


def test_decide_reject_when_score_below_t_reject() -> None:
    decision, _, reasons = decide(_bench_row(quality_score=0.2), _profile())
    assert decision == "reject"
    assert any("t_reject" in r for r in reasons)


def test_decide_review_in_grey_band() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=1.0), _profile())
    assert decision == "review"
    assert grade == "fair"
    assert any("grey band" in r for r in reasons)


def test_decide_review_when_score_missing() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=None), _profile())
    assert decision == "review"
    assert grade is None
    assert any("missing" in r for r in reasons)


def test_decide_ignores_disabled_blockers() -> None:
    profile = ThresholdProfile(
        name="t", version="1.0.0", created_at="2026-05-22", description="",
        t_publish=2.0, t_reject=0.5,
        grade_boundaries={"excellent": 2.5, "good": 1.5, "fair": 0.5},
        disabled_blockers=frozenset({"repetition_loop"}),
    )
    bad = _bench_row(
        quality_score=2.5,
        cascade_attempts=[{
            "stage": "mupdf", "decision": "publish",
            "blockers": {"empty_output": False, "too_short": False,
                         "high_replacement_chars": False, "high_garbage_chars": False,
                         "repetition_loop": True},
            "metrics": {}, "error": None, "wall_ms": 1.0,
        }],
    )
    decision, _, _ = decide(bad, profile)
    assert decision == "publish"


def test_grade_for_score_thresholds() -> None:
    profile = _profile()
    assert grade_for_score(2.5, profile) == "excellent"
    assert grade_for_score(2.49, profile) == "good"
    assert grade_for_score(1.5, profile) == "good"
    assert grade_for_score(0.5, profile) == "fair"
    assert grade_for_score(0.49, profile) == "poor"
    assert grade_for_score(None, profile) is None


def test_decide_publish_at_exact_t_publish() -> None:
    """score == t_publish must publish (`>= t_publish`)."""
    decision, _, _ = decide(_bench_row(quality_score=2.0), _profile())
    assert decision == "publish"


def test_decide_review_at_exact_t_reject() -> None:
    """score == t_reject is grey-band (`t_reject <= score < t_publish`)."""
    decision, _, _ = decide(_bench_row(quality_score=0.5), _profile())
    assert decision == "review"


def test_decide_reject_just_below_t_reject() -> None:
    """score just under t_reject must reject (`< t_reject`)."""
    decision, _, _ = decide(_bench_row(quality_score=0.4999), _profile())
    assert decision == "reject"


def test_threshold_profile_direct_construction_normalizes_grade_boundaries() -> None:
    """Even when caller passes a plain dict, the stored mapping is immutable."""
    profile = ThresholdProfile(
        name="t", version="1.0.0", created_at="2026-05-22", description="",
        t_publish=2.0, t_reject=0.5,
        grade_boundaries={"excellent": 2.5, "good": 1.5, "fair": 0.5},
        disabled_blockers=frozenset(),
    )
    with pytest.raises(TypeError):
        profile.grade_boundaries["excellent"] = 99.0  # type: ignore[index]
