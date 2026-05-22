"""Release-gate decision engine tests (Layer 4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pdfsys_bench.release_gate import ThresholdProfile, load_profile


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
