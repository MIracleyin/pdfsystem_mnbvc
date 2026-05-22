"""Release-gate decision engine tests (Layer 4)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from pdfsys_bench.release_gate import (
    ThresholdProfile,
    build_manifest_row,
    decide,
    grade_for_score,
    load_profile,
    main,
    run_gate,
)


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


def test_build_manifest_row_publish_with_cascade() -> None:
    profile = _profile()
    row = _bench_row(quality_score=2.3)
    manifest = build_manifest_row(row, profile)
    assert manifest["doc_id"] == "abc123"
    assert manifest["decision"] == "publish"
    assert manifest["doc_quality_score"] == 2.3
    assert manifest["doc_quality_grade"] == "good"
    assert manifest["blockers"] == {
        "empty_output": False, "too_short": False,
        "high_replacement_chars": False, "high_garbage_chars": False,
        "repetition_loop": False,
    }
    assert isinstance(manifest["reasons"], list) and manifest["reasons"]
    assert manifest["cascade_final_stage"] == "mupdf"
    # v1 page-level + Layer-3 fields are reserved as null
    assert manifest["page_quality_p05"] is None
    assert manifest["page_quality_min"] is None
    assert manifest["bad_page_ratio"] is None
    assert manifest["visual_alignment_score"] is None
    assert manifest["consensus_score"] is None
    # Traceability fields
    assert manifest["scorer_version"] == "release-gate-v0.1"
    assert manifest["threshold_profile"] == "t@1.0.0"
    # LLM fields are null until llm_review runs
    assert manifest["quality_score_llm"] is None
    assert manifest["quality_reason_llm"] is None
    assert manifest["quality_model_llm"] is None
    assert manifest["quality_parse_error_llm"] is None


def test_build_manifest_row_non_cascade_run() -> None:
    """When the bench was run without --cascade, blockers must be {} and
    cascade_final_stage None — decide() then falls through on the score."""
    profile = _profile()
    row = {
        "sha256": "noxcd",
        "quality_score": 2.5,
        "cascade_decision": None,
        "cascade_final_stage": None,
        "cascade_attempts": None,
    }
    manifest = build_manifest_row(row, profile)
    assert manifest["decision"] == "publish"
    assert manifest["blockers"] == {}
    assert manifest["cascade_final_stage"] is None


def test_run_gate_reads_jsonl_and_writes_manifest(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
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
    """), encoding="utf-8")

    bench_path = tmp_path / "bench.jsonl"
    rows = [
        _bench_row(quality_score=2.3),                   # publish
        _bench_row(quality_score=1.0),                   # review (grey)
        _bench_row(quality_score=0.1),                   # reject (score)
    ]
    rows[0]["sha256"] = "row0"
    rows[1]["sha256"] = "row1"
    rows[2]["sha256"] = "row2"
    bench_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "manifest.jsonl"
    summary = run_gate(bench_path, out_path, profile_path)

    assert summary["num_rows"] == 3
    assert summary["by_decision"] == {"publish": 1, "review": 1, "reject": 1}

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [m["decision"] for m in parsed] == ["publish", "review", "reject"]
    assert [m["doc_id"] for m in parsed] == ["row0", "row1", "row2"]


def test_run_gate_empty_bench_jsonl(tmp_path: Path) -> None:
    """Empty input produces an empty manifest and zero counts."""
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
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
    """), encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    bench.write_text("", encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    summary = run_gate(bench, out, profile_path)
    assert summary["num_rows"] == 0
    assert summary["by_decision"] == {}
    assert out.read_text(encoding="utf-8") == ""


def test_run_gate_malformed_json_raises_with_file_and_line(tmp_path: Path) -> None:
    """Parse errors must surface ``<path>:<line_no>: invalid JSON:`` — that
    format is part of the contract CI/log triage relies on."""
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
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
    """), encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"sha256": "ok", "quality_score": 2.5}) + "\n"
        "{not json}\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match=r"bench\.jsonl:2: invalid JSON:"):
        run_gate(bench, out, profile_path)
    # And the manifest must not exist (the .tmp must have been cleaned up).
    assert not out.exists()


def test_run_gate_null_score_routes_to_review(tmp_path: Path) -> None:
    """A bench row with ``quality_score: null`` (e.g. ``--no-quality`` run)
    must be routed to review with a null grade — not crash and not
    silently treated as zero."""
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
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
    """), encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({
            "sha256": "no-score",
            "quality_score": None,
            "cascade_final_stage": "mupdf",
            "cascade_attempts": [],
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"
    summary = run_gate(bench, out, profile_path)
    assert summary["by_decision"] == {"review": 1}
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["decision"] == "review"
    assert record["doc_quality_score"] is None
    assert record["doc_quality_grade"] is None


def test_main_via_argv_returns_zero_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entry must be callable with explicit argv, exit 0, and
    emit a JSON-parseable by_decision line."""
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
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
    """), encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(_bench_row(quality_score=2.3)) + "\n", encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    rc = main([
        "--bench-jsonl", str(bench),
        "--out", str(out),
        "--profile", str(profile_path),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[release-gate]" in captured
    # by_decision line must be JSON-parseable
    for line in captured.splitlines():
        if "by_decision" in line:
            payload = line.split("=", 1)[1].strip()
            parsed = json.loads(payload)
            assert parsed == {"publish": 1}
            break
    else:
        raise AssertionError("by_decision line not found in CLI output")
