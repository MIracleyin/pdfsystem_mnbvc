"""Cascade-with-early-exit orchestrator tests.

The cascade runs a sequence of parser stages (cheapest first), checks
each output against Layer-1 hard rules, and either publishes the first
acceptable result or escalates to the next stage. These tests pin the
control-flow contract independently of any real parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfsys_bench.cascade import CascadeStage, run_cascade
from pdfsys_core import Backend, ExtractedDoc


def _doc(markdown: str, sha: str = "abc") -> ExtractedDoc:
    return ExtractedDoc(
        sha256=sha,
        backend=Backend.MUPDF,
        segments=(),
        markdown=markdown,
        stats={},
    )


def _stage(
    name: str,
    markdown: str,
    *,
    raises: Exception | None = None,
    rules_kwargs: dict | None = None,
) -> CascadeStage:
    def extract(_path: Path) -> ExtractedDoc:
        if raises is not None:
            raise raises
        return _doc(markdown)

    return CascadeStage(name=name, extract=extract, rules_kwargs=rules_kwargs or {})


_CLEAN = "Clean extracted text with more than fifty characters of body content here."


def test_single_clean_stage_publishes():
    result = run_cascade(Path("dummy.pdf"), [_stage("mupdf", _CLEAN)])
    assert result.decision == "publish"
    assert result.final_stage == "mupdf"
    assert len(result.attempts) == 1
    assert result.attempts[0].decision == "publish"
    assert result.extracted is not None
    assert result.extracted.markdown == _CLEAN


def test_single_empty_stage_rejects():
    result = run_cascade(Path("dummy.pdf"), [_stage("mupdf", "")])
    assert result.decision == "reject"
    # Reject still records which stage produced the (bad) output so a
    # review queue can see "mupdf returned empty" without re-running.
    assert result.final_stage == "mupdf"
    assert result.attempts[0].blockers["empty_output"] is True


def test_first_stage_empty_escalates_to_second():
    stages = [_stage("mupdf", ""), _stage("pipeline", _CLEAN)]
    result = run_cascade(Path("dummy.pdf"), stages)
    assert result.decision == "publish"
    assert result.final_stage == "pipeline"
    assert [a.decision for a in result.attempts] == ["escalate", "publish"]


def test_first_stage_raises_escalates_to_second():
    stages = [
        _stage("mupdf", "", raises=RuntimeError("mupdf boom")),
        _stage("pipeline", _CLEAN),
    ]
    result = run_cascade(Path("dummy.pdf"), stages)
    assert result.decision == "publish"
    assert result.final_stage == "pipeline"
    assert "mupdf boom" in (result.attempts[0].error or "")


def test_all_stages_fail_rejects_and_keeps_last_output_for_review():
    stages = [
        _stage("mupdf", ""),
        _stage("pipeline", "tiny"),
        _stage("vlm", "x"),
    ]
    result = run_cascade(Path("dummy.pdf"), stages, num_pages=10)
    assert result.decision == "reject"
    assert len(result.attempts) == 3
    assert all(a.decision == "escalate" for a in result.attempts)
    # Last attempt's extracted is kept so a human-review queue has something
    # to look at; "reject" just means it should not enter the publish set.
    assert result.extracted is not None
    assert result.extracted.markdown == "x"
    assert result.final_stage == "vlm"


def test_empty_stages_raises_value_error():
    with pytest.raises(ValueError):
        run_cascade(Path("dummy.pdf"), [])


def test_per_stage_rules_kwargs_override_thresholds():
    # A stage that's expected to produce very short output (e.g. cover
    # page only) can relax the per-page minimum so it still publishes.
    stage = _stage("mupdf", "Short.", rules_kwargs={"min_chars_per_page": 3})
    result = run_cascade(Path("dummy.pdf"), [stage])
    assert result.decision == "publish"


def test_attempts_record_blockers_and_metrics_for_observability():
    result = run_cascade(Path("dummy.pdf"), [_stage("mupdf", "")])
    attempt = result.attempts[0]
    assert "empty_output" in attempt.blockers
    assert "chars_per_page" in attempt.metrics


def test_total_wall_ms_sums_attempt_walls():
    stages = [_stage("mupdf", ""), _stage("pipeline", _CLEAN)]
    result = run_cascade(Path("dummy.pdf"), stages)
    assert result.total_wall_ms == pytest.approx(
        sum(a.wall_ms for a in result.attempts)
    )


def test_publish_at_first_stage_does_not_run_later_stages():
    later_was_called = {"hit": False}

    def later_extract(_path: Path) -> ExtractedDoc:
        later_was_called["hit"] = True
        return _doc("never reached")

    stages = [
        _stage("mupdf", _CLEAN),
        CascadeStage(name="pipeline", extract=later_extract),
    ]
    result = run_cascade(Path("dummy.pdf"), stages)
    assert result.decision == "publish"
    assert later_was_called["hit"] is False
    assert len(result.attempts) == 1


def test_num_pages_propagates_to_hard_rules():
    # Same output, different page_count → too_short fires only at high count.
    text = "a" * 100
    one_page = run_cascade(Path("dummy.pdf"), [_stage("mupdf", text)], num_pages=1)
    many_pages = run_cascade(Path("dummy.pdf"), [_stage("mupdf", text)], num_pages=10)
    assert one_page.decision == "publish"
    assert many_pages.decision == "reject"
