"""LLM external-review tests.

The LLM client is mocked — these tests never hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdfsys_bench.llm_review import filter_manifest_rows, run_review


def _manifest_row(
    doc_id: str,
    decision: str = "publish",
    markdown_path: str = "md/x.md",
) -> dict:
    return {
        "doc_id": doc_id,
        "decision": decision,
        "doc_quality_score": 2.0,
        "doc_quality_grade": "good",
        "blockers": {},
        "reasons": [],
        "cascade_final_stage": "mupdf",
        "page_quality_p05": None,
        "page_quality_min": None,
        "bad_page_ratio": None,
        "visual_alignment_score": None,
        "consensus_score": None,
        "scorer_version": "release-gate-v0.1",
        "threshold_profile": "default-v1@0.1.0",
        "quality_score_llm": None,
        "quality_reason_llm": None,
        "quality_model_llm": None,
        # Auxiliary — kept from the bench JSONL by run_gate's caller in
        # the real pipeline. Tests inject it directly.
        "_markdown_path": markdown_path,
    }


def test_filter_scope_all_keeps_every_row() -> None:
    rows = [_manifest_row("a", "publish"), _manifest_row("b", "review"),
            _manifest_row("c", "reject")]
    kept = list(filter_manifest_rows(rows, scope="all"))
    assert [r["doc_id"] for r in kept] == ["a", "b", "c"]


def test_filter_scope_review_keeps_only_review_rows() -> None:
    rows = [_manifest_row("a", "publish"), _manifest_row("b", "review"),
            _manifest_row("c", "reject")]
    kept = list(filter_manifest_rows(rows, scope="review"))
    assert [r["doc_id"] for r in kept] == ["b"]


def test_filter_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        list(filter_manifest_rows([], scope="bogus"))


def test_run_review_patches_llm_fields(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("Some clean markdown.", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    rows = [_manifest_row("a", "publish", markdown_path="a.md")]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    scorer = MagicMock()
    fake_result = MagicMock()
    fake_result.score = 2.5
    fake_result.reason = "clean"
    fake_result.model = "mimo-v2.5-pro"
    fake_result.parse_error = None
    scorer.score.return_value = fake_result
    scorer.client.config.model = "mimo-v2.5-pro"

    summary = run_review(
        manifest_path=manifest,
        markdown_dir=md_dir,
        scope="all",
        scorer=scorer,
        workers=1,
    )

    assert summary["num_scored"] == 1
    assert scorer.score.call_count == 1
    patched = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert patched["quality_score_llm"] == 2.5
    assert patched["quality_reason_llm"] == "clean"
    assert patched["quality_model_llm"] == "mimo-v2.5-pro"
    assert patched["decision"] == "publish"  # untouched


def test_run_review_resume_skips_already_scored(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("aaa", encoding="utf-8")
    (md_dir / "b.md").write_text("bbb", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _manifest_row("a", "publish", markdown_path="a.md"),
        _manifest_row("b", "publish", markdown_path="b.md"),
    ]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # Pre-populate the checkpoint for "a".
    ckpt = tmp_path / "manifest.jsonl.llm.jsonl"
    ckpt.write_text(json.dumps({
        "doc_id": "a",
        "quality_score_llm": 2.5,
        "quality_reason_llm": "from ckpt",
        "quality_model_llm": "mimo-v2.5-pro",
    }) + "\n", encoding="utf-8")

    scorer = MagicMock()
    fake_result = MagicMock()
    fake_result.score = 1.0
    fake_result.reason = "fresh"
    fake_result.model = "mimo-v2.5-pro"
    fake_result.parse_error = None
    scorer.score.return_value = fake_result
    scorer.client.config.model = "mimo-v2.5-pro"

    summary = run_review(
        manifest_path=manifest,
        markdown_dir=md_dir,
        scope="all",
        scorer=scorer,
        workers=1,
        resume=True,
    )

    # Only "b" should have actually been scored
    assert scorer.score.call_count == 1
    assert summary["num_scored"] == 1
    assert summary["num_resumed"] == 1

    patched = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {r["doc_id"]: r for r in patched}
    assert by_id["a"]["quality_score_llm"] == 2.5   # from checkpoint
    assert by_id["a"]["quality_reason_llm"] == "from ckpt"
    assert by_id["b"]["quality_score_llm"] == 1.0   # fresh
    assert by_id["b"]["quality_reason_llm"] == "fresh"


def test_run_review_records_missing_markdown(tmp_path: Path) -> None:
    """If markdown_dir has no matching file, the row gets a sentinel
    record with parse_error='missing markdown' — but the manifest
    must still be rewritten cleanly."""
    md_dir = tmp_path / "md"
    md_dir.mkdir()  # empty dir — no markdown files
    manifest = tmp_path / "manifest.jsonl"
    rows = [_manifest_row("ghost", "publish", markdown_path="ghost.md")]
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    scorer = MagicMock()
    scorer.client.config.model = "mimo-v2.5-pro"
    # score() should NOT be called for a missing-markdown row.
    summary = run_review(
        manifest_path=manifest, markdown_dir=md_dir,
        scope="all", scorer=scorer, workers=1,
    )
    assert scorer.score.call_count == 0
    assert summary["num_scored"] == 1
    patched = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert patched["quality_score_llm"] is None
    assert "markdown not found" in patched["quality_reason_llm"]
    assert patched["quality_parse_error_llm"] == "missing markdown"


def test_run_review_captures_scorer_exception(tmp_path: Path) -> None:
    """A scorer exception on one row must NOT abort the batch — it
    becomes a sentinel record with parse_error=<exception> so the
    batch continues and the manifest is still rewritten."""
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("text", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_manifest_row("a", "publish", markdown_path="a.md")) + "\n",
        encoding="utf-8",
    )

    scorer = MagicMock()
    scorer.client.config.model = "mimo-v2.5-pro"
    scorer.score.side_effect = RuntimeError("transient API error")

    summary = run_review(
        manifest_path=manifest, markdown_dir=md_dir,
        scope="all", scorer=scorer, workers=1,
    )
    assert summary["num_scored"] == 1
    patched = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert patched["quality_score_llm"] is None
    assert "RuntimeError" in patched["quality_parse_error_llm"]
    assert "transient API error" in patched["quality_parse_error_llm"]
