"""Threshold-fitting tests on synthetic labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdfsys_bench.fit_profile import (
    FitResult,
    evaluate_thresholds,
    fit_thresholds,
)


def _bench(path: Path, items: list[tuple[str, float]]) -> None:
    """items = [(doc_id, quality_score)]"""
    path.write_text(
        "\n".join(json.dumps({"sha256": d, "quality_score": s}) for d, s in items),
        encoding="utf-8",
    )


def _labels(path: Path, items: list[tuple[str, bool]]) -> None:
    """items = [(doc_id, publishable)] — all 'source=human'."""
    path.write_text(
        "\n".join(
            json.dumps({
                "doc_id": d, "doc_publishable": pub,
                "doc_quality": 3 if pub else 1,
                "source": "human", "severity": "none" if pub else "minor",
                "issue_flags": [], "note": "",
                "labeled_by": "t", "labeled_at": "2026-05-22T00:00:00",
            })
            for d, pub in items
        ),
        encoding="utf-8",
    )


def test_evaluate_thresholds_counts_correctly(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    _bench(bench, [("a", 2.5), ("b", 1.0), ("c", 0.2)])
    _labels(labels, [("a", True), ("b", True), ("c", False)])

    res = evaluate_thresholds(
        bench_jsonl=bench, labels_jsonl=labels,
        t_publish=2.0, t_reject=0.5,
    )
    # a: score 2.5 >= 2.0 → publish (label publishable=True) → TP publish
    # b: score 1.0 in grey → review
    # c: score 0.2 < 0.5  → reject (label publishable=False) → correct reject
    assert res["counts"]["publish"] == 1
    assert res["counts"]["review"] == 1
    assert res["counts"]["reject"] == 1
    assert res["false_publish_rate"] == 0.0
    assert res["review_rate"] == pytest.approx(1 / 3)
    assert res["publish_of_publishable"] == 1


def test_fit_returns_best_thresholds_under_constraints(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    # Clear bimodal — score >=2 always publishable, score <=1 never publishable.
    _bench(bench, [
        ("a", 2.9), ("b", 2.5), ("c", 2.1),     # all publishable
        ("d", 0.9), ("e", 0.6), ("f", 0.1),     # all unpublishable
    ])
    _labels(labels, [
        ("a", True), ("b", True), ("c", True),
        ("d", False), ("e", False), ("f", False),
    ])

    fit: FitResult = fit_thresholds(
        bench_jsonl=bench, labels_jsonl=labels,
        false_publish_max=0.05, review_rate_max=0.30,
    )
    assert fit.t_reject < fit.t_publish
    assert fit.report["false_publish_rate"] <= 0.05
    assert fit.report["review_rate"] <= 0.30
    # With clean bimodality, all publishable should pass.
    assert fit.report["publish_of_publishable"] == 3


def test_fit_raises_when_no_feasible_thresholds(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    # Pathological: high-score docs all labeled unpublishable, no feasible cutoff.
    _bench(bench, [("a", 2.9), ("b", 2.8), ("c", 2.7)])
    _labels(labels, [("a", False), ("b", False), ("c", False)])

    with pytest.raises(RuntimeError, match="no feasible"):
        fit_thresholds(
            bench_jsonl=bench, labels_jsonl=labels,
            false_publish_max=0.0, review_rate_max=0.0,
        )


def test_load_human_labels_ignores_llm_draft_overrides(tmp_path: Path) -> None:
    """An llm_draft row for the same doc_id must NOT overwrite a prior
    human label. The classic re-calibration workflow appends llm_draft
    rows after humans have reviewed — humans win regardless of order.
    """
    from pdfsys_bench.fit_profile import _load_human_labels

    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        # Human row first — declares "a" unpublishable.
        json.dumps({
            "doc_id": "a", "doc_publishable": False, "source": "human",
            "doc_quality": 1, "severity": "minor", "issue_flags": [], "note": "",
            "labeled_by": "t", "labeled_at": "2026-05-22T01:00:00",
        }) + "\n"
        # llm_draft re-seed after — claims "a" is publishable.
        + json.dumps({
            "doc_id": "a", "doc_publishable": True, "source": "llm_draft",
            "draft_score_llm": 2.5, "draft_reason_llm": "looks fine",
            "labeled_by": "llm", "labeled_at": "2026-05-22T02:00:00",
        }) + "\n",
        encoding="utf-8",
    )
    labels = _load_human_labels(labels_path)
    # The human row must win — "a" → False — not the llm_draft → True.
    assert labels == {"a": False}


def test_emit_profile_escapes_quotes_in_description(tmp_path: Path) -> None:
    """A description containing a double-quote must NOT produce invalid
    TOML — release_gate.load_profile must accept the round-trip."""
    from pdfsys_bench.fit_profile import _emit_profile
    from pdfsys_bench.release_gate import load_profile

    out = tmp_path / "fitted.toml"
    _emit_profile(
        out,
        name="quoted-v1",
        version="0.2.0",
        description='fitted from "v1" run on 2026-05-22',
        t_publish=2.0,
        t_reject=0.5,
    )
    # If escaping is broken, load_profile raises a tomllib parse error.
    p = load_profile(out)
    assert p.name == "quoted-v1"
    assert p.t_publish == 2.0
    assert p.t_reject == 0.5
    assert '"v1"' in p.description  # description round-trips faithfully
