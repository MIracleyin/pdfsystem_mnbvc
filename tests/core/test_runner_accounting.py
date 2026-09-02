"""Tests for what a run *says it did*, as opposed to what it extracted.

Splitting the pipeline across a CPU box and a GPU box makes the run summary
load-bearing: it is the only thing that says whether a document was extracted,
handed on, or lost. Two defects made that impossible to trust.

``sha256`` used to be set only by the layout and extract stages, so a document
that was routed and then deliberately left for another machine had no doc_id —
no key to join it back on. And a document whose backend was never reached fell
through ``_stage_extract`` into a silent ``return None`` that the summary
counted as a successful extraction, so ``extracted=50 errors=0`` could mean
fifty documents extracted or zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from pdfsys_cli.config import RunConfig, apply_cli_overrides
from pdfsys_cli.runner import (
    Components,
    DocResult,
    _sha256_of_file,
    _stage_extract,
    run,
)


def _pdf(path: Path, text: str = "born digital") -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


class _ExplodingComponents:
    """Any parser touched here means the CPU box just tried to run MinerU."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @property
    def pipeline_parser(self):  # pragma: no cover - must never be reached
        raise AssertionError("pipeline parser constructed on the CPU lane")

    @property
    def vlm_parser(self):  # pragma: no cover - must never be reached
        raise AssertionError("vlm parser constructed on the CPU lane")


# ---------------------------------------------------------------------------
# 0.1 — every routed document gets its doc_id
# ---------------------------------------------------------------------------


def test_router_stamps_sha256_even_when_nothing_else_runs(tmp_path):
    pdf = _pdf(tmp_path / "pdfs" / "a.pdf")
    cfg = apply_cli_overrides(
        RunConfig(),
        stages="router",
        pdf_dir=str(tmp_path / "pdfs"),
        out_dir=str(tmp_path / "out"),
    )

    run(cfg)

    import json

    rows = [json.loads(x) for x in cfg.jsonl_path.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    # The doc_id has to be there with only the router run: it is the join key
    # for a worklist row about a document another machine will extract.
    assert rows[0]["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()


def test_sha256_of_file_matches_whole_file_hash(tmp_path):
    pdf = _pdf(tmp_path / "a.pdf")
    assert _sha256_of_file(pdf) == hashlib.sha256(pdf.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 0.2 — a skip is a skip, not an extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "want_reason"),
    [
        ("pipeline", "no-layout"),
        ("vlm", "no-layout"),
        ("deferred", "deferred"),
        ("mineru-next", "unknown-backend:mineru-next"),
    ],
)
def test_unreached_backends_record_why(backend, want_reason, tmp_path):
    """Three different intents used to collapse into one indistinguishable no-op."""
    row = DocResult(pdf_path="x.pdf", backend=backend)
    cfg = RunConfig()

    extracted = _stage_extract(
        row, Path("x.pdf"), None, _ExplodingComponents(cfg), cfg
    )

    assert extracted is None
    assert row.skip_reason == want_reason
    assert row.extract_backend == backend
    # Not an error: nothing failed. It was a routing decision.
    assert row.error_class is None


def _run_with_backends(tmp_path, backends: list[str], monkeypatch, **over):
    """Drive a real run() with the router's verdict pinned per document."""
    from pdfsys_core import Backend
    from pdfsys_router import RouterDecision

    src = tmp_path / "pdfs"
    for i, _ in enumerate(backends):
        _pdf(src / f"{i}.pdf")

    verdicts = iter(backends)

    class _Router:
        def classify(self, path):
            return RouterDecision(
                backend=Backend(next(verdicts)), ocr_prob=0.5, num_pages=1,
                is_form=False, garbled_text_ratio=0.0, is_encrypted=False,
                needs_password=False,
            )

    monkeypatch.setattr(
        type(Components(RunConfig())), "router", property(lambda self: _Router())
    )
    cfg = apply_cli_overrides(
        RunConfig(), stages="router,extract", pdf_dir=str(src),
        out_dir=str(tmp_path / "out"), **over,
    )
    return run(cfg)


def test_a_document_that_really_extracted_is_counted_as_extracted(
    tmp_path, monkeypatch
):
    """The positive half of the rewritten counter — without this, hard-wiring
    num_extracted to zero passes the whole suite."""
    summary = _run_with_backends(tmp_path, ["mupdf"], monkeypatch)

    assert summary["num_extracted"] == 1
    assert summary["num_skipped"] == 0
    assert summary["by_skip_reason"] == {}


def test_a_document_left_for_the_gpu_is_counted_as_skipped(tmp_path, monkeypatch):
    summary = _run_with_backends(tmp_path, ["pipeline"], monkeypatch)

    assert summary["num_extracted"] == 0
    assert summary["num_skipped"] == 1
    assert summary["by_skip_reason"] == {"no-layout": 1}


def test_the_two_counters_partition_a_mixed_corpus(tmp_path, monkeypatch):
    """The CPU lane's summary must not claim it extracted the GPU lane's work."""
    summary = _run_with_backends(
        tmp_path, ["mupdf", "pipeline", "mupdf", "deferred"], monkeypatch
    )

    assert summary["num_pdfs"] == 4
    assert summary["num_extracted"] == 2
    assert summary["num_skipped"] == 2
    assert summary["by_skip_reason"] == {"no-layout": 1, "deferred": 1}
    assert summary["num_errors"] == 0


# ---------------------------------------------------------------------------
# a failure is not a hand-off
# ---------------------------------------------------------------------------


def test_a_document_that_failed_is_not_also_reported_as_skipped(tmp_path):
    """A crashed layout stage used to be labelled skip_reason="no-layout" —
    the same label as a deliberate deferral — so phase 1 would queue a broken
    document onto the GPU worklist with no sign anything went wrong, and the
    summary counted it in both num_skipped and num_errors."""
    row = DocResult(pdf_path="x.pdf", backend="pipeline")
    row.error_class = "layout"
    row.error_message = "RuntimeError: CUDA out of memory"
    cfg = RunConfig()

    assert _stage_extract(row, Path("x.pdf"), None, _ExplodingComponents(cfg), cfg) is None
    assert row.skip_reason is None


def test_an_unreadable_pdf_is_an_error_not_a_deferral(tmp_path):
    """The case a real corpus hits: the router cannot read the file at all."""
    src = tmp_path / "pdfs"
    src.mkdir(parents=True)
    (src / "broken.pdf").write_bytes(b"not a pdf at all")
    cfg = apply_cli_overrides(
        RunConfig(), stages="router,extract", pdf_dir=str(src),
        out_dir=str(tmp_path / "out"),
    )

    summary = run(cfg)

    assert summary["num_errors"] == 1
    assert summary["num_skipped"] == 0, "a dead file must not look like a hand-off"
    assert summary["by_skip_reason"] == {}
