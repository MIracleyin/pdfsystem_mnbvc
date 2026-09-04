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
        ("deferred", "deferred"),
        ("mineru-next", "unknown-backend:mineru-next"),
    ],
)
def test_unreached_backends_record_why(backend, want_reason, tmp_path):
    """Different intents used to collapse into one indistinguishable no-op."""
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


@pytest.mark.parametrize("backend", ["pipeline", "vlm"])
def test_an_ocr_backend_no_longer_needs_our_layout(backend, tmp_path):
    """MinerU runs its own layout analysis and is handed only the PDF bytes, so
    a box that runs MinerU need not have run ours. The old `layout is not None`
    guard made the OCR lane unreachable without a layout stage — which is
    exactly what a GPU box doing extraction-only wants to run."""
    calls: list[str] = []

    class _Recording:
        def extract(self, path):
            calls.append(str(path))
            raise RuntimeError("reached the parser, which is the point")

    class _Comps:
        pipeline_parser = _Recording()
        vlm_parser = _Recording()

    row = DocResult(pdf_path="x.pdf", backend=backend)
    cfg = RunConfig(stages=["router", "extract"])

    _stage_extract(row, Path("x.pdf"), None, _Comps(), cfg)

    assert calls == ["x.pdf"]
    assert row.skip_reason is None


class _RecordingComponents:
    """Records parser calls instead of running them."""

    def __init__(self) -> None:
        self.calls: list[str] = []

        class _P:
            def extract(_self, path):
                self.calls.append(str(path))
                raise RuntimeError("reached the parser")

        self.pipeline_parser = _P()
        self.vlm_parser = _P()


@pytest.mark.parametrize("backend", ["pipeline", "vlm"])
def test_a_failed_layout_stage_still_blocks_extraction(backend, tmp_path):
    """Removing the guard must not mean extracting on top of a layout that
    crashed: that would report success for a document whose run went wrong."""
    row = DocResult(pdf_path="x.pdf", backend=backend)
    row.error_class = "layout"
    row.error_message = "RuntimeError: CUDA out of memory"
    cfg = RunConfig(stages=["router", "layout", "extract"])
    comps = _RecordingComponents()

    assert _stage_extract(row, Path("x.pdf"), None, comps, cfg) is None

    assert comps.calls == [], "the parser must not be reached"
    assert row.skip_reason is None, "a crash is an error, not a hand-off"
    assert row.error_class == "layout", "the original failure is preserved"


@pytest.mark.parametrize("backend", ["pipeline", "vlm"])
def test_a_layout_stage_that_returned_nothing_blocks_extraction(backend):
    """The other half: layout ran, produced no document, and did not raise."""
    row = DocResult(pdf_path="x.pdf", backend=backend)
    cfg = RunConfig(stages=["router", "layout", "extract"])
    comps = _RecordingComponents()

    assert _stage_extract(row, Path("x.pdf"), None, comps, cfg) is None
    assert comps.calls == []


def test_a_failed_document_is_never_labelled_a_lane_hand_off():
    """The hand-off worklist is built from skip_reason, so a document whose
    layout crashed must not be queued to another machine as routine work."""
    row = DocResult(pdf_path="x.pdf", backend="pipeline")
    row.error_class = "layout"
    cfg = RunConfig(stages=["router", "layout", "extract"], extract_backends=["mupdf"])
    comps = _RecordingComponents()

    _stage_extract(row, Path("x.pdf"), None, comps, cfg)

    assert row.skip_reason is None
    assert comps.calls == []


def test_layout_is_not_paid_for_on_documents_the_lane_will_drop(tmp_path, monkeypatch):
    """A CPU box on the mupdf lane would otherwise run DocLayout-YOLO on every
    document it is about to hand away."""
    from pdfsys_core import Backend
    from pdfsys_router import RouterDecision

    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")

    class _Router:
        def classify(self, path):
            return RouterDecision(
                backend=Backend.PIPELINE, ocr_prob=0.9, num_pages=1,
                is_form=False, garbled_text_ratio=0.0, is_encrypted=False,
                needs_password=False,
            )

    class _Analyser:
        def analyse(self, path):  # pragma: no cover - must not be reached
            raise AssertionError("layout ran for a document outside the lane")

    monkeypatch.setattr(Components, "router", property(lambda self: _Router()))
    monkeypatch.setattr(Components, "analyser", property(lambda self: _Analyser()))

    summary = run(apply_cli_overrides(
        RunConfig(), stages="router,layout,extract", pdf_dir=str(src),
        out_dir=str(tmp_path / "out"), extract_backends="mupdf",
    ))

    assert summary["by_skip_reason"] == {"lane-filter": 1}


# ---------------------------------------------------------------------------
# lane filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lane", "backend", "runs"),
    [
        (["mupdf"], "pipeline", False),
        (["mupdf"], "vlm", False),
        (["pipeline"], "pipeline", True),
        (["pipeline", "vlm"], "vlm", True),
        (["pipeline"], "mupdf", False),
    ],
)
def test_the_lane_filter_decides_which_backends_this_box_runs(lane, backend, runs):
    calls: list[str] = []

    class _Recording:
        def extract(self, path):
            calls.append(str(path))
            raise RuntimeError("reached the parser")

    class _Comps:
        pipeline_parser = _Recording()
        vlm_parser = _Recording()

    row = DocResult(pdf_path="x.pdf", backend=backend)
    cfg = RunConfig(stages=["router", "extract"], extract_backends=lane)

    _stage_extract(row, Path("x.pdf"), None, _Comps(), cfg)

    assert bool(calls) is runs
    if runs:
        assert row.skip_reason is None
    else:
        assert row.skip_reason == "lane-filter"
        # The filtered row still names the lane it belongs to, so the other
        # machine's worklist can be built from it.
        assert row.extract_backend == backend


def test_a_deferred_document_is_reported_as_deferred_not_filtered():
    """Stage-B declining is a different fact from this box not owning the lane,
    and the more informative one."""
    row = DocResult(pdf_path="x.pdf", backend="deferred")
    cfg = RunConfig(stages=["router", "extract"], extract_backends=["mupdf"])

    _stage_extract(row, Path("x.pdf"), None, _ExplodingComponents(cfg), cfg)

    assert row.skip_reason == "deferred"


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
        out_dir=str(tmp_path / "out"),
        # The CPU lane, stated explicitly. Without it this would reach for
        # MinerU on every OCR-routed document.
        extract_backends="mupdf",
        **over,
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
    assert summary["by_skip_reason"] == {"lane-filter": 1}


def test_the_two_counters_partition_a_mixed_corpus(tmp_path, monkeypatch):
    """The CPU lane's summary must not claim it extracted the GPU lane's work."""
    summary = _run_with_backends(
        tmp_path, ["mupdf", "pipeline", "mupdf", "deferred"], monkeypatch
    )

    assert summary["num_pdfs"] == 4
    assert summary["num_extracted"] == 2
    assert summary["num_skipped"] == 2
    assert summary["by_skip_reason"] == {"lane-filter": 1, "deferred": 1}
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
