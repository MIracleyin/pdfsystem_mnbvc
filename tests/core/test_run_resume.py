"""Tests for resuming a run instead of starting it over.

``run()`` opened results.jsonl with mode ``"w"``, so a crash at hour six of a
218k-document run erased everything it had done. Worse under a split: that same
file is the worklist another machine is waiting on, so restarting the CPU box
destroyed the GPU box's queue.

Two things have to hold. Work already done is not redone — including across a
crash that left a half-written final line, which is the shape a kill -9 leaves.
And the summary describes the whole run, not the leg that happened to finish:
a resumed run that processes one document must not report ``num_pdfs=1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from pdfsys_cli.config import RunConfig, apply_cli_overrides
from pdfsys_cli.runner import run


def _pdf(path: Path, text: str = "hello") -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def _corpus(root: Path, n: int) -> Path:
    for i in range(n):
        _pdf(root / f"{i:02d}.pdf", f"document number {i}")
    return root


def _cfg(tmp_path, **over):
    return apply_cli_overrides(
        RunConfig(),
        stages="router",
        pdf_dir=str(tmp_path / "corpus"),
        out_dir=str(tmp_path / "out"),
        **over,
    )


def _rows(cfg):
    return [
        json.loads(line)
        for line in cfg.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------


def test_without_resume_a_rerun_starts_over(tmp_path):
    _corpus(tmp_path / "corpus", 4)
    run(_cfg(tmp_path, limit=2))
    summary = run(_cfg(tmp_path))

    assert summary["num_pdfs"] == 4
    assert len(_rows(_cfg(tmp_path))) == 4, "truncated and rewritten, not appended"


def test_resume_skips_what_is_already_done(tmp_path):
    _corpus(tmp_path / "corpus", 5)
    first = run(_cfg(tmp_path, limit=2))
    assert first["num_pdfs"] == 2

    second = run(_cfg(tmp_path, resume=True))

    assert second["resumed_rows"] == 2
    assert second["num_skipped_as_done"] == 2
    assert second["num_pdfs"] == 5, "the summary covers the run, not just this leg"
    rows = _rows(_cfg(tmp_path))
    assert len(rows) == 5
    assert len({r["pdf_path"] for r in rows}) == 5, "nothing processed twice"


def test_resume_recovers_from_a_half_written_final_line(tmp_path):
    """The shape a kill -9 mid-write leaves behind."""
    cfg = _cfg(tmp_path, limit=2)
    _corpus(tmp_path / "corpus", 4)
    run(cfg)
    before = cfg.jsonl_path.read_text(encoding="utf-8")
    with cfg.jsonl_path.open("a", encoding="utf-8") as f:
        f.write('{"pdf_path": "truncated", "sha')

    summary = run(_cfg(tmp_path, resume=True))

    rows = _rows(cfg)
    assert len(rows) == 4, "the partial line is dropped, not appended to"
    assert summary["num_pdfs"] == 4
    assert "truncated" not in {r["pdf_path"] for r in rows}
    # Only a real resume can satisfy these: the two rows from leg 1 survive
    # byte-for-byte, and leg 2 skipped rather than redid them.
    assert cfg.jsonl_path.read_text(encoding="utf-8").startswith(before)
    assert summary["resumed_rows"] == 2
    assert summary["num_skipped_as_done"] == 2
    assert summary["repaired_tail_bytes"] == 30


def test_a_final_line_that_lost_only_its_newline_is_not_spliced(tmp_path):
    """Complete JSON, no terminator — the byte count has to land on a record
    boundary or the next append concatenates two records into one."""
    cfg = _cfg(tmp_path, limit=2)
    _corpus(tmp_path / "corpus", 4)
    run(cfg)
    text = cfg.jsonl_path.read_text(encoding="utf-8")
    cfg.jsonl_path.write_text(text.rstrip("\n"), encoding="utf-8")

    summary = run(_cfg(tmp_path, resume=True))

    rows = _rows(cfg)
    assert len(rows) == 4, "every line still parses on its own"
    assert summary["num_pdfs"] == 4
    assert len({r["pdf_path"] for r in rows}) == 4


def test_damage_in_the_middle_refuses_rather_than_deleting_the_rest(tmp_path):
    """JSONL records are framed independently, so a bad line says nothing about
    the intact rows after it. Truncating to the prefix would delete real work."""
    from pdfsys_cli.runner import CorruptResultsError

    cfg = _cfg(tmp_path)
    _corpus(tmp_path / "corpus", 4)
    run(cfg)
    lines = cfg.jsonl_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"pdf_path": "corrup'  # a torn line with good rows after it
    cfg.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    size_before = cfg.jsonl_path.stat().st_size

    with pytest.raises(CorruptResultsError):
        run(_cfg(tmp_path, resume=True))

    assert cfg.jsonl_path.stat().st_size == size_before, "nothing was deleted"


def test_resume_matches_when_the_run_is_restarted_from_a_different_directory(
    tmp_path, monkeypatch
):
    """A supervisor restarting a multi-day batch rarely lands in the same cwd.
    Recorded paths are absolute so the match does not depend on it."""
    corpus = _corpus(tmp_path / "corpus", 3)
    monkeypatch.chdir(tmp_path)
    run(_cfg(tmp_path, limit=2))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    summary = run(
        apply_cli_overrides(
            RunConfig(), stages="router", pdf_dir=str(corpus),
            out_dir=str(tmp_path / "out"), resume=True,
        )
    )

    assert summary["num_skipped_as_done"] == 2
    assert summary["num_pdfs"] == 3
    assert len({r["pdf_path"] for r in _rows(_cfg(tmp_path))}) == 3


def test_a_relative_pdf_dir_still_records_absolute_paths(tmp_path, monkeypatch):
    _corpus(tmp_path / "corpus", 2)
    monkeypatch.chdir(tmp_path)
    cfg = apply_cli_overrides(
        RunConfig(), stages="router", pdf_dir="corpus", out_dir=str(tmp_path / "out")
    )

    run(cfg)

    assert all(Path(r["pdf_path"]).is_absolute() for r in _rows(cfg))


def test_resuming_into_another_lanes_output_is_detected(tmp_path, monkeypatch):
    """Run the CPU lane, then point the GPU lane at the same --out-dir with
    --resume: every document the CPU lane handed over is already a row, so
    resume skips it and the run reports the CPU lane's numbers having done no
    GPU work at all."""
    from pdfsys_core import Backend
    from pdfsys_router import RouterDecision

    corpus = _corpus(tmp_path / "corpus", 3)

    class _Router:
        def classify(self, path):
            return RouterDecision(
                backend=Backend.PIPELINE, ocr_prob=0.9, num_pages=1,
                is_form=False, garbled_text_ratio=0.0, is_encrypted=False,
                needs_password=False,
            )

    from pdfsys_cli.runner import Components

    monkeypatch.setattr(
        Components, "router", property(lambda self: _Router())
    )

    from pdfsys_cli.runner import LaneConflictError

    cpu = apply_cli_overrides(
        RunConfig(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), extract_backends="mupdf",
    )
    first = run(cpu)
    assert first["by_skip_reason"] == {"lane-filter": 3}
    before = cpu.jsonl_path.read_bytes()
    summary_before = cpu.jsonl_path.with_suffix(".summary.json").read_bytes()

    gpu = apply_cli_overrides(
        RunConfig(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), extract_backends="pipeline", resume=True,
    )
    with pytest.raises(LaneConflictError, match="filtered out by an earlier lane"):
        run(gpu)

    # Raised before any work: the rejected leg leaves the first lane's output
    # exactly as it found it, summary included.
    assert cpu.jsonl_path.read_bytes() == before
    assert cpu.jsonl_path.with_suffix(".summary.json").read_bytes() == summary_before


def test_a_catch_up_pass_with_no_lane_is_also_a_conflict(tmp_path, monkeypatch):
    """No lane means "run every backend" — the widest claim, so it owns the
    handed-over documents more certainly than any named lane does. Reading it
    as "no lane configured, nothing to check" gets the test backwards."""
    from pdfsys_cli.runner import Components, LaneConflictError
    from pdfsys_core import Backend
    from pdfsys_router import RouterDecision

    corpus = _corpus(tmp_path / "corpus", 3)

    class _Router:
        def classify(self, path):
            return RouterDecision(
                backend=Backend.PIPELINE, ocr_prob=0.9, num_pages=1,
                is_form=False, garbled_text_ratio=0.0, is_encrypted=False,
                needs_password=False,
            )

    monkeypatch.setattr(Components, "router", property(lambda self: _Router()))

    run(apply_cli_overrides(
        RunConfig(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), extract_backends="mupdf",
    ))

    with pytest.raises(LaneConflictError):
        run(apply_cli_overrides(
            RunConfig(), stages="router,extract", pdf_dir=str(corpus),
            out_dir=str(tmp_path / "out"), resume=True,  # no lane at all
        ))


def test_resuming_the_same_lane_is_not_a_conflict(tmp_path):
    _corpus(tmp_path / "corpus", 3)
    cfg = _cfg(tmp_path, limit=2, extract_backends="mupdf")
    cfg.stages = ["router", "extract"]
    run(cfg)

    resumed = _cfg(tmp_path, resume=True, extract_backends="mupdf")
    resumed.stages = ["router", "extract"]
    summary = run(resumed)

    assert "resumed_lane_conflicts" not in summary


def test_resume_on_a_finished_run_is_a_no_op(tmp_path):
    _corpus(tmp_path / "corpus", 3)
    run(_cfg(tmp_path))

    summary = run(_cfg(tmp_path, resume=True))

    assert summary["num_skipped_as_done"] == 3
    assert summary["num_pdfs"] == 3
    assert len(_rows(_cfg(tmp_path))) == 3


def test_resume_with_no_existing_file_is_a_plain_run(tmp_path):
    _corpus(tmp_path / "corpus", 2)
    summary = run(_cfg(tmp_path, resume=True))
    assert summary["num_pdfs"] == 2
    assert summary.get("resumed_rows") is None


def test_resume_matches_a_worklist_against_a_directory_scan(tmp_path):
    """The paths are spelled differently in the two legs — relative entries
    resolved through --path-root versus a scan — and must still match."""
    corpus = _corpus(tmp_path / "corpus", 3)
    run(_cfg(tmp_path))  # leg 1: directory scan, absolute paths recorded

    listing = tmp_path / "work.txt"
    listing.write_text("00.pdf\n01.pdf\n02.pdf\n", encoding="utf-8")
    summary = run(
        apply_cli_overrides(
            RunConfig(), stages="router", out_dir=str(tmp_path / "out"),
            pdf_list=str(listing), path_root=str(corpus), resume=True,
        )
    )

    assert summary["num_skipped_as_done"] == 3
    assert summary["num_pdfs"] == 3


def test_limit_names_the_same_slice_on_every_invocation(tmp_path):
    """--limit applies before the resume filter, so it is not "N more each
    time" — resuming a limited run cannot walk off the end of the slice."""
    _corpus(tmp_path / "corpus", 6)
    run(_cfg(tmp_path, limit=2))

    summary = run(_cfg(tmp_path, limit=4, resume=True))

    assert summary["num_pdfs"] == 4
    assert summary["num_skipped_as_done"] == 2


def test_the_summary_arithmetic_survives_a_resume(tmp_path):
    _corpus(tmp_path / "corpus", 4)
    run(_cfg(tmp_path, limit=1))
    cfg = apply_cli_overrides(
        RunConfig(), stages="router,extract", extract_backends="mupdf",
        pdf_dir=str(tmp_path / "corpus"), out_dir=str(tmp_path / "out"),
        resume=True,
    )

    summary = run(cfg)

    assert summary["num_pdfs"] == 4
    assert sum(summary["by_backend"].values()) == summary["num_pdfs"]


def test_resuming_with_a_different_stage_list_is_reported(tmp_path):
    """Resume skips whole documents, so a longer stage list does not go back
    and fill in the rows already written — the shard would silently mix two
    depths of processing."""
    _corpus(tmp_path / "corpus", 3)
    run(_cfg(tmp_path, limit=1))  # router only

    summary = run(
        apply_cli_overrides(
            RunConfig(), stages="router,extract", extract_backends="mupdf",
            pdf_dir=str(tmp_path / "corpus"), out_dir=str(tmp_path / "out"),
            resume=True,
        )
    )

    assert summary["resumed_stage_mismatch"] == ["router"]


def test_resuming_with_the_same_stages_says_nothing(tmp_path):
    _corpus(tmp_path / "corpus", 3)
    run(_cfg(tmp_path, limit=1))

    summary = run(_cfg(tmp_path, resume=True))

    assert "resumed_stage_mismatch" not in summary
