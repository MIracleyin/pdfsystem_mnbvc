"""Tests for ``pdfsys score`` — scoring a finished run, not producing one.

Quality scoring used to be reachable only as a continuation of extraction: the
runner handed the scorer an ``ExtractedDoc`` still in memory, so the only way
to get a quality column was to re-run the pipeline. Split across machines that
is the wrong shape — the CPU box extracts most of the corpus, the GPU box
extracts the rest, and one ModernBERT on one GPU has to serve both.

What this pass must not do is lose rows. It reads a run's results.jsonl and
writes the same rows back with four columns filled; a document with no text to
score keeps its nulls and stays in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdfsys_cli.score import ScoreModelMismatch, score_run


class _Score:
    def __init__(self, text: str, model: str) -> None:
        self.score = min(3.0, len(text) / 100)
        self.num_chars = len(text)
        self.num_tokens = len(text) // 4
        self.model = model


class _FakeScorer:
    """Stands in for OcrQualityScorer. Records what it was asked to score."""

    model = "test/model"

    def __init__(self, *_a, fail_on: set[str] | None = None, **_kw) -> None:
        self.seen: list[str] = []
        self.fail_on = fail_on or set()
        self.closed = False

    def _ensure_server(self):
        return "http://stub"

    def score(self, text: str) -> _Score:
        self.seen.append(text)
        if text[:20] in self.fail_on:
            raise RuntimeError("scorer said no")
        return _Score(text, self.model)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def scorer(monkeypatch):
    made: list[_FakeScorer] = []

    def _factory(*a, **kw):
        s = _FakeScorer(*a, **kw)
        made.append(s)
        return s

    import pdfsys_bench.quality

    monkeypatch.setattr(pdfsys_bench.quality, "OcrQualityScorer", _factory)
    # No QUALITY_URL: the health check has no server to ask, so it is a no-op
    # and the tests exercise the scoring path rather than the network.
    monkeypatch.delenv("QUALITY_URL", raising=False)
    return made


def _run(tmp_path, rows: list[dict], markdown: dict[str, str]) -> tuple[Path, Path, Path]:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    md_dir = tmp_path / "markdown"
    md_dir.mkdir()
    for sha, text in markdown.items():
        (md_dir / f"{sha}.md").write_text(text, encoding="utf-8")
    return results, md_dir, tmp_path / "scored.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64


# ---------------------------------------------------------------------------
# no row is lost
# ---------------------------------------------------------------------------


def test_every_input_row_reaches_the_output(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [
            {"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None},
            {"pdf_path": "2.pdf", "sha256": SHA_B, "quality_score": None,
             "skip_reason": "lane-filter"},
            {"pdf_path": "3.pdf", "sha256": None, "quality_score": None},
        ],
        {SHA_A: "some extracted text"},
    )

    report = score_run(results, md, out)

    assert report.rows == 3
    assert len(_rows(out)) == 3
    assert [r["pdf_path"] for r in _rows(out)] == ["1.pdf", "2.pdf", "3.pdf"]


def test_columns_other_than_quality_are_untouched(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None,
          "backend": "mupdf", "markdown_chars": 19, "extract_stats": {"a": 1}}],
        {SHA_A: "some extracted text"},
    )

    score_run(results, md, out)

    row = _rows(out)[0]
    assert row["backend"] == "mupdf"
    assert row["markdown_chars"] == 19
    assert row["extract_stats"] == {"a": 1}


def test_a_document_with_text_is_scored(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "x" * 250},
    )

    report = score_run(results, md, out)

    assert report.scored == 1
    row = _rows(out)[0]
    assert row["quality_score"] == 2.5
    assert row["quality_num_chars"] == 250
    assert row["quality_model"] == "test/model"


# ---------------------------------------------------------------------------
# a document with nothing to score is not a defect
# ---------------------------------------------------------------------------


def test_a_handed_off_document_is_not_reported_as_missing(tmp_path, scorer):
    """The CPU lane's normal output: rows queued for the GPU box have no text
    yet. Calling that a missing file would make every CPU run look broken."""
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None,
          "skip_reason": "lane-filter"},
         {"pdf_path": "2.pdf", "sha256": SHA_B, "quality_score": None,
          "error_class": "extract_pipeline"},
         {"pdf_path": "3.pdf", "sha256": SHA_C, "quality_score": None},
         {"pdf_path": "4.pdf", "sha256": "d" * 64, "quality_score": None}],
        {SHA_C: "text"},
    )

    report = score_run(results, md, out)

    assert report.handed_off == 2
    assert report.missing_markdown == 1, "the row that claims an extraction"
    assert report.scored == 1


def test_a_row_with_no_sha256_is_counted_and_kept(tmp_path, scorer):
    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": None}], {}
    )

    report = score_run(results, md, out)

    assert report.no_key == 1
    assert len(_rows(out)) == 1


# ---------------------------------------------------------------------------
# already-scored rows
# ---------------------------------------------------------------------------


def test_a_row_that_already_has_a_score_is_left_alone(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": 1.25}],
        {SHA_A: "x" * 250},
    )

    report = score_run(results, md, out)

    assert report.already == 1
    assert report.scored == 0
    assert _rows(out)[0]["quality_score"] == 1.25


def test_rescore_overrides_an_existing_score(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": 1.25}],
        {SHA_A: "x" * 250},
    )

    report = score_run(results, md, out, rescore=True)

    assert report.scored == 1
    assert _rows(out)[0]["quality_score"] == 2.5


# ---------------------------------------------------------------------------
# crash safety
# ---------------------------------------------------------------------------


def test_resume_carries_the_checkpoint_and_does_not_rescore(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None},
         {"pdf_path": "2.pdf", "sha256": SHA_B, "quality_score": None}],
        {SHA_A: "aaa", SHA_B: "bbb"},
    )
    checkpoint = out.with_suffix(out.suffix + ".partial")
    checkpoint.write_text(
        json.dumps({"sha256": SHA_A, "quality_score": 9.0, "quality_num_chars": 3,
                    "quality_num_tokens": 0, "quality_model": "earlier/model"}) + "\n",
        encoding="utf-8",
    )

    report = score_run(results, md, out, resume=True)

    assert report.already == 1
    assert report.scored == 1
    assert scorer[0].seen == ["bbb"], "the carried document was not re-sent"
    by_sha = {r["sha256"]: r for r in _rows(out)}
    assert by_sha[SHA_A]["quality_score"] == 9.0
    assert by_sha[SHA_A]["quality_model"] == "earlier/model"


def test_the_checkpoint_is_removed_once_the_output_is_written(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "text"},
    )

    score_run(results, md, out)

    assert not out.with_suffix(out.suffix + ".partial").exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_a_stale_checkpoint_is_discarded_without_resume(tmp_path, scorer):
    """Without --resume the run starts over, so a checkpoint from a different
    attempt must not leak its scores into this one."""
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "text"},
    )
    out.with_suffix(out.suffix + ".partial").write_text(
        json.dumps({"sha256": SHA_A, "quality_score": 9.0}) + "\n", encoding="utf-8"
    )

    score_run(results, md, out)

    assert _rows(out)[0]["quality_score"] != 9.0


def test_a_truncated_results_file_stops_at_the_last_whole_row(tmp_path, scorer):
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "text"},
    )
    with results.open("a", encoding="utf-8") as f:
        f.write('{"pdf_path": "2.pdf", "sha')

    report = score_run(results, md, out)

    assert report.rows == 1
    assert len(_rows(out)) == 1


# ---------------------------------------------------------------------------
# failures are data
# ---------------------------------------------------------------------------


def test_one_failing_document_does_not_end_the_batch(tmp_path, monkeypatch):
    import pdfsys_bench.quality

    monkeypatch.delenv("QUALITY_URL", raising=False)
    monkeypatch.setattr(
        pdfsys_bench.quality, "OcrQualityScorer",
        lambda *a, **kw: _FakeScorer(fail_on={"poison"}),
    )
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None},
         {"pdf_path": "2.pdf", "sha256": SHA_B, "quality_score": None}],
        {SHA_A: "poison", SHA_B: "fine"},
    )

    report = score_run(results, md, out)

    assert report.failed == 1
    assert report.scored == 1
    assert len(report.failures) == 1
    by_sha = {r["sha256"]: r for r in _rows(out)}
    assert by_sha[SHA_A]["quality_score"] is None
    assert by_sha[SHA_B]["quality_score"] is not None


# ---------------------------------------------------------------------------
# two models must not share one column
# ---------------------------------------------------------------------------


def test_a_scorer_serving_a_different_model_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("QUALITY_URL", "http://scorer.invalid")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"ok": True, "model": "actually/other"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    results, md, out = _run(tmp_path, [], {})

    with pytest.raises(ScoreModelMismatch, match="two different scales"):
        score_run(results, md, out, model="expected/model")

    assert not out.exists(), "refused before writing anything"


def test_a_matching_model_proceeds(tmp_path, monkeypatch, scorer):
    monkeypatch.setenv("QUALITY_URL", "http://scorer.invalid")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"ok": True, "model": "expected/model"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "text"},
    )

    report = score_run(results, md, out, model="expected/model")

    assert report.model == "expected/model"
    assert report.scored == 1


# ---------------------------------------------------------------------------
# the real scorer, not the stub
# ---------------------------------------------------------------------------


def test_the_real_scorer_is_constructed_with_a_usable_model_name(monkeypatch):
    """Every other test in this file substitutes a stub, so none of them
    construct the real OcrQualityScorer — which is how `model_name=... or None`
    shipped: it overrides the class default rather than falling back to it, and
    the local path then hands None to subprocess.Popen."""
    from pdfsys_bench.quality import OcrQualityScorer

    monkeypatch.delenv("QUALITY_URL", raising=False)
    built: list[OcrQualityScorer] = []

    import pdfsys_bench.quality

    def _capture(**kw):
        s = OcrQualityScorer(**kw)
        built.append(s)
        s._ensure_server = lambda: "http://stub"  # do not really spawn one
        s.score = lambda text: _Score(text, s.model_name)
        s.close = lambda: None
        return s

    monkeypatch.setattr(pdfsys_bench.quality, "OcrQualityScorer", _capture)

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    results, md, out = _run(
        tmp, [{"pdf_path": "1.pdf", "sha256": SHA_A}], {SHA_A: "text"}
    )
    score_run(results, md, out)

    assert len(built) == 1
    assert isinstance(built[0].model_name, str) and built[0].model_name, (
        "model_name must be a real string — Popen rejects None"
    )


# ---------------------------------------------------------------------------
# damaged input is refused, not silently truncated
# ---------------------------------------------------------------------------


def test_damage_in_the_middle_of_results_is_refused(tmp_path, scorer):
    """Pass 2 regenerates the output from the same reader, so stopping early
    would delete every row after the bad line from the output file."""
    from pdfsys_cli.runner import CorruptResultsError

    results, md, out = _run(
        tmp_path,
        [{"pdf_path": f"{i}.pdf", "sha256": chr(97 + i) * 64} for i in range(3)],
        {},
    )
    lines = results.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"pdf_path": "torn'
    results.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CorruptResultsError):
        score_run(results, md, out)

    assert not out.exists()


def test_a_torn_checkpoint_tail_is_repaired_before_appending(tmp_path, scorer):
    """Otherwise the next score is spliced onto the half-written line and the
    whole checkpoint stops parsing on the leg after this one."""
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "1.pdf", "sha256": SHA_A}, {"pdf_path": "2.pdf", "sha256": SHA_B}],
        {SHA_A: "aaa", SHA_B: "bbb"},
    )
    ckpt = out.with_suffix(out.suffix + ".partial")
    ckpt.write_text(
        json.dumps({"sha256": SHA_A, "quality_score": 1.0}) + "\n"
        + '{"sha256": "torn',
        encoding="utf-8",
    )

    report = score_run(results, md, out, resume=True)

    assert report.already == 1
    assert report.scored == 1
    assert len(_rows(out)) == 2


# ---------------------------------------------------------------------------
# accounting that the exit code depends on
# ---------------------------------------------------------------------------


def test_a_checkpoint_for_a_different_corpus_does_not_count_as_work(tmp_path, scorer):
    """`already` fed the "did anything happen" test, so a stale checkpoint
    whose keys match nothing could make a no-op run report success."""
    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A, "skip_reason": "deferred"}], {}
    )
    out.with_suffix(out.suffix + ".partial").write_text(
        json.dumps({"sha256": "f" * 64, "quality_score": 2.0}) + "\n", encoding="utf-8"
    )

    report = score_run(results, md, out, resume=True)

    assert report.already == 0
    assert not report.ok


def test_a_scorer_that_fails_more_than_it_succeeds_is_not_a_success(tmp_path, monkeypatch):
    import pdfsys_bench.quality

    monkeypatch.delenv("QUALITY_URL", raising=False)
    monkeypatch.setattr(
        pdfsys_bench.quality, "OcrQualityScorer",
        lambda *a, **kw: _FakeScorer(fail_on={"bad1", "bad2"}),
    )
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": f"{i}.pdf", "sha256": chr(97 + i) * 64} for i in range(3)],
        {SHA_A: "bad1", SHA_B: "bad2", SHA_C: "fine"},
    )

    report = score_run(results, md, out)

    assert report.scored == 1
    assert report.failed == 2
    assert not report.ok, "a dying scorer must not report success"


def test_a_duplicate_document_does_not_overwrite_an_untouched_score(tmp_path, scorer):
    """The same PDF twice in a corpus: one row already scored, one not. The
    merge is keyed on sha256, so the fresh score would land on both."""
    results, md, out = _run(
        tmp_path,
        [{"pdf_path": "first.pdf", "sha256": SHA_A, "quality_score": 0.5,
          "quality_model": "earlier/model"},
         {"pdf_path": "copy.pdf", "sha256": SHA_A, "quality_score": None}],
        {SHA_A: "x" * 250},
    )

    report = score_run(results, md, out)

    assert report.already == 1
    rows = _rows(out)
    assert rows[0]["quality_score"] == 0.5, "the report said this one was left alone"
    assert rows[0]["quality_model"] == "earlier/model"


def test_rescore_wins_over_resume(tmp_path, scorer):
    """--rescore means the previous scores are wrong, and the checkpoint holds
    previous scores."""
    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A}], {SHA_A: "x" * 250}
    )
    out.with_suffix(out.suffix + ".partial").write_text(
        json.dumps({"sha256": SHA_A, "quality_score": 9.0}) + "\n", encoding="utf-8"
    )

    report = score_run(results, md, out, resume=True, rescore=True)

    assert report.scored == 1
    assert _rows(out)[0]["quality_score"] == 2.5


# ---------------------------------------------------------------------------
# reaching the scorer
# ---------------------------------------------------------------------------


def test_an_unreachable_scorer_is_an_error_not_a_traceback(tmp_path, monkeypatch):
    from pdfsys_cli.score import ScorerUnreachable

    monkeypatch.setenv("QUALITY_URL", "http://scorer.invalid")

    import httpx

    def _boom(*a, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "get", _boom)
    results, md, out = _run(tmp_path, [], {})

    with pytest.raises(ScorerUnreachable, match="could not ask"):
        score_run(results, md, out)


def test_a_server_that_will_not_name_its_model_fails_closed(tmp_path, monkeypatch):
    """The flag exists for exactly this: a server that will not identify itself
    is the case to refuse, not the case to wave through."""
    monkeypatch.setenv("QUALITY_URL", "http://scorer.invalid")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    results, md, out = _run(tmp_path, [], {})

    with pytest.raises(ScoreModelMismatch, match="would not say"):
        score_run(results, md, out, model="expected/model")


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def test_a_row_carrying_a_lone_surrogate_survives_the_merge(tmp_path, scorer):
    """A filename the OS handed over as undecodable bytes reaches results.jsonl
    as a ``\\udcXX`` escape — the file itself is ASCII. json.loads turns it back
    into a lone surrogate, and re-encoding it strictly would abort pass 2 after
    every document had already been scored."""
    results = tmp_path / "results.jsonl"
    # Written as the escape, so the bytes on disk are plain ASCII.
    results.write_text(
        '{"pdf_path": "bad\\udce9name.pdf", "sha256": "' + SHA_A + '"}\n',
        encoding="ascii",
    )
    md = tmp_path / "markdown"
    md.mkdir()
    (md / f"{SHA_A}.md").write_text("text", encoding="utf-8")
    out = tmp_path / "scored.jsonl"

    report = score_run(results, md, out)

    assert report.scored == 1
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()
    assert json.loads(out.read_text(encoding="utf-8", errors="surrogatepass"))[
        "quality_score"
    ] is not None


def test_the_runner_can_also_write_a_surrogate_filename(tmp_path):
    """Same hazard one step earlier: os.walk and the worklist reader both hand
    back undecodable filenames as surrogates, and results.jsonl quotes them."""
    from pdfsys_cli.runner import DocResult, _write_summary

    row = DocResult(pdf_path="bad\udce9name.pdf", sha256=SHA_A)
    target = tmp_path / "results.jsonl"
    with target.open("w", encoding="utf-8", errors="surrogatepass") as f:
        f.write(row.to_json_line() + "\n")

    _write_summary(tmp_path / "s.json", {"pdf_dir": "bad\udce9dir"})

    assert target.exists() and (tmp_path / "s.json").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_scoring_nothing_exits_nonzero(tmp_path, scorer, capsys):
    from pdfsys_cli.__main__ import main

    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A, "skip_reason": "deferred"}], {}
    )

    rc = main(["score", "--results", str(results), "--markdown-dir", str(md),
               "--out", str(out)])

    assert rc == 1
    assert "not one row was scored" in capsys.readouterr().err


def test_an_existing_out_is_not_clobbered(tmp_path, scorer):
    from pdfsys_cli.__main__ import main

    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A}], {SHA_A: "text"}
    )
    out.write_text("previous\n", encoding="utf-8")

    rc = main(["score", "--results", str(results), "--markdown-dir", str(md),
               "--out", str(out)])

    assert rc == 1
    assert out.read_text(encoding="utf-8") == "previous\n"


def test_overwrite_replaces_it(tmp_path, scorer):
    from pdfsys_cli.__main__ import main

    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A}], {SHA_A: "text"}
    )
    out.write_text("previous\n", encoding="utf-8")

    rc = main(["score", "--results", str(results), "--markdown-dir", str(md),
               "--out", str(out), "--overwrite"])

    assert rc == 0
    assert _rows(out)[0]["quality_score"] is not None


@pytest.mark.parametrize(
    ("missing", "want"), [("results", "--results"), ("markdown", "--markdown-dir")]
)
def test_a_missing_input_exits_nonzero(tmp_path, missing, want, capsys):
    from pdfsys_cli.__main__ import main

    results, md, out = _run(
        tmp_path, [{"pdf_path": "1.pdf", "sha256": SHA_A}], {SHA_A: "text"}
    )
    gone = tmp_path / "nope"
    rc = main([
        "score",
        "--results", str(gone if missing == "results" else results),
        "--markdown-dir", str(gone if missing == "markdown" else md),
        "--out", str(out),
    ])

    assert rc == 1
    err = capsys.readouterr().err
    assert want in err and "nope" in err
    assert not out.exists()
