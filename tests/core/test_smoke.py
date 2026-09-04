"""Tests for ``pdfsys smoke`` — the end-to-end check itself.

The smoke run is what stands between "the split works" and "six commands each
exited 0". It is only worth having if it would notice a break, so these pin
that: the corpus really does split across both lanes, and a broken phase turns
into a failed check rather than a green one.
"""

from __future__ import annotations

import pytest

from pdfsys_cli.smoke import build_corpus, run_smoke
from pdfsys_router.xgb_model import default_weights_path

# Every test here routes real PDFs, and the router's weights are gitignored —
# they are FinePDFs' to distribute. Without them classify() returns `deferred`
# with an error for every document instead of raising, so these would fail as
# "the corpus only exercises one lane" and say nothing about the real cause.
pytestmark = pytest.mark.skipif(
    not default_weights_path().is_file(),
    reason="router weights absent — run `python -m pdfsys_router.download_weights`",
)


def test_the_corpus_covers_both_lanes_and_the_awkward_shapes(tmp_path):
    from pdfsys_router import Router

    made = build_corpus(tmp_path / "corpus")
    router = Router()
    routed = {
        name: router.classify(p).backend.value
        for name, p in made.items()
        if name not in ("decoy", "encrypted")
    }

    assert {v for v in routed.values()} >= {"mupdf", "pipeline"}, (
        "a corpus that only exercises one lane cannot check a split"
    )
    assert routed["scan_a"] == "pipeline"
    assert routed["text_a"] == "mupdf"
    # The shapes that have actually broken discovery here.
    assert made["text_upper"].name == "UPPER.PDF"
    assert not made["text_noext"].suffix
    assert made["duplicate"].read_bytes() == made["text_a"].read_bytes()


def test_the_scanned_pages_carry_real_glyphs(tmp_path):
    """A page of grey bars also routes to OCR — the router only sees the absent
    text layer — but a real MinerU returns nothing for it, and the check would
    then be measuring the stub rather than the service. Found by running the
    smoke against the live GPU box: 0/2 extracted, "empty markdown"."""
    import pymupdf

    made = build_corpus(tmp_path / "corpus")
    doc = pymupdf.open(made["scan_a"])
    page = doc[0]

    assert not page.get_text().strip(), "a text layer would route it to mupdf"
    pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    ink = sum(1 for b in pix.samples if b < 128) / len(pix.samples)
    doc.close()
    assert 0.005 < ink < 0.5, f"ink fraction {ink:.4f} — blank page or solid block"


def test_a_real_scorer_is_not_asked_to_be_the_stub(tmp_path, monkeypatch):
    """--model is asserted only when we know the answer. Against a real server
    it would demand the stub's name and fail every scoring pass."""
    import pdfsys_cli.smoke as smoke_mod

    seen: list[list[str]] = []

    def _spy(argv):
        seen.append(argv)
        return 0

    monkeypatch.setattr(smoke_mod, "build_corpus", lambda root: {})
    monkeypatch.setattr(
        smoke_mod, "stub_services",
        lambda m, q: __import__("contextlib").nullcontext(
            (m or "http://stub-m", q or "http://stub-q")
        ),
    )
    monkeypatch.setattr(smoke_mod, "_phases", lambda *a, **kw: seen.append(kw))

    run_smoke(tmp_path / "w1", quality_url="http://real:8765")
    run_smoke(tmp_path / "w2")

    assert seen[0]["expect_model"] is None, "a real server serves a real model"
    assert seen[1]["expect_model"] == smoke_mod.SMOKE_MODEL


def test_the_corpus_is_small(tmp_path):
    """The whole point is that this is free to run. The real bench corpus is
    17 MB; if this creeps toward that, nobody runs it every time either."""
    made = build_corpus(tmp_path / "corpus")
    total = sum(p.stat().st_size for p in made.values())
    assert total < 200_000, f"{total} bytes"


def test_a_healthy_pipeline_passes_every_check(tmp_path):
    result = run_smoke(tmp_path / "work")

    assert result.ok, result.report()
    assert len(result.steps) >= 10, "the check should actually check things"


def test_a_broken_phase_is_reported_not_swallowed(tmp_path, monkeypatch):
    """If a phase silently produced nothing, the smoke must go red — that is
    the entire failure mode it exists to catch."""
    import pdfsys_cli.smoke as smoke_mod

    real = smoke_mod.build_corpus

    def _text_only(root):
        made = real(root)
        # Delete the scans: now nothing reaches the GPU lane, so phase 2
        # extracts zero documents where the check expects some.
        for key in ("scan_a", "scan_b"):
            made[key].unlink()
        return made

    monkeypatch.setattr(smoke_mod, "build_corpus", _text_only)

    result = run_smoke(tmp_path / "work")

    assert not result.ok
    by_name = {name: ok for name, ok, _ in result.steps}
    assert not by_name["phase 2 (GPU lane)"]
    # Nothing extracted means nothing to have persisted, so `0 == 0` must not
    # read as a pass — that is how a dead GPU lane looks green.
    assert not by_name["MinerU sidecars persisted"]


def test_the_workdir_is_cleaned_up_unless_asked_to_keep(tmp_path):
    work = tmp_path / "gone"
    run_smoke(work)
    assert not work.exists()

    kept = tmp_path / "kept"
    run_smoke(kept, keep=True)
    assert (kept / "dataset" / "pages").is_dir()


def test_the_cli_exits_nonzero_when_a_check_fails(tmp_path, monkeypatch, capsys):
    import pdfsys_cli.smoke as smoke_mod
    from pdfsys_cli.__main__ import main

    monkeypatch.setattr(
        smoke_mod, "run_smoke",
        lambda *a, **kw: smoke_mod.SmokeResult(
            steps=[("phase 1", False, "nothing happened")], chatter="detail here"
        ),
    )

    rc = main(["smoke"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "1 check(s) failed" in err
    assert "detail here" in err, "the swallowed output is shown when it matters"
