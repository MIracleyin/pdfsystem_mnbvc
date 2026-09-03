"""Tests for ``pdfsys smoke`` — the end-to-end check itself.

The smoke run is what stands between "the split works" and "six commands each
exited 0". It is only worth having if it would notice a break, so these pin
that: the corpus really does split across both lanes, and a broken phase turns
into a failed check rather than a green one.
"""

from __future__ import annotations

from pdfsys_cli.smoke import build_corpus, run_smoke


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
    assert any(not ok for name, ok, _ in result.steps if "phase 1" in name or "phase 2" in name)


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
