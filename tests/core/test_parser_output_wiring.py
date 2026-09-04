"""Tests that the run actually keeps what the OCR parsers hand back.

The parsers have always known how to persist MinerU's sidecars — middle.json,
content_list.json and the figure crops — into the ``<sha256>/`` layout that
``pdfsys dataset --from-mineru`` reads. ``pdfsys run`` simply never told them
where to put it: ``PipelineConfig``/``VlmConfig`` were built without
``output_dir``, so it stayed ``None`` and every sidecar was dropped on arrival.

The markdown survived, which is why nothing looked wrong. But the only other
copy is mineru-api's own task directory, which it garbage-collects and for
which the containerised deployment mounts no volume — so after a real run
there was nothing left to package, and ``--from-mineru`` reported an empty
directory rather than a lost one.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdfsys_cli.config import apply_cli_overrides, default_config
from pdfsys_cli.runner import Components


def _pdf(path: Path) -> Path:
    """A page with enough real text that the router sends it down the mupdf
    lane. These tests are about the pre-flight warning, not about extraction —
    a document that reached MinerU would just wait for a server that is not
    there."""
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(30):
        page.insert_text(
            (72, 72 + i * 20),
            f"Line {i}: this page carries a real text layer, extracted directly.",
            fontsize=11,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# the wiring itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["pipeline_parser", "vlm_parser"])
def test_the_parser_is_told_where_to_keep_its_sidecars(which, tmp_path):
    cfg = apply_cli_overrides(
        default_config(), parser_output_dir=str(tmp_path / "sidecars")
    )

    parser = getattr(Components(cfg), which)

    assert parser.config.output_dir == tmp_path / "sidecars"


@pytest.mark.parametrize("which", ["pipeline_parser", "vlm_parser"])
def test_without_the_flag_the_parser_keeps_nothing(which):
    parser = getattr(Components(default_config()), which)
    assert parser.config.output_dir is None


@pytest.mark.parametrize("which", ["pipeline_parser", "vlm_parser"])
def test_crops_are_requested_by_default(which):
    assert getattr(Components(default_config()), which).config.return_images is True


@pytest.mark.parametrize("which", ["pipeline_parser", "vlm_parser"])
def test_crops_can_be_declined(which):
    cfg = apply_cli_overrides(default_config(), no_parser_images=True)
    assert getattr(Components(cfg), which).config.return_images is False


def test_one_flag_covers_both_parsers(tmp_path):
    """Safe within a run because stage-B sends each document to exactly one of
    them. The layout carries no backend in the path, so the same document
    re-parsed by the other backend into the same directory would overwrite
    rather than sit beside — see the caveat on the flag."""
    cfg = apply_cli_overrides(
        default_config(), parser_output_dir=str(tmp_path / "s")
    )
    assert cfg.pipeline.output_dir == cfg.vlm.output_dir


def test_the_two_backends_share_a_path_for_one_document(tmp_path):
    """Pins the overwrite hazard the flag's help warns about, so nobody
    'fixes' the docs by assuming the layout separates them."""
    from pdfsys_parser_pipeline.extract import _persist_sidecar

    sha = "a" * 64
    first = _persist_sidecar({"a": 1}, tmp_path, sha, f"{sha}_middle.json")
    second = _persist_sidecar({"b": 2}, tmp_path, sha, f"{sha}_middle.json")

    assert first == second, "no backend dimension in the path"


def test_the_yaml_key_is_not_silently_dropped(tmp_path):
    """PipelineCfg had no such field, and _fill_dataclass ignores unknown keys —
    so an output_dir written into a config file did exactly nothing."""
    from pdfsys_cli.config import load_config

    conf = tmp_path / "c.yaml"
    conf.write_text(
        "pipeline:\n  output_dir: /data/sidecars\n  return_images: false\n",
        encoding="utf-8",
    )

    cfg = load_config(conf)

    assert cfg.pipeline.output_dir == "/data/sidecars"
    assert cfg.pipeline.return_images is False


# ---------------------------------------------------------------------------
# what the parsers write is what the packager reads
# ---------------------------------------------------------------------------


def test_the_layout_the_parser_writes_is_the_one_the_packager_globs(tmp_path):
    """Spans two packages: parser-pipeline decides the on-disk shape and
    pdfsys-cli decides what it looks for. They have to agree, or a run produces
    sidecars that `--from-mineru` walks straight past."""
    from pdfsys_cli.dataset_build import iter_mineru_dirs
    from pdfsys_parser_pipeline.extract import _persist_sidecar

    out = tmp_path / "sidecars"
    sha = "a" * 64
    _persist_sidecar([{"type": "text", "text": "hi"}], out, sha,
                     f"{sha}_content_list.json")
    _persist_sidecar({"pdf_info": []}, out, sha, f"{sha}_middle.json")

    assert [d.name for d in iter_mineru_dirs(out)] == [sha]


# ---------------------------------------------------------------------------
# the discard announces itself
# ---------------------------------------------------------------------------


def _run_cli(tmp_path, *extra):
    from pdfsys_cli.__main__ import main

    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    return main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(tmp_path / "out"),
        "--stages", "router", *extra,
    ])


def test_a_run_that_would_discard_sidecars_says_so(tmp_path, capsys):
    _run_cli(tmp_path, "--stages", "router,extract", "--extract-backends", "mupdf,pipeline")
    assert "no output_dir for pipeline" in capsys.readouterr().err


def test_a_mupdf_only_lane_is_not_nagged(tmp_path, capsys):
    """It never calls MinerU, so it has no sidecars to lose."""
    _run_cli(tmp_path, "--stages", "router,extract", "--extract-backends", "mupdf")
    assert "no output_dir for" not in capsys.readouterr().err


def test_a_run_without_the_extract_stage_is_not_nagged(tmp_path, capsys):
    _run_cli(tmp_path)
    assert "no output_dir for" not in capsys.readouterr().err


def test_the_configured_location_is_printed(tmp_path, capsys):
    _run_cli(
        tmp_path, "--stages", "router,extract",
        "--parser-output-dir", str(tmp_path / "sidecars"),
    )
    assert "sidecars:" in capsys.readouterr().out


def test_a_vlm_only_lane_is_also_checked(tmp_path, capsys):
    """The two parsers are configured independently, so reading only
    `pipeline.output_dir` gave a VLM lane a false all-clear."""
    from pdfsys_cli.config import load_config

    conf = tmp_path / "c.yaml"
    conf.write_text(
        "stages: [router, extract]\n"
        "pipeline:\n  output_dir: /tmp/only-pipeline\n",
        encoding="utf-8",
    )
    cfg = load_config(conf)
    cfg.extract_backends = ["vlm"]

    from pdfsys_cli.runner import parser_output_dirs

    assert parser_output_dirs(cfg) == {"vlm": None}, "pipeline's dir is irrelevant here"


# ---------------------------------------------------------------------------
# a bad directory costs one message, not the whole run
# ---------------------------------------------------------------------------


def test_a_sidecar_dir_that_is_actually_a_file_is_refused_up_front(tmp_path):
    """Persisting happens after each document is parsed, so an unusable path
    would raise once per document — after the OCR — and take the markdown with
    it. It has to fail before the first document instead."""
    from pdfsys_cli.runner import ParserOutputDirError, run

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")

    cfg = apply_cli_overrides(
        default_config(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), parser_output_dir=str(blocker),
    )

    with pytest.raises(ParserOutputDirError, match="not usable"):
        run(cfg)


def test_an_unwritable_sidecar_dir_is_refused_up_front(tmp_path):
    """The ordinary container case: /data bind-mounted root-owned."""
    from pdfsys_cli.runner import ParserOutputDirError, run

    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o500)
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")

    cfg = apply_cli_overrides(
        default_config(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"),
        parser_output_dir=str(parent / "sidecars"),
    )
    try:
        with pytest.raises(ParserOutputDirError):
            run(cfg)
    finally:
        parent.chmod(0o755)


def test_a_usable_sidecar_dir_is_created_before_the_run(tmp_path):
    from pdfsys_cli.runner import run

    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    where = tmp_path / "deep" / "sidecars"

    run(apply_cli_overrides(
        default_config(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), parser_output_dir=str(where),
    ))

    assert where.is_dir()
    assert not (where / ".pdfsys-write-probe").exists(), "the probe is cleaned up"


def test_no_sidecar_dir_configured_needs_no_check(tmp_path):
    """The default path must not start inventing directories."""
    from pdfsys_cli.runner import run

    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    run(apply_cli_overrides(
        default_config(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(tmp_path / "out"), extract_backends="mupdf",
    ))


# ---------------------------------------------------------------------------
# the state every fresh checkout starts in
# ---------------------------------------------------------------------------


def test_missing_router_weights_stop_the_run_instead_of_deferring_everything(tmp_path):
    """``models/`` is gitignored — the XGBoost weights are FinePDFs' to
    distribute, so a fresh clone has none. ``classify()`` promises never to
    raise, so without this check the weights being absent is not an error but
    a *verdict*: every document comes back ``deferred``, every row counts as
    an error, nothing is extracted, and the run exits 0. A whole corpus can go
    through that and report success. Caught by CI, where it looked like the
    smoke corpus had stopped covering both lanes."""
    from pdfsys_cli.runner import RouterWeightsError, run

    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    out = tmp_path / "out"

    cfg = apply_cli_overrides(
        default_config(), stages="router,extract", pdf_dir=str(corpus),
        out_dir=str(out), router_weights=str(tmp_path / "absent.ubj"),
    )

    with pytest.raises(RouterWeightsError, match="download_weights"):
        run(cfg)

    # Before the first document, so the leg leaves no half-written results.
    assert not (out / "results.jsonl").exists()


def test_the_weights_check_is_skipped_when_the_router_stage_is_not_running(tmp_path):
    """Refusing to start for want of a router that will not be used would be a
    new way to fail. Every ``--stages`` value resolves to something containing
    `router`, so this is reachable only programmatically — which is exactly
    when a guard nobody can see is worth pinning."""
    from pdfsys_cli.runner import Components, _check_router_weights

    cfg = apply_cli_overrides(
        default_config(), stages="extract", pdf_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"), router_weights=str(tmp_path / "absent.ubj"),
    )
    from pdfsys_cli.runner import RouterWeightsError

    assert "router" in cfg.stages, "the CLI always implies router"
    with pytest.raises(RouterWeightsError):
        _check_router_weights(cfg, Components(cfg))  # the guard is live here

    cfg.stages = [s for s in cfg.stages if s != "router"]
    _check_router_weights(cfg, Components(cfg))  # must not raise
