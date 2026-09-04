"""Tests for the CLI's refusals and its reporting of what it silently dropped.

Every case here used to exit 0. Running the pipeline as a fleet of per-bucket
jobs across two machines turns each one into a job that reports success and
produces nothing: a wrong ``--pdf-dir`` (``rglob("*.pdf")`` is case-sensitive),
a reused ``--shard`` name (``pq.ParquetWriter`` truncates, so the second lane
replaces the first), or a ``--meta`` file that covers a different corpus than
the shard being built.
"""

from __future__ import annotations

import json

import pymupdf
import pytest

from pdfsys_cli.__main__ import _apply_run_meta, _load_run_meta, main
from pdfsys_core import PageRecord

DOC = "a" * 64


def _pdf(path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# an empty run is a failed run
# ---------------------------------------------------------------------------


def test_run_over_a_directory_with_no_pdfs_exits_nonzero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main([
        "run", "--pdf-dir", str(empty), "--out-dir", str(tmp_path / "out"),
        "--stages", "router",
    ])
    assert rc == 1


def test_a_missing_pdf_list_exits_nonzero(tmp_path):
    rc = main([
        "run", "--pdf-list", str(tmp_path / "nope.txt"),
        "--out-dir", str(tmp_path / "out"), "--stages", "router",
    ])
    assert rc == 1


def test_a_worklist_whose_every_entry_is_missing_exits_nonzero(tmp_path):
    """The signature of a wrong --path-root, which otherwise looks like an
    empty corpus and marches on."""
    listing = tmp_path / "work.txt"
    listing.write_text("a.pdf\nb.pdf\n", encoding="utf-8")
    rc = main([
        "run", "--pdf-list", str(listing), "--path-root", str(tmp_path / "wrong"),
        "--out-dir", str(tmp_path / "out"), "--stages", "router",
    ])
    assert rc == 1


def test_a_worklist_run_reaches_the_files_through_path_root(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    _pdf(corpus / "nested" / "b.pdf")
    listing = tmp_path / "work.txt"
    listing.write_text("a.pdf\nnested/b.pdf\nghost.pdf\n", encoding="utf-8")

    rc = main([
        "run", "--pdf-list", str(listing), "--path-root", str(corpus),
        "--out-dir", str(tmp_path / "out"), "--stages", "router",
    ])

    assert rc == 0
    out = capsys.readouterr()
    assert "processed 2 PDFs" in out.out
    assert "1/3 listed paths do not exist" in out.err


def test_pdf_list_takes_precedence_over_pdf_dir_and_says_so(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    _pdf(corpus / "b.pdf")
    listing = tmp_path / "work.txt"
    listing.write_text(str(corpus / "a.pdf") + "\n", encoding="utf-8")

    rc = main([
        "run", "--pdf-list", str(listing), "--pdf-dir", str(corpus),
        "--out-dir", str(tmp_path / "out"), "--stages", "router",
    ])

    assert rc == 0
    out = capsys.readouterr()
    assert "--pdf-list wins" in out.err
    assert "processed 1 PDFs" in out.out


def test_a_resumed_leg_that_matched_nothing_warns(tmp_path, capsys):
    """Carried rows but zero skips means the paths do not line up — the leg is
    silently redoing every document."""
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    out_dir = tmp_path / "out"
    assert main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(out_dir),
        "--stages", "router",
    ]) == 0
    # Rewrite the recorded path so nothing matches on the next leg.
    rows = out_dir / "results.jsonl"
    rows.write_text(
        json.dumps({**json.loads(rows.read_text()), "pdf_path": "/gone/a.pdf"}) + "\n",
        encoding="utf-8",
    )

    main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(out_dir),
        "--stages", "router", "--resume",
    ])

    assert "skipped nothing" in capsys.readouterr().err


def test_an_unknown_backend_exits_1_without_a_traceback(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    rc = main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(tmp_path / "out"),
        "--stages", "router", "--extract-backends", "tesseract",
    ])
    assert rc == 1
    assert "Unknown extract backend" in capsys.readouterr().err


def test_an_unknown_backend_in_a_config_file_also_exits_1(tmp_path, capsys):
    """The YAML path raises the same ValueError and must read the same way."""
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")
    conf = tmp_path / "c.yaml"
    conf.write_text(
        f"stages: [router]\ninput:\n  pdf_dir: {corpus}\n"
        f"output:\n  dir: {tmp_path / 'out'}\nextract_backends: [tesseract]\n",
        encoding="utf-8",
    )

    rc = main(["run", "-c", str(conf)])

    assert rc == 1
    assert "Unknown extract backend" in capsys.readouterr().err


def test_a_scalar_lane_in_yaml_is_read_as_one_backend(tmp_path):
    """`extract_backends: mupdf` used to become ['m','u','p','d','f']."""
    from pdfsys_cli.config import load_config

    conf = tmp_path / "c.yaml"
    conf.write_text("extract_backends: mupdf\n", encoding="utf-8")
    assert load_config(conf).extract_backends == ["mupdf"]


def test_an_empty_vlm_lane_is_flagged_before_any_work(tmp_path, capsys):
    """Only stage-B says `vlm`, and only when vlm.enabled — otherwise the lane
    is empty by construction and the run would filter the whole corpus."""
    corpus = tmp_path / "corpus"
    _pdf(corpus / "a.pdf")

    main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(tmp_path / "out"),
        "--stages", "router,extract", "--extract-backends", "vlm",
    ])

    assert "vlm.enabled is false" in capsys.readouterr().err


def test_run_that_found_pdfs_still_exits_zero(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")
    rc = main([
        "run", "--pdf-dir", str(src), "--out-dir", str(tmp_path / "out"),
        "--stages", "router",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# one shard name, one writer
# ---------------------------------------------------------------------------


def test_building_over_an_existing_shard_is_refused(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")
    out = tmp_path / "ds"
    args = [
        "dataset", "--from-pdf-dir", str(src), "--to", str(out),
        "--shard", "cpu-00", "--images", "none",
    ]
    assert main(args) == 0
    assert main(args) == 1, "a second lane must not silently truncate the first"


def test_overwrite_replaces_the_whole_shard_not_just_its_pages(tmp_path):
    """A shard is up to four parquets plus a descriptor. The media writers open
    lazily, so truncating pages/ alone would splice the new pages table onto the
    previous build's images/ and page_images/."""
    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")
    out = tmp_path / "ds"
    base = ["dataset", "--from-pdf-dir", str(src), "--to", str(out), "--shard", "s"]

    # First build stores whole-page rasters, so page_images/s.parquet exists.
    assert main([*base, "--images", "pages"]) == 0
    assert (out / "page_images" / "s.parquet").exists()

    # Rebuild the same shard storing no pixels at all.
    assert main([*base, "--images", "none", "--overwrite"]) == 0
    assert not (out / "page_images" / "s.parquet").exists(), (
        "the previous build's rasters survived into a --images none shard"
    )
    assert json.loads((out / "s.meta.json").read_text())["images_mode"] == "none"


def test_a_collision_in_any_of_a_shards_files_is_refused(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")
    out = tmp_path / "ds"
    base = ["dataset", "--from-pdf-dir", str(src), "--to", str(out), "--shard", "s"]
    assert main([*base, "--images", "pages"]) == 0

    # pages/s.parquet gone but page_images/s.parquet still there — still a clash.
    (out / "pages" / "s.parquet").unlink()
    assert main([*base, "--images", "none"]) == 1


def test_a_different_shard_name_is_never_blocked(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "a.pdf")
    out = tmp_path / "ds"
    base = [
        "dataset", "--from-pdf-dir", str(src), "--to", str(out), "--images", "none",
    ]
    assert main([*base, "--shard", "cpu-00"]) == 0
    assert main([*base, "--shard", "gpu-00"]) == 0


# ---------------------------------------------------------------------------
# --meta joins, and says how well it joined
# ---------------------------------------------------------------------------


def test_meta_rows_without_a_key_are_dropped_and_reported(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"sha256": DOC, "quality_score": 3.0},
                {"sha256": None, "pdf_path": "unrouted.pdf"},
                {"sha256": DOC, "quality_score": 4.0},
            ]
        ),
        encoding="utf-8",
    )

    meta = _load_run_meta(path)

    assert set(meta) == {DOC}
    assert meta[DOC]["quality_score"] == 4.0, "last row wins"
    err = capsys.readouterr().err
    assert "重复" in err
    assert "没有 sha256" in err


def test_meta_does_not_overwrite_the_extractor_the_builder_determined():
    """middle.json says what produced these pages; the run row is a fallback."""
    page = PageRecord(doc_id=DOC, page_index=0, extractor="vlm")
    merged = _apply_run_meta(page, {"extract_backend": "pipeline"})
    assert merged.extractor == "vlm"


def test_meta_fills_an_extractor_the_builder_could_not_determine():
    page = PageRecord(doc_id=DOC, page_index=0, extractor="")
    merged = _apply_run_meta(page, {"extract_backend": "pipeline"})
    assert merged.extractor == "pipeline"


def test_meta_carries_the_layout_model_across():
    page = PageRecord(doc_id=DOC, page_index=0)
    merged = _apply_run_meta(page, {"layout_model": "doclayout-yolo@1.0"})
    assert merged.layout_model == "doclayout-yolo@1.0"


@pytest.mark.parametrize("row", [None, {}])
def test_no_meta_row_leaves_the_page_alone(row):
    page = PageRecord(doc_id=DOC, page_index=0, extractor="mupdf")
    assert _apply_run_meta(page, row) is page


def test_partial_meta_coverage_is_reported(tmp_path, capsys):
    """A shard whose quality columns are all null looks identical whether the
    scorer never ran or the --meta file covers a different corpus."""
    import hashlib

    src = tmp_path / "pdfs"
    covered = _pdf(src / "a.pdf")
    _pdf(src / "b.pdf")
    meta = tmp_path / "results.jsonl"
    meta.write_text(
        json.dumps({
            "sha256": hashlib.sha256(covered.read_bytes()).hexdigest(),
            "quality_score": 3.5,
        }),
        encoding="utf-8",
    )

    rc = main([
        "dataset", "--from-pdf-dir", str(src), "--to", str(tmp_path / "ds"),
        "--shard", "s", "--images", "none", "--meta", str(meta),
    ])

    assert rc == 0
    out = capsys.readouterr()
    assert "meta matched 1/2" in out.out
    assert "没有对应行" in out.err


def test_packaging_nothing_exits_nonzero(tmp_path):
    """Mirrors cmd_run's empty-corpus guard on the command each lane packages
    with — a per-bucket fleet job that writes no documents is not a success."""
    src = tmp_path / "mineru"
    (src / "aaa").mkdir(parents=True)
    # A content list that is present but yields no document.
    (src / "aaa" / "x_content_list.json").write_text("[]", encoding="utf-8")

    rc = main([
        "dataset", "--from-mineru", str(src), "--to", str(tmp_path / "ds"),
        "--shard", "s", "--images", "none",
    ])
    assert rc == 1
