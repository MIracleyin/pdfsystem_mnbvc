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
