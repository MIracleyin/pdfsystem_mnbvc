"""Tests for the mupdf lane's route into a ``pdfsys.page/v2`` shard.

The lane that handles most documents had no way into L2 at all: it leaves no
MinerU sidecars behind, and its page structure is persisted nowhere — a run
writes one merged ``markdown/<sha256>.md`` with no page boundaries, and
``segments_excerpt`` is filled only on the VLM branch and truncated. So this
path re-extracts, and these tests pin what re-extraction has to get right:
page boundaries, per-page geometry, the document's identity, and the fact
that it can never produce crops.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from pdfsys_cli.__main__ import main
from pdfsys_cli.dataset_build import build_from_pdf, iter_pdfs, select_pdfs
from pdfsys_cli.dataset_validate import validate_shard


def _pdf(path: Path, texts: list[str]) -> Path:
    """A PDF with one page per string, every page a different size.

    Differing sizes are the point: geometry must be read per page, not copied
    from the first one.
    """
    doc = pymupdf.open()
    for i, text in enumerate(texts):
        page = doc.new_page(width=400 + i * 20, height=600 + i * 10)
        page.insert_text((72, 72), text, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# build_from_pdf
# ---------------------------------------------------------------------------


def test_a_pdf_becomes_one_row_per_page(tmp_path):
    pdf = _pdf(tmp_path / "three.pdf", ["alpha here", "bravo here", "charlie here"])

    pages, blobs = build_from_pdf(pdf)

    assert [p.page_index for p in pages] == [0, 1, 2]
    assert all(p.doc_n_pages == 3 for p in pages)
    assert "alpha" in pages[0].text
    assert "bravo" in pages[1].text
    assert "bravo" not in pages[0].text, "page boundary leaked"
    assert blobs == [], "the mupdf lane reads a text layer; it has no crops to store"


def test_each_page_carries_its_own_size(tmp_path):
    pdf = _pdf(tmp_path / "sizes.pdf", ["a", "b", "c"])

    pages, _ = build_from_pdf(pdf)

    assert [round(p.width_pt) for p in pages] == [400, 420, 440]
    assert [round(p.height_pt) for p in pages] == [600, 610, 620]


def test_doc_id_is_the_source_pdf_sha256(tmp_path):
    """The whole shard is keyed on this, and it is content-addressed — a
    doc_id derived from anything else (a filename, a run id) would not survive
    re-ingesting the same PDF."""
    pdf = _pdf(tmp_path / "one.pdf", ["only"])

    (page,), _ = build_from_pdf(pdf)

    assert page.doc_id == hashlib.sha256(pdf.read_bytes()).hexdigest()


def test_source_uri_defaults_to_the_pdf_path(tmp_path):
    """This lane knows where the PDF is, so it should say so without --meta."""
    pdf = _pdf(tmp_path / "one.pdf", ["only"])

    (page,), _ = build_from_pdf(pdf)

    assert page.source_uri == str(pdf)


def test_an_explicit_source_uri_wins(tmp_path):
    """The local path is a fallback, not the truth — a corpus ingested from
    object storage should record where it really came from."""
    pdf = _pdf(tmp_path / "one.pdf", ["only"])

    (page,), _ = build_from_pdf(pdf, source_uri="s3://corpus/one.pdf")

    assert page.source_uri == "s3://corpus/one.pdf"


# ---------------------------------------------------------------------------
# discovery and selection
# ---------------------------------------------------------------------------


def test_iter_pdfs_recurses_and_ignores_everything_else(tmp_path):
    _pdf(tmp_path / "top.pdf", ["a"])
    _pdf(tmp_path / "sub" / "deep.pdf", ["b"])
    (tmp_path / "notes.txt").write_text("not a pdf", encoding="utf-8")

    assert {p.name for p in iter_pdfs(tmp_path)} == {"top.pdf", "deep.pdf"}


def test_identical_pdfs_are_deduplicated_and_the_drop_is_reported(tmp_path):
    """Same bytes under two names is normal in a crawled corpus, and
    ``(doc_id, page_index)`` is the primary key — keeping both writes a shard
    that violates its own contract."""
    a = _pdf(tmp_path / "a.pdf", ["same"])
    b = tmp_path / "b.pdf"
    b.write_bytes(a.read_bytes())

    kept, dropped = select_pdfs([a, b])

    assert len(kept) == 1
    assert len(dropped) == 1, "a silent drop is one you find out about far too late"
    assert dropped[0][0] in (a, b)


def test_selection_is_ordered_by_doc_id_not_by_filename(tmp_path):
    """DatasetWriter requires ascending doc_id, and doc_id is a content hash —
    so filename order says nothing about it."""
    paths = [_pdf(tmp_path / f"{name}.pdf", [f"content {name}"]) for name in "zyx"]

    kept, _ = select_pdfs(paths)

    ids = [doc_id for doc_id, _ in kept]
    assert ids == sorted(ids)


def test_an_unreadable_pdf_is_dropped_with_a_reason(tmp_path):
    kept, dropped = select_pdfs([tmp_path / "does-not-exist.pdf"])

    assert kept == []
    assert len(dropped) == 1 and dropped[0][2]


# ---------------------------------------------------------------------------
# through the CLI
# ---------------------------------------------------------------------------


def test_cli_packages_the_mupdf_lane_into_a_valid_shard(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "one.pdf", ["first page", "second page"])
    _pdf(src / "two.pdf", ["other document"])
    out = tmp_path / "shard"

    assert main(["dataset", "--from-pdf-dir", str(src), "--to", str(out)]) == 0

    report = validate_shard(out)
    assert report.ok, [str(f) for f in report.findings]
    assert report.stats["pages"] == 3
    assert report.stats["by_extractor"] == {"mupdf": 3}
    assert report.stats["page_images"] == 3, "the lane's default is one raster per page"
    assert report.stats["images"] == 0


def test_cli_rejects_crops_on_the_mupdf_lane(tmp_path, capsys):
    """Failing loudly beats writing a shard with no pixels in it and calling
    the mode 'crops'."""
    src = tmp_path / "pdfs"
    _pdf(src / "one.pdf", ["only"])

    rc = main(["dataset", "--from-pdf-dir", str(src), "--to", str(tmp_path / "s"),
               "--images", "crops"])

    assert rc == 1
    assert "crops" in capsys.readouterr().err


def test_cli_images_none_stores_no_pixels(tmp_path):
    src = tmp_path / "pdfs"
    _pdf(src / "one.pdf", ["only"])
    out = tmp_path / "shard"

    assert main(["dataset", "--from-pdf-dir", str(src), "--to", str(out),
                 "--images", "none"]) == 0

    report = validate_shard(out)
    assert report.ok, [str(f) for f in report.findings]
    assert (report.stats["images"], report.stats["page_images"]) == (0, 0)


def test_cli_refuses_a_directory_with_no_pdfs(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main(["dataset", "--from-pdf-dir", str(empty), "--to", str(tmp_path / "s")]) == 1
    assert "no PDF found" in capsys.readouterr().err


def test_the_two_source_flags_are_mutually_exclusive(tmp_path):
    """They select different lanes with different inputs; accepting both would
    have to silently pick one."""
    with pytest.raises(SystemExit):
        main(["dataset", "--from-pdf-dir", str(tmp_path),
              "--from-mineru", str(tmp_path), "--to", str(tmp_path / "s")])


def test_a_source_flag_is_required(tmp_path):
    with pytest.raises(SystemExit):
        main(["dataset", "--to", str(tmp_path / "s")])
