"""Tests for what counts as a PDF, and for reading a worklist of them.

Three entry points take a corpus directory — ``pdfsys run``, ``pdfsys
dataset --from-pdf-dir`` and ``_index_pdfs_by_sha256`` — and a shard is only
honest if they agree. They used to each spell ``rglob("*.pdf")``, which matches
case-sensitively on Linux and macOS and never sees a file with no extension.
Real corpora carry both, and neither the miss nor its size was reported.

The worklist is the other half: it is how a machine is handed a slice of a
corpus that lives on a different disk than the machine that chose the slice.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdfsys_core import PDF_MAGIC, iter_pdf_paths, read_pdf_list, take_inventory


def _pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def corpus(tmp_path):
    """One of each thing a real corpus turns out to contain."""
    root = tmp_path / "corpus"
    _pdf(root / "lower.pdf")
    _pdf(root / "UPPER.PDF")
    _pdf(root / "Mixed.Pdf")
    _pdf(root / "nested" / "deep.pdf")
    _pdf(root / "extensionless")
    (root / "notes.txt").write_bytes(b"just text")
    (root / "decoy").write_bytes(b"not a pdf at all")
    return root


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------


def test_any_case_of_pdf_is_found(corpus):
    names = {p.name for p in iter_pdf_paths(corpus)}
    assert {"lower.pdf", "UPPER.PDF", "Mixed.Pdf", "deep.pdf"} <= names


def test_an_extensionless_pdf_is_found_by_its_header(corpus):
    assert "extensionless" in {p.name for p in iter_pdf_paths(corpus)}


def test_a_non_pdf_is_not_picked_up_however_it_is_named(corpus):
    names = {p.name for p in iter_pdf_paths(corpus)}
    assert "decoy" not in names
    assert "notes.txt" not in names


def test_a_file_with_a_suffix_is_taken_at_its_word(tmp_path):
    """Sniffing is confined to extensionless files: a .txt holding PDF bytes is
    not silently reinterpreted, and we do not read every file in the tree."""
    (tmp_path / "x.txt").write_bytes(PDF_MAGIC + b"1.4 rest")
    assert list(iter_pdf_paths(tmp_path)) == []


def test_the_inventory_says_how_each_file_was_recognised(corpus):
    inv = take_inventory(corpus)
    assert len(inv.by_suffix) == 4
    assert [p.name for p in inv.by_magic] == ["extensionless"]
    assert len(inv) == 5
    assert inv.paths == tuple(sorted(inv.paths)), "callers rely on a stable order"


def test_sniffing_can_be_turned_off(corpus):
    inv = take_inventory(corpus, sniff_extensionless=False)
    assert inv.by_magic == ()
    assert len(inv) == 4


def test_an_unreadable_file_does_not_stop_the_scan(tmp_path):
    _pdf(tmp_path / "good.pdf")
    bad = tmp_path / "locked"
    bad.write_bytes(b"%PDF-1.4")
    bad.chmod(0o000)
    try:
        assert [p.name for p in iter_pdf_paths(tmp_path)] == ["good.pdf"]
    finally:
        bad.chmod(0o644)


def test_a_suffix_with_stray_whitespace_is_still_a_pdf(tmp_path):
    """A CRLF-mangled unzip produces "scan.pdf\\r"; a hand-typed list produces
    "scan.PDF ". Both used to fall between the two branches and vanish."""
    _pdf(tmp_path / "trailing space.PDF ")
    _pdf(tmp_path / "carriage.pdf\r")
    assert len(take_inventory(tmp_path)) == 2


def test_a_directory_that_cannot_be_entered_is_reported_not_ignored(tmp_path):
    """A corpus is not smaller because part of it was unreadable. rglob
    swallowed the PermissionError and simply yielded nothing."""
    _pdf(tmp_path / "visible.pdf")
    hidden = tmp_path / "locked"
    hidden.mkdir()
    _pdf(hidden / "invisible.pdf")
    hidden.chmod(0o000)
    try:
        inv = take_inventory(tmp_path)
        assert [p.name for p in inv.paths] == ["visible.pdf"]
        assert len(inv.unreadable_dirs) == 1
        assert "读不进去" in inv.describe()
    finally:
        hidden.chmod(0o755)


def test_a_readable_corpus_reports_no_unreadable_directories(corpus):
    assert take_inventory(corpus).unreadable_dirs == ()


def test_every_entry_point_discovers_the_same_set(corpus):
    """A shard must cover exactly the corpus the run processed, so the four
    places that scan a --pdf-dir have to agree. They each used to spell their
    own glob; this asserts they now share one."""
    from pdfsys_bench.loop import _iter_pdfs as bench_iter
    from pdfsys_cli.dataset_build import iter_pdfs as packager_iter

    canonical = sorted(iter_pdf_paths(corpus))
    assert canonical, "fixture must find something, or this proves nothing"
    assert sorted(packager_iter(corpus)) == canonical
    assert sorted(bench_iter(corpus, None)) == canonical


# ---------------------------------------------------------------------------
# worklists
# ---------------------------------------------------------------------------


def test_relative_entries_are_resolved_against_path_root(corpus, tmp_path):
    """The point of the whole flag: the box that wrote the list and the box
    that reads it mounted the corpus at different paths."""
    listing = tmp_path / "work.txt"
    listing.write_text("lower.pdf\nnested/deep.pdf\n", encoding="utf-8")

    result = read_pdf_list(listing, path_root=corpus)

    assert [p.name for p in result.paths] == ["lower.pdf", "deep.pdf"]
    assert all(p.is_file() for p in result.paths)


def test_absolute_entries_ignore_path_root(corpus, tmp_path):
    listing = tmp_path / "work.txt"
    listing.write_text(f"{corpus / 'lower.pdf'}\n", encoding="utf-8")

    result = read_pdf_list(listing, path_root=tmp_path / "somewhere-else")

    assert [p.name for p in result.paths] == ["lower.pdf"]


def test_list_order_is_preserved(corpus, tmp_path):
    """A worklist is usually a deliberate slice; re-sorting regroups the work."""
    listing = tmp_path / "work.txt"
    listing.write_text("nested/deep.pdf\nlower.pdf\n", encoding="utf-8")

    result = read_pdf_list(listing, path_root=corpus)

    assert [p.name for p in result.paths] == ["deep.pdf", "lower.pdf"]


def test_every_entry_is_accounted_for(corpus, tmp_path):
    listing = tmp_path / "work.txt"
    listing.write_text(
        "lower.pdf\n\nghost.pdf\nlower.pdf\nnested/deep.pdf\n", encoding="utf-8"
    )

    result = read_pdf_list(listing, path_root=corpus)

    assert result.entries == 4, "blank lines are not entries"
    assert len(result.paths) == 2
    assert result.missing == ("ghost.pdf",)
    assert result.duplicates == ("lower.pdf",)
    assert len(result.paths) + len(result.missing) + len(result.duplicates) == result.entries


def test_a_bom_does_not_eat_the_first_entry(corpus, tmp_path):
    """Worklists get opened in Windows editors."""
    listing = tmp_path / "work.txt"
    listing.write_bytes("﻿lower.pdf\nnested/deep.pdf\n".encode())

    result = read_pdf_list(listing, path_root=corpus)

    assert [p.name for p in result.paths] == ["lower.pdf", "deep.pdf"]


def test_crlf_line_endings_are_handled(corpus, tmp_path):
    listing = tmp_path / "work.txt"
    listing.write_bytes(b"lower.pdf\r\nnested/deep.pdf\r\n")

    result = read_pdf_list(listing, path_root=corpus)

    assert [p.name for p in result.paths] == ["lower.pdf", "deep.pdf"]


def test_one_undecodable_line_costs_one_line_not_the_run(corpus, tmp_path):
    """Filenames are bytes. A path the locale cannot decode must not abort a
    218k-document batch."""
    listing = tmp_path / "work.txt"
    listing.write_bytes(b"lower.pdf\n\xff\xfe-broken.pdf\nnested/deep.pdf\n")

    result = read_pdf_list(listing, path_root=corpus)

    assert [p.name for p in result.paths] == ["lower.pdf", "deep.pdf"]
    assert len(result.missing) == 1
    assert result.entries == 3


def test_a_repeat_is_dropped_even_spelled_differently(corpus, tmp_path):
    """Two spellings of one file would extract it twice and, downstream, break
    the (doc_id, page_index) primary key."""
    listing = tmp_path / "work.txt"
    listing.write_text(f"lower.pdf\n{corpus / 'lower.pdf'}\n", encoding="utf-8")

    result = read_pdf_list(listing, path_root=corpus)

    assert len(result.paths) == 1
    assert len(result.duplicates) == 1
