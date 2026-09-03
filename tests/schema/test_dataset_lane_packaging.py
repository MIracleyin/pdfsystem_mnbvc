"""Tests for packaging one lane's documents rather than a whole corpus.

``--from-pdf-dir`` scans a directory, which is right for a single-machine run
and wrong for a split one: the corpus root also holds the documents the OCR
lane owns, and mupdf will happily re-extract a scanned PDF into empty pages.
Those pages carry the same doc_id the GPU shard already used, and
``(doc_id, page_index)`` is the primary key — so the merged dataset is invalid,
and nothing in either shard says which copy is the real one.

``--from-pdf-list`` packages exactly what a lane extracted. And when ``--meta``
is there to say which lane owns what, packaging someone else's documents is an
error rather than something to discover at validation time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from pdfsys_cli.__main__ import main


def _pdf(path: Path, text: str = "born digital text") -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    """Two documents this lane extracted, one the OCR lane took."""
    root = tmp_path / "corpus"
    mine_a = _pdf(root / "a.pdf", "alpha")
    mine_b = _pdf(root / "nested" / "b.pdf", "beta")
    theirs = _pdf(root / "scan.pdf", "gamma")
    meta = tmp_path / "results.jsonl"
    meta.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"pdf_path": str(mine_a), "sha256": _sha(mine_a),
                 "extract_backend": "mupdf", "skip_reason": None,
                 "error_class": None, "quality_score": 2.0},
                {"pdf_path": str(mine_b), "sha256": _sha(mine_b),
                 "extract_backend": "mupdf", "skip_reason": None,
                 "error_class": None, "quality_score": 2.5},
                {"pdf_path": str(theirs), "sha256": _sha(theirs),
                 "extract_backend": "pipeline", "skip_reason": "lane-filter",
                 "error_class": None, "quality_score": None},
            ]
        ) + "\n",
        encoding="utf-8",
    )
    listing = tmp_path / "cpu_lane.txt"
    listing.write_text(f"{mine_a}\n{mine_b}\n", encoding="utf-8")
    return {"root": root, "meta": meta, "listing": listing,
            "mine": [mine_a, mine_b], "theirs": theirs, "tmp": tmp_path}


def _docs(shard: Path) -> set[str]:
    import pyarrow.parquet as pq

    return {
        r["doc_id"]
        for p in (shard / "pages").glob("*.parquet")
        for r in pq.read_table(p).to_pylist()
    }


# ---------------------------------------------------------------------------
# packaging by list
# ---------------------------------------------------------------------------


def test_a_list_packages_exactly_its_documents(corpus):
    out = corpus["tmp"] / "ds"
    rc = main([
        "dataset", "--from-pdf-list", str(corpus["listing"]),
        "--to", str(out), "--shard", "cpu-00", "--images", "none",
    ])

    assert rc == 0
    assert _docs(out) == {_sha(p) for p in corpus["mine"]}


def test_relative_entries_resolve_against_path_root(corpus):
    listing = corpus["tmp"] / "rel.txt"
    listing.write_text("a.pdf\nnested/b.pdf\n", encoding="utf-8")
    out = corpus["tmp"] / "ds"

    rc = main([
        "dataset", "--from-pdf-list", str(listing), "--path-root", str(corpus["root"]),
        "--to", str(out), "--shard", "cpu-00", "--images", "none",
    ])

    assert rc == 0
    assert _docs(out) == {_sha(p) for p in corpus["mine"]}


def test_a_missing_list_exits_nonzero(corpus, capsys):
    rc = main([
        "dataset", "--from-pdf-list", str(corpus["tmp"] / "nope.txt"),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])
    assert rc == 1
    assert "--from-pdf-list not found" in capsys.readouterr().err


def test_a_list_whose_entries_all_vanished_exits_nonzero(corpus, capsys):
    listing = corpus["tmp"] / "gone.txt"
    listing.write_text("ghost-a.pdf\nghost-b.pdf\n", encoding="utf-8")

    rc = main([
        "dataset", "--from-pdf-list", str(listing), "--path-root", str(corpus["root"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])

    assert rc == 1
    assert "do not exist" in capsys.readouterr().err


def test_the_list_lane_still_refuses_crops(corpus, capsys):
    """mupdf reads a text layer; it never cuts figures out of a page."""
    rc = main([
        "dataset", "--from-pdf-list", str(corpus["listing"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "crops",
    ])
    assert rc == 1
    assert "mupdf lane" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the collision this exists to prevent
# ---------------------------------------------------------------------------


def test_scanning_the_corpus_root_is_refused_when_meta_names_another_lane(
    corpus, capsys
):
    """The whole point: --from-pdf-dir over a split corpus would mupdf-extract
    the OCR lane's documents into empty pages carrying doc_ids the GPU shard
    already used."""
    rc = main([
        "dataset", "--from-pdf-dir", str(corpus["root"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "cpu-00", "--images", "none",
        "--meta", str(corpus["meta"]),
    ])

    assert rc == 1
    err = capsys.readouterr().err
    assert "mineru lane" in err
    assert "primary key" in err
    assert not (corpus["tmp"] / "ds" / "pages").exists(), "refused before writing"


@pytest.mark.parametrize(
    "row",
    [
        # Extracted by the other lane on a single machine: no skip_reason.
        {"extract_backend": "pipeline", "skip_reason": None},
        {"extract_backend": "vlm", "skip_reason": None},
        # Handed over on a split run: the other lane will package it.
        {"extract_backend": "pipeline", "skip_reason": "lane-filter"},
    ],
    ids=["pipeline-extracted", "vlm-extracted", "handed-over"],
)
def test_each_way_a_document_can_belong_elsewhere_is_caught(corpus, row):
    """The predicate is a disjunction; a fixture that sets both arms at once
    would let either one be deleted without a test noticing."""
    meta = corpus["tmp"] / "one.jsonl"
    meta.write_text(
        json.dumps({"sha256": _sha(corpus["mine"][0]), "error_class": None, **row})
        + "\n",
        encoding="utf-8",
    )
    listing = corpus["tmp"] / "one.txt"
    listing.write_text(f"{corpus['mine'][0]}\n", encoding="utf-8")

    rc = main([
        "dataset", "--from-pdf-list", str(listing),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
        "--meta", str(meta),
    ])

    assert rc == 1


@pytest.mark.parametrize(
    "row",
    [
        # Stage-B held it back: no shard holds it, so nothing to collide with.
        {"extract_backend": "deferred", "skip_reason": "deferred"},
        # The router could not open it — encrypted, corrupt. Certain at scale.
        {"extract_backend": "deferred", "skip_reason": None, "error_class": "router"},
        {"extract_backend": "unknown-backend:x", "skip_reason": "unknown-backend:x"},
        # Extraction was attempted by the other lane and failed: no output.
        {"extract_backend": "pipeline", "skip_reason": None,
         "error_class": "extract_pipeline"},
    ],
    ids=["deferred", "router-error", "unknown-backend", "other-lane-failed"],
)
def test_a_document_no_lane_produced_does_not_block_packaging(corpus, row):
    """An encrypted PDF is a certainty in a 218k corpus. Refusing the whole
    packaging run over one is far worse than the per-document failure it used
    to be — and there is no rival shard to collide with anyway."""
    meta = corpus["tmp"] / "one.jsonl"
    meta.write_text(
        json.dumps({"sha256": _sha(corpus["mine"][0]), "error_class": None, **row})
        + "\n",
        encoding="utf-8",
    )
    listing = corpus["tmp"] / "one.txt"
    listing.write_text(f"{corpus['mine'][0]}\n", encoding="utf-8")

    rc = main([
        "dataset", "--from-pdf-list", str(listing),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
        "--meta", str(meta),
    ])

    assert rc == 0


def test_the_mineru_lane_is_checked_too(corpus):
    """The mirror hazard: packaging a MinerU directory for a document the mupdf
    lane already shipped. The directory name is the sha256, so it is free."""
    doc = corpus["tmp"] / "mineru" / _sha(corpus["mine"][0])
    doc.mkdir(parents=True)
    (doc / f"{doc.name}_content_list.json").write_text("[]", encoding="utf-8")
    (doc / f"{doc.name}_middle.json").write_text(
        json.dumps({"pdf_info": [{"page_size": [612.0, 792.0]}]}), encoding="utf-8"
    )

    rc = main([
        "dataset", "--from-mineru", str(corpus["tmp"] / "mineru"),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "gpu-00", "--images", "none",
        "--meta", str(corpus["meta"]),
    ])

    assert rc == 1


def test_the_lane_list_passes_the_same_check(corpus):
    rc = main([
        "dataset", "--from-pdf-list", str(corpus["listing"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "cpu-00", "--images", "none",
        "--meta", str(corpus["meta"]),
    ])
    assert rc == 0


def test_allow_other_lanes_is_the_escape_hatch(corpus, capsys):
    rc = main([
        "dataset", "--from-pdf-dir", str(corpus["root"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "cpu-00", "--images", "none",
        "--meta", str(corpus["meta"]), "--allow-other-lanes",
    ])

    assert rc == 0
    assert "another lane extracted" in capsys.readouterr().err
    assert _sha(corpus["theirs"]) in _docs(corpus["tmp"] / "ds")


def test_a_meta_row_that_names_no_backend_is_not_treated_as_another_lane(corpus):
    """A router-only run, or a hand-made meta, says nothing about lanes.
    Treating silence as 'someone else owns it' would refuse ordinary runs."""
    quiet = corpus["tmp"] / "quiet.jsonl"
    quiet.write_text(
        "\n".join(
            json.dumps({"sha256": _sha(p), "quality_score": 1.0})
            for p in [*corpus["mine"], corpus["theirs"]]
        ) + "\n",
        encoding="utf-8",
    )

    rc = main([
        "dataset", "--from-pdf-dir", str(corpus["root"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
        "--meta", str(quiet),
    ])

    assert rc == 0


def test_without_meta_there_is_nothing_to_check_against(corpus):
    """No --meta means no lane information; the scan is the old behaviour."""
    rc = main([
        "dataset", "--from-pdf-dir", str(corpus["root"]),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])
    assert rc == 0
    assert len(_docs(corpus["tmp"] / "ds")) == 3


def test_a_non_pdf_entry_is_skipped_not_packaged(corpus, capsys):
    """A list can name anything. The scan applies the shared discovery rule;
    the list lane has to apply the same one, or a text file becomes a
    'document' whose content is whatever mupdf makes of it."""
    junk = corpus["root"] / "notes.txt"
    junk.write_text("not a pdf", encoding="utf-8")
    listing = corpus["tmp"] / "mixed.txt"
    listing.write_text(f"{corpus['mine'][0]}\n{junk}\n", encoding="utf-8")

    rc = main([
        "dataset", "--from-pdf-list", str(listing),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])

    assert rc == 0
    assert "not PDFs" in capsys.readouterr().err
    assert _docs(corpus["tmp"] / "ds") == {_sha(corpus["mine"][0])}


def test_a_list_of_only_non_pdfs_exits_nonzero(corpus, capsys):
    junk = corpus["root"] / "notes.txt"
    junk.write_text("not a pdf", encoding="utf-8")
    listing = corpus["tmp"] / "junk.txt"
    listing.write_text(f"{junk}\n", encoding="utf-8")

    rc = main([
        "dataset", "--from-pdf-list", str(listing),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])

    assert rc == 1
    assert "no readable PDF" in capsys.readouterr().err


def test_a_repeated_entry_is_packaged_once_and_reported(corpus, capsys):
    listing = corpus["tmp"] / "dup.txt"
    listing.write_text(
        f"{corpus['mine'][0]}\n{corpus['mine'][0]}\n", encoding="utf-8"
    )

    rc = main([
        "dataset", "--from-pdf-list", str(listing),
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])

    assert rc == 0
    assert "are repeats" in capsys.readouterr().err
    assert len(_docs(corpus["tmp"] / "ds")) == 1


def test_the_descriptor_records_what_the_list_asked_for(corpus):
    """On disk a shard built from a half-resolved list is otherwise
    indistinguishable from one built from a complete short list."""
    listing = corpus["tmp"] / "partial.txt"
    listing.write_text(
        f"{corpus['mine'][0]}\n{corpus['tmp']}/ghost.pdf\n", encoding="utf-8"
    )
    out = corpus["tmp"] / "ds"

    assert main([
        "dataset", "--from-pdf-list", str(listing), "--to", str(out),
        "--shard", "s", "--images", "none",
    ]) == 0

    desc = json.loads((out / "s.meta.json").read_text())
    assert desc["list_entries"] == 2
    assert desc["list_missing"] == 1
    assert desc["documents"] == 1


def test_path_root_on_the_scan_lane_says_it_is_ignored(corpus, capsys):
    main([
        "dataset", "--from-pdf-dir", str(corpus["root"]), "--path-root", "/tmp",
        "--to", str(corpus["tmp"] / "ds"), "--shard", "s", "--images", "none",
    ])
    assert "only applies to --from-pdf-list" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the two lanes merge into one valid dataset
# ---------------------------------------------------------------------------


def test_the_two_lanes_produce_one_valid_dataset(corpus):
    """The end state: a mupdf shard and a (simulated) OCR shard side by side,
    no doc_id in both, validator clean."""
    from pdfsys_cli.dataset_validate import validate_shard

    out = corpus["tmp"] / "ds"
    assert main([
        "dataset", "--from-pdf-list", str(corpus["listing"]),
        "--to", str(out), "--shard", "cpu-00", "--images", "none",
        "--meta", str(corpus["meta"]),
    ]) == 0

    # The OCR lane's document, packaged through its own (here: also mupdf, but
    # from a list holding only it) shard.
    theirs = corpus["tmp"] / "gpu_lane.txt"
    theirs.write_text(f"{corpus['theirs']}\n", encoding="utf-8")
    assert main([
        "dataset", "--from-pdf-list", str(theirs),
        "--to", str(out), "--shard", "gpu-00", "--images", "none",
        "--allow-other-lanes", "--meta", str(corpus["meta"]),
    ]) == 0

    report = validate_shard(out)
    assert report.ok, [str(f) for f in report.findings]
    assert len(_docs(out)) == 3, "every document exactly once"
