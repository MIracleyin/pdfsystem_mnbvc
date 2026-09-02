"""Tests for a dataset directory that more than one writer contributed to.

Once extraction is split across machines, the two lanes each produce their own
shard and both land in one directory: ``pages/cpu-00.parquet`` beside
``pages/gpu-00.parquet``. Three parts of the format's enforcement assumed a
single writer and got this wrong.

The sortedness rule is per file, because a reader scans one file at a time;
demanding the *concatenation* be sorted additionally requires the files to
partition the doc_id space in filename order, which nothing ever promised.
A repeated ``image_id`` is an error inside one file and merely wasteful across
two, since content addressing guarantees the bytes are identical. And the page
raster table is keyed on ``(doc_id, page_index, render_dpi)``, so that is the
key the writer has to dedupe on — keying on ``image_id`` drops the second of
two pages that happen to render to the same pixels.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pdfsys_cli.dataset_validate import validate_shard
from pdfsys_cli.dataset_writer import PAGE_SCHEMA, DatasetWriter
from pdfsys_core import Block, BlockType, ImageBlob, PageRecord, render_markdown

LOGO_BYTES = b"\x89PNG\r\n\x1a\n" + b"shared-letterhead"
LOGO = ImageBlob(
    image_id=hashlib.sha256(LOGO_BYTES).hexdigest(),
    data=LOGO_BYTES, format="png", width=4, height=3,
)
BLANK_BYTES = b"\xff\xd8blank-page-raster"
BLANK = ImageBlob(
    image_id=hashlib.sha256(BLANK_BYTES).hexdigest(),
    data=BLANK_BYTES, format="jpeg", width=8, height=11,
)


def _page(doc_id: str, page_index: int, *, n_pages: int = 1, **kw) -> PageRecord:
    blocks = [Block(idx=page_index, page=page_index, type=BlockType.TEXT, text="正文")]
    defaults = dict(
        doc_id=doc_id, page_index=page_index, width_pt=612.0, height_pt=792.0,
        text=render_markdown(blocks), blocks=tuple(blocks),
        extractor="mupdf", doc_n_pages=n_pages,
    )
    defaults.update(kw)
    return PageRecord(**defaults)


# ---------------------------------------------------------------------------
# 0.4 — sortedness is per file
# ---------------------------------------------------------------------------


def test_two_lanes_interleaving_doc_ids_validate_clean(tmp_path):
    """The exact shape the CPU/GPU split produces: neither shard's range nests."""
    root = tmp_path / "shard"
    # Interleaved on purpose — cpu holds aa/cc, gpu holds bb/dd. Each file is
    # sorted; the concatenation in filename order is not.
    with DatasetWriter(root, shard="cpu-00") as w:
        for doc in ("a" * 64, "c" * 64):
            w.write([_page(doc, 0)])
    with DatasetWriter(root, shard="gpu-00") as w:
        for doc in ("b" * 64, "d" * 64):
            w.write([_page(doc, 0)])

    keys = [
        (r["doc_id"], r["page_index"])
        for p in sorted((root / "pages").glob("*.parquet"))
        for r in pq.read_table(p).to_pylist()
    ]
    assert keys != sorted(keys), "fixture must actually interleave, or it proves nothing"

    report = validate_shard(root)
    assert report.ok, [str(f) for f in report.findings]


def test_a_single_unsorted_file_is_still_an_error(tmp_path):
    """Relaxing the cross-file rule must not relax the per-file one."""
    root = tmp_path / "shard"
    with DatasetWriter(root, shard="only") as w:
        for doc in ("a" * 64, "b" * 64):
            w.write([_page(doc, 0)])

    path = next((root / "pages").glob("*.parquet"))
    rows = pq.read_table(path).to_pylist()
    rows.reverse()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PAGE_SCHEMA).replace_schema_metadata(
            {b"pdfsys.schema": b"pdfsys.page/2"}
        ),
        path,
    )

    report = validate_shard(root)
    assert "order" in {f.check for f in report.findings if f.severity == "error"}


def test_the_same_document_in_two_shards_is_still_an_error(tmp_path):
    """Relaxing sortedness leaves directory-wide key uniqueness as the ONLY
    thing stopping two lanes from both claiming a document. It has to hold."""
    root = tmp_path / "shard"
    for shard in ("cpu-00", "gpu-00"):
        with DatasetWriter(root, shard=shard) as w:
            w.write([_page("a" * 64, 0)])

    report = validate_shard(root)
    assert "key" in {f.check for f in report.findings if f.severity == "error"}


def test_a_crop_repeated_inside_one_file_is_an_error(tmp_path):
    """image_id IS the key of the images table — the other half of the
    keyed/unkeyed split, whose warn side is covered below."""
    root = tmp_path / "shard"
    with DatasetWriter(root, shard="only") as w:
        blocks = [
            Block(idx=0, page=0, type=BlockType.IMAGE, image_id=LOGO.image_id,
                  bbox=(0.1, 0.1, 0.5, 0.5)),
        ]
        page = _page("a" * 64, 0, text=render_markdown(blocks),
                     blocks=tuple(blocks), image_ids=(LOGO.image_id,))
        w.write([page], [LOGO])

    path = next((root / "images").glob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    pq.write_table(
        pa.Table.from_pylist(rows + rows, schema=table.schema), path
    )

    report = validate_shard(root)
    assert "images" in {f.check for f in report.findings if f.severity == "error"}


def test_a_truncated_run_of_warnings_cannot_hide_an_error(tmp_path):
    """The display cap is per (check, severity). Sharing it across severities
    let a burst of warnings eat the slots an error needed, and `ok` was read
    off the printed list — so a broken shard reported 通过."""
    from pdfsys_cli.dataset_validate import Report

    report = Report()
    for i in range(50):
        report.warn("images", f"warning {i}")
    report.error("images", "the one that matters")

    assert report.n_errors == 1
    assert not report.ok
    assert any(
        f.severity == "error" and "matters" in f.message for f in report.findings
    ), "the error must survive into the printed output too"


def test_a_crop_shared_by_two_shards_warns_but_passes(tmp_path):
    """A letterhead in both lanes costs storage, not correctness."""
    root = tmp_path / "shard"
    for shard, doc in (("cpu-00", "a" * 64), ("gpu-00", "b" * 64)):
        with DatasetWriter(root, shard=shard) as w:
            blocks = [
                Block(idx=0, page=0, type=BlockType.IMAGE, image_id=LOGO.image_id,
                      bbox=(0.1, 0.1, 0.5, 0.5)),
            ]
            page = _page(
                doc, 0, text=render_markdown(blocks), blocks=tuple(blocks),
                image_ids=(LOGO.image_id,),
            )
            w.write([page], [LOGO])

    report = validate_shard(root)
    assert report.ok, [str(f) for f in report.findings]
    assert any(f.check == "images" and f.severity == "warn" for f in report.findings)


# ---------------------------------------------------------------------------
# 0.3 — writer counters and dedupe keys
# ---------------------------------------------------------------------------


def test_a_write_with_no_pages_is_not_a_document(tmp_path):
    with DatasetWriter(tmp_path / "s") as w:
        w.write([])
        w.write([_page("a" * 64, 0)])
        assert w.docs_written == 1


def test_the_same_crop_twice_in_one_call_is_written_once(tmp_path):
    """A logo repeating down one page arrives as one batch of identical blobs."""
    root = tmp_path / "s"
    with DatasetWriter(root) as w:
        w.write([_page("a" * 64, 0)], [LOGO, LOGO, LOGO])
        assert w.images_written == 1

    rows = pq.read_table(root / "images").to_pylist()
    assert [r["image_id"] for r in rows] == [LOGO.image_id]


def test_two_pages_rendering_to_identical_pixels_both_keep_a_raster(tmp_path):
    """Guards the new tuple key: it must include render_dpi and must not
    collapse back to image_id. (This case also passed before the fix, because
    the old within-call bug happened to mask the keying bug —
    ``test_two_documents_with_an_identical_blank_page_both_keep_a_raster`` is
    the one that isolates it.)"""
    root = tmp_path / "s"
    doc = "a" * 64
    pages = [
        _page(doc, i, n_pages=2, page_image_id=BLANK.image_id, render_dpi=200)
        for i in range(2)
    ]
    with DatasetWriter(root) as w:
        w.write(pages, [], [(p, BLANK) for p in pages])
        assert w.page_images_written == 2

    rows = pq.read_table(root / "page_images").to_pylist()
    assert sorted(r["page_index"] for r in rows) == [0, 1]
    # …and the format's real key stays unique, so the validator is satisfied.
    keys = {(r["doc_id"], r["page_index"], r["render_dpi"]) for r in rows}
    assert len(keys) == 2


def test_the_same_page_raster_offered_twice_is_written_once(tmp_path):
    root = tmp_path / "s"
    doc = "a" * 64
    page = _page(doc, 0, page_image_id=BLANK.image_id, render_dpi=200)
    with DatasetWriter(root) as w:
        w.write([page], [], [(page, BLANK), (page, BLANK)])
        assert w.page_images_written == 1


def test_two_documents_with_an_identical_blank_page_both_keep_a_raster(tmp_path):
    """The sharp case: the collision spans two write() calls, so no within-call
    dedupe hides it. Scanned corpora are full of identical blank pages, and
    dropping the second one leaves that page's bbox:// markers dangling."""
    root = tmp_path / "s"
    with DatasetWriter(root) as w:
        for doc in ("a" * 64, "b" * 64):
            page = _page(doc, 0, page_image_id=BLANK.image_id, render_dpi=200)
            w.write([page], [], [(page, BLANK)])
        assert w.page_images_written == 2

    rows = pq.read_table(root / "page_images").to_pylist()
    assert sorted(r["doc_id"] for r in rows) == ["a" * 64, "b" * 64]
    report = validate_shard(root)
    assert report.ok, [str(f) for f in report.findings]


@pytest.mark.parametrize("dpi", [150, 300])
def test_the_same_page_at_two_dpis_is_two_rasters(tmp_path, dpi):
    """render_dpi is part of the table's key, so it must be part of the dedupe
    key. Guards against the key being narrowed back to (doc_id, page_index)."""
    root = tmp_path / f"s{dpi}"
    doc = "a" * 64
    lo = _page(doc, 0, page_image_id=BLANK.image_id, render_dpi=200)
    hi = _page(doc, 0, page_image_id=BLANK.image_id, render_dpi=dpi)
    with DatasetWriter(root) as w:
        w.write([lo], [], [(lo, BLANK), (hi, BLANK)])
        assert w.page_images_written == 2
