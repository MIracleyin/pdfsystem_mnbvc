"""Tests for the shard validator.

A validator that passes everything is worse than none, so every check gets a
deliberately broken shard and has to catch it. The happy-path test guards the
other direction: a correct shard must come back clean.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pdfsys_cli.dataset_validate import validate_shard
from pdfsys_cli.dataset_writer import PAGE_SCHEMA, DatasetWriter
from pdfsys_core import Block, BlockType, ImageBlob, PageRecord, image_ref, render_markdown

PNG = b"\x89PNG\r\n\x1a\n" + b"payload" * 4
IMG = ImageBlob(
    image_id=hashlib.sha256(PNG).hexdigest(), data=PNG, format="png", width=4, height=3
)
RASTER_BYTES = b"\xff\xd8raster"
RASTER = ImageBlob(
    image_id=hashlib.sha256(RASTER_BYTES).hexdigest(),
    data=RASTER_BYTES, format="jpeg", width=8, height=11,
)


def _page(doc_id: str, page_index: int, *, with_image: bool = True, **kw) -> PageRecord:
    blocks = [Block(idx=page_index * 10, page=page_index, type=BlockType.TEXT, text="正文")]
    if with_image:
        blocks.append(
            Block(idx=page_index * 10 + 1, page=page_index, type=BlockType.IMAGE,
                  caption="图 1", image_id=IMG.image_id, bbox=(0.1, 0.1, 0.5, 0.5))
        )
    defaults = dict(
        doc_id=doc_id, page_index=page_index, width_pt=612.0, height_pt=792.0,
        text=render_markdown(blocks), blocks=tuple(blocks),
        image_ids=(IMG.image_id,) if with_image else (),
        extractor="vlm", doc_n_pages=2,
    )
    defaults.update(kw)
    return PageRecord(**defaults)


@pytest.fixture
def good(tmp_path):
    """Two documents, two pages each, one shared image."""
    root = tmp_path / "shard"
    with DatasetWriter(root) as w:
        for doc in ("1" * 64, "2" * 64):
            pages = [_page(doc, i) for i in range(2)]
            w.write(pages, [IMG])
    return root


def _rewrite_pages(root, mutate):
    """Read pages back, mutate the row dicts, write them out again."""
    rows = pq.read_table(root / "pages").to_pylist()
    mutate(rows)
    path = next((root / "pages").glob("*.parquet"))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PAGE_SCHEMA).replace_schema_metadata(
            {b"pdfsys.schema": b"pdfsys.page/2"}
        ),
        path,
    )


def _checks(report):
    return {f.check for f in report.findings if f.severity == "error"}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_a_correct_shard_passes(good):
    report = validate_shard(good)
    assert report.ok, [str(f) for f in report.findings]
    assert report.stats["documents"] == 2
    assert report.stats["pages"] == 4
    assert report.stats["images"] == 1, "shared image stored once"


def test_missing_pages_directory_is_an_error(tmp_path):
    assert not validate_shard(tmp_path).ok


# ---------------------------------------------------------------------------
# keys and ordering
# ---------------------------------------------------------------------------


def test_duplicate_primary_key_is_caught(good):
    _rewrite_pages(good, lambda rows: rows.append(dict(rows[0])))
    assert "key" in _checks(validate_shard(good))


def test_unsorted_rows_are_caught(good):
    _rewrite_pages(good, lambda rows: rows.reverse())
    assert "order" in _checks(validate_shard(good))


def test_page_gap_is_caught(good):
    def mutate(rows):
        rows[1]["page_index"] = 5
    _rewrite_pages(good, mutate)
    assert "key" in _checks(validate_shard(good))


def test_doc_n_pages_disagreeing_with_row_count_is_caught(good):
    def mutate(rows):
        for r in rows:
            r["doc_n_pages"] = 9
    _rewrite_pages(good, mutate)
    assert "key" in _checks(validate_shard(good))


def test_denormalized_column_differing_across_pages_is_caught(good):
    def mutate(rows):
        rows[0]["doc_lang"] = "eng_Latn"
        rows[1]["doc_lang"] = "zho_Hans"
    _rewrite_pages(good, mutate)
    assert "denorm" in _checks(validate_shard(good))


# ---------------------------------------------------------------------------
# counters, bboxes, blocks
# ---------------------------------------------------------------------------


def test_wrong_n_chars_is_caught(good):
    def mutate(rows):
        rows[0]["n_chars"] = 99999
    _rewrite_pages(good, mutate)
    assert "counter" in _checks(validate_shard(good))


def test_wrong_n_images_is_caught(good):
    def mutate(rows):
        rows[0]["n_images"] = 7
    _rewrite_pages(good, mutate)
    assert "counter" in _checks(validate_shard(good))


@pytest.mark.parametrize("bbox", [
    {"x0": -0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5},   # 越界
    {"x0": 0.9, "y0": 0.1, "x1": 0.2, "y1": 0.5},    # 反向
])
def test_bad_bbox_is_caught(good, bbox):
    def mutate(rows):
        rows[0]["blocks"][1]["bbox"] = bbox
    _rewrite_pages(good, mutate)
    assert "bbox" in _checks(validate_shard(good))


def test_duplicate_block_idx_within_a_document_is_caught(good):
    def mutate(rows):
        rows[1]["blocks"][0]["idx"] = rows[0]["blocks"][0]["idx"]
    _rewrite_pages(good, mutate)
    assert "block" in _checks(validate_shard(good))


def test_mention_pointing_at_a_missing_block_is_caught(good):
    def mutate(rows):
        rows[0]["blocks"][0]["mentions"] = [4242]
    _rewrite_pages(good, mutate)
    assert "mention" in _checks(validate_shard(good))


# ---------------------------------------------------------------------------
# referential integrity
# ---------------------------------------------------------------------------


def test_image_id_not_in_the_images_table_is_caught(good):
    def mutate(rows):
        rows[0]["image_ids"] = ["f" * 64]
    _rewrite_pages(good, mutate)
    assert "ref" in _checks(validate_shard(good))


def test_marker_pointing_at_a_missing_image_is_caught(good):
    def mutate(rows):
        rows[0]["text"] = image_ref("e" * 64)
        rows[0]["n_chars"] = len(rows[0]["text"])
    _rewrite_pages(good, mutate)
    assert "marker" in _checks(validate_shard(good))


def test_region_marker_without_a_page_raster_is_caught(good):
    def mutate(rows):
        rows[0]["text"] = "![](bbox://0.1,0.1,0.5,0.5)"
        rows[0]["n_chars"] = len(rows[0]["text"])
    _rewrite_pages(good, mutate)
    assert "marker" in _checks(validate_shard(good))


def test_block_image_not_listed_in_image_ids_is_caught(good):
    def mutate(rows):
        rows[0]["image_ids"] = []
        rows[0]["text"] = "正文"
        rows[0]["n_chars"] = 2
    _rewrite_pages(good, mutate)
    assert "image_ids" in _checks(validate_shard(good))


def test_page_image_id_without_render_dpi_is_caught(good):
    def mutate(rows):
        rows[0]["page_image_id"] = RASTER.image_id
    _rewrite_pages(good, mutate)
    assert "ref" in _checks(validate_shard(good))


# ---------------------------------------------------------------------------
# content addressing
# ---------------------------------------------------------------------------


def test_image_id_that_is_not_the_hash_of_its_bytes_is_caught(tmp_path):
    """内容寻址不能只是个约定，得真的对得上。"""
    root = tmp_path / "shard"
    lying = ImageBlob(image_id="a" * 64, data=PNG, format="png", width=4, height=3)
    with DatasetWriter(root) as w:
        w.write([_page("1" * 64, 0, image_ids=(lying.image_id,), doc_n_pages=1)], [lying])
    assert "images" in _checks(validate_shard(root))
    # 关掉重算就查不出来 —— 这正是 --no-hash 的取舍
    assert "images" not in _checks(validate_shard(root, verify_hashes=False))


def test_wrong_n_bytes_is_caught(tmp_path):
    root = tmp_path / "shard"
    with DatasetWriter(root) as w:
        w.write([_page("1" * 64, 0, doc_n_pages=1)], [IMG])
    path = next((root / "images").glob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["n_bytes"] = 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    assert "images" in _checks(validate_shard(root))


# ---------------------------------------------------------------------------
# warnings, not errors
# ---------------------------------------------------------------------------


def test_one_document_storing_both_crops_and_a_raster_is_a_warning(tmp_path):
    root = tmp_path / "shard"
    page = _page("1" * 64, 0, doc_n_pages=1, page_image_id=RASTER.image_id, render_dpi=200)
    with DatasetWriter(root) as w:
        w.write([page], [IMG], [(page, RASTER)])
    report = validate_shard(root)
    assert report.ok, "同时存两张表是浪费，不是违约"
    assert any(f.check == "images-mode" for f in report.findings)


def test_different_documents_using_different_image_modes_is_fine(tmp_path):
    """一个 shard 里混了 crops 链路和 pages 链路的产物，没有任何像素存两遍。"""
    root = tmp_path / "shard"
    raster_page = _page("2" * 64, 0, with_image=False, doc_n_pages=1,
                        page_image_id=RASTER.image_id, render_dpi=200)
    with DatasetWriter(root) as w:
        w.write([_page("1" * 64, 0, doc_n_pages=1)], [IMG])
        w.write([raster_page], [], [(raster_page, RASTER)])
    report = validate_shard(root)
    assert report.ok
    assert not any(f.check == "images-mode" for f in report.findings)


def test_orphan_image_is_a_warning(tmp_path):
    root = tmp_path / "shard"
    with DatasetWriter(root) as w:
        w.write([_page("1" * 64, 0, with_image=False, doc_n_pages=1)], [IMG])
    report = validate_shard(root)
    assert report.ok
    assert any(f.check == "orphan" for f in report.findings)


def test_replacement_chars_are_a_statistic_not_a_finding(tmp_path):
    """U+FFFD 是语料的属性，不是编码的缺陷。"""
    root = tmp_path / "shard"
    with DatasetWriter(root) as w:
        w.write([_page("1" * 64, 0, with_image=False, doc_n_pages=1,
                       text="数学符号丢了 ��", image_ids=())], [])
    report = validate_shard(root)
    assert report.stats["replacement_chars"] == 2
    assert not any("fffd" in f.message.lower() for f in report.findings)


# ---------------------------------------------------------------------------
# the writer refuses to emit an unsorted shard
# ---------------------------------------------------------------------------


def test_writer_rejects_documents_out_of_order(tmp_path):
    with DatasetWriter(tmp_path / "shard") as w:
        w.write([_page("2" * 64, 0, doc_n_pages=1)], [IMG])
        with pytest.raises(ValueError, match="doc_id 必须递增"):
            w.write([_page("1" * 64, 0, doc_n_pages=1)], [IMG])


def test_writer_rejects_mixing_documents_in_one_call(tmp_path):
    with DatasetWriter(tmp_path / "shard") as w, pytest.raises(ValueError, match="一个文档"):
        w.write([_page("1" * 64, 0, doc_n_pages=1), _page("2" * 64, 0, doc_n_pages=1)])
