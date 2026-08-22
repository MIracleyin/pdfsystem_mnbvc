"""Tests for exporting ``pdfsys.page/v2`` to the MNBVC multimodal block format.

What the export has to guarantee:

1. `legacy` really is what mm_template_mnbvc writes — same columns, same
   types, images base64 in a string column.
2. The two upstream bugs it repairs stay repaired: 块ID increments, 页ID is
   populated.
3. Nothing from the v2 page row is silently dropped; the columns that have no
   home in mmDataBlock land in 扩展字段.
4. `v2` images are loadable as a HuggingFace Image; `legacy` ones are not,
   which is the whole reason the dialect exists.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pdfsys_cli.dataset_writer import DatasetWriter
from pdfsys_cli.mnbvc_export import (
    LEGACY_SCHEMA,
    V2_SCHEMA,
    export_shard,
    page_row_to_block,
    schema_for,
)
from pdfsys_core import Block, BlockType, ImageBlob, PageRecord, render_markdown

# A real 4x3 JPEG — small enough to inline, valid enough that PIL will
# decode it, which is what the HuggingFace round-trip test needs.
JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300281c1e231e19282321232d"
    "2b28303c64413c37373c7b585d4964918099968f808c8aa0b4e6c3a0aadaad8a8cc8ffcb"
    "daeef5ffffff9bc1fffffffaffe6fdfff8ffdb0043012b2d2d3c353c76414176f8a58ca5"
    "f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8"
    "f8f8f8f8f8f8f8f8f8f8f8f8f8f8ffc00011080003000403012200021101031101ffc400"
    "1f0000010501010101010100000000000000000102030405060708090a0bffc400b51000"
    "02010303020403050504040000017d010203000411051221314106135161072271143281"
    "91a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a"
    "434445464748494a535455565758595a636465666768696a737475767778797a83848586"
    "8788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6"
    "c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc400"
    "1f0100030101010101010101010000000000000102030405060708090a0bffc400b51100"
    "020102040403040705040400010277000102031104052131061241510761711322328108"
    "144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a3536373839"
    "3a434445464748494a535455565758595a636465666768696a737475767778797a828384"
    "85868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4"
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda00"
    "0c03010002110311003f00ad45145647a07fffd9"
)
RASTER = ImageBlob(image_id="c" * 64, data=JPEG, format="jpeg", width=4, height=3)


def _page(doc_id: str, page_index: int, *, with_raster: bool) -> PageRecord:
    blocks = (
        Block(idx=page_index * 5, page=page_index, type=BlockType.TEXT, text="正文内容"),
        Block(
            idx=page_index * 5 + 1,
            page=page_index,
            type=BlockType.IMAGE,
            caption="图 1 说明",
            bbox=(0.1, 0.2, 0.5, 0.6),
        ),
    )
    return PageRecord(
        doc_id=doc_id,
        page_index=page_index,
        width_pt=612.0,
        height_pt=792.0,
        text=render_markdown(blocks),
        blocks=blocks,
        extractor="vlm",
        layout_model="pp-doclayoutv3@1.0",
        doc_n_pages=3,
        doc_lang="zho_Hans",
        doc_quality_score=2.5,
        source_uri="s3://bucket/a.pdf",
        page_image_id=RASTER.image_id if with_raster else None,
        render_dpi=200 if with_raster else None,
    )


@pytest.fixture
def shard(tmp_path):
    """One 3-page document, every page carrying a raster."""
    root = tmp_path / "v2"
    pages = [_page("a" * 64, i, with_raster=True) for i in range(3)]
    with DatasetWriter(root) as w:
        w.write(pages, [], [(p, RASTER) for p in pages])
    return root


def _export(shard, tmp_path, dialect="legacy", **kwargs):
    out = tmp_path / f"mnbvc_{dialect}.parquet"
    stats = export_shard(shard, out, dialect=dialect, timestamp="20260822", **kwargs)
    return pq.read_table(out), stats


# ---------------------------------------------------------------------------
# the legacy dialect is what upstream writes
# ---------------------------------------------------------------------------


def test_legacy_columns_are_the_mmdatablock_fields_in_order():
    assert LEGACY_SCHEMA.names == [
        "实体ID", "md5", "块ID", "块类型", "扩展字段", "时间",
        "页ID", "文本", "图片", "视频", "音频", "OCR文本", "STT文本",
    ]
    assert V2_SCHEMA.names == LEGACY_SCHEMA.names


def test_legacy_stores_images_as_base64_text_like_upstream(shard, tmp_path):
    table, _ = _export(shard, tmp_path)
    assert table.schema.field("图片").type == pa.large_string()
    row = table.to_pylist()[0]
    assert base64.b64decode(row["图片"]) == JPEG


def test_legacy_page_id_is_a_string_matching_the_declared_optional_str(shard, tmp_path):
    table, _ = _export(shard, tmp_path)
    assert table.schema.field("页ID").type == pa.string()
    assert [r["页ID"] for r in table.to_pylist()] == ["0", "1", "2"]


def test_legacy_md5_keeps_upstreams_name_hash_semantics(shard, tmp_path):
    row = _export(shard, tmp_path)[0].to_pylist()[0]
    assert row["md5"] == hashlib.md5(row["实体ID"].encode()).hexdigest()


# ---------------------------------------------------------------------------
# the two upstream bugs stay fixed
# ---------------------------------------------------------------------------


def test_block_id_increments_instead_of_staying_zero(shard, tmp_path):
    """Upstream sets block_id = 0 and never advances it, so every block in a
    document shares id 0."""
    table, _ = _export(shard, tmp_path)
    assert [r["块ID"] for r in table.to_pylist()] == [0, 1, 2]


def test_block_id_restarts_per_document(tmp_path):
    root = tmp_path / "v2"
    with DatasetWriter(root) as w:
        for doc in ("a" * 64, "b" * 64):
            pages = [_page(doc, i, with_raster=True) for i in range(2)]
            w.write(pages, [], [(p, RASTER) for p in pages])
    table, _ = _export(root, tmp_path)
    rows = table.to_pylist()
    assert [r["块ID"] for r in rows] == [0, 1, 0, 1]


def test_page_id_column_is_populated_not_only_buried_in_the_json(shard, tmp_path):
    """Upstream puts page_id in 扩展字段 and leaves the 页ID column None."""
    table, _ = _export(shard, tmp_path)
    for row in table.to_pylist():
        assert row["页ID"] is not None
        assert json.loads(row["扩展字段"])["page_id"] == int(row["页ID"])


# ---------------------------------------------------------------------------
# nothing is silently dropped
# ---------------------------------------------------------------------------


def test_v2_columns_with_no_mmdatablock_home_land_in_the_extension_field(shard, tmp_path):
    row = _export(shard, tmp_path)[0].to_pylist()[0]
    extra = json.loads(row["扩展字段"])
    for key in ("doc_id", "page_index", "width_pt", "extractor", "layout_model",
                "doc_quality_score", "source_uri", "doc_n_pages"):
        assert key in extra, key


def test_extension_field_keeps_upstreams_own_keys(shard, tmp_path):
    """A reader written against the reference implementation still finds what
    it expects."""
    extra = json.loads(_export(shard, tmp_path)[0].to_pylist()[0]["扩展字段"])
    assert extra["page_image_size"] == {"width": 4, "height": 3}
    assert extra["page_text_length"] == len(_export(shard, tmp_path)[0].to_pylist()[0]["文本"])


def test_extension_field_is_always_a_string(shard, tmp_path):
    """Upstream passes a dict on the pdf path and a JSON string on the
    image-text-pair path, so the column type drifts between shards."""
    for dialect in ("legacy", "v2"):
        table, _ = _export(shard, tmp_path, dialect=dialect)
        assert table.schema.field("扩展字段").type == pa.string()
        assert all(isinstance(r["扩展字段"], str) for r in table.to_pylist())


def test_text_and_ocr_text_both_carry_the_page_markdown_for_ocr_lanes(shard, tmp_path):
    row = _export(shard, tmp_path)[0].to_pylist()[0]
    assert "正文内容" in row["文本"]
    assert row["OCR文本"] == row["文本"], "extractor=vlm means the text came from OCR"


def test_mupdf_pages_leave_ocr_text_null(tmp_path):
    import dataclasses

    root = tmp_path / "v2"
    page = dataclasses.replace(_page("a" * 64, 0, with_raster=True), extractor="mupdf")
    with DatasetWriter(root) as w:
        w.write([page], [], [(page, RASTER)])
    row = _export(root, tmp_path)[0].to_pylist()[0]
    assert row["文本"]
    assert row["OCR文本"] is None, "mupdf reads an embedded text layer, not OCR"


# ---------------------------------------------------------------------------
# the v2 dialect
# ---------------------------------------------------------------------------


def test_v2_image_column_is_the_huggingface_wire_struct(shard, tmp_path):
    table, _ = _export(shard, tmp_path, dialect="v2")
    assert pa.types.is_struct(table.schema.field("图片").type)
    row = table.to_pylist()[0]
    assert row["图片"]["bytes"] == JPEG
    assert row["图片"]["path"].endswith(".jpeg")


def test_v2_page_id_is_an_integer(shard, tmp_path):
    table, _ = _export(shard, tmp_path, dialect="v2")
    assert table.schema.field("页ID").type == pa.int32()
    assert [r["页ID"] for r in table.to_pylist()] == [0, 1, 2]


def test_v2_md5_is_over_content_not_over_a_name(shard, tmp_path):
    """Upstream hashes the image filename, which can neither dedupe nor
    verify."""
    row = _export(shard, tmp_path, dialect="v2")[0].to_pylist()[0]
    expected = hashlib.md5()
    expected.update(JPEG)
    expected.update(row["文本"].encode("utf-8"))
    assert row["md5"] == expected.hexdigest()
    assert row["md5"] != hashlib.md5(row["实体ID"].encode()).hexdigest()


def test_both_dialects_declare_their_schema_so_shards_concatenate(shard, tmp_path):
    """Upstream infers the schema per batch via from_pandas: a batch where
    every 视频 is None types the column null, the next types it binary, and
    the two shards will not concatenate."""
    for dialect in ("legacy", "v2"):
        table, _ = _export(shard, tmp_path, dialect=dialect)
        declared = schema_for(dialect)
        assert table.schema.names == declared.names
        for name in declared.names:
            written, want = table.schema.field(name).type, declared.field(name).type
            if pa.types.is_dictionary(want):
                # Parquet normalises dictionary index width on the way out.
                assert pa.types.is_dictionary(written), name
            else:
                assert written == want, name
        # 视频 / 音频 are all-null here and must still keep their declared type,
        # which is exactly what per-batch inference gets wrong.
        assert not pa.types.is_null(table.schema.field("视频").type)
        assert not pa.types.is_null(table.schema.field("音频").type)


def test_v2_images_decode_through_huggingface_and_legacy_ones_do_not(shard, tmp_path):
    """The practical reason the v2 dialect exists. Not a storage argument:
    measured on a real shard the base64 column actually compressed 0.7 %
    *smaller* than binary, because zstd takes back nearly all of base64's
    one-third inflation."""
    datasets = pytest.importorskip("datasets")

    out = tmp_path / "v2.parquet"
    export_shard(shard, out, dialect="v2", timestamp="20260822")
    ds = datasets.load_dataset("parquet", data_files=str(out), split="train")
    ds = ds.cast_column("图片", datasets.Image())
    assert ds[0]["图片"].size == (4, 3)

    legacy = tmp_path / "legacy.parquet"
    export_shard(shard, legacy, dialect="legacy", timestamp="20260822")
    ds2 = datasets.load_dataset("parquet", data_files=str(legacy), split="train")
    with pytest.raises(Exception, match="(?i)cast|struct"):
        ds2.cast_column("图片", datasets.Image())[0]


# ---------------------------------------------------------------------------
# missing rasters
# ---------------------------------------------------------------------------


def test_pages_without_a_raster_still_export_and_are_counted(tmp_path):
    root = tmp_path / "v2"
    with DatasetWriter(root) as w:
        w.write([_page("a" * 64, 0, with_raster=False)])
    table, stats = _export(root, tmp_path)
    assert stats["pages_without_image"] == 1
    row = table.to_pylist()[0]
    assert row["图片"] is None
    assert row["文本"], "the text is still worth exporting"


def test_export_rejects_a_directory_that_is_not_a_shard(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_shard(tmp_path, tmp_path / "x.parquet", timestamp="20260822")


def test_unknown_dialect_is_rejected():
    with pytest.raises(ValueError, match="dialect"):
        schema_for("nope")


# ---------------------------------------------------------------------------
# row mapping in isolation
# ---------------------------------------------------------------------------


def test_entity_id_is_content_addressed_rather_than_a_filename():
    row = page_row_to_block(
        {"doc_id": "a" * 64, "page_index": 7, "text": "x", "extractor": "vlm"},
        None,
        dialect="v2",
        block_id=7,
        timestamp="20260822",
    )
    assert row["实体ID"] == f"{'a' * 64}-page-7"
