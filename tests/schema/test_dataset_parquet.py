"""Tests for the Parquet encoding of ``pdfsys.page/v2``.

The format's promises that only the writer can keep:

* pages, image crops and page rasters are separate files, joinable by id
* an image blob is stored once per shard no matter how often it is referenced
* rows come out sorted by ``(doc_id, page_index)``
* dropping ``blocks`` keeps the interleaving, because it lives in ``text``
* the ``image`` column is the HF ``datasets.Image`` wire struct
* the schema version travels in the file's key-value metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pdfsys_cli.dataset_writer import (
    BLOCK_TYPE,
    IMAGE_SCHEMA,
    IMAGE_TYPE,
    PAGE_IMAGE_SCHEMA,
    PAGE_SCHEMA,
    PAIR_SCHEMA,
    DatasetWriter,
    pairs_table,
)
from pdfsys_core import (
    DATASET_SCHEMA_VERSION,
    Block,
    BlockType,
    ImageBlob,
    PageRecord,
    render_markdown,
    to_interleaved,
)

IMG_A = ImageBlob(image_id="a" * 64, data=b"\xff\xd8fake-a", format="jpeg", width=4, height=8)
IMG_B = ImageBlob(image_id="b" * 64, data=b"\x89PNGfake-b", format="png", width=2, height=3)
PAGE_RASTER = ImageBlob(
    image_id="c" * 64, data=b"\xff\xd8fake-page", format="jpeg", width=1275, height=1650
)


def _page(doc_id: str, page_index: int, *image_ids: str) -> PageRecord:
    blocks = [Block(idx=page_index * 10, page=page_index, type=BlockType.TEXT, text="正文内容")]
    for n, iid in enumerate(image_ids, start=1):
        blocks.append(
            Block(
                idx=page_index * 10 + n,
                page=page_index,
                type=BlockType.IMAGE,
                caption=f"图 {n} 说明文字",
                image_id=iid,
                bbox=(0.1, 0.1, 0.5, 0.5),
            )
        )
    return PageRecord(
        doc_id=doc_id,
        page_index=page_index,
        width_pt=595.0,
        height_pt=842.0,
        text=render_markdown(blocks),
        image_ids=tuple(image_ids),
        blocks=tuple(blocks),
        extractor="vlm",
        layout_model="pp-doclayoutv3@1.0",
        doc_n_pages=2,
        doc_lang="zho_Hans",
        doc_quality_score=2.5,
    )


@pytest.fixture
def shard(tmp_path):
    """Two documents. Doc 1 has two pages sharing image A; doc 2 uses A and B."""
    with DatasetWriter(tmp_path, shard="shard-00007") as w:
        w.write(
            [_page("1" * 64, 1, IMG_A.image_id), _page("1" * 64, 0, IMG_A.image_id)],
            [IMG_A],
        )
        w.write([_page("2" * 64, 0, IMG_A.image_id, IMG_B.image_id)], [IMG_A, IMG_B])
        stats = (w.docs_written, w.pages_written, w.images_written)
    return tmp_path, stats


# ---------------------------------------------------------------------------
# layout + schema
# ---------------------------------------------------------------------------


def test_writes_pages_and_images_as_separate_files(shard):
    root, _ = shard
    assert (root / "pages" / "shard-00007.parquet").is_file()
    assert (root / "images" / "shard-00007.parquet").is_file()


def test_page_schema_matches_the_declared_constant(shard):
    root, _ = shard
    assert pq.read_table(root / "pages").schema.names == PAGE_SCHEMA.names


def test_schema_version_travels_in_file_metadata(shard):
    root, _ = shard
    for sub in ("pages", "images"):
        path = next((root / sub).glob("*.parquet"))
        meta = pq.ParquetFile(str(path)).schema_arrow.metadata
        assert meta[b"pdfsys.schema"] == DATASET_SCHEMA_VERSION.encode()


def test_image_column_is_the_huggingface_image_struct():
    assert IMAGE_SCHEMA.field("image").type == IMAGE_TYPE
    assert PAGE_IMAGE_SCHEMA.field("image").type == IMAGE_TYPE
    assert IMAGE_TYPE.field("bytes").type == pa.large_binary()
    assert IMAGE_TYPE.field("path").type == pa.string()


def test_block_struct_omits_page_because_the_row_already_is_one():
    assert "page" not in [f.name for f in BLOCK_TYPE]
    assert "page_index" in PAGE_SCHEMA.names


# ---------------------------------------------------------------------------
# row unit + ordering
# ---------------------------------------------------------------------------


def test_row_count_is_pages_not_documents(shard):
    _, (n_docs, n_pages, _) = shard
    assert (n_docs, n_pages) == (2, 3)


def test_rows_come_out_sorted_by_doc_then_page(shard):
    root, _ = shard
    rows = pq.read_table(root / "pages").to_pylist()
    keys = [(r["doc_id"], r["page_index"]) for r in rows]
    assert keys == sorted(keys), "written out of order — reassembly must stay a scan"


def test_page_geometry_survives(shard):
    root, _ = shard
    row = pq.read_table(root / "pages").to_pylist()[0]
    assert (row["width_pt"], row["height_pt"]) == pytest.approx((595.0, 842.0))


def test_document_level_columns_are_denormalized_onto_every_page(shard):
    root, _ = shard
    doc1 = [r for r in pq.read_table(root / "pages").to_pylist() if r["doc_id"] == "1" * 64]
    assert len(doc1) == 2
    assert all(r["doc_lang"] == "zho_Hans" for r in doc1)
    assert all(r["doc_quality_score"] == pytest.approx(2.5) for r in doc1)
    assert all(r["doc_n_pages"] == 2 for r in doc1)


def test_page_level_quality_stays_null_until_a_page_scorer_fills_it(shard):
    root, _ = shard
    rows = pq.read_table(root / "pages").to_pylist()
    assert all(r["quality_score"] is None for r in rows)
    assert all(r["doc_quality_score"] is not None for r in rows)


# ---------------------------------------------------------------------------
# dedup + join
# ---------------------------------------------------------------------------


def test_shared_image_is_stored_once_per_shard(shard):
    root, (_, _, n_images) = shard
    assert n_images == 2, "image A referenced by three pages, stored once"
    images = pq.read_table(root / "images").to_pylist()
    assert sorted(r["image_id"] for r in images) == [IMG_A.image_id, IMG_B.image_id]


def test_every_referenced_image_id_resolves_in_the_images_table(shard):
    root, _ = shard
    pages = pq.read_table(root / "pages").to_pylist()
    known = {r["image_id"] for r in pq.read_table(root / "images").to_pylist()}
    referenced = {i for p in pages for i in p["image_ids"]}
    assert referenced and referenced <= known


def test_image_bytes_round_trip_intact(shard):
    root, _ = shard
    by_id = {r["image_id"]: r for r in pq.read_table(root / "images").to_pylist()}
    assert by_id[IMG_A.image_id]["image"]["bytes"] == IMG_A.data
    assert by_id[IMG_A.image_id]["image"]["path"] == f"{IMG_A.image_id}.jpeg"
    assert by_id[IMG_B.image_id]["n_bytes"] == len(IMG_B.data)


# ---------------------------------------------------------------------------
# page rasters
# ---------------------------------------------------------------------------


def test_no_page_images_file_unless_rasters_are_supplied(shard):
    root, _ = shard
    assert not (root / "page_images").exists()


def test_page_rasters_land_in_their_own_table(tmp_path):
    page = _page("1" * 64, 0)
    import dataclasses

    stamped = dataclasses.replace(
        page, page_image_id=PAGE_RASTER.image_id, render_dpi=150
    )
    with DatasetWriter(tmp_path) as w:
        w.write([stamped], [], [(stamped, PAGE_RASTER)])
        assert w.page_images_written == 1

    raster = pq.read_table(tmp_path / "page_images").to_pylist()[0]
    assert raster["image_id"] == PAGE_RASTER.image_id
    assert (raster["doc_id"], raster["page_index"], raster["render_dpi"]) == (
        "1" * 64,
        0,
        150,
    )
    row = pq.read_table(tmp_path / "pages").to_pylist()[0]
    assert row["page_image_id"] == PAGE_RASTER.image_id
    assert row["render_dpi"] == 150


def test_page_raster_ids_are_not_mixed_into_the_crops_table(tmp_path):
    page = _page("1" * 64, 0, IMG_A.image_id)
    with DatasetWriter(tmp_path) as w:
        w.write([page], [IMG_A], [(page, PAGE_RASTER)])
    crops = {r["image_id"] for r in pq.read_table(tmp_path / "images").to_pylist()}
    assert crops == {IMG_A.image_id}


def test_a_page_addressed_only_by_bbox_needs_no_crops_table(tmp_path):
    """images="pages": the figure is a rectangle of the page raster, so no
    crop blob is stored and `text` carries a region reference instead."""
    import dataclasses

    from pdfsys_core import IMAGE_REF_RE, parse_image_ref, render_markdown

    blocks = (
        Block(idx=0, page=0, type=BlockType.TEXT, text="正文内容"),
        Block(idx=1, page=0, type=BlockType.IMAGE, caption="图 1", bbox=(0.1, 0.1, 0.5, 0.5)),
    )
    page = dataclasses.replace(
        _page("1" * 64, 0),
        blocks=blocks,
        text=render_markdown(blocks),
        image_ids=(),
        page_image_id=PAGE_RASTER.image_id,
        render_dpi=200,
    )
    with DatasetWriter(tmp_path) as w:
        w.write([page], [], [(page, PAGE_RASTER)])

    assert not (tmp_path / "images").exists(), "no duplicated pixels"
    row = pq.read_table(tmp_path / "pages").to_pylist()[0]
    assert row["image_ids"] == []
    assert row["page_image_id"] == PAGE_RASTER.image_id
    (ref,) = [parse_image_ref(r) for r in IMAGE_REF_RE.findall(row["text"])]
    assert ref.kind == "region"
    assert ref.bbox == pytest.approx((0.1, 0.1, 0.5, 0.5))
    # Everything needed to cut the figure out is on this row alone.
    assert row["blocks"][1]["bbox"] is not None


# ---------------------------------------------------------------------------
# blocks are droppable, the interleaving is not
# ---------------------------------------------------------------------------


def test_no_blocks_keeps_text_and_the_interleaving(tmp_path):
    with DatasetWriter(tmp_path, include_blocks=False) as w:
        w.write([_page("1" * 64, 0, IMG_A.image_id)], [IMG_A])

    table = pq.read_table(tmp_path / "pages")
    row = table.to_pylist()[0]
    assert table.schema.names == PAGE_SCHEMA.names, "schema is stable either way"
    assert row["blocks"] is None
    # The whole point: the view still works.
    images, _texts = to_interleaved(row["text"])
    assert f"img://{IMG_A.image_id}" in images
    assert row["image_ids"] == [IMG_A.image_id]
    assert row["n_chars"] > 0


def test_counts_are_populated_even_when_blocks_are_dropped(tmp_path):
    with DatasetWriter(tmp_path, include_blocks=False) as w:
        w.write([_page("1" * 64, 0, IMG_A.image_id)], [IMG_A])
    row = pq.read_table(tmp_path / "pages").to_pylist()[0]
    assert (row["n_blocks"], row["n_images"]) == (2, 1)


# ---------------------------------------------------------------------------
# row content
# ---------------------------------------------------------------------------


def test_derived_count_columns_agree_with_blocks(shard):
    root, _ = shard
    for row in pq.read_table(root / "pages").to_pylist():
        assert row["n_blocks"] == len(row["blocks"])
        assert row["n_chars"] == len(row["text"])
        assert row["n_images"] == sum(
            1 for b in row["blocks"] if b["type"] in ("image", "chart")
        )


def test_blocks_survive_the_round_trip_with_bbox_and_caption(shard):
    root, _ = shard
    row = pq.read_table(root / "pages").to_pylist()[0]
    image_block = next(b for b in row["blocks"] if b["type"] == "image")
    assert image_block["caption"] == "图 1 说明文字"
    assert image_block["bbox"] == pytest.approx(
        {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}
    )
    assert image_block["image_id"] == IMG_A.image_id


def test_provenance_columns_are_carried_through(shard):
    root, _ = shard
    row = pq.read_table(root / "pages").to_pylist()[0]
    assert row["extractor"] == "vlm"
    assert row["layout_model"] == "pp-doclayoutv3@1.0"


def test_row_group_buffering_does_not_drop_rows(tmp_path):
    with DatasetWriter(tmp_path, row_group_size=2) as w:
        for i in range(5):
            w.write([_page(str(i) * 64, 0)])
    assert pq.read_table(tmp_path / "pages").num_rows == 5


def test_no_images_file_when_the_shard_has_no_images(tmp_path):
    with DatasetWriter(tmp_path) as w:
        w.write([_page("1" * 64, 0)])
    assert not (tmp_path / "images").exists()


# ---------------------------------------------------------------------------
# pair view
# ---------------------------------------------------------------------------


def test_pairs_table_shape_and_content():
    table = pairs_table([[_page("1" * 64, 0, IMG_A.image_id, IMG_B.image_id)]])
    rows = table.to_pylist()
    assert table.schema.names == PAIR_SCHEMA.names
    assert {r["image_id"] for r in rows} == {IMG_A.image_id, IMG_B.image_id}
    assert all(r["source"] == "caption" for r in rows)
    assert all(r["page_index"] == 0 for r in rows)


def test_pairs_table_is_empty_not_broken_when_there_are_no_images():
    table = pairs_table([[_page("1" * 64, 0)]])
    assert table.num_rows == 0
    assert table.schema.names[0] == "doc_id"


# ---------------------------------------------------------------------------
# the checked-in JSON Schema must describe what we actually write
# ---------------------------------------------------------------------------

_JSON_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "schema" / "doc_dataset.v2.json"
)


@pytest.fixture(scope="module")
def json_schema():
    return json.loads(_JSON_SCHEMA_PATH.read_text(encoding="utf-8"))


def _props(schema, name):
    return list(schema["$defs"][name]["properties"])


def test_json_schema_page_fields_match_the_arrow_schema(json_schema):
    assert _props(json_schema, "Page") == PAGE_SCHEMA.names


def test_json_schema_block_fields_match_the_arrow_struct(json_schema):
    assert _props(json_schema, "Block") == [f.name for f in BLOCK_TYPE]


def test_json_schema_image_and_pair_fields_match(json_schema):
    assert _props(json_schema, "Image") == IMAGE_SCHEMA.names
    assert _props(json_schema, "PageImage") == PAGE_IMAGE_SCHEMA.names
    assert _props(json_schema, "Pair") == PAIR_SCHEMA.names


def test_json_schema_block_type_enum_matches_the_python_enum(json_schema):
    assert json_schema["$defs"]["BlockType"]["enum"] == [str(t) for t in BlockType]


def test_json_schema_title_is_the_version_constant(json_schema):
    assert json_schema["title"] == DATASET_SCHEMA_VERSION
