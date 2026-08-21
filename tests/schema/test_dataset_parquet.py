"""Tests for the Parquet encoding of ``pdfsys.doc/v1``.

The format's promises that only the writer can keep:

* documents and images are separate files, joinable on ``image_id``
* an image blob is stored once per shard no matter how often it is referenced
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
    DOC_SCHEMA,
    IMAGE_SCHEMA,
    IMAGE_TYPE,
    PAIR_SCHEMA,
    DatasetWriter,
    pairs_table,
)
from pdfsys_core import (
    DATASET_SCHEMA_VERSION,
    Block,
    BlockType,
    DocRecord,
    ImageBlob,
    render_markdown,
)

IMG_A = ImageBlob(image_id="a" * 64, data=b"\xff\xd8fake-a", format="jpeg", width=4, height=8)
IMG_B = ImageBlob(image_id="b" * 64, data=b"\x89PNGfake-b", format="png", width=2, height=3)


def _doc(doc_id: str, *image_ids: str) -> DocRecord:
    blocks = [Block(idx=0, page=0, type=BlockType.TEXT, text="正文内容")]
    for n, iid in enumerate(image_ids, start=1):
        blocks.append(
            Block(
                idx=n,
                page=0,
                type=BlockType.IMAGE,
                caption=f"图 {n} 说明文字",
                image_id=iid,
                bbox=(0.1, 0.1, 0.5, 0.5),
            )
        )
    text, page_ends = render_markdown(blocks)
    return DocRecord(
        id=doc_id,
        blocks=tuple(blocks),
        text=text,
        page_ends=page_ends,
        backend="vlm",
        n_pages=1,
        lang="zho_Hans",
        quality_score=2.5,
    )


@pytest.fixture
def shard(tmp_path):
    """Two documents that share image A; only doc 2 uses image B."""
    with DatasetWriter(tmp_path, shard="shard-00007") as w:
        w.write(_doc("1" * 64, IMG_A.image_id), [IMG_A])
        w.write(_doc("2" * 64, IMG_A.image_id, IMG_B.image_id), [IMG_A, IMG_B])
        stats = (w.docs_written, w.images_written)
    return tmp_path, stats


# ---------------------------------------------------------------------------
# layout + schema
# ---------------------------------------------------------------------------


def test_writes_documents_and_images_as_separate_files(shard):
    root, _ = shard
    assert (root / "documents" / "shard-00007.parquet").is_file()
    assert (root / "images" / "shard-00007.parquet").is_file()


def test_document_schema_matches_the_declared_constant(shard):
    root, _ = shard
    table = pq.read_table(root / "documents")
    assert table.schema.names == DOC_SCHEMA.names


def test_schema_version_travels_in_file_metadata(shard):
    root, _ = shard
    for sub in ("documents", "images"):
        path = next((root / sub).glob("*.parquet"))
        meta = pq.ParquetFile(str(path)).schema_arrow.metadata
        assert meta[b"pdfsys.schema"] == DATASET_SCHEMA_VERSION.encode()


def test_image_column_is_the_huggingface_image_struct():
    assert IMAGE_SCHEMA.field("image").type == IMAGE_TYPE
    assert IMAGE_TYPE.field("bytes").type == pa.large_binary()
    assert IMAGE_TYPE.field("path").type == pa.string()


# ---------------------------------------------------------------------------
# the checked-in JSON Schema must describe what we actually write
# ---------------------------------------------------------------------------

_JSON_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "schema" / "doc_dataset.v1.json"
)


@pytest.fixture(scope="module")
def json_schema():
    return json.loads(_JSON_SCHEMA_PATH.read_text(encoding="utf-8"))


def _props(schema, name):
    return list(schema["$defs"][name]["properties"])


def test_json_schema_document_fields_match_the_arrow_schema(json_schema):
    documented = [f for f in _props(json_schema, "Document") if f != "images"]
    assert documented == DOC_SCHEMA.names


def test_json_schema_block_fields_match_the_arrow_struct(json_schema):
    assert _props(json_schema, "Block") == [f.name for f in BLOCK_TYPE]


def test_json_schema_image_and_pair_fields_match(json_schema):
    assert _props(json_schema, "Image") == IMAGE_SCHEMA.names
    assert _props(json_schema, "Pair") == PAIR_SCHEMA.names


def test_json_schema_block_type_enum_matches_the_python_enum(json_schema):
    assert json_schema["$defs"]["BlockType"]["enum"] == [str(t) for t in BlockType]


def test_json_schema_title_is_the_version_constant(json_schema):
    assert json_schema["title"] == DATASET_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# dedup + join
# ---------------------------------------------------------------------------


def test_shared_image_is_stored_once_per_shard(shard):
    root, (n_docs, n_images) = shard
    assert (n_docs, n_images) == (2, 2), "image A referenced twice, stored once"
    images = pq.read_table(root / "images").to_pylist()
    assert sorted(r["image_id"] for r in images) == [IMG_A.image_id, IMG_B.image_id]


def test_every_referenced_image_id_resolves_in_the_images_table(shard):
    root, _ = shard
    docs = pq.read_table(root / "documents").to_pylist()
    known = {r["image_id"] for r in pq.read_table(root / "images").to_pylist()}
    referenced = {i for d in docs for i in d["image_ids"]}
    assert referenced and referenced <= known


def test_image_bytes_round_trip_intact(shard):
    root, _ = shard
    by_id = {r["image_id"]: r for r in pq.read_table(root / "images").to_pylist()}
    assert by_id[IMG_A.image_id]["image"]["bytes"] == IMG_A.data
    assert by_id[IMG_A.image_id]["image"]["path"] == f"{IMG_A.image_id}.jpeg"
    assert by_id[IMG_B.image_id]["n_bytes"] == len(IMG_B.data)


# ---------------------------------------------------------------------------
# row content
# ---------------------------------------------------------------------------


def test_derived_count_columns_agree_with_blocks(shard):
    root, _ = shard
    for row in pq.read_table(root / "documents").to_pylist():
        assert row["n_blocks"] == len(row["blocks"])
        assert row["n_chars"] == len(row["text"])
        assert row["n_images"] == sum(
            1 for b in row["blocks"] if b["type"] in ("image", "chart")
        )


def test_blocks_survive_the_round_trip_with_bbox_and_caption(shard):
    root, _ = shard
    row = pq.read_table(root / "documents").to_pylist()[0]
    image_block = next(b for b in row["blocks"] if b["type"] == "image")
    assert image_block["caption"] == "图 1 说明文字"
    assert image_block["bbox"] == pytest.approx(
        {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}
    )
    assert image_block["image_id"] == IMG_A.image_id


def test_pipeline_signal_columns_are_carried_through(shard):
    root, _ = shard
    row = pq.read_table(root / "documents").to_pylist()[0]
    assert row["backend"] == "vlm"
    assert row["lang"] == "zho_Hans"
    assert row["quality_score"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_row_group_buffering_does_not_drop_rows(tmp_path):
    with DatasetWriter(tmp_path, row_group_size=2) as w:
        for i in range(5):
            w.write(_doc(str(i) * 64))
    assert pq.read_table(tmp_path / "documents").num_rows == 5


def test_no_images_file_when_the_shard_has_no_images(tmp_path):
    with DatasetWriter(tmp_path) as w:
        w.write(_doc("1" * 64))
    assert not (tmp_path / "images").exists()


def test_include_text_false_nulls_the_column_without_changing_the_schema(tmp_path):
    with DatasetWriter(tmp_path, include_text=False) as w:
        w.write(_doc("1" * 64))
    table = pq.read_table(tmp_path / "documents")
    row = table.to_pylist()[0]
    assert table.schema.names == DOC_SCHEMA.names
    assert row["text"] is None
    # The content is still there — it just has to come from blocks.
    assert row["n_chars"] > 0
    assert row["blocks"][0]["text"] == "正文内容"


def test_embed_images_inlines_blobs_into_the_documents_table(tmp_path):
    with DatasetWriter(tmp_path, embed_images=True) as w:
        w.write(_doc("1" * 64, IMG_A.image_id), [IMG_A])
    table = pq.read_table(tmp_path / "documents")
    assert "images" in table.schema.names
    assert table.to_pylist()[0]["images"][0]["bytes"] == IMG_A.data
    # Self-contained by construction — no side table to join.
    assert not (tmp_path / "images").exists()


# ---------------------------------------------------------------------------
# pair view
# ---------------------------------------------------------------------------


def test_pairs_table_shape_and_content():
    table = pairs_table([_doc("1" * 64, IMG_A.image_id, IMG_B.image_id)])
    rows = table.to_pylist()
    assert table.schema.names == [
        "doc_id",
        "image_id",
        "block_idx",
        "page",
        "text",
        "source",
    ]
    assert {r["image_id"] for r in rows} == {IMG_A.image_id, IMG_B.image_id}
    assert all(r["source"] == "caption" for r in rows)


def test_pairs_table_is_empty_not_broken_when_there_are_no_images():
    table = pairs_table([_doc("1" * 64)])
    assert table.num_rows == 0
    assert table.schema.names[0] == "doc_id"
