"""Export a ``pdfsys.page/v2`` shard to the MNBVC multimodal block format.

The target is ``mmDataBlock`` from `mm_template_mnbvc
<https://github.com/MIracleyin/mm_template_mnbvc>`_ — one row per block, all
field names in Chinese, binary payloads inline.

The two formats agree on the thing that matters. That repo's
``chinaxiv_to_image_text_pair_blocks`` emits one ``块类型="image-text-pair"``
row per page, carrying the page PNG and that page's Markdown. That is exactly
a ``pdfsys.page/v2`` page row, so the mapping is a rename rather than a
restructure — and it is why the export needs a shard built with
``--images pages``: the historical format's unit of image *is* the page
raster.

Two dialects
------------
``legacy``
    Same column names and types the existing repo writes, so anything reading
    those Parquet files today reads ours. Two things are populated that the
    reference implementation leaves empty, because both are bugs rather than
    conventions and filling them cannot break a reader: ``块ID`` actually
    increments (upstream initialises ``block_id = 0`` and never advances it,
    so every block in a document shares id 0), and ``页ID`` is filled from the
    page index (upstream buries ``page_id`` in the ``扩展字段`` JSON while the
    dedicated column stays ``None``).

``v2``
    Same row semantics, four repairs that do change the wire format — see
    ``docs/schema/mnbvc-mm-compat.md``. In short: images as binary rather than
    base64 text, ``md5`` over content rather than over a filename, ``页ID``
    typed as an integer, and a declared Arrow schema instead of one inferred
    per batch.

    The binary change is *not* a disk saving, contrary to the obvious guess:
    base64 inflates by a third but zstd takes almost all of it back, and
    measured on a real shard the base64 column compressed 0.7 % *smaller*.
    What it buys is that ``cast_column("图片", Image())`` works at all —
    on the legacy column it raises ``ArrowNotImplementedError`` — plus 33 %
    less uncompressed memory and ~0.8 ms/page of base64 decoding, which is
    close to an hour of single-core time per pass over a 4-million-page
    corpus.

Both dialects declare their schema explicitly. The reference implementation
builds each shard with ``pa.Table.from_pandas``, which infers column types
from whatever that batch happens to contain — a batch where every ``视频`` is
None types the column as null, the next one types it as binary, and the two
shards will not concatenate.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "DIALECTS",
    "LEGACY_SCHEMA",
    "V2_SCHEMA",
    "schema_for",
    "page_row_to_block",
    "export_shard",
]

DIALECTS = ("legacy", "v2")

#: Field order matches ``mmDataBlock``'s declaration order, which is what the
#: reference implementation's ``to_dict()`` walks.
_FIELDS = (
    "实体ID",
    "md5",
    "块ID",
    "块类型",
    "扩展字段",
    "时间",
    "页ID",
    "文本",
    "图片",
    "视频",
    "音频",
    "OCR文本",
    "STT文本",
)

#: Byte-compatible with what the reference implementation writes today:
#: binaries arrive base64-encoded because ``to_dict()`` encodes them before
#: pandas ever sees them, so the Parquet column is a string.
LEGACY_SCHEMA = pa.schema(
    [
        ("实体ID", pa.string()),
        ("md5", pa.string()),
        ("块ID", pa.int64()),
        ("块类型", pa.string()),
        ("扩展字段", pa.string()),
        ("时间", pa.string()),
        ("页ID", pa.string()),
        ("文本", pa.large_string()),
        ("图片", pa.large_string()),  # base64
        ("视频", pa.large_string()),  # base64
        ("音频", pa.large_string()),  # base64
        ("OCR文本", pa.large_string()),
        ("STT文本", pa.large_string()),
    ]
)

#: HuggingFace ``datasets.Image`` wire struct — the reason ``v2`` exists.
_IMAGE_TYPE = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])

#: Must stay byte-identical to ``mmdata_block.BLOCK_SCHEMA`` in
#: mm_template_mnbvc — the whole point of the v2 dialect is that the code, the
#: published example dataset and this exporter finally agree on one schema.
#: Three columns were out of line here and have been brought over:
#: ``块类型`` is a plain string (not dictionary-encoded), and ``视频`` / ``音频``
#: use the same ``struct<bytes, path>`` as ``图片`` — HuggingFace's Audio and
#: Video features use that struct too, so there is no reason for the media
#: columns to disagree with each other.
V2_SCHEMA = pa.schema(
    [
        ("实体ID", pa.string()),
        ("md5", pa.string()),
        ("块ID", pa.int32()),
        ("块类型", pa.string()),
        ("扩展字段", pa.string()),
        ("时间", pa.string()),
        ("页ID", pa.int32()),
        ("文本", pa.large_string()),
        ("图片", _IMAGE_TYPE),
        ("视频", _IMAGE_TYPE),
        ("音频", _IMAGE_TYPE),
        ("OCR文本", pa.large_string()),
        ("STT文本", pa.large_string()),
    ]
)


def schema_for(dialect: str) -> pa.Schema:
    if dialect == "legacy":
        return LEGACY_SCHEMA
    if dialect == "v2":
        return V2_SCHEMA
    raise ValueError(f"dialect must be one of {DIALECTS}, got {dialect!r}")


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

#: pdfsys.page/v2 columns that have no home in mmDataBlock. They go into
#: 扩展字段 rather than being dropped — that column is the historical format's
#: only extension point, and losing provenance on export would be worse than
#: a JSON blob.
_EXTRA_COLUMNS = (
    "doc_id",
    "page_index",
    "doc_n_pages",
    "width_pt",
    "height_pt",
    "rotation",
    "extractor",
    "layout_model",
    "n_chars",
    "n_blocks",
    "n_images",
    "n_tables",
    "n_formulas",
    "lang",
    "lang_score",
    "quality_score",
    "doc_lang",
    "doc_quality_score",
    "router_ocr_prob",
    "source_uri",
    "image_ids",
)


def page_row_to_block(
    page: dict[str, Any],
    raster: dict[str, Any] | None,
    *,
    dialect: str,
    block_id: int,
    timestamp: str,
    block_type: str = "image-text-pair",
) -> dict[str, Any]:
    """Map one ``pages`` row (+ its raster) onto one ``mmDataBlock`` row."""
    image_bytes = (raster or {}).get("image", {}).get("bytes")

    # 实体ID: upstream uses the page image's filename. We use the document's
    # sha256 plus the page number — same shape, but stable across re-ingests
    # and not dependent on a file ever having existed on disk.
    entity_id = f"{page['doc_id']}-page-{page['page_index']}"

    extra = {k: page.get(k) for k in _EXTRA_COLUMNS if page.get(k) is not None}
    # Keep upstream's own keys so a reader written against the reference
    # implementation still finds what it expects.
    extra["page_id"] = page["page_index"]
    if raster:
        extra["page_image_size"] = {
            "width": raster.get("width"),
            "height": raster.get("height"),
        }
        extra["render_dpi"] = raster.get("render_dpi")
    extra["page_text_length"] = len(page.get("text") or "")

    row: dict[str, Any] = {f: None for f in _FIELDS}
    row.update(
        {
            "实体ID": entity_id,
            "块ID": block_id,
            "块类型": block_type,
            "扩展字段": json.dumps(extra, ensure_ascii=False),
            "时间": timestamp,
            "文本": page.get("text"),
            # The pipeline knows whether the text came from OCR; the mupdf lane
            # reads an embedded text layer, the others do not.
            "OCR文本": (
                page.get("text") if page.get("extractor") in ("pipeline", "vlm") else None
            ),
        }
    )

    if dialect == "legacy":
        row["页ID"] = str(page["page_index"])
        row["md5"] = hashlib.md5(entity_id.encode()).hexdigest()
        row["图片"] = (
            base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
        )
    else:
        row["页ID"] = page["page_index"]
        # Content hash, not a hash of a name: this one can dedupe and verify.
        digest = hashlib.md5()
        digest.update(image_bytes or b"")
        digest.update((page.get("text") or "").encode("utf-8"))
        row["md5"] = digest.hexdigest()
        row["图片"] = (
            {"bytes": image_bytes, "path": f"{entity_id}.jpeg"} if image_bytes else None
        )
    return row


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_shard(
    shard_dir: Path,
    out_path: Path,
    *,
    dialect: str = "legacy",
    timestamp: str,
    block_type: str = "image-text-pair",
    compression: str = "zstd",
    row_group_size: int = 1024,
) -> dict[str, int]:
    """Convert a ``pdfsys.page/v2`` shard into one MNBVC block Parquet file.

    Returns counts, including ``pages_without_image`` — a page whose raster is
    missing still exports, because dropping it would lose the text, but the
    count is reported so a run that silently produced text-only
    "image-text-pair" blocks is visible rather than assumed fine.
    """
    schema = schema_for(dialect)
    shard_dir = Path(shard_dir)
    pages_dir = shard_dir / "pages"
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"no pages/ under {shard_dir}")

    rasters = _index_rasters(shard_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"blocks": 0, "pages_without_image": 0}
    writer = pq.ParquetWriter(str(out_path), schema, compression=compression)
    try:
        buffer: list[dict[str, Any]] = []
        block_id = 0
        current_doc: str | None = None
        for page in _iter_pages(pages_dir):
            # 块ID is scoped to the document and actually increments; upstream
            # leaves it at 0 for every block.
            if page["doc_id"] != current_doc:
                current_doc, block_id = page["doc_id"], 0
            raster = rasters.get((page["doc_id"], page["page_index"]))
            if raster is None:
                stats["pages_without_image"] += 1
            buffer.append(
                page_row_to_block(
                    page,
                    raster,
                    dialect=dialect,
                    block_id=block_id,
                    timestamp=timestamp,
                    block_type=block_type,
                )
            )
            block_id += 1
            stats["blocks"] += 1
            if len(buffer) >= row_group_size:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    finally:
        writer.close()
    return stats


def _iter_pages(pages_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(pages_dir.glob("*.parquet")):
        table = pq.read_table(path)
        yield from table.to_pylist()


def _index_rasters(shard_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    raster_dir = shard_dir / "page_images"
    if not raster_dir.is_dir():
        return {}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(raster_dir.glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index[(row["doc_id"], row["page_index"])] = row
    return index


def blocks_to_table(rows: Iterable[dict[str, Any]], dialect: str) -> pa.Table:
    """Build one Arrow table from already-mapped rows. Used by tests."""
    return pa.Table.from_pylist(list(rows), schema=schema_for(dialect))
