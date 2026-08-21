"""Parquet encoding for ``pdfsys.doc/v1`` — the L2 publish format.

Two physical tables, joined on ``image_id``:

* ``documents/`` — one row per PDF. Carries the whole reading-order ``blocks``
  list plus the pre-rendered ``text``. Text-only and metadata scans never
  touch image bytes, which is the entire point of the split: at PB scale a
  1 KB text row and a 400 KB JPEG must not share a row group.
* ``images/`` — one row per *unique* image blob, content-addressed by
  SHA-256. Repeated letterheads / stamps / logos collapse to one copy, and
  the ``image`` column is a ``struct<bytes, path>`` so
  ``datasets.load_dataset(...).cast_column("image", Image())`` decodes it
  without conversion.

For small datasets ``embed_images=True`` inlines the same struct into the
documents table so a single file is self-contained (handy for a 150-PDF bench
shard); at corpus scale leave it off.

The schema version lives in the Parquet file-level key-value metadata under
``pdfsys.schema`` so a reader can dispatch on it without a side-channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pdfsys_core import (
    DATASET_SCHEMA_VERSION,
    Block,
    BlockType,
    DocRecord,
    ImageBlob,
    iter_pairs,
)

__all__ = [
    "BBOX_TYPE",
    "BLOCK_TYPE",
    "IMAGE_TYPE",
    "DOC_SCHEMA",
    "IMAGE_SCHEMA",
    "PAIR_SCHEMA",
    "DatasetWriter",
    "doc_to_row",
    "image_to_row",
    "pairs_table",
]

# ---------------------------------------------------------------------------
# Arrow types
# ---------------------------------------------------------------------------

BBOX_TYPE = pa.struct(
    [
        ("x0", pa.float32()),
        ("y0", pa.float32()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
    ]
)

#: HF ``datasets.Image`` wire format. ``path`` stays nullable — we address
#: images by content, not location.
IMAGE_TYPE = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])

#: One block of a document. ``type`` is dictionary-encoded: a dozen distinct
#: values over billions of rows compresses to nothing and keeps predicate
#: pushdown (``blocks.type == 'image'``) cheap.
BLOCK_TYPE = pa.struct(
    [
        ("idx", pa.int32()),
        ("page", pa.int32()),
        ("type", pa.dictionary(pa.int8(), pa.string())),
        ("text", pa.large_string()),
        ("level", pa.int8()),
        ("caption", pa.string()),
        ("footnote", pa.string()),
        ("alt", pa.string()),
        ("bbox", BBOX_TYPE),
        ("image_id", pa.string()),
        ("mentions", pa.list_(pa.int32())),
    ]
)

DOC_SCHEMA = pa.schema(
    [
        # -- identity / provenance -------------------------------------------
        ("id", pa.string()),  # sha256 of the source PDF
        ("source_uri", pa.string()),
        ("provenance", pa.string()),  # opaque upstream JSON (license, batch)
        # -- text view -------------------------------------------------------
        ("text", pa.large_string()),
        ("page_ends", pa.list_(pa.int32())),
        # -- interleaved view ------------------------------------------------
        ("blocks", pa.list_(BLOCK_TYPE)),
        ("image_ids", pa.list_(pa.string())),
        # -- cheap filter columns (all derivable; stored to avoid nested scans)
        ("n_pages", pa.int32()),
        ("n_blocks", pa.int32()),
        ("n_chars", pa.int32()),
        ("n_images", pa.int32()),
        ("n_tables", pa.int32()),
        ("n_formulas", pa.int32()),
        # -- pipeline signals -------------------------------------------------
        ("backend", pa.dictionary(pa.int8(), pa.string())),
        ("router_ocr_prob", pa.float32()),
        ("quality_score", pa.float32()),
        ("quality_model", pa.string()),
        ("lang", pa.dictionary(pa.int16(), pa.string())),
        ("lang_score", pa.float32()),
    ]
)

IMAGE_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("image", IMAGE_TYPE),
        ("format", pa.dictionary(pa.int8(), pa.string())),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("n_bytes", pa.int32()),
    ]
)

#: Materialized image-text pair view. Not written by default — build it from
#: ``documents`` + ``images`` when a pair-shaped dataset is what you want.
PAIR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("image_id", pa.string()),
        ("block_idx", pa.int32()),
        ("page", pa.int32()),
        ("text", pa.large_string()),
        ("source", pa.dictionary(pa.int8(), pa.string())),
    ]
)

_KV_METADATA = {b"pdfsys.schema": DATASET_SCHEMA_VERSION.encode()}


# ---------------------------------------------------------------------------
# Row encoding
# ---------------------------------------------------------------------------


def _block_to_dict(b: Block) -> dict[str, Any]:
    return {
        "idx": b.idx,
        "page": b.page,
        "type": str(b.type),
        "text": b.text,
        "level": b.level,
        "caption": b.caption,
        "footnote": b.footnote,
        "alt": b.alt,
        "bbox": (
            None
            if b.bbox is None
            else {"x0": b.bbox[0], "y0": b.bbox[1], "x1": b.bbox[2], "y1": b.bbox[3]}
        ),
        "image_id": b.image_id,
        "mentions": list(b.mentions),
    }


def doc_to_row(
    doc: DocRecord, *, embed_images: bool = False, include_text: bool = True
) -> dict[str, Any]:
    """Encode one :class:`DocRecord` as a ``DOC_SCHEMA`` row dict.

    ``text`` duplicates the content already in ``blocks[].text`` — measured at
    ~40 % of a documents file, i.e. ~4 % of a shard once image bytes are in
    the picture. It is stored anyway because text-only consumers are the
    majority and re-rendering costs a full nested-column read. Set
    ``include_text=False`` for an image-free corpus, where the duplication is
    no longer amortized against image bytes; the column stays in the schema
    and is written null.
    """
    row: dict[str, Any] = {
        "id": doc.id,
        "source_uri": doc.source_uri or None,
        "provenance": doc.provenance or None,
        "text": doc.text if include_text else None,
        "page_ends": list(doc.page_ends),
        "blocks": [_block_to_dict(b) for b in doc.blocks],
        "image_ids": list(doc.image_ids),
        "n_pages": doc.n_pages or len(doc.page_ends),
        "n_blocks": len(doc.blocks),
        "n_chars": len(doc.text),
        "n_images": doc.count(BlockType.IMAGE, BlockType.CHART),
        "n_tables": doc.count(BlockType.TABLE),
        "n_formulas": doc.count(BlockType.FORMULA),
        "backend": doc.backend or None,
        "router_ocr_prob": doc.router_ocr_prob,
        "quality_score": doc.quality_score,
        "quality_model": doc.quality_model or None,
        "lang": doc.lang or None,
        "lang_score": doc.lang_score,
    }
    if embed_images:
        row["images"] = None  # filled by DatasetWriter, which owns the blobs
    return row


def image_to_row(blob: ImageBlob) -> dict[str, Any]:
    return {
        "image_id": blob.image_id,
        "image": {"bytes": blob.data, "path": f"{blob.image_id}.{blob.format}"},
        "format": blob.format,
        "width": blob.width,
        "height": blob.height,
        "n_bytes": blob.n_bytes,
    }


def pairs_table(docs: Iterable[DocRecord], **kwargs: Any) -> pa.Table:
    """Build the image-text pair view from documents.

    ``kwargs`` are forwarded to :func:`pdfsys_core.iter_pairs`
    (``context_window``, ``min_chars``).
    """
    rows = [
        {
            "doc_id": p.doc_id,
            "image_id": p.image_id,
            "block_idx": p.block_idx,
            "page": p.page,
            "text": p.text,
            "source": p.source,
        }
        for doc in docs
        for p in iter_pairs(doc, **kwargs)
    ]
    return pa.Table.from_pylist(rows, schema=PAIR_SCHEMA)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class DatasetWriter:
    """Streaming writer for a ``pdfsys.doc/v1`` shard.

    Usage::

        with DatasetWriter(Path("out/v1")) as w:
            for doc, blobs in produce():
                w.write(doc, blobs)

    Image blobs are deduped across the whole shard by ``image_id``; calling
    ``write`` with a blob already seen is free.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        shard: str = "shard-00000",
        compression: str = "zstd",
        embed_images: bool = False,
        include_text: bool = True,
        row_group_size: int = 512,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.shard = shard
        self.compression = compression
        self.embed_images = embed_images
        self.include_text = include_text
        self.row_group_size = row_group_size

        self._doc_schema = _doc_schema_for(embed_images)
        (self.out_dir / "documents").mkdir(parents=True, exist_ok=True)
        self._docs = pq.ParquetWriter(
            str(self.out_dir / "documents" / f"{shard}.parquet"),
            self._doc_schema.with_metadata(_KV_METADATA),
            compression=compression,
        )
        self._images: pq.ParquetWriter | None = None
        self._seen_images: set[str] = set()
        self._doc_buffer: list[dict[str, Any]] = []
        self.docs_written = 0
        self.images_written = 0

    # -- public ------------------------------------------------------------

    def write(self, doc: DocRecord, blobs: Sequence[ImageBlob] = ()) -> None:
        by_id = {b.image_id: b for b in blobs}
        row = doc_to_row(
            doc, embed_images=self.embed_images, include_text=self.include_text
        )
        if self.embed_images:
            row["images"] = [
                image_to_row(by_id[i])["image"] for i in doc.image_ids if i in by_id
            ]
        else:
            self._write_images(by_id.values())

        self._doc_buffer.append(row)
        self.docs_written += 1
        if len(self._doc_buffer) >= self.row_group_size:
            self._flush_docs()

    def close(self) -> None:
        self._flush_docs()
        self._docs.close()
        if self._images is not None:
            self._images.close()
            self._images = None

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _flush_docs(self) -> None:
        if not self._doc_buffer:
            return
        table = pa.Table.from_pylist(self._doc_buffer, schema=self._doc_schema)
        self._docs.write_table(table)
        self._doc_buffer.clear()

    def _write_images(self, blobs: Iterable[ImageBlob]) -> None:
        fresh = [b for b in blobs if b.image_id not in self._seen_images]
        if not fresh:
            return
        if self._images is None:
            (self.out_dir / "images").mkdir(parents=True, exist_ok=True)
            self._images = pq.ParquetWriter(
                str(self.out_dir / "images" / f"{self.shard}.parquet"),
                IMAGE_SCHEMA.with_metadata(_KV_METADATA),
                compression=self.compression,
            )
        self._images.write_table(
            pa.Table.from_pylist([image_to_row(b) for b in fresh], schema=IMAGE_SCHEMA)
        )
        self._seen_images.update(b.image_id for b in fresh)
        self.images_written += len(fresh)


def _doc_schema_for(embed_images: bool) -> pa.Schema:
    if not embed_images:
        return DOC_SCHEMA
    return DOC_SCHEMA.append(pa.field("images", pa.list_(IMAGE_TYPE)))
