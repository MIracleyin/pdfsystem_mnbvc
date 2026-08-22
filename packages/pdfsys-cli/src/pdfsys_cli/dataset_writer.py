"""Parquet encoding for ``pdfsys.page/v2`` — the L2 publish format.

Three tables. Only the first is required; the other two are joined by content
address and can be built, rebuilt or dropped independently.

``pages/``
    One row per page, keyed ``(doc_id, page_index)``. Carries the page text
    (with the image interleaving inline), the model-derived ``blocks``, and
    document-level columns denormalized so the common filter needs no join.
    Rows are written sorted by ``(doc_id, page_index)`` so that reassembling a
    document is a sequential scan.
``images/``
    One row per *unique* image crop, content-addressed by SHA-256. Repeated
    letterheads / stamps / logos collapse to one copy.
``page_images/``
    One row per full-page raster. Separate from ``images/`` on purpose: a
    200-dpi page is one to two orders of magnitude larger than a crop, so
    mixing them wrecks row-group sizing, and keeping them apart means the
    whole table can be built later — or never — without rewriting anything.

Why the split at all: at PB scale a 1 KB text row and a 400 KB JPEG must not
share a row group, or every text-only scan pays for pixels it will not read.

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
    ImageBlob,
    PageRecord,
    iter_pairs,
)

__all__ = [
    "BBOX_TYPE",
    "BLOCK_TYPE",
    "IMAGE_TYPE",
    "PAGE_SCHEMA",
    "IMAGE_SCHEMA",
    "PAGE_IMAGE_SCHEMA",
    "PAIR_SCHEMA",
    "DatasetWriter",
    "page_to_row",
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

#: One block of a page. ``type`` is dictionary-encoded: a dozen distinct values
#: over billions of rows compresses to nothing and keeps predicate pushdown
#: (``blocks.type == 'image'``) cheap. ``Block.page`` is deliberately absent —
#: it is a constant per row, equal to ``page_index``.
BLOCK_TYPE = pa.struct(
    [
        ("idx", pa.int32()),
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

PAGE_SCHEMA = pa.schema(
    [
        # -- identity: from the PDF, no model involved -----------------------
        ("doc_id", pa.string()),  # sha256 of the source PDF
        ("page_index", pa.int32()),
        ("width_pt", pa.float32()),
        ("height_pt", pa.float32()),
        ("rotation", pa.int16()),
        # -- content ---------------------------------------------------------
        ("text", pa.large_string()),  # image interleaving encoded inline
        ("image_ids", pa.list_(pa.string())),
        ("page_image_id", pa.string()),
        ("render_dpi", pa.int16()),
        ("blocks", pa.list_(BLOCK_TYPE)),  # model-derived, droppable
        # -- provenance -------------------------------------------------------
        ("extractor", pa.dictionary(pa.int8(), pa.string())),
        ("layout_model", pa.dictionary(pa.int8(), pa.string())),
        # -- cheap filter columns (all derivable; stored to avoid nested scans)
        ("n_chars", pa.int32()),
        ("n_blocks", pa.int32()),
        ("n_images", pa.int32()),
        ("n_tables", pa.int32()),
        ("n_formulas", pa.int32()),
        # -- page-level signals ------------------------------------------------
        ("lang", pa.dictionary(pa.int16(), pa.string())),
        ("lang_score", pa.float32()),
        ("quality_score", pa.float32()),
        ("quality_model", pa.dictionary(pa.int8(), pa.string())),
        # -- document-level, denormalized onto every page ----------------------
        ("doc_n_pages", pa.int32()),
        ("source_uri", pa.string()),
        ("provenance", pa.string()),
        ("doc_lang", pa.dictionary(pa.int16(), pa.string())),
        ("doc_quality_score", pa.float32()),
        ("router_ocr_prob", pa.float32()),
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

#: Full-page rasters. Same columns as ``images``, plus the page it renders and
#: at what DPI — a page can legitimately appear at several resolutions.
PAGE_IMAGE_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("doc_id", pa.string()),
        ("page_index", pa.int32()),
        ("render_dpi", pa.int16()),
        ("image", IMAGE_TYPE),
        ("format", pa.dictionary(pa.int8(), pa.string())),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("n_bytes", pa.int32()),
    ]
)

#: Materialized image-text pair view. Not written by default — build it from
#: ``pages`` + ``images`` when a pair-shaped dataset is what you want.
PAIR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("page_index", pa.int32()),
        # Exactly one of these addresses the pixels: a stored crop, or a
        # rectangle of the page raster. Mirrors the two inline marker kinds.
        ("image_id", pa.string()),
        ("bbox", BBOX_TYPE),
        ("block_idx", pa.int32()),
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


def page_to_row(page: PageRecord, *, include_blocks: bool = True) -> dict[str, Any]:
    """Encode one :class:`PageRecord` as a ``PAGE_SCHEMA`` row dict.

    ``include_blocks=False`` writes the column null. The counts derived from
    blocks are still populated, and ``text`` still carries the interleaving —
    what you lose is bboxes, captions, types and mention links.
    """
    return {
        "doc_id": page.doc_id,
        "page_index": page.page_index,
        "width_pt": page.width_pt or None,
        "height_pt": page.height_pt or None,
        "rotation": page.rotation,
        "text": page.text,
        "image_ids": list(page.image_ids),
        "page_image_id": page.page_image_id,
        "render_dpi": page.render_dpi,
        "blocks": (
            [_block_to_dict(b) for b in page.blocks] if include_blocks else None
        ),
        "extractor": page.extractor or None,
        "layout_model": page.layout_model or None,
        "n_chars": page.n_chars,
        "n_blocks": len(page.blocks),
        "n_images": page.count(BlockType.IMAGE, BlockType.CHART),
        "n_tables": page.count(BlockType.TABLE),
        "n_formulas": page.count(BlockType.FORMULA),
        "lang": page.lang or None,
        "lang_score": page.lang_score,
        "quality_score": page.quality_score,
        "quality_model": page.quality_model or None,
        "doc_n_pages": page.doc_n_pages,
        "source_uri": page.source_uri or None,
        "provenance": page.provenance or None,
        "doc_lang": page.doc_lang or None,
        "doc_quality_score": page.doc_quality_score,
        "router_ocr_prob": page.router_ocr_prob,
    }


def image_to_row(blob: ImageBlob) -> dict[str, Any]:
    return {
        "image_id": blob.image_id,
        "image": {"bytes": blob.data, "path": f"{blob.image_id}.{blob.format}"},
        "format": blob.format,
        "width": blob.width,
        "height": blob.height,
        "n_bytes": blob.n_bytes,
    }


def pairs_table(docs: Iterable[Sequence[PageRecord]], **kwargs: Any) -> pa.Table:
    """Build the image-text pair view.

    Takes an iterable of *documents*, each a sequence of its pages — mention
    links cross page boundaries, so pairs cannot be built one page at a time.
    ``kwargs`` are forwarded to :func:`pdfsys_core.iter_pairs`.
    """
    rows = [
        {
            "doc_id": p.doc_id,
            "page_index": p.page_index,
            "image_id": p.image_id,
            "bbox": (
                None
                if p.bbox is None
                else {"x0": p.bbox[0], "y0": p.bbox[1], "x1": p.bbox[2], "y1": p.bbox[3]}
            ),
            "block_idx": p.block_idx,
            "text": p.text,
            "source": p.source,
        }
        for pages in docs
        for p in iter_pairs(pages, **kwargs)
    ]
    return pa.Table.from_pylist(rows, schema=PAIR_SCHEMA)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class DatasetWriter:
    """Streaming writer for a ``pdfsys.page/v2`` shard.

    Usage::

        with DatasetWriter(Path("out/v2")) as w:
            for pages, blobs in produce():
                w.write(pages, blobs)

    ``write`` takes all pages of one document at a time so the file stays
    sorted by ``(doc_id, page_index)``. Image blobs are deduped across the
    whole shard by ``image_id``.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        shard: str = "shard-00000",
        compression: str = "zstd",
        include_blocks: bool = True,
        row_group_size: int = 2048,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.shard = shard
        self.compression = compression
        self.include_blocks = include_blocks
        self.row_group_size = row_group_size

        (self.out_dir / "pages").mkdir(parents=True, exist_ok=True)
        self._pages = pq.ParquetWriter(
            str(self.out_dir / "pages" / f"{shard}.parquet"),
            PAGE_SCHEMA.with_metadata(_KV_METADATA),
            compression=compression,
        )
        self._images: pq.ParquetWriter | None = None
        self._page_images: pq.ParquetWriter | None = None
        self._seen_images: set[str] = set()
        self._seen_page_images: set[str] = set()
        self._last_doc_id: str | None = None
        self._buffer: list[dict[str, Any]] = []
        self.pages_written = 0
        self.docs_written = 0
        self.images_written = 0
        self.page_images_written = 0

    # -- public ------------------------------------------------------------

    def write(
        self,
        pages: Sequence[PageRecord],
        blobs: Sequence[ImageBlob] = (),
        page_rasters: Sequence[tuple[PageRecord, ImageBlob]] = (),
    ) -> None:
        """Append one document: all its pages, its crops, and any page rasters.

        Documents must arrive in ascending ``doc_id`` order — the shard
        promises to be sorted by ``(doc_id, page_index)`` so that reassembling
        a document is a sequential scan, and sorting the whole shard at close
        time would mean buffering it. Out-of-order calls raise instead of
        quietly producing a shard that violates its own contract.
        """
        doc_ids = {p.doc_id for p in pages}
        if len(doc_ids) > 1:
            raise ValueError(f"write() 一次只接受一个文档的页，收到 {len(doc_ids)} 个 doc_id")
        if doc_ids:
            doc_id = doc_ids.pop()
            if self._last_doc_id is not None and doc_id <= self._last_doc_id:
                raise ValueError(
                    f"doc_id 必须递增：收到 {doc_id[:12]}… 但上一个是 "
                    f"{self._last_doc_id[:12]}…。写入前请按 doc_id 排序。"
                )
            self._last_doc_id = doc_id

        self._write_images(blobs)
        self._write_page_images(page_rasters)

        for page in sorted(pages, key=lambda p: p.page_index):
            self._buffer.append(page_to_row(page, include_blocks=self.include_blocks))
            self.pages_written += 1
        self.docs_written += 1

        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._pages.close()
        for writer in (self._images, self._page_images):
            if writer is not None:
                writer.close()
        self._images = self._page_images = None

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._pages.write_table(
            pa.Table.from_pylist(self._buffer, schema=PAGE_SCHEMA)
        )
        self._buffer.clear()

    def _write_images(self, blobs: Iterable[ImageBlob]) -> None:
        fresh = [b for b in blobs if b.image_id not in self._seen_images]
        if not fresh:
            return
        if self._images is None:
            self._images = self._open("images", IMAGE_SCHEMA)
        self._images.write_table(
            pa.Table.from_pylist([image_to_row(b) for b in fresh], schema=IMAGE_SCHEMA)
        )
        self._seen_images.update(b.image_id for b in fresh)
        self.images_written += len(fresh)

    def _write_page_images(
        self, rasters: Iterable[tuple[PageRecord, ImageBlob]]
    ) -> None:
        rows = [
            {
                **image_to_row(blob),
                "doc_id": page.doc_id,
                "page_index": page.page_index,
                "render_dpi": page.render_dpi,
            }
            for page, blob in rasters
            if blob.image_id not in self._seen_page_images
        ]
        if not rows:
            return
        if self._page_images is None:
            self._page_images = self._open("page_images", PAGE_IMAGE_SCHEMA)
        self._page_images.write_table(
            pa.Table.from_pylist(rows, schema=PAGE_IMAGE_SCHEMA)
        )
        self._seen_page_images.update(r["image_id"] for r in rows)
        self.page_images_written += len(rows)

    def _open(self, subdir: str, schema: pa.Schema) -> pq.ParquetWriter:
        (self.out_dir / subdir).mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(
            str(self.out_dir / subdir / f"{self.shard}.parquet"),
            schema.with_metadata(_KV_METADATA),
            compression=self.compression,
        )
