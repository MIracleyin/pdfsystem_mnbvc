"""``pdfsys.page/v2`` — the published dataset format (L2), stdlib-only.

This module defines *what a released page looks like*, independent of Parquet.
The Arrow schemas and the writer live in :mod:`pdfsys_cli.dataset_writer`;
everything here is a plain dataclass or a pure function so that
``pdfsys-core`` keeps its zero-dependency guarantee
(``docs/golden-principles/ZERO_DEP_CORE.md``).

Design in one sentence
----------------------
**One row per page, keyed by ``(doc_id, page_index)`` — an identity the PDF
gives us, not one a model invents — carrying the page's text with the image
interleaving encoded inline; model-derived structure is one droppable column
beside it, and image bytes live in content-addressed side tables.**

Why the page, not the document
------------------------------
v1 used one row per document with a nested ``blocks`` list. Three things went
wrong at scale:

* **Row identity was model-defined.** The only fine-grained handle was
  ``blocks[].idx``, entirely a product of whichever layout model ran.
  ``(doc_id, page_index)`` comes from the PDF itself.
* **Rows were wildly uneven.** A 500-page book is one row with ~20k blocks and
  megabytes of text; Parquet row groups skew and a predicate is all-or-nothing.
* **``page_ends`` existed at all.** That column, and the page-tracking loop in
  the renderer, existed only to recover page boundaries from a document-level
  text blob. Making the page the row deletes the whole mechanism — a good sign
  it was the natural unit all along.

FinePDFs stores ``per_page_languages`` / ``ocr_quality_scores`` as *arrays* on
a document row for the same reason. With page rows they are scalars.

The stability ladder
--------------------
Nothing about reading order is truly model-independent for a scanned PDF — a
layout model decided where the figure sits between paragraphs. So the format
is layered by how much a consumer has to trust:

1. ``doc_id`` / ``page_index`` / ``width_pt`` / ``height_pt`` / the page
   raster — from the PDF, no model involved.
2. ``text`` + the image blobs it references — extractor-dependent, but at the
   tool level, and stamped with ``extractor``.
3. ``blocks`` — layout-model-dependent: types, reading order, bboxes,
   captions. Stamped with ``layout_model``, nullable, droppable as a column.

A consumer can stop at any rung. Critically, the **image/text interleaving
lives in ``text``** as inline ``![](img://<sha256>)`` markers, not in
``blocks`` — so dropping rung 3 costs you bboxes and captions, never the
interleaving itself. :func:`to_interleaved` operates on the string alone.

Views (all derivable, none stored twice)
----------------------------------------
* **plain text** — ``PageRecord.text``, or strip the markers with
  :data:`IMAGE_REF_RE`.
* **interleaved** — :func:`to_interleaved`, OBELICS/MINT-1T shape.
* **image–text pairs** — :func:`iter_pairs`, caption / model description /
  figure-referencing body text (the last following PMC-InterCPT, which found
  reference context materially better than captions alone).

See ``docs/superpowers/specs/2026-08-22-page-level-parquet-dataset-design.md``
for the full rationale and the survey of prior formats.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "BlockType",
    "FURNITURE_TYPES",
    "IMAGE_REF_RE",
    "Block",
    "ImageBlob",
    "ImageRef",
    "PageRecord",
    "ImageTextPair",
    "image_ref",
    "bbox_ref",
    "parse_image_ref",
    "crop_region",
    "strip_image_refs",
    "image_id_for",
    "probe_image",
    "blocks_from_content_list",
    "blocks_from_segments",
    "link_mentions",
    "render_markdown",
    "split_pages",
    "to_interleaved",
    "iter_pairs",
]

#: Bumped on any breaking change to the page/block field set. Written into the
#: Parquet file-level key-value metadata by the writer.
DATASET_SCHEMA_VERSION = "pdfsys.page/2"


class BlockType(StrEnum):
    """Block-level classification.

    Deliberately coarser than a full document model (cf. DoclingDocument's
    tree of ``texts`` / ``tables`` / ``pictures`` / ``groups``): we are
    building a pretraining corpus, not reconstructing an editable document.
    Flat + typed beats hierarchical + faithful when the consumer is a
    tokenizer.
    """

    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    CHART = "chart"
    # Page furniture — excluded from the rendered text by default.
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PAGE_NUMBER = "page_number"
    ASIDE = "aside"


#: Block types that can carry pixels — a figure, a chart, or a table (whose
#: crop is worth pairing with its HTML transcription).
_PIXEL_BEARING_TYPES = frozenset(
    {BlockType.IMAGE, BlockType.CHART, BlockType.TABLE}
)

#: Blocks that are page decoration rather than document content. Docling calls
#: this "furniture"; keeping the distinction lets us drop running headers and
#: page numbers from the training text without losing them from the record.
FURNITURE_TYPES = frozenset(
    {
        BlockType.PAGE_HEADER,
        BlockType.PAGE_FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE,
    }
)

#: The inline marker that carries the image/text interleaving inside ``text``.
#: This regex is part of the format contract: splitting a page's text on it
#: yields the interleaved sequence without touching ``blocks``. Group 1 is the
#: *reference*, of which there are two kinds — see :func:`parse_image_ref`.
IMAGE_REF_RE = re.compile(
    r"!\[[^\]]*\]\((img://[0-9a-f]{64}|bbox://[\d.]+,[\d.]+,[\d.]+,[\d.]+)\)"
)


@dataclass(frozen=True, slots=True)
class ImageRef:
    """A resolved inline image reference.

    ``kind="blob"``
        Addresses a stored crop in the ``images`` table by content hash.
    ``kind="region"``
        Addresses a rectangle of this page's raster. No crop is stored; the
        pixels are obtained by cropping ``page_images`` at ``bbox``. This
        loses almost nothing because MinerU's crops are *already*
        sub-rectangles of a 200-dpi page render — storing both is storing the
        same pixels twice. "Almost": bboxes live on a 0-1000 integer grid, so
        a derived crop can differ from MinerU's by roughly a pixel per edge.
    """

    kind: str  # "blob" | "region"
    image_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None


def image_ref(image_id: str) -> str:
    """Marker addressing a stored crop by content hash."""
    return f"![](img://{image_id})"


def bbox_ref(bbox: tuple[float, float, float, float]) -> str:
    """Marker addressing a rectangle of this page's raster.

    Only the geometry goes in: the row already carries ``doc_id``,
    ``page_index`` and ``page_image_id``, so repeating them per marker would
    be redundant. The marker still resolves without reading ``blocks``, which
    is the property that matters.
    """
    return "![](bbox://" + ",".join(f"{v:.4f}" for v in bbox) + ")"


def parse_image_ref(ref: str) -> ImageRef | None:
    """Resolve a reference captured by :data:`IMAGE_REF_RE` (group 1)."""
    if ref.startswith("img://"):
        return ImageRef(kind="blob", image_id=ref[len("img://") :])
    if ref.startswith("bbox://"):
        try:
            x0, y0, x1, y1 = (float(v) for v in ref[len("bbox://") :].split(","))
        except ValueError:
            return None
        return ImageRef(kind="region", bbox=(x0, y0, x1, y1))
    return None


def crop_region(
    bbox: tuple[float, float, float, float], width_px: int, height_px: int
) -> tuple[int, int, int, int]:
    """Normalized bbox → integer pixel box for cropping a page raster.

    The one operation every consumer of a region reference needs. Matches
    PIL's ``Image.crop`` argument order.
    """
    x0, y0, x1, y1 = bbox
    return (
        round(x0 * width_px),
        round(y0 * height_px),
        round(x1 * width_px),
        round(y1 * height_px),
    )


def strip_image_refs(text: str) -> str:
    """Drop image markers, leaving plain text. Captions survive — they are
    emitted as their own paragraphs, never inside the marker's alt slot."""
    return re.sub(r"\n{3,}", "\n\n", IMAGE_REF_RE.sub("", text)).strip()


@dataclass(frozen=True, slots=True)
class Block:
    """One block-level unit in document reading order — rung 3, model-derived.

    ``text`` encoding follows the same contract as :class:`pdfsys_core.Segment`
    so no re-interpretation is needed across the L1→L2 boundary:
    TEXT/TITLE/LIST = Markdown, TABLE = HTML, FORMULA = LaTeX, IMAGE/CHART =
    ``None`` (the pixels live in ``image_id``).
    """

    #: Reading-order index, scoped to the *document*, not the page — so that
    #: ``mentions`` can point across a page break. Dense over the document.
    idx: int
    #: Page this block sits on. Equals the containing row's ``page_index``;
    #: carried on the dataclass because blocks are built and linked at
    #: document scope before being split into page rows, and dropped from the
    #: Arrow struct where it would be a constant per row.
    page: int
    type: BlockType
    text: str | None = None
    #: Heading level (1-6) for TITLE blocks, else ``None``.
    level: int | None = None
    #: Human-authored caption attached to an image / chart / table.
    caption: str | None = None
    #: Footnote attached to an image / chart / table.
    footnote: str | None = None
    #: Model-generated description of the image (MinerU VLM ``content``).
    #: Kept separate from ``caption`` so that a consumer can choose between
    #: human ground truth and synthetic text.
    alt: str | None = None
    #: Normalized ``(x0, y0, x1, y1)`` in [0, 1], top-left origin.
    bbox: tuple[float, float, float, float] | None = None
    #: SHA-256 of the image bytes; join key into the images table.
    image_id: str | None = None
    #: ``idx`` of blocks in the same *document* whose text references this
    #: figure/table ("as shown in Fig. 3"). May point to another page.
    mentions: tuple[int, ...] = ()

    @property
    def is_furniture(self) -> bool:
        return self.type in FURNITURE_TYPES

    @property
    def has_image(self) -> bool:
        return self.image_id is not None


@dataclass(frozen=True, slots=True)
class ImageBlob:
    """One unique image, addressed by the SHA-256 of its encoded bytes.

    Content addressing is not a nicety here: scanned corpora repeat the same
    logo/stamp/letterhead across thousands of documents, and the images table
    dedupes them for free.
    """

    image_id: str
    data: bytes
    format: str  # "jpeg" | "png" | "webp" | "gif" | "unknown"
    width: int
    height: int

    @property
    def n_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class PageRecord:
    """One published page — one row of the ``pages`` table.

    Deliberately self-contained: document-level fields are denormalized onto
    every page so that the common query ("give me text where quality > 2 and
    lang = zho_Hans") needs no join. Dictionary encoding makes the repetition
    nearly free, and a page row that answers questions on its own is what
    "complete" means at this scale.
    """

    # -- identity: from the PDF, no model involved -------------------------
    #: SHA-256 of the source PDF bytes.
    doc_id: str
    #: Zero-based page number within the source PDF.
    page_index: int
    #: Page size in PDF points (1/72 inch), as the PDF declares it.
    width_pt: float = 0.0
    height_pt: float = 0.0
    rotation: int = 0

    # -- content ------------------------------------------------------------
    #: Page rendered as Markdown. Images appear as ``![](img://<sha256>)``;
    #: this string alone carries the interleaving.
    text: str = ""
    #: Image crops referenced by this page, in first-use order.
    image_ids: tuple[str, ...] = ()
    #: Full-page raster, if one was built. Joins ``page_images``. Null by
    #: default: rasters are rebuildable from L0 at any DPI, so committing to
    #: one now would be paying TB for a choice we can defer.
    page_image_id: str | None = None
    render_dpi: int | None = None
    #: Model-derived structure. Empty when the shard was written without it;
    #: dropping it costs bboxes, captions and types, never the interleaving.
    blocks: tuple[Block, ...] = ()

    # -- provenance ---------------------------------------------------------
    extractor: str = ""  # mupdf | pipeline | vlm
    layout_model: str = ""

    # -- page-level signals -------------------------------------------------
    lang: str = ""
    lang_score: float | None = None
    quality_score: float | None = None
    quality_model: str = ""

    # -- document-level, denormalized onto every page -----------------------
    doc_n_pages: int = 0
    source_uri: str = ""
    provenance: str = ""  # opaque upstream JSON (license, crawl batch, ...)
    doc_lang: str = ""
    doc_quality_score: float | None = None
    router_ocr_prob: float | None = None

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def count(self, *types: BlockType) -> int:
        wanted = set(types)
        return sum(1 for b in self.blocks if b.type in wanted)


@dataclass(frozen=True, slots=True)
class ImageTextPair:
    """One image–text pair extracted from a page.

    The image is addressed one of two ways, mirroring the inline markers:
    ``image_id`` names a stored crop, or ``bbox`` names a rectangle of the
    page raster. Exactly one is set.
    """

    doc_id: str
    page_index: int
    block_idx: int
    text: str
    #: Which tier supplied the text — see :func:`iter_pairs`.
    source: str
    image_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None


# ---------------------------------------------------------------------------
# Image identity + dimensions (no Pillow — core stays zero-dep)
# ---------------------------------------------------------------------------


def image_id_for(data: bytes) -> str:
    """Content address of an image blob."""
    return hashlib.sha256(data).hexdigest()


def probe_image(data: bytes) -> tuple[str, int, int]:
    """Return ``(format, width, height)`` by reading container headers only.

    Supports the encodings MinerU emits (JPEG, PNG) plus GIF/WebP. Unknown or
    truncated input yields ``("unknown", 0, 0)`` rather than raising — a
    corpus run must never die on one bad crop.
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return "png", int(w), int(h)

        if data[:2] == b"\xff\xd8":
            return ("jpeg", *_jpeg_size(data))

        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return "gif", int(w), int(h)

        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ("webp", *_webp_size(data))
    except (struct.error, IndexError, ValueError):
        pass
    return "unknown", 0, 0


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Walk JPEG markers to the first SOF segment and read its dimensions."""
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # Standalone markers carry no length field.
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        # SOF0..SOF15, excluding DHT (C4), JPG (C8) and DAC (CC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return int(w), int(h)
        i += 2 + seg_len
    return 0, 0


def _webp_size(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        return (
            struct.unpack("<H", data[26:28])[0] & 0x3FFF,
            struct.unpack("<H", data[28:30])[0] & 0x3FFF,
        )
    return 0, 0


# ---------------------------------------------------------------------------
# Ingest: MinerU content_list.json → blocks
# ---------------------------------------------------------------------------

#: MinerU ``content_list`` type → our BlockType. Anything unlisted falls back
#: to TEXT, which is the safe default (content is kept, just not classified).
_MINERU_TYPE_MAP: dict[str, BlockType] = {
    "text": BlockType.TEXT,
    "list": BlockType.LIST,
    "code": BlockType.CODE,
    "algorithm": BlockType.CODE,
    "table": BlockType.TABLE,
    "equation": BlockType.FORMULA,
    "interline_equation": BlockType.FORMULA,
    "image": BlockType.IMAGE,
    "chart": BlockType.CHART,
    "header": BlockType.PAGE_HEADER,
    "page_header": BlockType.PAGE_HEADER,
    "footer": BlockType.PAGE_FOOTER,
    "page_footer": BlockType.PAGE_FOOTER,
    "page_number": BlockType.PAGE_NUMBER,
    "page_footnote": BlockType.PAGE_FOOTER,
    "aside_text": BlockType.ASIDE,
}


def blocks_from_content_list(
    items: Sequence[Mapping[str, object]],
    *,
    bbox_scale: float = 1000.0,
    image_ids: Mapping[str, str] | None = None,
) -> tuple[Block, ...]:
    """Convert MinerU's ``content_list.json`` into document-scoped blocks.

    ``bbox_scale`` is the coordinate space MinerU's bboxes live in. MinerU maps
    them onto a 0–1000 grid per axis, *independent of page size* — do not try
    to normalize against ``middle.json``'s ``page_size``, which is the page
    geometry in PDF points, a different quantity entirely (observed:
    ``page_size=[558, 773]`` with bboxes reaching 940).

    ``image_ids`` maps MinerU's ``img_path`` to the content address of the
    corresponding blob.
    """
    blocks: list[Block] = []
    for idx, item in enumerate(items):
        raw_type = str(item.get("type") or "text")
        btype = _MINERU_TYPE_MAP.get(raw_type, BlockType.TEXT)

        text = _clean(item.get("text"))
        level = None
        if btype is BlockType.TEXT and item.get("text_level"):
            btype = BlockType.TITLE
            level = _as_int(item.get("text_level"), default=1)
        if btype is BlockType.LIST and text is None:
            text = _join_lines(item.get("list_items"))
        if btype is BlockType.TABLE:
            text = _clean(item.get("table_body")) or text
        if btype in (BlockType.IMAGE, BlockType.CHART):
            # MinerU puts the VLM-generated description in `content`; the OCR'd
            # text of an image block, if any, is in `text`.
            text = None

        img_path = item.get("img_path")
        image_id = None
        if isinstance(img_path, str) and img_path:
            image_id = (image_ids or {}).get(img_path)

        blocks.append(
            Block(
                idx=idx,
                page=_as_int(item.get("page_idx"), default=0),
                type=btype,
                text=text,
                level=level,
                caption=_join_lines(
                    item.get("image_caption")
                    or item.get("table_caption")
                    or item.get("chart_caption")
                ),
                footnote=_join_lines(
                    item.get("image_footnote")
                    or item.get("table_footnote")
                    or item.get("chart_footnote")
                ),
                alt=_clean(item.get("content"))
                if btype in (BlockType.IMAGE, BlockType.CHART)
                else None,
                bbox=_norm_bbox(item.get("bbox"), bbox_scale),
                image_id=image_id,
            )
        )
    return tuple(blocks)


def blocks_from_segments(segments: Iterable[object]) -> tuple[Block, ...]:
    """Convert :class:`pdfsys_core.Segment` objects (the mupdf fast path).

    Duck-typed on purpose: ``Segment`` lives in the ``pdfsys-types`` submodule
    and importing it here would make the dataset format depend on the parser
    contract in the wrong direction.
    """
    seg_type_map = {
        "text": BlockType.TEXT,
        "image": BlockType.IMAGE,
        "table": BlockType.TABLE,
        "formula": BlockType.FORMULA,
    }
    blocks: list[Block] = []
    for idx, seg in enumerate(segments):
        raw = getattr(seg, "type", None)
        raw_value = getattr(raw, "value", raw)
        btype = seg_type_map.get(str(raw_value), BlockType.TEXT)
        bbox = getattr(seg, "bbox", None)
        blocks.append(
            Block(
                idx=idx,
                page=int(getattr(seg, "page_index", 0) or 0),
                type=btype,
                text=_clean(getattr(seg, "content", None)),
                bbox=(
                    (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox is not None else None
                ),
            )
        )
    return tuple(blocks)


# ---------------------------------------------------------------------------
# Figure-mention linking (document-scoped — a figure is often referenced from
# the facing page)
# ---------------------------------------------------------------------------

#: "图 3" / "图3-1" / "Figure 3" / "Fig. 3" / "表 2" / "Table 2".
_LABEL_RE = re.compile(
    r"(?P<kind>图表|图|表|Figure|Fig\.?|Table|Tab\.?)\s*"
    r"(?P<num>\d{1,3}(?:[.\-−–]\d{1,3})?)",
    re.IGNORECASE,
)

_KIND_CANON = {
    "图": "figure",
    "图表": "figure",
    "figure": "figure",
    "fig": "figure",
    "fig.": "figure",
    "表": "table",
    "table": "table",
    "tab": "table",
    "tab.": "table",
}


def link_mentions(blocks: Sequence[Block]) -> tuple[Block, ...]:
    """Attach figure-referencing body text to the block it references.

    PMC-InterCPT (2026) found that captions alone under-describe a figure and
    that the paragraphs which *reference* it carry the actual explanation. We
    do the cheap version: parse ``图 N`` / ``Figure N`` / ``表 N`` out of each
    caption, then scan body text for the same label.

    Runs over a whole document's blocks, not one page's — a figure and the
    paragraph discussing it routinely sit on different pages.

    Returns a new tuple; blocks with no match are returned unchanged.
    """
    # label -> block indices that own it (from captions)
    owners: dict[tuple[str, str], list[int]] = {}
    for b in blocks:
        if b.caption is None:
            continue
        m = _LABEL_RE.search(b.caption)
        if m:
            owners.setdefault(_canon_label(m), []).append(b.idx)

    if not owners:
        return tuple(blocks)

    mentions: dict[int, list[int]] = {}
    for b in blocks:
        if b.text is None or b.type in (BlockType.TABLE, BlockType.FORMULA):
            continue
        for m in _LABEL_RE.finditer(b.text):
            for owner_idx in owners.get(_canon_label(m), ()):
                if owner_idx == b.idx:
                    continue
                bucket = mentions.setdefault(owner_idx, [])
                if b.idx not in bucket:
                    bucket.append(b.idx)

    if not mentions:
        return tuple(blocks)
    return tuple(
        (
            b
            if b.idx not in mentions
            else Block(
                idx=b.idx,
                page=b.page,
                type=b.type,
                text=b.text,
                level=b.level,
                caption=b.caption,
                footnote=b.footnote,
                alt=b.alt,
                bbox=b.bbox,
                image_id=b.image_id,
                mentions=tuple(mentions[b.idx]),
            )
        )
        for b in blocks
    )


def _canon_label(match: re.Match[str]) -> tuple[str, str]:
    kind = _KIND_CANON.get(match.group("kind").lower(), "figure")
    num = re.sub(r"[.\-−–]", "-", match.group("num"))
    return kind, num


# ---------------------------------------------------------------------------
# Rendering + page splitting
# ---------------------------------------------------------------------------


def render_markdown(
    blocks: Sequence[Block],
    *,
    drop_furniture: bool = True,
    region_refs: bool = True,
) -> str:
    """Render one page's blocks to Markdown.

    Images become ``![](img://<image_id>)`` with the caption following as its
    own paragraph. The empty alt slot is deliberate: put the caption in the
    marker *and* in the paragraph and every caption is counted twice, and a
    consumer stripping markers loses it entirely. ``Block.alt`` is
    model-generated and is never rendered — this string carries human/OCR
    content only.

    ``region_refs=False`` suppresses ``![](bbox://…)`` markers. A region
    reference addresses pixels inside a page raster, so it only resolves when
    the shard stores one; emit it without and every such marker dangles. The
    bbox stays on the block either way, so the pixels remain reconstructible
    from the source PDF.
    """
    parts = [
        chunk
        for b in blocks
        if not (drop_furniture and b.is_furniture)
        and (chunk := _render_block(b, region_refs=region_refs))
    ]
    return "\n\n".join(parts)


def _render_block(b: Block, *, region_refs: bool = True) -> str:
    if b.type is BlockType.TITLE:
        level = min(max(b.level or 1, 1), 6)
        return f"{'#' * level} {b.text}".strip() if b.text else ""
    if b.type is BlockType.FORMULA:
        return f"$$\n{b.text}\n$$" if b.text else ""
    if b.type in (BlockType.IMAGE, BlockType.CHART):
        # Blob reference when a crop was stored, region reference when the
        # pixels are to be cut out of the page raster instead. No mode flag
        # needed — which one applies is visible on the block itself.
        # No marker at all when there is nothing to address. An empty `![]()`
        # is not matched by IMAGE_REF_RE, so strip_image_refs cannot remove it
        # and every consumer inherits a literal that resolves to nothing. The
        # caption and footnote are real text and stay either way.
        if b.image_id:
            lines = [image_ref(b.image_id)]
        elif b.bbox is not None and region_refs:
            lines = [bbox_ref(b.bbox)]
        else:
            lines = []
        if b.caption:
            lines.append(b.caption)
        if b.footnote:
            lines.append(b.footnote)
        return "\n\n".join(lines)
    if b.type is BlockType.TABLE:
        lines = [x for x in (b.caption, b.text, b.footnote) if x]
        return "\n\n".join(lines)
    return b.text or ""


def split_pages(
    blocks: Sequence[Block],
    *,
    n_pages: int = 0,
    drop_furniture: bool = True,
    region_refs: bool = True,
    **page_fields: object,
) -> tuple[PageRecord, ...]:
    """Group document-scoped blocks into one :class:`PageRecord` per page.

    ``n_pages`` (from the PDF) makes pages that produced no block at all — a
    blank scan, a full-page figure the extractor skipped — appear as empty
    rows rather than vanishing. A page that silently disappears is a page you
    cannot later notice is missing.

    Remaining keyword arguments are copied onto every page (this is where the
    denormalized document-level columns come from).
    """
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    last = max(by_page, default=-1)
    total = max(n_pages, last + 1)

    pages = []
    for index in range(total):
        page_blocks = tuple(by_page.get(index, ()))
        text = render_markdown(
            page_blocks, drop_furniture=drop_furniture, region_refs=region_refs
        )
        pages.append(
            PageRecord(
                page_index=index,
                text=text,
                image_ids=_distinct_image_ids(page_blocks),
                blocks=page_blocks,
                doc_n_pages=total,
                **page_fields,  # type: ignore[arg-type]
            )
        )
    return tuple(pages)


def _distinct_image_ids(blocks: Sequence[Block]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for b in blocks:
        if b.image_id is not None:
            seen.setdefault(b.image_id, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def to_interleaved(text: str) -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    """Project a page's ``text`` onto the OBELICS / MINT-1T parallel arrays.

    Returns ``(images, texts)`` of equal length where exactly one side is
    non-null at each position, so code written against those datasets runs
    unchanged. Takes the *string*, not the blocks: the interleaving is encoded
    in the text, which is the whole point — this view survives a shard written
    without ``blocks``.
    """
    images: list[str | None] = []
    texts: list[str | None] = []
    cursor = 0
    for match in IMAGE_REF_RE.finditer(text):
        chunk = text[cursor : match.start()].strip()
        if chunk:
            images.append(None)
            texts.append(chunk)
        images.append(match.group(1))
        texts.append(None)
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        images.append(None)
        texts.append(tail)
    return tuple(images), tuple(texts)


def iter_pairs(
    pages: Iterable[PageRecord],
    *,
    context_window: int = 1,
    min_chars: int = 8,
) -> Iterator[ImageTextPair]:
    """Yield image–text pairs from one document's pages.

    Pass every page of a document together: ``mentions`` routinely point at a
    paragraph on the facing page, and a per-page call would silently drop
    those. Requires ``blocks`` — a shard written without them can only be
    paired by neighbouring text, which this function cannot see from ``text``
    alone.

    Precedence per image, highest first:

    ``content``
        The block's own transcription plus its caption — a table crop paired
        with its HTML. This is the table-recognition pair shape, and it beats
        the caption alone whenever the block has one.
    ``caption``
        Human-authored caption (+ footnote). Ground truth.
    ``alt``
        Model-generated description. Synthetic — filter with
        ``WHERE source != 'alt'`` if that matters to you.
    ``mention``
        Body text that references the figure by label.
    ``context``
        Adjacent text blocks. Weakest signal, last resort.

    Each image yields at most one pair, tagged with the tier that won.
    """
    pages = list(pages)
    by_idx = {b.idx: b for page in pages for b in page.blocks}
    order = sorted(by_idx)

    for page in pages:
        # In `pages` mode there are no crop blobs: a figure is a rectangle of
        # the page raster. Pair it anyway — the pixels are just as reachable,
        # and skipping them would silently drop every table-crop/HTML pair.
        has_raster = page.page_image_id is not None
        for b in page.blocks:
            region = (
                has_raster
                and b.image_id is None
                and b.bbox is not None
                and b.type in _PIXEL_BEARING_TYPES
            )
            if not (b.has_image or region):
                continue
            candidates: list[tuple[str, str | None]] = []
            if b.text:
                # A crop that also has a transcription (table → HTML). Pair the
                # image with caption + transcription together.
                candidates.append(
                    (
                        "content",
                        "\n\n".join(x for x in (b.caption, b.text, b.footnote) if x)
                        or None,
                    )
                )
            candidates += [
                (
                    "caption",
                    "\n".join(x for x in (b.caption, b.footnote) if x) or None,
                ),
                ("alt", b.alt),
                (
                    "mention",
                    "\n\n".join(
                        t
                        for i in b.mentions
                        if (t := (by_idx[i].text if i in by_idx else None))
                    )
                    or None,
                ),
                ("context", _neighbour_text(b, by_idx, order, context_window)),
            ]
            for source, text in candidates:
                if text and len(text.strip()) >= min_chars:
                    yield ImageTextPair(
                        doc_id=page.doc_id,
                        page_index=page.page_index,
                        block_idx=b.idx,
                        text=text.strip(),
                        source=source,
                        image_id=b.image_id,
                        bbox=None if b.image_id else b.bbox,
                    )
                    break


def _neighbour_text(
    block: Block,
    by_idx: Mapping[int, Block],
    order: Sequence[int],
    window: int,
) -> str | None:
    if window <= 0:
        return None
    pos = order.index(block.idx)
    picked: list[str] = []
    for offset in (*range(-window, 0), *range(1, window + 1)):
        j = pos + offset
        if not (0 <= j < len(order)):
            continue
        nb = by_idx[order[j]]
        if nb.type in (BlockType.TEXT, BlockType.TITLE, BlockType.LIST) and nb.text:
            picked.append(nb.text)
    return "\n\n".join(picked) or None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _join_lines(value: object) -> str | None:
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, (list, tuple)):
        lines = [s.strip() for s in value if isinstance(s, str) and s.strip()]
        return "\n".join(lines) or None
    return None


def _as_int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _norm_bbox(
    value: object, scale: float
) -> tuple[float, float, float, float] | None:
    """Rescale a bbox from its source grid onto [0, 1].

    Returns ``None`` when the input is malformed, out of range, or degenerate —
    a missing bbox is honest, a wrong one silently corrupts every downstream
    crop. Note the out-of-range check is a rejection, not a clamp: a box that
    does not fit the declared ``scale`` means the scale is wrong, and clamping
    would hide that.
    """
    if scale <= 0:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(v) / scale for v in value)
    except (TypeError, ValueError):
        return None
    if any(v < 0.0 or v > 1.0 for v in box):
        return None
    if box[2] < box[0] or box[3] < box[1]:
        return None
    return box  # type: ignore[return-value]
