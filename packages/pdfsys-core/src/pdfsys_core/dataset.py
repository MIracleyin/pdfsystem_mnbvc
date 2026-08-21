"""``pdfsys.doc/v1`` — the published dataset format (L2), stdlib-only.

This module defines *what a released document looks like*, independent of
Parquet. The Arrow schemas and the writer live in
:mod:`pdfsys_cli.dataset_writer`; everything here is a plain dataclass or a
pure function so that ``pdfsys-core`` keeps its zero-dependency guarantee
(``docs/golden-principles/ZERO_DEP_CORE.md``).

Design in one sentence
----------------------
**One row per document, with a single ordered ``blocks`` list — the reading
order IS the interleaving — plus a content-addressed side table for image
bytes.**

Why not the OBELICS / MINT-1T shape
-----------------------------------
OBELICS and MINT-1T-PDF store two parallel arrays (``images`` / ``texts``) in
which exactly one side is non-null at each position. That works for web pages,
where a document is literally a run of paragraphs and ``<img>`` tags, but it
throws away everything a PDF gives us for free: page number, bbox, heading
level, table structure, caption↔figure attachment. Half of every array is also
null padding.

A document is not a web page — so we keep one array of *typed blocks*. The
OBELICS view is a two-line projection over it (:func:`to_interleaved`), so
nothing is lost for consumers that want that shape; the reverse is not true.

Views (all derivable, none stored twice)
----------------------------------------
* **plain text** — ``DocRecord.text``, materialized because ~90 % of consumers
  are text-only and re-deriving it costs a full nested-column read.
* **interleaved** — :func:`to_interleaved`, OBELICS/MINT-1T shape.
* **image–text pairs** — :func:`iter_pairs`, caption / model description /
  figure-referencing body text (the last one following PMC-InterCPT, which
  found reference context materially better than captions alone).

See ``docs/superpowers/specs/2026-08-22-interleaved-parquet-dataset-design.md``
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
    "Block",
    "ImageBlob",
    "DocRecord",
    "ImageTextPair",
    "image_id_for",
    "probe_image",
    "blocks_from_content_list",
    "blocks_from_segments",
    "link_mentions",
    "render_markdown",
    "to_interleaved",
    "iter_pairs",
]

#: Bumped on any breaking change to the block/doc field set. Written into the
#: Parquet file-level key-value metadata by the writer.
DATASET_SCHEMA_VERSION = "pdfsys.doc/1"


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


@dataclass(frozen=True, slots=True)
class Block:
    """One block-level unit in document reading order.

    ``text`` encoding follows the same contract as :class:`pdfsys_core.Segment`
    so no re-interpretation is needed across the L1→L2 boundary:
    TEXT/TITLE/LIST = Markdown, TABLE = HTML, FORMULA = LaTeX, IMAGE/CHART =
    ``None`` (the pixels live in ``image_id``).
    """

    idx: int
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
    #: Indices of blocks in the same document whose text references this
    #: figure/table ("as shown in Fig. 3"). Empty when unknown.
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
class DocRecord:
    """One published document — one row of the ``documents`` table."""

    #: SHA-256 of the source PDF bytes. Same identity as ``ExtractedDoc.sha256``.
    id: str
    blocks: tuple[Block, ...]
    #: Markdown rendering of ``blocks`` (furniture dropped, images as
    #: ``![caption](img://<image_id>)``). Redundant with ``blocks`` by design.
    text: str = ""
    #: Character offset in ``text`` at which each page ends. Same trick as
    #: FinePDFs' ``page_ends`` — recovers page granularity without a second
    #: copy of the text.
    page_ends: tuple[int, ...] = ()
    source_uri: str = ""
    backend: str = ""  # mupdf | pipeline | vlm
    n_pages: int = 0
    lang: str = ""
    lang_score: float | None = None
    quality_score: float | None = None
    quality_model: str = ""
    router_ocr_prob: float | None = None
    provenance: str = ""  # opaque upstream JSON (license, crawl batch, ...)

    @property
    def image_ids(self) -> tuple[str, ...]:
        """Distinct image ids referenced by this document, in first-use order."""
        seen: dict[str, None] = {}
        for b in self.blocks:
            if b.image_id is not None:
                seen.setdefault(b.image_id, None)
        return tuple(seen)

    def count(self, *types: BlockType) -> int:
        wanted = set(types)
        return sum(1 for b in self.blocks if b.type in wanted)


@dataclass(frozen=True, slots=True)
class ImageTextPair:
    """One image–text pair extracted from a document."""

    doc_id: str
    image_id: str
    block_idx: int
    page: int
    text: str
    #: Where ``text`` came from: ``caption`` | ``alt`` | ``mention`` | ``context``.
    source: str


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
    page_sizes: Sequence[tuple[float, float]] | None = None,
    image_ids: Mapping[str, str] | None = None,
) -> tuple[Block, ...]:
    """Convert MinerU's ``content_list.json`` into blocks.

    ``page_sizes`` comes from ``middle.json``'s ``pdf_info[i].page_size`` and is
    what MinerU's pixel bboxes are relative to; without it bboxes are dropped
    rather than emitted unnormalized (repo convention: bbox is always in
    [0, 1]). ``image_ids`` maps MinerU's ``img_path`` to the content address of
    the corresponding blob.
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

        page = _as_int(item.get("page_idx"), default=0)
        img_path = item.get("img_path")
        image_id = None
        if isinstance(img_path, str) and img_path:
            image_id = (image_ids or {}).get(img_path)

        blocks.append(
            Block(
                idx=idx,
                page=page,
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
                bbox=_norm_bbox(item.get("bbox"), page, page_sizes),
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
# Figure-mention linking
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
# Views
# ---------------------------------------------------------------------------


def render_markdown(
    blocks: Sequence[Block], *, drop_furniture: bool = True
) -> tuple[str, tuple[int, ...]]:
    """Render blocks to Markdown; return ``(text, page_ends)``.

    ``page_ends[i]`` is the character offset in ``text`` at which page ``i``
    ends, so a consumer can slice the text back into pages without a second
    stored copy (FinePDFs uses the same encoding).
    """
    parts: list[str] = []
    page_ends: list[int] = []
    length = 0
    current_page = 0

    def close_pages_through(page: int) -> None:
        nonlocal current_page
        while current_page < page:
            page_ends.append(length)
            current_page += 1

    for b in blocks:
        if drop_furniture and b.is_furniture:
            close_pages_through(b.page)
            continue
        chunk = _render_block(b)
        if not chunk:
            continue
        close_pages_through(b.page)
        parts.append(chunk)
        # +2 for the "\n\n" separator that join() will insert after this chunk.
        length += len(chunk) + 2

    text = "\n\n".join(parts)
    # The final page ends at the end of the text; so does every trailing page
    # that produced no renderable block.
    page_ends.append(len(text))
    return text, tuple(page_ends)


def _render_block(b: Block) -> str:
    if b.type is BlockType.TITLE:
        level = min(max(b.level or 1, 1), 6)
        return f"{'#' * level} {b.text}".strip() if b.text else ""
    if b.type is BlockType.FORMULA:
        return f"$$\n{b.text}\n$$" if b.text else ""
    if b.type in (BlockType.IMAGE, BlockType.CHART):
        alt = (b.caption or b.alt or "").replace("]", "\\]").replace("\n", " ")
        ref = f"img://{b.image_id}" if b.image_id else ""
        lines = [f"![{alt}]({ref})"]
        if b.caption:
            lines.append(b.caption)
        if b.footnote:
            lines.append(b.footnote)
        return "\n\n".join(lines)
    if b.type is BlockType.TABLE:
        lines = [x for x in (b.caption, b.text, b.footnote) if x]
        return "\n\n".join(lines)
    return b.text or ""


def to_interleaved(
    blocks: Sequence[Block], *, drop_furniture: bool = True
) -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    """Project blocks onto the OBELICS / MINT-1T parallel-array shape.

    Returns ``(images, texts)`` of equal length where exactly one side is
    non-null at each position. Provided so that code written against those
    datasets runs unchanged on ours.
    """
    images: list[str | None] = []
    texts: list[str | None] = []
    for b in blocks:
        if drop_furniture and b.is_furniture:
            continue
        if b.has_image:
            images.append(b.image_id)
            texts.append(None)
            # The caption is text that belongs *after* the image, exactly as
            # PMC-InterCPT lays out its samples.
            caption = "\n".join(x for x in (b.caption, b.footnote) if x)
            if caption:
                images.append(None)
                texts.append(caption)
            continue
        chunk = _render_block(b)
        if chunk:
            images.append(None)
            texts.append(chunk)
    return tuple(images), tuple(texts)


def iter_pairs(
    doc: DocRecord,
    *,
    context_window: int = 1,
    min_chars: int = 8,
) -> Iterator[ImageTextPair]:
    """Yield image–text pairs, best text source first.

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
        Body text that references the figure by label (PMC-InterCPT's
        observation that this beats captions for explanatory content).
    ``context``
        Adjacent text blocks. Weakest signal, last resort.

    Each image yields at most one pair, tagged with the tier that won.
    """
    by_idx = {b.idx: b for b in doc.blocks}
    order = [b.idx for b in doc.blocks]

    for b in doc.blocks:
        if not b.has_image:
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
            ("caption", "\n".join(x for x in (b.caption, b.footnote) if x) or None),
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
                    doc_id=doc.id,
                    image_id=b.image_id,  # type: ignore[arg-type]
                    block_idx=b.idx,
                    page=b.page,
                    text=text.strip(),
                    source=source,
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
    value: object,
    page: int,
    page_sizes: Sequence[tuple[float, float]] | None,
) -> tuple[float, float, float, float] | None:
    """Normalize MinerU's pixel bbox against the page it came from.

    Returns ``None`` when the page size is unknown or the box is degenerate —
    a missing bbox is honest, an unnormalized one silently corrupts every
    downstream crop.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not page_sizes or not (0 <= page < len(page_sizes)):
        return None
    width, height = page_sizes[page]
    if width <= 0 or height <= 0:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    box = (
        min(max(x0 / width, 0.0), 1.0),
        min(max(y0 / height, 0.0), 1.0),
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
    )
    if box[2] < box[0] or box[3] < box[1]:
        return None
    return box
