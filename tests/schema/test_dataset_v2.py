"""Tests for the ``pdfsys.page/v2`` format definition (pdfsys_core.dataset).

The promises that make or break the format:

1. The row unit is the page, and no page ever silently disappears.
2. The image/text interleaving lives in ``text`` — every view of it must
   survive a shard written without ``blocks``.
3. MinerU ``content_list`` → blocks is lossless for the fields we keep.
4. bboxes are always normalized or absent — never raw source coordinates.
5. Figure mentions link across page boundaries.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from pdfsys_core import (
    IMAGE_REF_RE,
    Block,
    BlockType,
    PageRecord,
    bbox_ref,
    blocks_from_content_list,
    blocks_from_segments,
    crop_region,
    image_id_for,
    image_ref,
    iter_pairs,
    link_mentions,
    parse_image_ref,
    probe_image,
    render_markdown,
    split_pages,
    strip_image_refs,
    to_interleaved,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

IMG_A = "a" * 64
IMG_B = "b" * 64

# MinerU maps content_list bboxes onto a 0-1000 grid per axis, independent of
# page size — see test_bbox_scale_is_independent_of_page_size.
CONTENT_LIST = [
    # -- page 0 ------------------------------------------------------------
    {"type": "text", "text": "章节标题", "text_level": 1, "bbox": [100, 100, 900, 130], "page_idx": 0},
    {"type": "text", "text": "正文第一段，见图 1 的说明。", "bbox": [100, 150, 900, 180], "page_idx": 0},
    {"type": "page_number", "text": "- 1 -", "bbox": [480, 975, 520, 990], "page_idx": 0},
    # -- page 1: the figure the previous page referred to -------------------
    {
        "type": "image",
        "img_path": "images/a.jpg",
        "image_caption": ["图 1 系统架构"],
        "image_footnote": ["数据来源：内部"],
        "content": "A block diagram with three boxes",
        "bbox": [100, 200, 500, 400],
        "page_idx": 1,
    },
    {
        "type": "table",
        "img_path": "images/b.jpg",
        "table_caption": ["表 2 结果"],
        "table_footnote": [],
        "table_body": "<table><tr><td>1</td></tr></table>",
        "bbox": [100, 450, 900, 700],
        "page_idx": 1,
    },
    {"type": "equation", "text": "E = mc^2", "bbox": [300, 750, 700, 800], "page_idx": 1},
    {"type": "footer", "text": "confidential", "bbox": [100, 950, 900, 975], "page_idx": 1},
]

IMAGE_IDS = {"images/a.jpg": IMG_A, "images/b.jpg": IMG_B}


@pytest.fixture
def blocks() -> tuple[Block, ...]:
    return link_mentions(blocks_from_content_list(CONTENT_LIST, image_ids=IMAGE_IDS))


@pytest.fixture
def pages(blocks) -> tuple[PageRecord, ...]:
    return split_pages(blocks, doc_id="d" * 64, extractor="vlm")


# ---------------------------------------------------------------------------
# the row unit is the page
# ---------------------------------------------------------------------------


def test_one_row_per_page_keyed_by_doc_and_index(pages):
    assert [p.page_index for p in pages] == [0, 1]
    assert {p.doc_id for p in pages} == {"d" * 64}


def test_blocks_land_on_their_own_page(pages):
    assert "章节标题" in pages[0].text
    assert "章节标题" not in pages[1].text
    assert "E = mc^2" in pages[1].text
    assert "E = mc^2" not in pages[0].text


def test_page_carries_only_its_own_image_ids(pages):
    assert pages[0].image_ids == ()
    assert pages[1].image_ids == (IMG_A, IMG_B)


def test_doc_n_pages_is_stamped_on_every_page(pages):
    assert all(p.doc_n_pages == 2 for p in pages)


def test_a_page_with_no_blocks_still_produces_a_row():
    """A blank scan or a page the extractor skipped must be visible as an
    empty row — a page that vanishes is one you cannot notice is missing."""
    items = [{"type": "text", "text": "p0", "page_idx": 0},
             {"type": "text", "text": "p2", "page_idx": 2}]
    pages = split_pages(blocks_from_content_list(items), doc_id="x" * 64)
    assert [p.page_index for p in pages] == [0, 1, 2]
    assert pages[1].text == "" and pages[1].blocks == ()


def test_trailing_blank_pages_come_from_the_pdf_page_count():
    items = [{"type": "text", "text": "p0", "page_idx": 0}]
    pages = split_pages(blocks_from_content_list(items), n_pages=4, doc_id="x" * 64)
    assert [p.page_index for p in pages] == [0, 1, 2, 3]
    assert all(p.doc_n_pages == 4 for p in pages)


def test_declared_page_count_never_truncates_observed_pages():
    items = [{"type": "text", "text": "p5", "page_idx": 5}]
    pages = split_pages(blocks_from_content_list(items), n_pages=2, doc_id="x" * 64)
    assert len(pages) == 6, "a block on page 5 outranks a page_count of 2"


# ---------------------------------------------------------------------------
# the interleaving lives in `text`
# ---------------------------------------------------------------------------


def test_text_carries_the_image_marker(pages):
    assert image_ref(IMG_A) in pages[1].text
    assert IMAGE_REF_RE.findall(pages[1].text) == [f"img://{IMG_A}"]


def test_interleaved_view_needs_only_the_string(pages):
    """The whole point of the v2 encoding: this view must work on a shard
    written with --no-blocks."""
    images, texts = to_interleaved(pages[1].text)
    assert len(images) == len(texts)
    for img, txt in zip(images, texts, strict=True):
        assert (img is None) != (txt is None)
    refs = [parse_image_ref(i) for i in images if i]
    assert [r.image_id for r in refs] == [IMG_A]


def test_interleaved_places_caption_immediately_after_its_image(pages):
    images, texts = to_interleaved(pages[1].text)
    pos = images.index(f"img://{IMG_A}")
    assert texts[pos + 1].startswith("图 1 系统架构")


def test_interleaved_of_pure_text_is_a_single_chunk():
    images, texts = to_interleaved("just words")
    assert images == (None,)
    assert texts == ("just words",)


def test_interleaved_of_empty_text_is_empty():
    assert to_interleaved("") == ((), ())


def test_strip_image_refs_keeps_captions():
    text = f"before\n\n{image_ref(IMG_A)}\n\n图 1 系统架构\n\nafter"
    plain = strip_image_refs(text)
    assert "img://" not in plain
    assert "图 1 系统架构" in plain
    assert plain.startswith("before") and plain.endswith("after")


def test_image_ref_and_regex_are_inverses():
    ref = parse_image_ref(IMAGE_REF_RE.fullmatch(image_ref(IMG_A)).group(1))
    assert (ref.kind, ref.image_id) == ("blob", IMG_A)


# ---------------------------------------------------------------------------
# region references: a figure addressed as a rectangle of the page raster
# ---------------------------------------------------------------------------

BOX = (0.134, 0.222, 0.874, 0.369)


def test_bbox_ref_and_regex_are_inverses():
    ref = parse_image_ref(IMAGE_REF_RE.fullmatch(bbox_ref(BOX)).group(1))
    assert ref.kind == "region"
    assert ref.bbox == pytest.approx(BOX)


def test_renderer_emits_a_region_ref_when_no_crop_was_stored():
    """images="pages" drops the crop blob and lets the bbox address the
    pixels — MinerU's crops are sub-rectangles of a 200-dpi page render, so
    storing both is storing the same pixels twice."""
    block = Block(idx=0, page=0, type=BlockType.IMAGE, bbox=BOX, caption="图 1 x")
    text = render_markdown([block])
    (ref,) = [parse_image_ref(r) for r in IMAGE_REF_RE.findall(text)]
    assert ref.kind == "region"
    assert ref.bbox == pytest.approx(BOX)
    assert "图 1 x" in text, "the caption still follows the marker"


def test_renderer_prefers_the_blob_ref_when_a_crop_exists():
    block = Block(idx=0, page=0, type=BlockType.IMAGE, bbox=BOX, image_id=IMG_A)
    (ref,) = [parse_image_ref(r) for r in IMAGE_REF_RE.findall(render_markdown([block]))]
    assert (ref.kind, ref.image_id) == ("blob", IMG_A)


def test_interleaved_and_strip_handle_region_refs():
    text = f"before\n\n{bbox_ref(BOX)}\n\nafter"
    images, _ = to_interleaved(text)
    assert [i for i in images if i] == [bbox_ref(BOX)[4:-1]]
    assert "bbox://" not in strip_image_refs(text)
    assert strip_image_refs(text) == "before\n\nafter"


def test_crop_region_reproduces_the_crop_mineru_would_have_stored():
    """A 558x773pt page at 200 dpi is 1550x2147 px; MinerU's stored crop for
    this bbox is 1148x318.

    We land on 1147x315 — off by 1 and 3 px. That is the bbox quantization,
    not a bug: bboxes live on a 0-1000 integer grid, so each edge carries up
    to +/-0.5/1000 of the page, about 1 px per edge at this resolution. Worth
    asserting the size of the gap so a real regression cannot hide inside it.
    """
    left, top, right, bottom = crop_region(BOX, 1550, 2147)
    assert abs((right - left) - 1148) <= 4
    assert abs((bottom - top) - 318) <= 4


@pytest.mark.parametrize("bad", ["img://short", "bbox://1,2,3", "http://x", "bbox://a,b,c,d"])
def test_parse_image_ref_rejects_junk(bad):
    assert parse_image_ref(bad) is None or parse_image_ref(bad).bbox is None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_render_drops_furniture_by_default(pages):
    assert "- 1 -" not in pages[0].text
    assert "confidential" not in pages[1].text


def test_render_keeps_furniture_when_asked(blocks):
    text = render_markdown([b for b in blocks if b.page == 1], drop_furniture=False)
    assert "confidential" in text


def test_render_uses_markdown_conventions(pages):
    assert pages[0].text.startswith("# 章节标题")
    assert "$$\nE = mc^2\n$$" in pages[1].text


def test_render_emits_each_caption_exactly_once(pages):
    assert pages[1].text.count("图 1 系统架构") == 1
    assert pages[1].text.count("表 2 结果") == 1


def test_render_never_leaks_model_generated_descriptions_into_text(pages):
    assert "A block diagram with three boxes" not in pages[1].text


# ---------------------------------------------------------------------------
# content_list → blocks
# ---------------------------------------------------------------------------


def test_block_types_and_heading_level(blocks):
    assert [b.type for b in blocks] == [
        BlockType.TITLE,
        BlockType.TEXT,
        BlockType.PAGE_NUMBER,
        BlockType.IMAGE,
        BlockType.TABLE,
        BlockType.FORMULA,
        BlockType.PAGE_FOOTER,
    ]
    assert blocks[0].level == 1
    assert blocks[1].level is None


def test_block_idx_is_document_scoped_not_page_scoped(blocks):
    """mentions point at idx, and a mention routinely crosses a page break —
    so idx has to be unique across the document, not restart per page."""
    assert [b.idx for b in blocks] == list(range(len(blocks)))
    page1 = [b.idx for b in blocks if b.page == 1]
    assert min(page1) == 3, "page 1 continues the document's numbering"


def test_image_block_keeps_caption_footnote_and_model_description(blocks):
    img = blocks[3]
    assert img.caption == "图 1 系统架构"
    assert img.footnote == "数据来源：内部"
    assert img.alt == "A block diagram with three boxes"
    # Pixels live in the images table, never in `text`.
    assert img.text is None
    assert img.image_id == IMG_A


def test_table_block_carries_html_and_its_crop(blocks):
    table = blocks[4]
    assert table.text == "<table><tr><td>1</td></tr></table>"
    assert table.caption == "表 2 结果"
    assert table.footnote is None
    assert table.image_id == IMG_B


def test_unknown_mineru_type_falls_back_to_text_without_losing_content():
    (block,) = blocks_from_content_list([{"type": "wat", "text": "hi", "page_idx": 0}])
    assert block.type is BlockType.TEXT
    assert block.text == "hi"


# ---------------------------------------------------------------------------
# bbox normalization
# ---------------------------------------------------------------------------


def test_bbox_is_rescaled_from_the_0_1000_grid(blocks):
    assert blocks[0].bbox == pytest.approx((0.1, 0.1, 0.9, 0.13))


def test_all_bboxes_are_within_unit_square(blocks):
    for b in blocks:
        if b.bbox is not None:
            assert all(0.0 <= v <= 1.0 for v in b.bbox), b


def test_bbox_scale_is_independent_of_page_size():
    """Regression: MinerU bboxes are NOT pixels relative to middle.json's
    page_size. Observed in real output: page_size=[558, 773] with bboxes
    reaching 940 — normalizing against page_size clamps everything to 1.0.
    """
    box = [155, 567, 460, 767]
    (block,) = blocks_from_content_list(
        [{"type": "text", "text": "x", "bbox": box, "page_idx": 0}]
    )
    assert block.bbox == pytest.approx((0.155, 0.567, 0.460, 0.767))


def test_bbox_rejected_rather_than_clamped_when_out_of_scale():
    (block,) = blocks_from_content_list(
        [{"type": "text", "text": "x", "bbox": [100, 200, 300, 1400], "page_idx": 0}]
    )
    assert block.bbox is None, "a box past the declared scale means the scale is wrong"


def test_bbox_scale_is_configurable():
    items = [{"type": "text", "text": "x", "bbox": [50, 100, 150, 200], "page_idx": 0}]
    (block,) = blocks_from_content_list(items, bbox_scale=200.0)
    assert block.bbox == pytest.approx((0.25, 0.5, 0.75, 1.0))


@pytest.mark.parametrize(
    "bad", [None, [1, 2, 3], "nope", [1, 2, "x", 4], [500, 100, 100, 200]]
)
def test_malformed_bbox_becomes_null(bad):
    (block,) = blocks_from_content_list(
        [{"type": "text", "text": "x", "bbox": bad, "page_idx": 0}]
    )
    assert block.bbox is None


# ---------------------------------------------------------------------------
# figure-mention linking
# ---------------------------------------------------------------------------


def test_mentions_link_across_a_page_break(blocks):
    """The figure is on page 1; the paragraph that discusses it is on page 0."""
    figure = blocks[3]
    assert figure.page == 1
    assert figure.mentions == (1,)
    assert blocks[1].page == 0, "the mentioning paragraph is on the previous page"


def test_mentions_survive_the_page_split(pages):
    figure = next(b for b in pages[1].blocks if b.image_id == IMG_A)
    assert figure.mentions == (1,)
    # The referenced block is reachable from the sibling page row.
    assert any(b.idx == 1 for b in pages[0].blocks)


def test_link_mentions_matches_fullwidth_and_latin_labels():
    items = [
        {"type": "text", "text": "As shown in Fig. 3, the loss drops.", "page_idx": 0},
        {"type": "image", "img_path": "i", "image_caption": ["Figure 3: loss curve"], "page_idx": 0},
    ]
    linked = link_mentions(blocks_from_content_list(items, image_ids={"i": "c" * 64}))
    assert linked[1].mentions == (0,)


def test_link_mentions_does_not_link_a_figure_to_its_own_caption():
    items = [{"type": "image", "img_path": "i", "image_caption": ["图 1 x"], "page_idx": 0}]
    linked = link_mentions(blocks_from_content_list(items, image_ids={"i": "c" * 64}))
    assert linked[0].mentions == ()


def test_link_mentions_is_a_noop_without_captions():
    items = [{"type": "text", "text": "见图 1", "page_idx": 0}]
    original = blocks_from_content_list(items)
    assert link_mentions(original) == original


# ---------------------------------------------------------------------------
# image-text pairs
# ---------------------------------------------------------------------------


def test_pairs_prefer_human_caption(pages):
    by_image = {p.image_id: p for p in iter_pairs(pages)}
    assert by_image[IMG_A].source == "caption"
    assert by_image[IMG_A].text.startswith("图 1 系统架构")


def test_table_crop_pairs_with_its_html_transcription(pages):
    table = {p.image_id: p for p in iter_pairs(pages)}[IMG_B]
    assert table.source == "content"
    assert "表 2 结果" in table.text
    assert "<table>" in table.text


def test_pairs_carry_the_page_they_came_from(pages):
    assert all(p.page_index == 1 for p in iter_pairs(pages))


def test_pairs_resolve_mentions_that_live_on_another_page():
    items = [
        {"type": "text", "text": "结果见图 7，误差显著下降。", "page_idx": 0},
        {"type": "image", "img_path": "y", "image_caption": ["图 7"], "page_idx": 1},
    ]
    blocks = link_mentions(blocks_from_content_list(items, image_ids={"y": "y" * 64}))
    pages = split_pages(blocks, doc_id="z" * 64)
    pair = next(iter_pairs(pages))
    assert pair.source == "mention"
    assert "误差显著下降" in pair.text
    # Passing only the figure's page cannot reach the mentioning paragraph, and
    # "图 7" alone is too short to stand as a caption — so the pair is lost
    # entirely. That is why iter_pairs takes a document, not a page.
    assert list(iter_pairs(pages[1:])) == []


def test_pairs_fall_back_to_model_description():
    items = [{"type": "image", "img_path": "x", "content": "a scatter plot", "page_idx": 0}]
    pages = split_pages(
        blocks_from_content_list(items, image_ids={"x": "x" * 64}), doc_id="z" * 64
    )
    assert next(iter_pairs(pages)).source == "alt"


def test_pairs_emit_at_most_one_row_per_image(pages):
    ids = [p.image_id for p in iter_pairs(pages)]
    assert len(ids) == len(set(ids)) == 2


def test_pairs_survive_dropping_the_crop_blobs(pages):
    """images="pages" nulls image_id and lets bbox address the pixels. Pairs
    must keep coming — otherwise every table-crop/HTML pair silently vanishes
    the moment you stop storing crops."""
    import dataclasses

    stripped = tuple(
        dataclasses.replace(
            p,
            page_image_id="c" * 64,
            render_dpi=200,
            image_ids=(),
            blocks=tuple(
                dataclasses.replace(b, image_id=None) if b.image_id else b
                for b in p.blocks
            ),
        )
        for p in pages
    )
    crops = {(p.image_id, p.source) for p in iter_pairs(pages)}
    regions = list(iter_pairs(stripped))
    assert len(regions) == len(crops) == 2
    assert {p.source for p in regions} == {s for _, s in crops}
    assert all(p.image_id is None and p.bbox is not None for p in regions)


def test_pairs_are_not_invented_for_regions_without_a_raster(pages):
    """No page raster and no crop means no reachable pixels — pointing a pair
    at a rectangle of an image that does not exist would be worse than none."""
    import dataclasses

    stripped = tuple(
        dataclasses.replace(
            p,
            blocks=tuple(
                dataclasses.replace(b, image_id=None) if b.image_id else b
                for b in p.blocks
            ),
        )
        for p in pages
    )
    assert list(iter_pairs(stripped)) == []


def test_pairs_skip_images_with_no_usable_text():
    items = [{"type": "image", "img_path": "x", "page_idx": 0}]
    pages = split_pages(
        blocks_from_content_list(items, image_ids={"x": "x" * 64}), doc_id="z" * 64
    )
    assert list(iter_pairs(pages)) == []


# ---------------------------------------------------------------------------
# PageRecord derived values
# ---------------------------------------------------------------------------


def test_n_chars_tracks_text(pages):
    assert pages[0].n_chars == len(pages[0].text)


def test_count_by_block_type(pages):
    assert pages[1].count(BlockType.IMAGE, BlockType.CHART) == 1
    assert pages[1].count(BlockType.TABLE) == 1
    assert pages[1].count(BlockType.FORMULA) == 1
    assert pages[0].count(BlockType.TABLE) == 0


# ---------------------------------------------------------------------------
# segments (mupdf lane)
# ---------------------------------------------------------------------------


def test_blocks_from_segments_uses_already_normalized_bboxes():
    from pdfsys_core import Backend, BBox, RegionType, Segment

    segs = (
        Segment(
            index=0,
            backend=Backend.MUPDF,
            page_index=3,
            type=RegionType.TEXT,
            content="hello",
            bbox=BBox(0.1, 0.2, 0.3, 0.4),
        ),
    )
    (block,) = blocks_from_segments(segs)
    assert block.type is BlockType.TEXT
    assert block.page == 3
    assert block.bbox == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_segments_split_into_pages_by_page_index():
    from pdfsys_core import Backend, RegionType, Segment

    segs = tuple(
        Segment(index=i, backend=Backend.MUPDF, page_index=i // 2,
                type=RegionType.TEXT, content=f"s{i}")
        for i in range(4)
    )
    pages = split_pages(blocks_from_segments(segs), doc_id="m" * 64)
    assert [p.page_index for p in pages] == [0, 1]
    assert pages[0].text == "s0\n\ns1"


# ---------------------------------------------------------------------------
# image identity + header probing (no Pillow — core is zero-dep)
# ---------------------------------------------------------------------------


def _png(width: int, height: int) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x00" * 3
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_png(7, 11), ("png", 7, 11)),
        (_jpeg(640, 480), ("jpeg", 640, 480)),
        (b"GIF89a" + struct.pack("<HH", 3, 5) + b"\x00" * 8, ("gif", 3, 5)),
    ],
)
def test_probe_image_reads_dimensions_from_headers(data, expected):
    assert probe_image(data) == expected


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8truncated"])
def test_probe_image_never_raises_on_garbage(data):
    fmt, w, h = probe_image(data)
    assert (w, h) == (0, 0)
    assert fmt in ("unknown", "jpeg")


def test_image_id_is_content_addressed():
    assert image_id_for(b"abc") == image_id_for(b"abc")
    assert image_id_for(b"abc") != image_id_for(b"abd")
    assert len(image_id_for(b"abc")) == 64
