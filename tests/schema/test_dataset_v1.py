"""Tests for the ``pdfsys.doc/v1`` format definition (pdfsys_core.dataset).

Covers the four things that make or break the format:

1. MinerU ``content_list`` → blocks is lossless for the fields we keep.
2. bboxes are always normalized or absent — never raw pixels.
3. The interleaved and pair views are faithful projections of ``blocks``.
4. ``page_ends`` really does slice ``text`` back into pages.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from pdfsys_core import (
    Block,
    BlockType,
    DocRecord,
    blocks_from_content_list,
    blocks_from_segments,
    image_id_for,
    iter_pairs,
    link_mentions,
    probe_image,
    render_markdown,
    to_interleaved,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# MinerU maps content_list bboxes onto a 0-1000 grid per axis, independent of
# page size — see test_bbox_scale_is_independent_of_page_size.
CONTENT_LIST = [
    {"type": "text", "text": "章节标题", "text_level": 1, "bbox": [100, 100, 900, 130], "page_idx": 0},
    {"type": "text", "text": "正文第一段，见图 1 的说明。", "bbox": [100, 150, 900, 180], "page_idx": 0},
    {
        "type": "image",
        "img_path": "images/a.jpg",
        "image_caption": ["图 1 系统架构"],
        "image_footnote": ["数据来源：内部"],
        "content": "A block diagram with three boxes",
        "bbox": [100, 200, 500, 400],
        "page_idx": 0,
    },
    {"type": "page_number", "text": "- 1 -", "bbox": [480, 975, 520, 990], "page_idx": 0},
    {
        "type": "table",
        "img_path": "images/b.jpg",
        "table_caption": ["表 2 结果"],
        "table_footnote": [],
        "table_body": "<table><tr><td>1</td></tr></table>",
        "bbox": [100, 50, 900, 350],
        "page_idx": 1,
    },
    {"type": "equation", "text": "E = mc^2", "bbox": [300, 400, 700, 450], "page_idx": 1},
    {"type": "footer", "text": "confidential", "bbox": [100, 950, 900, 975], "page_idx": 1},
]

IMAGE_IDS = {"images/a.jpg": "a" * 64, "images/b.jpg": "b" * 64}


@pytest.fixture
def blocks() -> tuple[Block, ...]:
    return blocks_from_content_list(CONTENT_LIST, image_ids=IMAGE_IDS)


# ---------------------------------------------------------------------------
# content_list → blocks
# ---------------------------------------------------------------------------


def test_block_types_and_heading_level(blocks):
    assert [b.type for b in blocks] == [
        BlockType.TITLE,
        BlockType.TEXT,
        BlockType.IMAGE,
        BlockType.PAGE_NUMBER,
        BlockType.TABLE,
        BlockType.FORMULA,
        BlockType.PAGE_FOOTER,
    ]
    assert blocks[0].level == 1
    assert blocks[1].level is None


def test_reading_order_index_is_dense_and_matches_position(blocks):
    assert [b.idx for b in blocks] == list(range(len(blocks)))


def test_image_block_keeps_caption_footnote_and_model_description(blocks):
    img = blocks[2]
    assert img.caption == "图 1 系统架构"
    assert img.footnote == "数据来源：内部"
    assert img.alt == "A block diagram with three boxes"
    # Pixels live in the images table, never in `text`.
    assert img.text is None
    assert img.image_id == "a" * 64


def test_table_block_carries_html_and_its_crop(blocks):
    table = blocks[4]
    assert table.text == "<table><tr><td>1</td></tr></table>"
    assert table.caption == "表 2 结果"
    assert table.footnote is None
    assert table.image_id == "b" * 64


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
    landscape = blocks_from_content_list(
        [{"type": "text", "text": "x", "bbox": [155, 567, 460, 767], "page_idx": 0}]
    )
    portrait = blocks_from_content_list(
        [{"type": "text", "text": "x", "bbox": [155, 567, 460, 767], "page_idx": 0}]
    )
    assert landscape[0].bbox == portrait[0].bbox == pytest.approx(
        (0.155, 0.567, 0.460, 0.767)
    )


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


def test_link_mentions_attaches_referencing_paragraph_to_the_figure(blocks):
    linked = link_mentions(blocks)
    figure = linked[2]
    assert figure.mentions == (1,), "paragraph 1 says '见图 1'"
    assert linked[1].mentions == ()


def test_link_mentions_matches_fullwidth_and_latin_labels():
    items = [
        {"type": "text", "text": "As shown in Fig. 3, the loss drops.", "page_idx": 0},
        {"type": "image", "img_path": "i", "image_caption": ["Figure 3: loss curve"], "page_idx": 0},
    ]
    linked = link_mentions(blocks_from_content_list(items, image_ids={"i": "c" * 64}))
    assert linked[1].mentions == (0,)


def test_link_mentions_does_not_link_a_figure_to_its_own_caption():
    items = [
        {"type": "image", "img_path": "i", "image_caption": ["图 1 x"], "page_idx": 0},
    ]
    linked = link_mentions(blocks_from_content_list(items, image_ids={"i": "c" * 64}))
    assert linked[0].mentions == ()


def test_link_mentions_is_a_noop_without_captions():
    items = [{"type": "text", "text": "见图 1", "page_idx": 0}]
    original = blocks_from_content_list(items)
    assert link_mentions(original) == original


# ---------------------------------------------------------------------------
# text rendering + page_ends
# ---------------------------------------------------------------------------


def test_render_drops_furniture_by_default(blocks):
    text, _ = render_markdown(blocks)
    assert "- 1 -" not in text
    assert "confidential" not in text


def test_render_keeps_furniture_when_asked(blocks):
    text, _ = render_markdown(blocks, drop_furniture=False)
    assert "confidential" in text


def test_render_uses_markdown_conventions(blocks):
    text, _ = render_markdown(blocks)
    assert text.startswith("# 章节标题")
    assert f"![](img://{'a' * 64})" in text
    assert "$$\nE = mc^2\n$$" in text


def test_render_emits_each_caption_exactly_once(blocks):
    text, _ = render_markdown(blocks)
    assert text.count("图 1 系统架构") == 1
    assert text.count("表 2 结果") == 1


def test_render_never_leaks_model_generated_descriptions_into_text(blocks):
    text, _ = render_markdown(blocks)
    assert "A block diagram with three boxes" not in text


def test_page_ends_slices_text_back_into_pages(blocks):
    text, page_ends = render_markdown(blocks)
    assert len(page_ends) == 2
    page0 = text[: page_ends[0]]
    page1 = text[page_ends[0] : page_ends[1]]
    assert "章节标题" in page0 and "E = mc^2" not in page0
    assert "E = mc^2" in page1 and "章节标题" not in page1
    assert page_ends[-1] == len(text)


def test_page_ends_has_one_entry_per_page_even_when_a_page_renders_empty():
    items = [
        {"type": "text", "text": "p0", "page_idx": 0},
        # page 1 contributes only furniture → renders to nothing
        {"type": "page_number", "text": "2", "page_idx": 1},
        {"type": "text", "text": "p2", "page_idx": 2},
    ]
    text, page_ends = render_markdown(blocks_from_content_list(items))
    assert len(page_ends) == 3
    assert text[: page_ends[0]].strip() == "p0"
    assert text[page_ends[0] : page_ends[1]].strip() == ""


# ---------------------------------------------------------------------------
# interleaved view (OBELICS / MINT-1T shape)
# ---------------------------------------------------------------------------


def test_interleaved_has_exactly_one_non_null_per_position(blocks):
    images, texts = to_interleaved(blocks)
    assert len(images) == len(texts)
    for img, txt in zip(images, texts, strict=True):
        assert (img is None) != (txt is None)


def test_interleaved_places_caption_immediately_after_its_image(blocks):
    images, texts = to_interleaved(blocks)
    pos = images.index("a" * 64)
    assert texts[pos + 1].startswith("图 1 系统架构")


def test_interleaved_drops_furniture(blocks):
    _, texts = to_interleaved(blocks)
    assert all("confidential" not in (t or "") for t in texts)


# ---------------------------------------------------------------------------
# image-text pairs
# ---------------------------------------------------------------------------


def _doc(blocks) -> DocRecord:
    text, page_ends = render_markdown(blocks)
    return DocRecord(id="d" * 64, blocks=blocks, text=text, page_ends=page_ends)


def test_pairs_prefer_human_caption(blocks):
    pairs = {p.image_id: p for p in iter_pairs(_doc(link_mentions(blocks)))}
    assert pairs["a" * 64].source == "caption"
    assert pairs["a" * 64].text.startswith("图 1 系统架构")


def test_pairs_fall_back_to_model_description_then_mention():
    items = [
        {"type": "text", "text": "结果见图 7，误差显著下降。", "page_idx": 0},
        {"type": "image", "img_path": "x", "content": "a scatter plot", "page_idx": 0},
        {"type": "image", "img_path": "y", "image_caption": ["图 7"], "page_idx": 0},
    ]
    blocks = link_mentions(
        blocks_from_content_list(items, image_ids={"x": "x" * 64, "y": "y" * 64})
    )
    pairs = {p.image_id: p for p in iter_pairs(_doc(blocks))}
    assert pairs["x" * 64].source == "alt"
    # "图 7" alone is shorter than min_chars, so the referencing paragraph wins.
    assert pairs["y" * 64].source == "mention"
    assert "误差显著下降" in pairs["y" * 64].text


def test_pairs_emit_at_most_one_row_per_image(blocks):
    doc = _doc(link_mentions(blocks))
    ids = [p.image_id for p in iter_pairs(doc)]
    assert len(ids) == len(set(ids)) == len(doc.image_ids)


def test_table_crop_pairs_with_its_html_transcription(blocks):
    pairs = {p.image_id: p for p in iter_pairs(_doc(link_mentions(blocks)))}
    table = pairs["b" * 64]
    assert table.source == "content"
    assert "表 2 结果" in table.text
    assert "<table>" in table.text


def test_pairs_skip_images_with_no_usable_text():
    items = [{"type": "image", "img_path": "x", "page_idx": 0}]
    blocks = blocks_from_content_list(items, image_ids={"x": "x" * 64})
    assert list(iter_pairs(_doc(blocks))) == []


# ---------------------------------------------------------------------------
# DocRecord derived columns
# ---------------------------------------------------------------------------


def test_image_ids_are_distinct_and_in_first_use_order(blocks):
    doc = _doc(blocks + blocks)  # same images referenced twice
    assert doc.image_ids == ("a" * 64, "b" * 64)


def test_count_by_block_type(blocks):
    doc = _doc(blocks)
    assert doc.count(BlockType.IMAGE, BlockType.CHART) == 1
    assert doc.count(BlockType.TABLE) == 1
    assert doc.count(BlockType.FORMULA) == 1


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
