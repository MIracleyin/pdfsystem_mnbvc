"""Tests for the three image modes on the MinerU lane.

``crops``/``pages``/``none`` decide which pixels a shard stores, and the text
column has to stay consistent with that choice. An image block renders as
``![](img://…)`` when its crop is stored and ``![](bbox://…)`` when the pixels
are to be cut out of a page raster instead — so under ``none``, where neither
exists, it must render as neither. Emitting the region marker anyway produced
a shard that ``dataset-validate`` rejects: every marker pointed into a raster
the shard does not contain.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from pdfsys_cli.dataset_build import build_from_mineru_dir
from pdfsys_core import IMAGE_REF_RE

DOC = "a" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"crop-bytes" * 4

CONTENT_LIST = [
    {"type": "text", "text": "开头一段", "page_idx": 0},
    {
        "type": "image",
        "img_path": "images/fig1.jpg",
        "image_caption": ["图 1 示意"],
        "bbox": [100, 200, 800, 600],
        "page_idx": 0,
    },
]


@pytest.fixture
def mineru_dir(tmp_path):
    """One document in the on-disk shape our parsers persist."""
    d = tmp_path / DOC
    (d / "images").mkdir(parents=True)
    (d / "images" / "fig1.jpg").write_bytes(JPEG)
    (d / f"{DOC}_content_list.json").write_text(
        json.dumps(CONTENT_LIST), encoding="utf-8"
    )
    (d / f"{DOC}_middle.json").write_text(
        json.dumps(
            {
                "backend": "pipeline",
                "pdf_info": [{"page_size": [612.0, 792.0]}],
            }
        ),
        encoding="utf-8",
    )
    return d


def _markers(pages):
    return [m for p in pages for m in IMAGE_REF_RE.findall(p.text)]


def test_crops_mode_addresses_the_stored_blob(mineru_dir):
    pages, blobs = build_from_mineru_dir(mineru_dir, images="crops")
    assert [b.image_id for b in blobs] == [hashlib.sha256(JPEG).hexdigest()]
    assert _markers(pages) == [f"img://{hashlib.sha256(JPEG).hexdigest()}"]


def test_pages_mode_addresses_the_region(mineru_dir):
    pages, blobs = build_from_mineru_dir(mineru_dir, images="pages")
    assert blobs == []
    (marker,) = _markers(pages)
    assert marker.startswith("bbox://")


def test_none_mode_emits_no_marker_at_all(mineru_dir):
    """Neither a blob to point at nor a raster to cut from — so no marker."""
    pages, blobs = build_from_mineru_dir(mineru_dir, images="none")
    assert blobs == []
    assert _markers(pages) == []


def test_none_mode_keeps_the_block_and_its_bbox(mineru_dir):
    """The pixels stay reconstructible from the source PDF, so don't drop them."""
    pages, _ = build_from_mineru_dir(mineru_dir, images="none")
    images = [b for p in pages for b in p.blocks if b.type.value == "image"]
    assert len(images) == 1
    assert images[0].bbox is not None
    assert images[0].image_id is None


def test_none_mode_keeps_the_caption(mineru_dir):
    """Dropping the marker must not drop the human text that followed it."""
    pages, _ = build_from_mineru_dir(mineru_dir, images="none")
    assert "图 1 示意" in pages[0].text
