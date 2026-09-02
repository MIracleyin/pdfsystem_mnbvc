"""Build ``pdfsys.page/v2`` page rows from what the pipeline already writes.

Two entry points, one per lane:

* :func:`build_from_mineru_dir` — the pipeline/vlm lane. MinerU's
  ``content_list.json`` is already a reading-order interleaved list with
  captions, table HTML and image crops; ``middle.json`` supplies the page
  count and each page's size in PDF points. This is a pure re-encoding, not a
  re-parse.
* :func:`build_from_extracted` — the mupdf fast lane, where
  ``ExtractedDoc.segments`` is the authority and there are no images.
  :func:`build_from_pdf` wraps it for callers holding a PDF rather than an
  already-extracted document.

Both return ``(pages, blobs)`` ready for
:class:`pdfsys_cli.dataset_writer.DatasetWriter`. Full-page rasters are opt-in
via :func:`render_page_images`, which needs the source PDF — they are
rebuildable from L0 at any DPI, so the default is not to build them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pdfsys_core import (
    ImageBlob,
    PageRecord,
    blocks_from_content_list,
    blocks_from_segments,
    image_id_for,
    link_mentions,
    probe_image,
    split_pages,
)

_LOG = logging.getLogger(__name__)

#: How image pixels are stored. These are mutually exclusive on purpose:
#: MinerU's crops are sub-rectangles of a 200-dpi page render (measured at
#: 200.1 dpi across 172 crops in 19 documents), so keeping both tables means
#: storing the same pixels twice.
IMAGE_MODES = ("crops", "pages", "none")

__all__ = [
    "IMAGE_MODES",
    "build_from_mineru_dir",
    "build_from_extracted",
    "build_from_pdf",
    "iter_mineru_dirs",
    "iter_pdfs",
    "select_documents",
    "select_pdfs",
    "page_geometry_from_pdf",
    "render_page_images",
]


def iter_mineru_dirs(root: Path) -> Iterator[Path]:
    """Yield every directory under ``root`` holding a MinerU content list."""
    for path in sorted(Path(root).rglob("*_content_list.json")):
        # `*_content_list_v2.json` also matches the glob; skip it — v2 nests
        # inline spans per paragraph, which we flatten from v1 anyway.
        if path.name.endswith("_content_list_v2.json"):
            continue
        yield path.parent


def iter_pdfs(root: Path) -> Iterator[Path]:
    """Yield every PDF under ``root``.

    Deliberately the same glob the runner discovers with, so
    ``pdfsys dataset --from-pdf-dir X`` packages exactly the set that
    ``pdfsys run --pdf-dir X`` processed. A glob that differed even slightly
    would produce a shard quietly covering a different corpus than the run it
    claims to come from.
    """
    yield from (p for p in sorted(Path(root).rglob("*.pdf")) if p.is_file())


def select_pdfs(
    paths: Iterable[Path],
) -> tuple[list[tuple[str, Path]], list[tuple[Path, str, str]]]:
    """一个 doc_id 只留一份 PDF，按 doc_id 排序。

    返回 ``(选中的 (doc_id, path), 丢弃的 (path, doc_id, 原因))``——和
    :func:`select_documents` 同形状，调用方一套代码处理两条链路。

    同一份 PDF 在语料里出现两次（不同文件名、同样内容）是常态，而
    ``(doc_id, page_index)`` 是主键，两份都写就是违约的 shard。排序也在这里
    做：``DatasetWriter`` 要求 doc_id 递增，而文件名和 sha256 没有关系。
    """
    import hashlib

    seen: dict[str, Path] = {}
    dropped: list[tuple[Path, str, str]] = []
    for raw in paths:
        path = Path(raw)
        try:
            doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as e:
            dropped.append((path, "", f"读不出来: {e}"))
            continue
        if doc_id in seen:
            dropped.append((path, doc_id, f"与 {seen[doc_id]} 内容相同"))
            continue
        seen[doc_id] = path
    return sorted(seen.items()), dropped


def build_from_pdf(
    pdf_path: Path,
    **page_fields: Any,
) -> tuple[tuple[PageRecord, ...], list[ImageBlob]]:
    """Extract one PDF through the mupdf fast lane and encode it as page rows.

    This re-extracts rather than reading a finished run's output, because the
    mupdf lane leaves its page structure nowhere on disk: it writes one merged
    ``markdown/<sha256>.md`` with no page boundaries, and ``segments_excerpt``
    in ``results.jsonl`` is populated only on the VLM branch and truncated to
    200 characters — a viz artifact, not a data source. Re-running mupdf costs
    ~10 ms/page, which is cheaper than inventing a format to persist what it
    already knows how to recompute.

    ``source_uri`` defaults to the PDF's path, so this lane needs no ``--meta``
    to produce a row that says where it came from.
    """
    from pdfsys_parser_mupdf import extract_doc

    pdf_path = Path(pdf_path)
    geometry = page_geometry_from_pdf(pdf_path)
    extracted = extract_doc(pdf_path)
    page_fields.setdefault("source_uri", str(pdf_path))
    return build_from_extracted(
        extracted,
        n_pages=len(geometry),
        page_sizes=geometry,
        **page_fields,
    )


#: 同一份 PDF 被多个 backend 跑过时，谁的产物进 shard。越靠前越优先——
#: VLM 走的是版面模型，块结构最全；pipeline 次之；mupdf 只有原生文本块。
#: 这只是没有质量分时的兜底启发式，有 --meta 的话应该按质量分选。
_EXTRACTOR_RANK = {"vlm": 0, "pipeline": 1, "mupdf": 2}


def select_documents(
    doc_dirs: Sequence[Path],
) -> tuple[list[Path], list[tuple[Path, str, str]]]:
    """一个 doc_id 只留一份产物。

    返回 ``(选中的目录, 丢弃的 (目录, doc_id, 原因))``。丢弃的必须被调用方
    打出来——静默截断会让人以为覆盖全了。

    不去重的后果是实打实的：``(doc_id, page_index)`` 是主键，同一份 PDF 被
    pipeline 和 vlm 各跑一遍就会产生两行同主键的记录，shard 直接违约。
    """
    candidates: dict[str, list[tuple[tuple, Path, str]]] = {}
    for doc_dir in doc_dirs:
        content = _one(doc_dir, "*_content_list.json", exclude="_content_list_v2.json")
        if content is None:
            continue
        doc_id = content.name[: -len("_content_list.json")]
        _, backend = _page_geometry(doc_dir / f"{doc_id}_middle.json")
        try:
            n_items = len(json.loads(content.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            n_items = 0
        rank = (_EXTRACTOR_RANK.get(backend, 99), -n_items, str(doc_dir))
        candidates.setdefault(doc_id, []).append((rank, doc_dir, backend))

    kept_by_id: list[tuple[str, Path]] = []
    dropped: list[tuple[Path, str, str]] = []
    for doc_id, options in candidates.items():
        options.sort(key=lambda o: o[0])
        kept_by_id.append((doc_id, options[0][1]))
        winner = options[0][2]
        for _, doc_dir, backend in options[1:]:
            dropped.append(
                (doc_dir, doc_id, f"同 doc_id 已选用 {winner} 的产物，这份是 {backend}")
            )
    # 按 doc_id 排序，而不是路径 —— shard 承诺按 (doc_id, page_index) 有序，
    # 目录名和 doc_id 没有关系。
    kept_by_id.sort()
    return [path for _, path in kept_by_id], dropped


def build_from_mineru_dir(
    doc_dir: Path,
    *,
    doc_id: str | None = None,
    extractor: str = "",
    link_figure_mentions: bool = True,
    images: str = "crops",
    **page_fields: Any,
) -> tuple[tuple[PageRecord, ...], list[ImageBlob]]:
    """Re-encode one MinerU output directory as page rows.

    ``doc_id`` defaults to the sha256 prefix MinerU uses for its filenames,
    which is the source PDF's sha256 as written by our parsers.

    ``images`` selects how image pixels are stored — see :data:`IMAGE_MODES`.
    ``"pages"`` returns no crop blobs at all (build the page rasters with
    :func:`render_page_images` instead); ``"none"`` stores no pixels.
    """
    if images not in IMAGE_MODES:
        raise ValueError(f"images must be one of {IMAGE_MODES}, got {images!r}")
    doc_dir = Path(doc_dir)
    content_path = _one(doc_dir, "*_content_list.json", exclude="_content_list_v2.json")
    if content_path is None:
        raise FileNotFoundError(f"no *_content_list.json under {doc_dir}")

    stem = content_path.name[: -len("_content_list.json")]
    items = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{content_path} is not a content list")

    middle_path = doc_dir / f"{stem}_middle.json"
    page_sizes, backend = _page_geometry(middle_path)

    blobs, path_to_id = _load_images(doc_dir)

    blocks = blocks_from_content_list(items, image_ids=path_to_id)
    if link_figure_mentions:
        blocks = link_mentions(blocks)

    if images == "pages":
        # The crop is a rectangle of the page raster; drop the duplicate blob
        # and let the bbox address it. Blocks whose bbox could not be
        # normalized keep their blob — otherwise the image becomes
        # unreachable, which is worse than a little redundancy.
        blocks = tuple(
            dataclasses.replace(b, image_id=None)
            if b.image_id and b.bbox is not None
            else b
            for b in blocks
        )
    elif images == "none":
        blocks = tuple(dataclasses.replace(b, image_id=None) for b in blocks)

    pages = split_pages(
        blocks,
        n_pages=len(page_sizes),
        # A bbox:// marker is only resolvable against a page raster, and only
        # the "pages" mode stores one. Under "none" the same marker would point
        # at pixels the shard does not contain — dataset-validate rejects it,
        # and rightly so. The bbox stays on the block, so the crop is still
        # reconstructible from the source PDF.
        region_refs=(images == "pages"),
        doc_id=doc_id or stem,
        extractor=extractor or backend,
        **page_fields,
    )
    pages = tuple(_with_geometry(p, page_sizes) for p in pages)

    # Only ship blobs some page actually references — a stale images/ dir
    # would otherwise inflate the shard.
    referenced = {i for p in pages for i in p.image_ids}
    return pages, [b for b in blobs if b.image_id in referenced]


def build_from_extracted(
    extracted: Any,
    *,
    n_pages: int = 0,
    page_sizes: Sequence[tuple[float, float, int]] | None = None,
    **page_fields: Any,
) -> tuple[tuple[PageRecord, ...], list[ImageBlob]]:
    """Encode an in-memory :class:`pdfsys_core.ExtractedDoc` (mupdf lane).

    ``page_sizes`` is ``(width_pt, height_pt, rotation)`` per page, normally
    from :func:`page_geometry_from_pdf`. Without it the geometry columns stay
    null — and they should not, because page size comes from the PDF itself
    and is the most model-independent thing in the whole row.

    Falls back to a single page holding the pre-merged ``markdown`` when a
    backend emitted no segments, so a document is never silently dropped.
    """
    extractor = str(getattr(extracted.backend, "value", extracted.backend))
    blocks = blocks_from_segments(extracted.segments)
    if not blocks:
        page = PageRecord(
            doc_id=extracted.sha256,
            page_index=0,
            text=extracted.markdown or "",
            extractor=extractor,
            doc_n_pages=max(n_pages, 1),
            **page_fields,
        )
        return (_with_geometry(page, page_sizes),), []

    pages = split_pages(
        blocks,
        n_pages=n_pages,
        doc_id=extracted.sha256,
        extractor=extractor,
        **page_fields,
    )
    return tuple(_with_geometry(p, page_sizes) for p in pages), []


def page_geometry_from_pdf(pdf_path: Path) -> list[tuple[float, float, int]]:
    """``(width_pt, height_pt, rotation)`` per page, straight from the PDF."""
    import pymupdf  # lazy: heavy, and only this path needs it

    with pymupdf.open(pdf_path) as doc:
        return [
            (float(page.rect.width), float(page.rect.height), int(page.rotation))
            for page in doc
        ]


def render_page_images(
    pdf_path: Path, pages: Sequence[PageRecord], *, dpi: int = 200
) -> list[tuple[PageRecord, ImageBlob]]:
    """Render full-page rasters for a document.

    The default of 200 dpi is not arbitrary: MinerU produces its crops by
    cutting them out of a 200-dpi page render (measured at 200.1 dpi median
    over 172 crops). Rendering at the same resolution makes a crop taken from
    this raster match the one MinerU would have stored to within the bbox
    quantization — bboxes are integers on a 0-1000 grid, so each edge can be
    off by about a pixel. That is what lets ``images="pages"`` drop the crop
    table without meaningfully losing anything.

    Opt-in and deliberately not part of the default build: a page raster is
    reproducible from the immutable L0 PDF at whatever DPI a downstream task
    turns out to need, whereas OCR text and layout cost GPU hours. Storing
    rasters now would mean paying terabytes to freeze a choice we can defer.

    Returns ``(page, blob)`` pairs with ``page`` updated to carry
    ``page_image_id`` / ``render_dpi``; the caller must use these updated
    records, not the originals.
    """
    import pymupdf  # lazy: heavy, and only this path needs it

    out: list[tuple[PageRecord, ImageBlob]] = []
    zoom = dpi / 72.0
    with pymupdf.open(pdf_path) as doc:
        for page in pages:
            if not (0 <= page.page_index < doc.page_count):
                _LOG.warning(
                    "page %d out of range for %s (%d pages)",
                    page.page_index,
                    pdf_path,
                    doc.page_count,
                )
                continue
            pix = doc[page.page_index].get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), alpha=False
            )
            data = pix.tobytes("jpeg", jpg_quality=85)
            iid = image_id_for(data)
            fmt, width, height = probe_image(data)
            out.append(
                (
                    dataclasses.replace(page, page_image_id=iid, render_dpi=dpi),
                    ImageBlob(
                        image_id=iid, data=data, format=fmt, width=width, height=height
                    ),
                )
            )
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _one(directory: Path, pattern: str, *, exclude: str | None = None) -> Path | None:
    for path in sorted(directory.glob(pattern)):
        if exclude and path.name.endswith(exclude):
            continue
        return path
    return None


def _page_geometry(middle_path: Path) -> tuple[list[tuple[float, float]], str]:
    """Read per-page size (PDF points) and the backend tag from ``middle.json``.

    ``page_size`` is the page geometry — ``[612, 792]`` is US Letter, ``[595,
    841]`` is A4. It is emphatically *not* the space ``content_list`` bboxes
    live in; those are on MinerU's 0–1000 grid.
    """
    if not middle_path.exists():
        return [], ""
    try:
        middle = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _LOG.warning("unreadable middle json %s: %s", middle_path, e)
        return [], ""

    sizes: list[tuple[float, float]] = []
    for page in middle.get("pdf_info") or ():
        size = page.get("page_size") or (0, 0)
        try:
            sizes.append((float(size[0]), float(size[1])))
        except (TypeError, ValueError, IndexError):
            sizes.append((0.0, 0.0))
    return sizes, str(middle.get("_backend") or "")


def _with_geometry(
    page: PageRecord, page_sizes: Sequence[Sequence[float]] | None
) -> PageRecord:
    """Stamp page geometry. Accepts ``(w, h)`` or ``(w, h, rotation)``."""
    if not page_sizes or not (0 <= page.page_index < len(page_sizes)):
        return page
    size = page_sizes[page.page_index]
    width, height = float(size[0]), float(size[1])
    rotation = int(size[2]) if len(size) > 2 else page.rotation
    return dataclasses.replace(
        page, width_pt=width, height_pt=height, rotation=rotation
    )


def _load_images(doc_dir: Path) -> tuple[list[ImageBlob], dict[str, str]]:
    """Load ``images/*`` and map MinerU's ``img_path`` to a content address.

    MinerU names crops by a hash of the *pre-encoding* pixels, so it is not
    the hash of the file on disk — we compute our own so ``image_id`` is a
    true content address of the bytes we ship.
    """
    images_dir = doc_dir / "images"
    if not images_dir.is_dir():
        return [], {}

    blobs: dict[str, ImageBlob] = {}
    path_to_id: dict[str, str] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as e:
            _LOG.warning("unreadable image %s: %s", path, e)
            continue
        if not data:
            continue
        iid = image_id_for(data)
        fmt, width, height = probe_image(data)
        blobs.setdefault(
            iid,
            ImageBlob(image_id=iid, data=data, format=fmt, width=width, height=height),
        )
        # content_list refers to crops as "images/<name>"; accept the bare
        # name too since MinerU has used both spellings.
        path_to_id[f"images/{path.name}"] = iid
        path_to_id[path.name] = iid
    return list(blobs.values()), path_to_id
