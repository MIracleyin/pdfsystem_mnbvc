"""Build ``pdfsys.doc/v1`` records from what the pipeline already writes.

Two entry points, one per lane:

* :func:`build_from_mineru_dir` — the pipeline/vlm lane. MinerU's
  ``content_list.json`` is already a reading-order interleaved list with
  captions, table HTML and image crops; ``middle.json`` supplies the page
  sizes its pixel bboxes are relative to. This is a pure re-encoding, not a
  re-parse.
* :func:`build_from_extracted` — the mupdf fast lane, where
  ``ExtractedDoc.segments`` is the authority and there are no images.

Both return ``(DocRecord, list[ImageBlob])`` ready for
:class:`pdfsys_cli.dataset_writer.DatasetWriter`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pdfsys_core import (
    DocRecord,
    ImageBlob,
    blocks_from_content_list,
    blocks_from_segments,
    image_id_for,
    link_mentions,
    probe_image,
    render_markdown,
)

_LOG = logging.getLogger(__name__)

__all__ = [
    "build_from_mineru_dir",
    "build_from_extracted",
    "iter_mineru_dirs",
]


def iter_mineru_dirs(root: Path) -> Iterator[Path]:
    """Yield every directory under ``root`` holding a MinerU content list."""
    for path in sorted(Path(root).rglob("*_content_list.json")):
        # `*_content_list_v2.json` also matches the glob; skip it — v2 nests
        # inline spans per paragraph, which we flatten from v1 anyway.
        if path.name.endswith("_content_list_v2.json"):
            continue
        yield path.parent


def build_from_mineru_dir(
    doc_dir: Path,
    *,
    doc_id: str | None = None,
    source_uri: str = "",
    backend: str = "",
    link_figure_mentions: bool = True,
    **doc_fields: Any,
) -> tuple[DocRecord, list[ImageBlob]]:
    """Re-encode one MinerU output directory as a :class:`DocRecord`.

    ``doc_id`` defaults to the sha256 prefix MinerU uses for its filenames,
    which is the source PDF's sha256 as written by our parsers.
    """
    doc_dir = Path(doc_dir)
    content_path = _one(doc_dir, "*_content_list.json", exclude="_content_list_v2.json")
    if content_path is None:
        raise FileNotFoundError(f"no *_content_list.json under {doc_dir}")

    stem = content_path.name[: -len("_content_list.json")]
    items = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{content_path} is not a content list")

    middle_path = doc_dir / f"{stem}_middle.json"
    n_pages_from_middle = _page_count(middle_path)

    blobs, path_to_id = _load_images(doc_dir)

    blocks = blocks_from_content_list(items, image_ids=path_to_id)
    if link_figure_mentions:
        blocks = link_mentions(blocks)
    text, page_ends = render_markdown(blocks)

    n_pages = n_pages_from_middle or len(page_ends)
    doc = DocRecord(
        id=doc_id or stem,
        blocks=blocks,
        text=text,
        page_ends=page_ends,
        source_uri=source_uri,
        backend=backend or _backend_from_middle(middle_path),
        n_pages=n_pages,
        **doc_fields,
    )
    # Only ship blobs this document actually references — a stale images/ dir
    # would otherwise inflate the shard.
    referenced = set(doc.image_ids)
    return doc, [b for b in blobs if b.image_id in referenced]


def build_from_extracted(
    extracted: Any,
    *,
    source_uri: str = "",
    n_pages: int = 0,
    **doc_fields: Any,
) -> tuple[DocRecord, list[ImageBlob]]:
    """Encode an in-memory :class:`pdfsys_core.ExtractedDoc` (mupdf lane).

    Falls back to the pre-merged ``markdown`` when a backend emitted no
    segments, so a document is never silently dropped.
    """
    blocks = blocks_from_segments(extracted.segments)
    if blocks:
        text, page_ends = render_markdown(blocks)
    else:
        text = extracted.markdown or ""
        page_ends = (len(text),)

    backend = getattr(extracted.backend, "value", extracted.backend)
    doc = DocRecord(
        id=extracted.sha256,
        blocks=blocks,
        text=text,
        page_ends=page_ends,
        source_uri=source_uri,
        backend=str(backend),
        n_pages=n_pages or len(page_ends),
        **doc_fields,
    )
    return doc, []


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _one(directory: Path, pattern: str, *, exclude: str | None = None) -> Path | None:
    for path in sorted(directory.glob(pattern)):
        if exclude and path.name.endswith(exclude):
            continue
        return path
    return None


def _page_count(middle_path: Path) -> int:
    """Page count from ``middle.json``, which sees pages that produced no text.

    Note this is the *only* thing we take from ``middle.json``: its
    ``page_size`` is NOT the space ``content_list`` bboxes live in — those are
    on MinerU's 0–1000 grid, independent of page size.
    """
    if not middle_path.exists():
        return 0
    try:
        middle = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _LOG.warning("unreadable middle json %s: %s", middle_path, e)
        return 0
    return len(middle.get("pdf_info") or ())


def _backend_from_middle(middle_path: Path) -> str:
    if not middle_path.exists():
        return ""
    try:
        middle = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(middle.get("_backend") or "")


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
            iid, ImageBlob(image_id=iid, data=data, format=fmt, width=width, height=height)
        )
        # content_list refers to crops as "images/<name>"; accept the bare
        # name too since MinerU has used both spellings.
        path_to_id[f"images/{path.name}"] = iid
        path_to_id[path.name] = iid
    return list(blobs.values()), path_to_id
