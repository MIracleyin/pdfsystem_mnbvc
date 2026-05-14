"""VLM extraction backend using mineru 3.x.

Replaces magic-pdf 1.x (removed 2026-05-14). The :class:`VlmParser`
public contract is preserved — only the model package underneath
changes. The two-path fallback (mineru_v2 + magic_pdf_v1) that lived
here previously is gone; there is one engine.

mineru is invoked via the high-level ``mineru.cli.common.do_parse``
function, which writes its artifacts to a tempdir as
``<name>_content_list.json`` and ``<name>.md``. We read those back and
map mineru's content-list items into our :class:`pdfsys_core.Segment`
contract.

Heavy dependencies (``mineru``, ``torch``) are imported lazily inside
:meth:`_invoke_mineru` so callers that don't actually need VLM don't
pay the import cost.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    BBox,
    ExtractedDoc,
    LayoutDocument,
    RegionType,
    Segment,
    VlmConfig,
    merge_segments_to_markdown,
)

# Mapping from mineru's content-type strings to our RegionType enum.
# Verified against the migration spike (out/mineru_spike_notes.md,
# 2026-05-14). Update the right-hand-side keys if mineru's type strings
# evolve.
_MINERU_TYPE_MAP: dict[str, RegionType] = {
    "text": RegionType.TEXT,
    "title": RegionType.TEXT,
    "equation": RegionType.FORMULA,
    "inline_equation": RegionType.FORMULA,
    "interline_equation": RegionType.FORMULA,
    "table": RegionType.TABLE,
    "image": RegionType.IMAGE,
    "figure": RegionType.IMAGE,
    "figure_caption": RegionType.TEXT,
    "table_caption": RegionType.TEXT,
    "table_footnote": RegionType.TEXT,
    "header": RegionType.TEXT,
    "footer": RegionType.TEXT,
    "page_number": RegionType.TEXT,
    "reference": RegionType.TEXT,
}


class VlmParser:
    """mineru-based VLM extraction parser.

    Wraps mineru 3.x to process PDF pages and produce
    :class:`pdfsys_core.ExtractedDoc` output.
    """

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()

    # ------------------------------------------------------------------ api

    def extract(
        self, pdf_path: str | Path, sha256: str | None = None
    ) -> ExtractedDoc:
        """Process an entire PDF through mineru and return ExtractedDoc."""
        path = Path(pdf_path)
        sha = sha256 or _sha256_of_file(path)
        with path.open("rb") as f:
            pdf_bytes = f.read()
        return self._run_mineru(pdf_bytes, sha, complex_pages=None)

    def extract_bytes(
        self, pdf_bytes: bytes, sha256: str | None = None
    ) -> ExtractedDoc:
        """Same as :meth:`extract`, but from an in-memory buffer."""
        sha = sha256 or hashlib.sha256(pdf_bytes).hexdigest()
        return self._run_mineru(pdf_bytes, sha, complex_pages=None)

    def extract_complex_pages(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Process only pages flagged as having complex content.

        Reads the :class:`LayoutDocument` to identify pages with TABLE
        or FORMULA regions, runs mineru on the full PDF, then filters
        segments to those pages. Simple pages are skipped (they should
        be handled by the pipeline parser).
        """
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)

        complex_pages: set[int] = set()
        for lp in layout.pages:
            for region in lp.regions:
                if region.type in (RegionType.TABLE, RegionType.FORMULA):
                    complex_pages.add(lp.index)
                    break

        if not complex_pages:
            return ExtractedDoc(
                sha256=sha,
                backend=Backend.VLM,
                segments=(),
                markdown="",
                stats={"complex_pages": 0, "reason": "no_complex_content"},
            )

        with path.open("rb") as f:
            pdf_bytes = f.read()

        return self._run_mineru(pdf_bytes, sha, complex_pages=complex_pages)

    # --------------------------------------------------------------- internal

    def _run_mineru(
        self,
        pdf_bytes: bytes,
        sha256: str,
        complex_pages: set[int] | None,
    ) -> ExtractedDoc:
        """Run mineru's VLM pipeline on raw PDF bytes.

        If ``complex_pages`` is given, returned segments are filtered to
        those page indices.
        """
        content_list, md_content, stats = self._invoke_mineru(pdf_bytes)

        segments = self._content_list_to_segments(content_list)

        if complex_pages is not None:
            segments = [s for s in segments if s.page_index in complex_pages]
            stats["complex_pages"] = len(complex_pages)
            stats["complex_page_indices"] = sorted(complex_pages)

        # Re-index after filtering so segment.index stays contiguous.
        segments = [
            Segment(
                index=i,
                backend=s.backend,
                page_index=s.page_index,
                type=s.type,
                content=s.content,
                bbox=s.bbox,
                source_region_id=s.source_region_id,
            )
            for i, s in enumerate(segments)
        ]
        seg_tuple = tuple(segments)

        # Prefer our assembled markdown for consistency across backends;
        # fall back to mineru's own markdown if ours is empty (e.g.
        # image-only pages).
        markdown = merge_segments_to_markdown(seg_tuple)
        if not markdown.strip() and md_content:
            markdown = md_content.strip() + "\n"

        stats["char_count"] = len(markdown)
        stats["segment_count"] = len(seg_tuple)
        stats["vlm_engine"] = "mineru-3.x"
        stats["vlm_model"] = self.config.model

        return ExtractedDoc(
            sha256=sha256,
            backend=Backend.VLM,
            segments=seg_tuple,
            markdown=markdown,
            stats=stats,
        )

    def _invoke_mineru(
        self, pdf_bytes: bytes
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """Call mineru's VLM entry and return (content_list, markdown, stats)."""
        # Lazy import — mineru pulls torch + transformers + many deps.
        from mineru.cli.common import do_parse  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="pdfsys_vlm_") as tmpdir:
            do_parse(
                output_dir=tmpdir,
                pdf_file_names=["doc"],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                backend="vlm-transformers",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                f_dump_md=True,
                f_dump_content_list=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
            )

            td = Path(tmpdir)
            content_list: list[dict[str, Any]] = []
            for cand in td.rglob("*_content_list.json"):
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        content_list = [it for it in data if isinstance(it, dict)]
                    break
                except (json.JSONDecodeError, OSError):
                    continue

            md_content = ""
            for cand in td.rglob("*.md"):
                try:
                    md_content = cand.read_text(encoding="utf-8")
                    break
                except OSError:
                    continue

        stats: dict[str, Any] = {
            "api": "mineru_v3",
            "backend": "vlm-transformers",
        }
        return content_list, md_content, stats

    def _content_list_to_segments(
        self, content_list: list[dict[str, Any]]
    ) -> list[Segment]:
        """Map mineru's content_list items to our Segment format.

        mineru's content_list shape (verified in spike):
        - type: text | equation | table | image | header | page_number | ...
        - text: content body (LaTeX-wrapped for equations, may be HTML for tables)
        - text_format: 'latex' | 'html' (optional)
        - bbox: [x0, y0, x1, y1] in PIXEL coords at rendering DPI
        - page_idx: 0-based page index (note: page_idx, not page_index)
        - img_path: relative path for image items
        """
        segments: list[Segment] = []

        for item in content_list:
            item_type = item.get("type", "text")
            region_type = _MINERU_TYPE_MAP.get(item_type, RegionType.TEXT)

            # Extract content. mineru puts everything in `text` even for
            # equations (already LaTeX-wrapped) and tables (HTML wrapped).
            # Fall through to legacy magic-pdf fields if mineru ever
            # emits them.
            content = ""
            if region_type == RegionType.IMAGE:
                content = item.get("img_path", "") or item.get("text", "") or "[image]"
            elif region_type == RegionType.TABLE:
                content = (
                    item.get("text", "")
                    or item.get("html", "")
                    or item.get("latex", "")
                    or item.get("md", "")
                )
            elif region_type == RegionType.FORMULA:
                content = (
                    item.get("text", "")
                    or item.get("latex", "")
                    or item.get("md", "")
                )
            else:
                content = item.get("text", "") or item.get("md", "")

            if not content:
                continue

            # mineru's bbox is in pixel coords at rendering DPI but does
            # not include page dimensions in content_list (those live in
            # middle_json). Without page dims we can't safely normalize,
            # so leave bbox=None for VLM segments. Downstream consumers
            # (parquet, JSONL) already treat bbox as nullable.
            bbox = None

            page_index = item.get("page_idx", item.get("page_index", 0))

            segments.append(
                Segment(
                    index=len(segments),
                    backend=Backend.VLM,
                    page_index=int(page_index),
                    type=region_type,
                    content=content.strip(),
                    bbox=bbox,
                    source_region_id=None,
                )
            )

        return segments


# ---------------------------------------------------------------- convenience

def extract_doc(
    pdf_path: str | Path,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: create a VLM parser, extract, return."""
    parser = VlmParser(config=config)
    return parser.extract(pdf_path, sha256=sha256)


def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: extract only complex pages identified by layout."""
    parser = VlmParser(config=config)
    return parser.extract_complex_pages(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
