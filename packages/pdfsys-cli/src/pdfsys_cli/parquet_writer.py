"""Single-shard Parquet sink for the pdfsys pipeline output.

One row per PDF. The full schema is the `SCHEMA` constant below — see
docs/superpowers/specs/2026-05-14-e2e-parquet-mac-mps-design.md §5 for
column semantics.

Markdown text is included as a string column (`include_markdown=True`)
to keep the dataset self-contained. For 150 PDFs the total compressed
size is well under 50 MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from .runner import DocResult


SCHEMA = pa.schema(
    [
        ("pdf_path", pa.string()),
        ("sha256", pa.string()),
        ("backend", pa.string()),
        ("stage_b_backend", pa.string()),
        ("ocr_prob", pa.float64()),
        ("num_pages", pa.int32()),
        ("is_form", pa.bool_()),
        ("garbled_text_ratio", pa.float64()),
        ("is_encrypted", pa.bool_()),
        ("layout_model", pa.string()),
        ("layout_num_regions", pa.int32()),
        ("layout_has_complex", pa.bool_()),
        ("extract_backend", pa.string()),
        ("markdown", pa.string()),
        ("markdown_chars", pa.int64()),
        ("quality_score", pa.float64()),
        ("quality_model", pa.string()),
        ("error_class", pa.string()),
        ("error_message", pa.string()),
        ("segments_excerpt", pa.string()),  # JSON-encoded list[dict]; null for non-VLM rows
        ("region_failures", pa.int32()),
        ("kept", pa.bool_()),
        ("wall_ms_total", pa.float64()),
    ]
)


class ParquetSink:
    """Streaming Parquet writer. Open once per run, write_row per PDF, close at end."""

    def __init__(
        self,
        path: Path,
        compression: str = "zstd",
        quality_threshold: float = 2.0,
        include_markdown: bool = True,
    ) -> None:
        self.path = path
        self.quality_threshold = quality_threshold
        self.include_markdown = include_markdown
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(path), SCHEMA, compression=compression
        )
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def write_row(self, row: "DocResult", markdown: str | None) -> None:
        kept = (
            row.error_class is None
            and row.quality_score is not None
            and row.quality_score >= self.quality_threshold
        )
        wall_total = (
            row.wall_ms_router
            + row.wall_ms_layout
            + row.wall_ms_extract
            + row.wall_ms_quality
        )

        md_value: str | None
        if not self.include_markdown:
            md_value = None
        elif markdown is None or markdown == "":
            md_value = None
        else:
            md_value = markdown

        data: dict[str, Any] = {
            "pdf_path": row.pdf_path,
            "sha256": row.sha256,
            "backend": row.backend,
            "stage_b_backend": row.stage_b_backend,
            "ocr_prob": row.ocr_prob,
            "num_pages": row.num_pages,
            "is_form": row.is_form,
            "garbled_text_ratio": row.garbled_text_ratio,
            "is_encrypted": row.is_encrypted,
            "layout_model": row.layout_model,
            "layout_num_regions": row.layout_num_regions,
            "layout_has_complex": row.layout_has_complex,
            "extract_backend": row.extract_backend,
            "markdown": md_value,
            "markdown_chars": row.markdown_chars,
            "quality_score": row.quality_score,
            "quality_model": row.quality_model,
            "error_class": row.error_class,
            "error_message": row.error_message,
            "segments_excerpt": (
                __import__("json").dumps(row.segments_excerpt, ensure_ascii=False)
                if row.segments_excerpt else None
            ),
            "region_failures": row.region_failures,
            "kept": kept,
            "wall_ms_total": wall_total,
        }

        # ParquetWriter expects a Table; one row per call is fine at our scale.
        table = pa.Table.from_pylist([data], schema=SCHEMA)
        self._writer.write_table(table)
        self._rows_written += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "ParquetSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
