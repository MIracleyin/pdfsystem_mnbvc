"""MVP closed-loop runner: router → parser → quality scorer → JSONL.

This is the tiniest possible end-to-end harness for the pdfsys pipeline.
Given a directory of PDFs, it:

1. runs :class:`pdfsys_router.Router` to pick a backend per document;
2. for PDFs routed to ``Backend.MUPDF``, runs :func:`pdfsys_parser_mupdf.extract_doc`
   to produce an :class:`pdfsys_core.ExtractedDoc`;
3. scores the resulting Markdown with :class:`pdfsys_bench.OcrQualityScorer`
   (the ModernBERT-large regression head from FinePDFs);
4. writes one JSON line per PDF to an output file with routing decision,
   extraction stats, and quality score.

PDFs routed to ``PIPELINE`` / ``VLM`` / ``DEFERRED`` are recorded with
their routing decision but skipped for extraction — those backends are
not implemented yet in this MVP.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pdfsys_core import Backend
from pdfsys_parser_mupdf import extract_doc
from pdfsys_router import Router

from .quality import OcrQualityScorer, QualityScore


@dataclass(slots=True)
class LoopResult:
    """Per-PDF result row, serialized to JSONL."""

    pdf_path: str
    sha256: str | None
    backend: str
    ocr_prob: float
    num_pages: int
    is_form: bool
    garbled_text_ratio: float
    router_error: str | None
    extract_stats: dict[str, Any] = field(default_factory=dict)
    extract_error: str | None = None
    quality_score: float | None = None
    quality_num_chars: int | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    markdown_chars: int = 0
    wall_ms_router: float = 0.0
    wall_ms_extract: float = 0.0
    wall_ms_quality: float = 0.0

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _iter_pdfs(root: Path, limit: int | None) -> Iterable[Path]:
    pdfs = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    if limit is not None:
        pdfs = pdfs[:limit]
    yield from pdfs


def run_loop(
    pdf_dir: str | Path,
    out_path: str | Path,
    *,
    limit: int | None = None,
    score_quality: bool = True,
    router_weights: str | Path | None = None,
    quality_model: str = "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn",
    markdown_dir: str | Path | None = None,
    ocr_threshold: float = 0.5,
) -> dict[str, Any]:
    """Drive the full MVP loop over a PDF directory.

    Returns an aggregate summary dict. Individual result rows are written
    to ``out_path`` as JSONL (one line per PDF, in input-order).
    """
    pdf_dir = Path(pdf_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    router = Router(model_path=router_weights, ocr_threshold=ocr_threshold)
    scorer = OcrQualityScorer(model_name=quality_model) if score_quality else None

    md_root = Path(markdown_dir) if markdown_dir else None
    if md_root is not None:
        md_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pdf_dir": str(pdf_dir),
        "out_path": str(out_path),
        "num_pdfs": 0,
        "by_backend": {},
        "num_extracted": 0,
        "num_scored": 0,
        "num_errors": 0,
        "sum_quality": 0.0,
        "started_at": time.time(),
    }

    with out_path.open("w", encoding="utf-8") as out_f:
        for pdf_path in _iter_pdfs(pdf_dir, limit):
            row = _run_one(
                pdf_path=pdf_path,
                router=router,
                scorer=scorer,
                md_root=md_root,
            )
            out_f.write(row.to_json_line() + "\n")
            out_f.flush()

            summary["num_pdfs"] += 1
            by_b = summary["by_backend"]
            by_b[row.backend] = by_b.get(row.backend, 0) + 1
            if row.extract_error is None and row.backend == Backend.MUPDF.value:
                summary["num_extracted"] += 1
            if row.quality_score is not None:
                summary["num_scored"] += 1
                summary["sum_quality"] += row.quality_score
            if row.router_error or row.extract_error:
                summary["num_errors"] += 1

    summary["finished_at"] = time.time()
    summary["wall_seconds"] = summary["finished_at"] - summary["started_at"]
    summary["avg_quality"] = (
        summary["sum_quality"] / summary["num_scored"] if summary["num_scored"] else None
    )

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    summary["summary_path"] = str(summary_path)

    return summary


def _run_one(
    *,
    pdf_path: Path,
    router: Router,
    scorer: OcrQualityScorer | None,
    md_root: Path | None,
) -> LoopResult:
    # -- Stage-A routing ------------------------------------------------------
    t0 = time.perf_counter()
    decision = router.classify(pdf_path)
    t1 = time.perf_counter()

    row = LoopResult(
        pdf_path=str(pdf_path),
        sha256=None,
        backend=decision.backend.value,
        ocr_prob=decision.ocr_prob,
        num_pages=decision.num_pages,
        is_form=decision.is_form,
        garbled_text_ratio=decision.garbled_text_ratio,
        router_error=decision.error,
        wall_ms_router=(t1 - t0) * 1000.0,
    )

    # -- MVP only extracts the text-ok fast path ------------------------------
    if decision.backend != Backend.MUPDF:
        return row

    try:
        t2 = time.perf_counter()
        extracted = extract_doc(pdf_path)
        t3 = time.perf_counter()
        row.sha256 = extracted.sha256
        row.extract_stats = dict(extracted.stats)
        row.markdown_chars = extracted.char_count
        row.wall_ms_extract = (t3 - t2) * 1000.0
    except Exception as e:  # noqa: BLE001
        row.extract_error = f"extract_failed: {e}"
        return row

    if md_root is not None and extracted.markdown:
        md_path = md_root / f"{extracted.sha256}.md"
        md_path.write_text(extracted.markdown, encoding="utf-8")

    # -- Quality scoring ------------------------------------------------------
    if scorer is not None and extracted.markdown:
        try:
            t4 = time.perf_counter()
            q: QualityScore = scorer.score(extracted.markdown)
            t5 = time.perf_counter()
            row.quality_score = q.score
            row.quality_num_chars = q.num_chars
            row.quality_num_tokens = q.num_tokens
            row.quality_model = q.model
            row.wall_ms_quality = (t5 - t4) * 1000.0
        except Exception as e:  # noqa: BLE001
            row.extract_error = f"quality_failed: {e}"

    return row
