"""Full-pipeline closed-loop runner.

Drives the complete pdfsys extraction pipeline on a directory of PDFs:

1. **Stage-A** routing (XGBoost) → MUPDF or needs-ocr.
2. For MUPDF: text extraction via PyMuPDF (fast path).
3. For needs-ocr: **layout analyser** → **Stage-B** decider → PIPELINE or VLM.
4. PIPELINE: region-level OCR via RapidOCR.
5. VLM: end-to-end extraction via MinerU 2.5.
6. Quality scoring (ModernBERT) on all extracted Markdown.
7. One JSONL row per PDF with full routing/extraction/quality stats.

When ``--full-pipeline`` is disabled (the default), only the MUPDF fast
path runs — same as the original MVP behaviour.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pdfsys_core import Backend, LayoutCache

from .cascade import CascadeStage, run_cascade
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
    # --- extraction ---
    extract_stats: dict[str, Any] = field(default_factory=dict)
    extract_error: str | None = None
    markdown_chars: int = 0
    # --- layout (Stage-B) ---
    layout_model: str | None = None
    layout_num_regions: int | None = None
    layout_has_complex: bool | None = None
    stage_b_backend: str | None = None
    # --- quality ---
    quality_score: float | None = None
    quality_num_chars: int | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    # --- cascade (only populated in cascade mode) ---
    cascade_decision: str | None = None
    cascade_final_stage: str | None = None
    cascade_attempts: list[dict[str, Any]] | None = None
    cascade_total_wall_ms: float | None = None
    # --- timing ---
    wall_ms_router: float = 0.0
    wall_ms_layout: float = 0.0
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
    quality_model: str = "miracleyin/mnbvc-pdf-quality-scorer-modernbert",
    markdown_dir: str | Path | None = None,
    ocr_threshold: float = 0.5,
    full_pipeline: bool = False,
    cache_dir: str | Path | None = None,
    vlm_enabled: bool = False,
    vlm_engine: str = "transformers",
    cascade: bool = False,
    cascade_skip_mupdf_threshold: float = 0.9,
    cascade_skip_pipeline: bool = False,
) -> dict[str, Any]:
    """Drive the full loop over a PDF directory.

    Parameters
    ----------
    full_pipeline:
        If True, needs-ocr PDFs go through layout analyser → Stage-B →
        pipeline/VLM parser. If False (default), only the MUPDF fast path
        runs (original MVP behaviour).
    cache_dir:
        Directory for LayoutCache. Defaults to ``out_path.parent / .cache``.
    vlm_enabled:
        Whether the VLM lane is open. Only relevant when ``full_pipeline``
        is True.
    cascade:
        If True, replace Stage-B with a quality-driven cascade: try the
        cheapest parser first, check its output with Layer-1 hard rules,
        and escalate to the next parser only if the gate rejects. Implies
        ``full_pipeline=True`` (layout analyser + pipeline parser are
        always loaded; VLM only if ``vlm_enabled``).
    cascade_skip_mupdf_threshold:
        When ``cascade`` is on, skip the MuPDF attempt entirely if
        Stage-A's ``ocr_prob`` is at or above this value (i.e. the
        router is highly confident the PDF needs OCR). Default 0.9.
    """
    from pdfsys_router import Router

    pdf_dir = Path(pdf_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    router = Router(model_path=router_weights, ocr_threshold=ocr_threshold)
    scorer = OcrQualityScorer(model_name=quality_model) if score_quality else None

    md_root = Path(markdown_dir) if markdown_dir else None
    if md_root is not None:
        md_root.mkdir(parents=True, exist_ok=True)

    # Cascade mode needs layout + pipeline parser unconditionally;
    # promote the flag so the loader block below loads them.
    if cascade:
        full_pipeline = True

    # Full pipeline components (lazy-loaded only if needed).
    layout_analyser = None
    pipeline_parser = None
    vlm_parser = None
    layout_cache = None

    if full_pipeline:
        cache_root = Path(cache_dir) if cache_dir else out_path.parent / ".cache"
        layout_cache = LayoutCache(cache_root / "layout")

        from pdfsys_layout_analyser import LayoutAnalyser

        layout_analyser = LayoutAnalyser()

        from pdfsys_parser_pipeline import PipelineParser

        pipeline_parser = PipelineParser()

        if vlm_enabled:
            from pdfsys_core import VlmConfig
            from pdfsys_parser_vlm import VlmParser

            vlm_parser = VlmParser(VlmConfig(engine=vlm_engine))

    summary: dict[str, Any] = {
        "pdf_dir": str(pdf_dir),
        "out_path": str(out_path),
        "full_pipeline": full_pipeline,
        "vlm_enabled": vlm_enabled,
        "cascade": cascade,
        "num_pdfs": 0,
        "by_backend": {},
        "by_stage_b": {},
        "by_cascade_decision": {},
        "num_extracted": 0,
        "num_scored": 0,
        "num_errors": 0,
        "sum_quality": 0.0,
        "started_at": time.time(),
    }

    with out_path.open("w", encoding="utf-8") as out_f:
        for pdf_path in _iter_pdfs(pdf_dir, limit):
            if cascade:
                row = _run_one_cascade(
                    pdf_path=pdf_path,
                    router=router,
                    scorer=scorer,
                    md_root=md_root,
                    layout_analyser=layout_analyser,
                    pipeline_parser=pipeline_parser,
                    vlm_parser=vlm_parser,
                    layout_cache=layout_cache,
                    vlm_enabled=vlm_enabled,
                    vlm_engine=vlm_engine,
                    skip_mupdf_threshold=cascade_skip_mupdf_threshold,
                    skip_pipeline=cascade_skip_pipeline,
                )
            else:
                row = _run_one(
                    pdf_path=pdf_path,
                    router=router,
                    scorer=scorer,
                    md_root=md_root,
                    full_pipeline=full_pipeline,
                    layout_analyser=layout_analyser,
                    pipeline_parser=pipeline_parser,
                    vlm_parser=vlm_parser,
                    layout_cache=layout_cache,
                    vlm_enabled=vlm_enabled,
                )
            out_f.write(row.to_json_line() + "\n")
            out_f.flush()

            summary["num_pdfs"] += 1
            by_b = summary["by_backend"]
            by_b[row.backend] = by_b.get(row.backend, 0) + 1
            if row.stage_b_backend:
                by_sb = summary["by_stage_b"]
                by_sb[row.stage_b_backend] = by_sb.get(row.stage_b_backend, 0) + 1
            if row.cascade_decision:
                by_cd = summary["by_cascade_decision"]
                by_cd[row.cascade_decision] = by_cd.get(row.cascade_decision, 0) + 1
            if row.extract_error is None and row.sha256 is not None:
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
    router: Any,
    scorer: OcrQualityScorer | None,
    md_root: Path | None,
    full_pipeline: bool,
    layout_analyser: Any,
    pipeline_parser: Any,
    vlm_parser: Any,
    layout_cache: LayoutCache | None,
    vlm_enabled: bool,
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

    # -- MUPDF fast path (text-ok) -------------------------------------------
    if decision.backend == Backend.MUPDF:
        return _extract_mupdf(row, pdf_path, scorer, md_root)

    # -- needs-ocr: if full pipeline is off, just record and skip -------------
    if not full_pipeline or layout_analyser is None:
        return row

    # -- Layout analysis ------------------------------------------------------
    extracted = None
    try:
        t2 = time.perf_counter()
        layout = layout_analyser.analyse(pdf_path)
        t3 = time.perf_counter()
        row.wall_ms_layout = (t3 - t2) * 1000.0
        row.sha256 = layout.sha256
        row.layout_model = layout.layout_model
        row.layout_has_complex = layout.has_complex_content

        total_regions = sum(len(p.regions) for p in layout.pages)
        row.layout_num_regions = total_regions

        # Save to cache.
        if layout_cache is not None:
            layout_cache.save(layout)

    except Exception as e:
        row.extract_error = f"layout_failed: {e}"
        return row

    # -- Stage-B decision -----------------------------------------------------
    from pdfsys_core import RouterConfig
    from pdfsys_router import decide

    stage_b = decide(layout, config=RouterConfig(vlm_enabled=vlm_enabled))
    row.stage_b_backend = stage_b.backend.value
    # Update the final backend to reflect Stage-B decision.
    row.backend = stage_b.backend.value

    # -- PIPELINE extraction --------------------------------------------------
    if stage_b.backend == Backend.PIPELINE and pipeline_parser is not None:
        try:
            t4 = time.perf_counter()
            extracted = pipeline_parser.extract(pdf_path)
            t5 = time.perf_counter()
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t5 - t4) * 1000.0
        except Exception as e:
            row.extract_error = f"pipeline_extract_failed: {e}"
            return row

    # -- VLM extraction -------------------------------------------------------
    elif stage_b.backend == Backend.VLM and vlm_parser is not None:
        try:
            t4 = time.perf_counter()
            extracted = vlm_parser.extract(pdf_path)
            t5 = time.perf_counter()
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t5 - t4) * 1000.0
        except Exception as e:
            row.extract_error = f"vlm_extract_failed: {e}"
            return row

    # -- DEFERRED: no extraction, just record ---------------------------------
    else:
        return row

    # -- Dump markdown --------------------------------------------------------
    if md_root is not None and extracted is not None and extracted.markdown:
        md_path = md_root / f"{extracted.sha256}.md"
        md_path.write_text(extracted.markdown, encoding="utf-8")

    # -- Quality scoring ------------------------------------------------------
    if scorer is not None and extracted is not None and extracted.markdown:
        try:
            t6 = time.perf_counter()
            q: QualityScore = scorer.score(extracted.markdown)
            t7 = time.perf_counter()
            row.quality_score = q.score
            row.quality_num_chars = q.num_chars
            row.quality_num_tokens = q.num_tokens
            row.quality_model = q.model
            row.wall_ms_quality = (t7 - t6) * 1000.0
        except Exception as e:
            row.extract_error = f"quality_failed: {e}"

    return row


def _run_one_cascade(
    *,
    pdf_path: Path,
    router: Any,
    scorer: OcrQualityScorer | None,
    md_root: Path | None,
    layout_analyser: Any,
    pipeline_parser: Any,
    vlm_parser: Any,
    layout_cache: LayoutCache | None,
    vlm_enabled: bool,
    vlm_engine: str = "transformers",
    skip_mupdf_threshold: float,
    skip_pipeline: bool = False,
) -> LoopResult:
    """Quality-driven cascade: cheapest parser → gate → escalate on fail.

    Always runs Stage-A first (cheap, ~10 ms) so the cascade can skip
    the MuPDF attempt entirely on PDFs the router is confident need OCR.
    Layout analysis is deferred until the first non-MuPDF stage that
    actually needs it, then shared between PIPELINE and VLM via a
    closure cache.
    """
    from pdfsys_parser_mupdf import extract_doc as mupdf_extract

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

    if decision.error:
        return row

    # Layout is computed at most once per PDF and shared across
    # pipeline/vlm stages via this closure cell.
    layout_holder: dict[str, Any] = {"layout": None, "wall_ms": 0.0}

    def _ensure_layout(path: Path) -> Any:
        if layout_holder["layout"] is None:
            t = time.perf_counter()
            layout = layout_analyser.analyse(path)
            layout_holder["wall_ms"] = (time.perf_counter() - t) * 1000.0
            if layout_cache is not None:
                layout_cache.save(layout)
            layout_holder["layout"] = layout
        return layout_holder["layout"]

    stages: list[CascadeStage] = []

    if decision.ocr_prob < skip_mupdf_threshold:
        stages.append(
            CascadeStage(name="mupdf", extract=lambda p: mupdf_extract(p))
        )

    if not skip_pipeline:
        def _pipeline_extract(p: Path) -> Any:
            return pipeline_parser.extract(p)

        stages.append(CascadeStage(name="pipeline", extract=_pipeline_extract))

    if vlm_enabled and vlm_parser is not None:
        def _vlm_extract(p: Path) -> Any:
            return vlm_parser.extract(p)

        stages.append(CascadeStage(name="vlm", extract=_vlm_extract))

    cascade_result = run_cascade(pdf_path, stages, num_pages=decision.num_pages)

    row.wall_ms_layout = layout_holder["wall_ms"]
    row.wall_ms_extract = cascade_result.total_wall_ms - row.wall_ms_layout
    row.cascade_decision = cascade_result.decision
    row.cascade_final_stage = cascade_result.final_stage
    row.cascade_attempts = [a.as_record() for a in cascade_result.attempts]
    row.cascade_total_wall_ms = cascade_result.total_wall_ms

    if cascade_result.final_stage is not None:
        row.backend = cascade_result.final_stage

    extracted = cascade_result.extracted
    if extracted is not None:
        row.sha256 = extracted.sha256
        row.extract_stats = dict(extracted.stats)
        row.markdown_chars = extracted.char_count

    # Bubble up the last attempt's error (if any) so existing JSONL
    # consumers that only look at extract_error still see "something
    # went wrong" on reject.
    if cascade_result.decision == "reject" and cascade_result.attempts:
        last_error = cascade_result.attempts[-1].error
        if last_error is not None:
            row.extract_error = f"cascade_reject: {last_error}"
        else:
            row.extract_error = "cascade_reject"

    if md_root is not None and extracted is not None and extracted.markdown:
        md_path = md_root / f"{extracted.sha256}.md"
        md_path.write_text(extracted.markdown, encoding="utf-8")

    # Quality-score the published output only — no point scoring rejects.
    if (
        cascade_result.decision == "publish"
        and scorer is not None
        and extracted is not None
        and extracted.markdown
    ):
        try:
            t_q = time.perf_counter()
            q: QualityScore = scorer.score(extracted.markdown)
            row.quality_score = q.score
            row.quality_num_chars = q.num_chars
            row.quality_num_tokens = q.num_tokens
            row.quality_model = q.model
            row.wall_ms_quality = (time.perf_counter() - t_q) * 1000.0
        except Exception as e:
            row.extract_error = f"quality_failed: {e}"

    return row


def _extract_mupdf(
    row: LoopResult,
    pdf_path: Path,
    scorer: OcrQualityScorer | None,
    md_root: Path | None,
) -> LoopResult:
    """Handle the MUPDF text-ok fast path (unchanged from MVP)."""
    from pdfsys_parser_mupdf import extract_doc

    try:
        t2 = time.perf_counter()
        extracted = extract_doc(pdf_path)
        t3 = time.perf_counter()
        row.sha256 = extracted.sha256
        row.extract_stats = dict(extracted.stats)
        row.markdown_chars = extracted.char_count
        row.wall_ms_extract = (t3 - t2) * 1000.0
    except Exception as e:
        row.extract_error = f"extract_failed: {e}"
        return row

    if md_root is not None and extracted.markdown:
        md_path = md_root / f"{extracted.sha256}.md"
        md_path.write_text(extracted.markdown, encoding="utf-8")

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
        except Exception as e:
            row.extract_error = f"quality_failed: {e}"

    return row
