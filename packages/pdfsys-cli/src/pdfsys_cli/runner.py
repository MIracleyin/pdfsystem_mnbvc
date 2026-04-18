"""Stage-aware pipeline runner.

Processes a directory of PDFs according to a :class:`RunConfig`. Each PDF
flows through only the stages specified in ``config.stages``, in canonical
order: router → layout → extract → quality.

All heavy dependencies are imported lazily at first use.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import RunConfig


@dataclass(slots=True)
class DocResult:
    """Per-PDF result row, serialized to JSONL."""

    pdf_path: str = ""
    sha256: str | None = None
    # router
    backend: str | None = None
    ocr_prob: float | None = None
    num_pages: int = 0
    is_form: bool = False
    garbled_text_ratio: float = 0.0
    router_error: str | None = None
    # layout
    layout_model: str | None = None
    layout_num_regions: int | None = None
    layout_has_complex: bool | None = None
    stage_b_backend: str | None = None
    # extract
    extract_backend: str | None = None
    extract_stats: dict[str, Any] = field(default_factory=dict)
    extract_error: str | None = None
    markdown_chars: int = 0
    # quality
    quality_score: float | None = None
    quality_num_chars: int | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    # timing
    wall_ms_router: float = 0.0
    wall_ms_layout: float = 0.0
    wall_ms_extract: float = 0.0
    wall_ms_quality: float = 0.0

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Components:
    """Lazy container for all pipeline components. Loads only what's needed."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._router: Any = None
        self._analyser: Any = None
        self._pipeline: Any = None
        self._vlm: Any = None
        self._scorer: Any = None
        self._layout_cache: Any = None

    @property
    def router(self) -> Any:
        if self._router is None:
            from pdfsys_router import Router  # noqa: PLC0415

            self._router = Router(
                model_path=self.cfg.router.weights,
                ocr_threshold=self.cfg.router.ocr_threshold,
            )
        return self._router

    @property
    def analyser(self) -> Any:
        if self._analyser is None:
            from pdfsys_layout_analyser import LayoutAnalyser  # noqa: PLC0415
            from pdfsys_core import LayoutConfig  # noqa: PLC0415

            lc = LayoutConfig(render_dpi=self.cfg.layout.render_dpi)
            self._analyser = LayoutAnalyser(
                config=lc,
                model_path=self.cfg.layout.model,
                conf_threshold=self.cfg.layout.conf_threshold,
                iou_threshold=self.cfg.layout.iou_threshold,
            )
        return self._analyser

    @property
    def pipeline_parser(self) -> Any:
        if self._pipeline is None:
            from pdfsys_parser_pipeline import PipelineParser  # noqa: PLC0415
            from pdfsys_core import PipelineConfig  # noqa: PLC0415

            pc = PipelineConfig(
                ocr_engine=self.cfg.pipeline.ocr_engine,
                languages=tuple(self.cfg.pipeline.languages),
                render_dpi=self.cfg.pipeline.render_dpi,
            )
            self._pipeline = PipelineParser(config=pc)
        return self._pipeline

    @property
    def vlm_parser(self) -> Any:
        if self._vlm is None:
            from pdfsys_parser_vlm import VlmParser  # noqa: PLC0415
            from pdfsys_core import VlmConfig  # noqa: PLC0415

            vc = VlmConfig(model=self.cfg.vlm.model)
            self._vlm = VlmParser(config=vc)
        return self._vlm

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            from pdfsys_bench.quality import OcrQualityScorer  # noqa: PLC0415

            self._scorer = OcrQualityScorer(
                model_name=self.cfg.quality.model,
                max_tokens=self.cfg.quality.max_tokens,
                device=self.cfg.quality.device,
            )
        return self._scorer

    @property
    def layout_cache(self) -> Any:
        if self._layout_cache is None:
            from pdfsys_core import LayoutCache  # noqa: PLC0415

            self._layout_cache = LayoutCache(self.cfg.cache_path / "layout")
        return self._layout_cache


def run(cfg: RunConfig) -> dict[str, Any]:
    """Execute the pipeline according to *cfg*. Returns summary dict."""
    # Set thread env vars before any torch import.
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.runtime.omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.runtime.omp_threads))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.markdown_path:
        cfg.markdown_path.mkdir(parents=True, exist_ok=True)

    comps = Components(cfg)

    summary: dict[str, Any] = {
        "config_stages": cfg.stages,
        "pdf_dir": cfg.input.pdf_dir,
        "num_pdfs": 0,
        "by_backend": {},
        "by_stage_b": {},
        "num_extracted": 0,
        "num_scored": 0,
        "num_errors": 0,
        "sum_quality": 0.0,
        "started_at": time.time(),
    }

    with cfg.jsonl_path.open("w", encoding="utf-8") as out_f:
        for pdf_path in _iter_pdfs(Path(cfg.input.pdf_dir), cfg.input.limit):
            row = _process_one(pdf_path, cfg, comps)
            out_f.write(row.to_json_line() + "\n")
            out_f.flush()

            summary["num_pdfs"] += 1
            if row.backend:
                by_b = summary["by_backend"]
                final = row.extract_backend or row.backend
                by_b[final] = by_b.get(final, 0) + 1
            if row.stage_b_backend:
                by_sb = summary["by_stage_b"]
                by_sb[row.stage_b_backend] = by_sb.get(row.stage_b_backend, 0) + 1
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

    summary_path = cfg.jsonl_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    summary["summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------- per-pdf

def _process_one(pdf_path: Path, cfg: RunConfig, comps: Components) -> DocResult:
    row = DocResult(pdf_path=str(pdf_path))

    # ---- router ----
    if cfg.has_stage("router"):
        _stage_router(row, pdf_path, comps)

    # ---- layout ----
    layout = None
    if cfg.has_stage("layout") and _needs_ocr(row):
        layout = _stage_layout(row, pdf_path, comps, cfg)

    # ---- extract ----
    extracted = None
    if cfg.has_stage("extract"):
        extracted = _stage_extract(row, pdf_path, layout, comps, cfg)

    # ---- quality ----
    if cfg.has_stage("quality") and cfg.quality.enabled and extracted is not None:
        _stage_quality(row, extracted, comps)

    return row


def _needs_ocr(row: DocResult) -> bool:
    from pdfsys_core import Backend  # noqa: PLC0415

    return row.backend is not None and row.backend != Backend.MUPDF.value


# ---------------------------------------------------------------- stages

def _stage_router(row: DocResult, pdf_path: Path, comps: Components) -> None:
    t0 = time.perf_counter()
    decision = comps.router.classify(pdf_path)
    t1 = time.perf_counter()

    row.backend = decision.backend.value
    row.ocr_prob = decision.ocr_prob
    row.num_pages = decision.num_pages
    row.is_form = decision.is_form
    row.garbled_text_ratio = decision.garbled_text_ratio
    row.router_error = decision.error
    row.wall_ms_router = (t1 - t0) * 1000.0


def _stage_layout(
    row: DocResult, pdf_path: Path, comps: Components, cfg: RunConfig
) -> Any:
    """Run layout analysis. Returns the LayoutDocument or None on error."""
    try:
        t0 = time.perf_counter()
        layout = comps.analyser.analyse(pdf_path)
        t1 = time.perf_counter()

        row.sha256 = layout.sha256
        row.layout_model = layout.layout_model
        row.layout_has_complex = layout.has_complex_content
        row.layout_num_regions = sum(len(p.regions) for p in layout.pages)
        row.wall_ms_layout = (t1 - t0) * 1000.0

        comps.layout_cache.save(layout)

        # Stage-B decision.
        from pdfsys_router import decide  # noqa: PLC0415
        from pdfsys_core import RouterConfig  # noqa: PLC0415

        sb = decide(layout, config=RouterConfig(vlm_enabled=cfg.vlm.enabled))
        row.stage_b_backend = sb.backend.value

        return layout
    except Exception as e:  # noqa: BLE001
        row.extract_error = f"layout_failed: {e}"
        return None


def _stage_extract(
    row: DocResult, pdf_path: Path, layout: Any, comps: Components, cfg: RunConfig
) -> Any:
    """Run extraction. Returns the ExtractedDoc or None on error."""
    from pdfsys_core import Backend  # noqa: PLC0415

    backend = row.stage_b_backend or row.backend
    extracted = None

    # MUPDF fast path.
    if backend == Backend.MUPDF.value or backend is None:
        try:
            from pdfsys_parser_mupdf import extract_doc  # noqa: PLC0415

            t0 = time.perf_counter()
            extracted = extract_doc(pdf_path)
            t1 = time.perf_counter()
            row.extract_backend = Backend.MUPDF.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            row.extract_error = f"mupdf_extract_failed: {e}"
            return None

    # Pipeline OCR path.
    elif backend == Backend.PIPELINE.value and layout is not None:
        try:
            t0 = time.perf_counter()
            extracted = comps.pipeline_parser.extract(pdf_path, layout)
            t1 = time.perf_counter()
            row.extract_backend = Backend.PIPELINE.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            row.extract_error = f"pipeline_extract_failed: {e}"
            return None

    # VLM path.
    elif backend == Backend.VLM.value and layout is not None:
        try:
            t0 = time.perf_counter()
            extracted = comps.vlm_parser.extract_complex_pages(pdf_path, layout)
            t1 = time.perf_counter()
            row.extract_backend = Backend.VLM.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            row.extract_error = f"vlm_extract_failed: {e}"
            return None

    # DEFERRED or no layout — skip extraction.
    else:
        row.extract_backend = backend
        return None

    # Dump markdown.
    if cfg.markdown_path and extracted is not None and extracted.markdown:
        md_path = cfg.markdown_path / f"{extracted.sha256}.md"
        md_path.write_text(extracted.markdown, encoding="utf-8")

    return extracted


def _stage_quality(row: DocResult, extracted: Any, comps: Components) -> None:
    if not extracted.markdown:
        return
    try:
        t0 = time.perf_counter()
        q = comps.scorer.score(extracted.markdown)
        t1 = time.perf_counter()
        row.quality_score = q.score
        row.quality_num_chars = q.num_chars
        row.quality_num_tokens = q.num_tokens
        row.quality_model = q.model
        row.wall_ms_quality = (t1 - t0) * 1000.0
    except Exception as e:  # noqa: BLE001
        row.extract_error = f"quality_failed: {e}"


# ---------------------------------------------------------------- util

def _iter_pdfs(root: Path, limit: int | None) -> Iterable[Path]:
    pdfs = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    if limit is not None:
        pdfs = pdfs[:limit]
    yield from pdfs
