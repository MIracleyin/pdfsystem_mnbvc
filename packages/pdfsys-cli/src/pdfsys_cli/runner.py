"""Stage-aware pipeline runner.

Processes a directory of PDFs according to a :class:`RunConfig`. Each PDF
flows through only the stages specified in ``config.stages``, in canonical
order: router → layout → extract → quality.

All heavy dependencies are imported lazily at first use.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import RunConfig
from .parquet_writer import ParquetSink


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
    is_encrypted: bool = False
    router_error: str | None = None
    # layout
    layout_model: str | None = None
    layout_num_regions: int | None = None
    layout_has_complex: bool | None = None
    stage_b_backend: str | None = None
    # extract
    extract_backend: str | None = None
    extract_stats: dict[str, Any] = field(default_factory=dict)
    markdown_chars: int = 0
    #: Why this row produced no text, when that was a decision rather than a
    #: failure. ``None`` on every row that actually extracted. Without it a
    #: deferred document and a document whose backend was never reached are
    #: indistinguishable from a successful extraction of an empty PDF.
    skip_reason: str | None = None
    # quality
    quality_score: float | None = None
    quality_num_chars: int | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    # vlm region-based (only populated for VLM rows)
    segments_excerpt: list[dict] = field(default_factory=list)
    region_failures: int | None = None
    # error capture
    error_class: str | None = None  # router | layout | extract_mupdf | extract_pipeline | extract_vlm | quality
    error_message: str | None = None  # f"{type(e).__name__}: {e}", truncated to 500 chars
    # timing
    wall_ms_router: float = 0.0
    wall_ms_layout: float = 0.0
    wall_ms_extract: float = 0.0
    wall_ms_quality: float = 0.0

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _as_path(value: str | None) -> Path | None:
    """Config carries paths as strings; the parser configs want Path or None."""
    return Path(value) if value else None


def parser_output_dirs(cfg: RunConfig) -> dict[str, str | None]:
    """Where each OCR backend this run can reach would keep its sidecars.

    Keyed by backend, because the two are configured independently in YAML —
    checking only ``pipeline`` gives a VLM lane a false all-clear.
    """
    return {
        b: getattr(cfg, b).output_dir
        for b in ("pipeline", "vlm")
        if cfg.has_stage("extract") and _in_lane(b, cfg)
    }


def _check_parser_output_dirs(cfg: RunConfig) -> None:
    """Make the sidecar directories now, so a bad path costs nothing.

    Persisting happens inside ``extract()``, after the PDF has been through
    MinerU. An unwritable path therefore raises *per document*, after the GPU
    work is done, and ``_stage_extract`` turns that into an error row — which
    also skips the markdown dump below it. So a mistyped flag would burn the
    whole run's GPU budget and produce nothing, not even the markdown that used
    to survive unconditionally. One mkdir up front turns that into one message.
    """
    for backend, raw in parser_output_dirs(cfg).items():
        if not raw:
            continue
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".pdfsys-write-probe"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as e:
            raise ParserOutputDirError(
                f"{backend} sidecar directory {path} is not usable: {e}. "
                f"Sidecars are written after each document is parsed, so this "
                f"would fail once per document — after the OCR work — and take "
                f"the markdown with it."
            ) from e


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
            from pdfsys_router import Router

            self._router = Router(
                model_path=self.cfg.router.weights,
                ocr_threshold=self.cfg.router.ocr_threshold,
            )
        return self._router

    @property
    def analyser(self) -> Any:
        if self._analyser is None:
            from pdfsys_core import LayoutConfig
            from pdfsys_layout_analyser import LayoutAnalyser

            lc = LayoutConfig(render_dpi=self.cfg.layout.render_dpi)
            self._analyser = LayoutAnalyser(
                config=lc,
                model_path=self.cfg.layout.model,
                backend=self.cfg.layout.backend,
                conf_threshold=self.cfg.layout.conf_threshold,
                iou_threshold=self.cfg.layout.iou_threshold,
            )
        return self._analyser

    @property
    def pipeline_parser(self) -> Any:
        if self._pipeline is None:
            from pdfsys_core import PipelineConfig
            from pdfsys_parser_pipeline import PipelineParser

            pc = PipelineConfig(
                formula_enable=self.cfg.pipeline.formula_enable,
                table_enable=self.cfg.pipeline.table_enable,
                p_lang=self.cfg.pipeline.p_lang,
                output_dir=_as_path(self.cfg.pipeline.output_dir),
                return_images=self.cfg.pipeline.return_images,
            )
            self._pipeline = PipelineParser(config=pc)
        return self._pipeline

    @property
    def vlm_parser(self) -> Any:
        if self._vlm is None:
            from pdfsys_core import VlmConfig
            from pdfsys_parser_vlm import VlmParser

            vc = VlmConfig(
                engine=self.cfg.vlm.engine,
                formula_enable=self.cfg.vlm.formula_enable,
                table_enable=self.cfg.vlm.table_enable,
                p_lang=self.cfg.vlm.p_lang,
                output_dir=_as_path(self.cfg.vlm.output_dir),
                return_images=self.cfg.vlm.return_images,
            )
            self._vlm = VlmParser(config=vc)
        return self._vlm

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            from pdfsys_bench.quality import OcrQualityScorer

            self._scorer = OcrQualityScorer(
                model_name=self.cfg.quality.model,
                max_tokens=self.cfg.quality.max_tokens,
                max_chars=self.cfg.quality.max_chars,
                device=self.cfg.quality.device,
            )
        return self._scorer

    @property
    def layout_cache(self) -> Any:
        if self._layout_cache is None:
            from pdfsys_core import LayoutCache

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
    _check_parser_output_dirs(cfg)

    comps = Components(cfg)
    inputs, discovery = resolve_inputs(cfg)

    summary: dict[str, Any] = {
        "config_stages": cfg.stages,
        "pdf_dir": cfg.input.pdf_dir,
        "pdf_list": cfg.input.pdf_list,
        "path_root": cfg.input.path_root,
        "resume": cfg.resume,
        "discovery": discovery,
        "num_pdfs": 0,
        "by_backend": {},
        "by_stage_b": {},
        "num_extracted": 0,
        "num_scored": 0,
        "num_errors": 0,
        "num_skipped": 0,
        "by_skip_reason": {},
        "sum_quality": 0.0,
        "started_at": time.time(),
    }

    # ---- optional parquet sink (opened lazily, closed via context) ----
    parquet_sink: ParquetSink | None = None
    if cfg.has_stage("parquet") and cfg.parquet.enabled:
        parquet_sink = ParquetSink(
            path=cfg.parquet_path,
            compression=cfg.parquet.compression,
            quality_threshold=cfg.parquet.quality_threshold,
            include_markdown=cfg.parquet.include_markdown,
        )

    # ---- resume: carry the existing rows into the summary, skip their PDFs ----
    done_keys: set[str] = set()
    if cfg.resume and cfg.jsonl_path.exists():

        conflicts: list[str] = []

        def _carry(row: dict[str, Any]) -> None:
            _tally(summary, row)
            done_keys.update(_path_keys(row.get("pdf_path")))
            # A row another lane filtered out is not "done" — it was handed to
            # whoever runs that backend. If that is us, resume is about to skip
            # the very documents we were started to extract.
            if row.get("skip_reason") == "lane-filter" and _in_lane(
                row.get("extract_backend"), cfg
            ):
                conflicts.append(row.get("pdf_path") or "")

        n_carried, good_bytes = _scan_completed(cfg.jsonl_path, _carry)
        if conflicts:
            # Raise here, before the summary is written or a single document is
            # touched. Detected up front but reported at the end, this would
            # run the whole leg and overwrite the other lane's summary.json on
            # the way to exiting non-zero.
            raise LaneConflictError(
                f"{len(conflicts)} documents in {cfg.jsonl_path} were filtered out "
                f"by an earlier lane and belong to this one "
                f"({', '.join(cfg.extract_backends or RUNNABLE_BACKENDS)}), but "
                f"--resume treats any row as done and would skip them "
                f"(e.g. {conflicts[0]}). Give each lane its own --out-dir; "
                f"results.jsonl is append-only, so one directory cannot hold two "
                f"passes over the same document."
            )
        size = cfg.jsonl_path.stat().st_size
        if good_bytes < size:
            # An interrupted write leaves a partial final line. Appending after
            # it would splice two records; truncating to the last complete one
            # costs at most the row we were mid-write on. _scan_completed has
            # already refused anything that is not tail damage.
            os.truncate(cfg.jsonl_path, good_bytes)
            summary["repaired_tail_bytes"] = size - good_bytes
        summary["resumed_rows"] = n_carried

        # Resume is per document, not per stage: a document already in the file
        # is skipped whole. So resuming with a longer stage list does not go
        # back and fill the earlier rows in, and the shard would silently mix
        # two depths of processing. Stages this invocation dropped are excluded
        # from the comparison — --resume drops `parquet` itself, and warning
        # about our own removal on every single resume would train the operator
        # to ignore the message.
        previous = _previous_stages(cfg.jsonl_path.with_suffix(".summary.json"))
        if previous is not None:
            before = [s for s in previous if s not in cfg.dropped_stages]
            if before != cfg.stages:
                summary["resumed_stage_mismatch"] = before

    paths = [p for p in inputs if not _path_keys(str(p)) & done_keys]
    summary["num_skipped_as_done"] = len(inputs) - len(paths)

    summary_path = cfg.jsonl_path.with_suffix(".summary.json")
    # Write it once before any work, not only on a clean exit. The stage list it
    # carries is what a later --resume compares against, and the case that
    # comparison exists for is precisely the leg that was killed and never
    # reached the end.
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    mode = "a" if cfg.resume else "w"
    try:
        with cfg.jsonl_path.open(mode, encoding="utf-8") as out_f:
            for pdf_path in paths:
                row, extracted = _process_one(pdf_path, cfg, comps)
                out_f.write(row.to_json_line() + "\n")
                out_f.flush()

                if parquet_sink is not None:
                    md = extracted.markdown if extracted is not None else None
                    parquet_sink.write_row(row, md)

                _tally(summary, asdict(row))
    finally:
        if parquet_sink is not None:
            parquet_sink.close()
            summary["parquet_rows"] = parquet_sink.rows_written
            summary["parquet_path"] = str(cfg.parquet_path)

    summary["finished_at"] = time.time()
    # This leg only — num_pdfs spans every leg of a resumed run, so pairing the
    # two would read as a throughput they never achieved.
    summary["wall_seconds"] = summary["finished_at"] - summary["started_at"]
    summary["leg_num_pdfs"] = len(paths)
    summary["avg_quality"] = (
        summary["sum_quality"] / summary["num_scored"] if summary["num_scored"] else None
    )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    summary["summary_path"] = str(summary_path)

    return summary


def _set_error(row: DocResult, error_class: str, exc: BaseException) -> None:
    """Record the first error to hit this row; later errors are dropped."""
    if row.error_class is not None:
        return
    row.error_class = error_class
    msg = f"{type(exc).__name__}: {exc}"
    row.error_message = msg[:500]


# ---------------------------------------------------------------- per-pdf

def _process_one(
    pdf_path: Path, cfg: RunConfig, comps: Components
) -> tuple[DocResult, Any]:
    """Run all configured stages for one PDF.

    Returns
    -------
    tuple
        ``(row, extracted)`` — extracted is the in-memory ExtractedDoc when
        extraction succeeded, else ``None``. The caller uses extracted to
        feed the parquet sink without re-reading markdown from disk.
    """
    row = DocResult(pdf_path=str(pdf_path))

    # ---- router ----
    if cfg.has_stage("router"):
        _stage_router(row, pdf_path, comps)

    # ---- layout ----
    # Skipped when no OCR backend is in this lane: layout exists to produce the
    # stage-B decision between pipeline and vlm, and if we run neither, that
    # decision cannot change what happens to the document. Running it anyway
    # means a CPU box paying DocLayout-YOLO for every document it is about to
    # hand away — precisely the documents it is not extracting.
    layout = None
    if (
        cfg.has_stage("layout")
        and _needs_ocr(row)
        and any(_in_lane(b, cfg) for b in ("pipeline", "vlm"))
    ):
        layout = _stage_layout(row, pdf_path, comps, cfg)

    # ---- extract ----
    extracted = None
    if cfg.has_stage("extract"):
        extracted = _stage_extract(row, pdf_path, layout, comps, cfg)

    # ---- quality ----
    if cfg.has_stage("quality") and cfg.quality.enabled and extracted is not None:
        _stage_quality(row, extracted, comps)

    return row, extracted


#: Backends a process can actually be asked to run. ``deferred`` is not one of
#: them — it is stage-B declining, and labelling that a lane filter would hide
#: the reason the document was held back.
RUNNABLE_BACKENDS = ("mupdf", "pipeline", "vlm")
_RUNNABLE_BACKENDS = frozenset(RUNNABLE_BACKENDS)


def _in_lane(backend: str | None, cfg: RunConfig) -> bool:
    """Does this process run *backend*?

    ``extract_backends is None`` means every backend — which makes it the
    *widest* lane, not an absent one. Reading it as "no lane configured, so
    nothing to check" gets the resume conflict test exactly backwards: the
    process willing to run everything is the one most certain to own a document
    an earlier lane handed away.
    """
    if cfg.extract_backends is None:
        return True
    return backend in cfg.extract_backends


def _needs_ocr(row: DocResult) -> bool:
    from pdfsys_core import Backend

    return row.backend is not None and row.backend != Backend.MUPDF.value


# ---------------------------------------------------------------- stages

def _stage_router(row: DocResult, pdf_path: Path, comps: Components) -> None:
    t0 = time.perf_counter()
    decision = comps.router.classify(pdf_path)
    t1 = time.perf_counter()

    # doc_id for every routed document, not just the ones that go on to layout
    # or extraction. It is the primary key of pdfsys.page/v2, so a row without
    # it cannot be joined to anything — which is exactly what a worklist row
    # for a document handed to another machine needs to be. The later stages
    # recompute the identical value and overwrite it.
    try:
        row.sha256 = _sha256_of_file(pdf_path)
    except OSError as e:
        # classify() reports unreadable files through decision.error rather
        # than raising, so this is the one place a dead file would otherwise
        # take the whole run down.
        _set_error(row, "router", e)

    row.backend = decision.backend.value
    row.ocr_prob = decision.ocr_prob
    row.num_pages = decision.num_pages
    row.is_form = decision.is_form
    row.garbled_text_ratio = decision.garbled_text_ratio
    row.is_encrypted = decision.is_encrypted
    row.router_error = decision.error
    if decision.error is not None:
        _set_error(row, "router", RuntimeError(decision.error))
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
        from pdfsys_core import RouterConfig
        from pdfsys_router import decide

        sb = decide(layout, config=RouterConfig(vlm_enabled=cfg.vlm.enabled))
        row.stage_b_backend = sb.backend.value

        return layout
    except Exception as e:
        _set_error(row, "layout", e)
        return None


def _stage_extract(
    row: DocResult, pdf_path: Path, layout: Any, comps: Components, cfg: RunConfig
) -> Any:
    """Run extraction. Returns the ExtractedDoc or None on error."""
    from pdfsys_core import Backend

    backend = row.stage_b_backend or row.backend
    extracted = None

    # An earlier stage already failed — the router could not read the file, or
    # the layout model died. Nothing after this is a routing decision, so the
    # row keeps its error and gets no skip label: the two counters stay
    # disjoint, and the hand-off worklist (built from skip_reason) does not
    # queue a broken document onto another machine as routine work.
    if row.error_class is not None:
        row.extract_backend = backend
        return None

    # Lane filter. This process runs only the backends it was told to; the rest
    # belong to another machine, or already ran on one. Saying so explicitly is
    # what lets the OCR branches below drop their `layout is not None` guard —
    # a guard that filtered lanes only by accident, and could not mean "mupdf
    # only" on one box and "MinerU only" on another.
    if backend in _RUNNABLE_BACKENDS and not _in_lane(backend, cfg):
        row.extract_backend = backend
        row.skip_reason = "lane-filter"
        return None

    # Layout was asked for and did not deliver, without raising. Do not extract
    # on top of a layout stage that produced nothing.
    if (
        cfg.has_stage("layout")
        and layout is None
        and backend in (Backend.PIPELINE.value, Backend.VLM.value)
    ):
        row.extract_backend = backend
        return None

    # MUPDF fast path.
    if backend == Backend.MUPDF.value or backend is None:
        try:
            from pdfsys_parser_mupdf import extract_doc

            t0 = time.perf_counter()
            extracted = extract_doc(pdf_path)
            t1 = time.perf_counter()
            row.extract_backend = Backend.MUPDF.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
        except Exception as e:
            _set_error(row, "extract_mupdf", e)
            return None

    # Pipeline OCR path. No layout requirement: MinerU does its own layout
    # analysis internally and is handed only the PDF bytes, so a box that runs
    # MinerU does not need to have run ours.
    elif backend == Backend.PIPELINE.value:
        try:
            t0 = time.perf_counter()
            extracted = comps.pipeline_parser.extract(pdf_path)
            t1 = time.perf_counter()
            row.extract_backend = Backend.PIPELINE.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
        except Exception as e:
            _set_error(row, "extract_pipeline", e)
            return None

    # VLM path. Only stage-B ever says "vlm", so reaching here already implies
    # the layout stage ran — but the branch no longer depends on that.
    elif backend == Backend.VLM.value:
        try:
            t0 = time.perf_counter()
            extracted = comps.vlm_parser.extract(pdf_path)
            t1 = time.perf_counter()
            row.extract_backend = Backend.VLM.value
            row.sha256 = extracted.sha256
            row.extract_stats = dict(extracted.stats)
            row.markdown_chars = extracted.char_count
            row.wall_ms_extract = (t1 - t0) * 1000.0
            # Region-based VLM exposes per-segment content + region failures
            # for viz consumption.
            row.region_failures = extracted.stats.get("region_failures")
            row.segments_excerpt = [
                {
                    "page_index": s.page_index,
                    "type": s.type.value,
                    "bbox": [s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1] if s.bbox else None,
                    "content": (s.content or "")[:200],
                }
                for s in extracted.segments
            ]
        except Exception as e:
            _set_error(row, "extract_vlm", e)
            return None

    # Stage-B held this document back, or named a backend we do not have. Say
    # which — both used to arrive here as a silent no-op that the summary
    # counted as a success.
    else:
        row.extract_backend = backend
        if backend == Backend.DEFERRED.value:
            row.skip_reason = "deferred"
        else:
            row.skip_reason = f"unknown-backend:{backend}"
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
    except Exception as e:
        _set_error(row, "quality", e)


# ---------------------------------------------------------------- inputs

def resolve_inputs(cfg: RunConfig) -> tuple[list[Path], dict[str, Any]]:
    """Work out which PDFs this run covers, and say how it decided.

    Either a worklist (``input.pdf_list``) or a directory scan
    (``input.pdf_dir``) — the worklist wins when both are set, because it is
    the more specific instruction. The returned dict goes into the run summary
    so a shard can be traced back to the exact corpus it covers.
    """
    from pdfsys_core import read_pdf_list, take_inventory

    info: dict[str, Any] = {}
    if cfg.input.pdf_list:
        worklist = read_pdf_list(cfg.input.pdf_list, path_root=cfg.input.path_root)
        paths = list(worklist.paths)
        info["source"] = "list"
        info["entries"] = worklist.entries
        info["missing"] = len(worklist.missing)
        info["missing_examples"] = list(worklist.missing[:5])
        info["duplicates"] = len(worklist.duplicates)
        info["duplicate_examples"] = list(worklist.duplicates[:5])
    else:
        inventory = take_inventory(cfg.input.pdf_dir)
        paths = list(inventory.paths)
        info["source"] = "scan"
        info["by_suffix"] = len(inventory.by_suffix)
        info["by_magic"] = len(inventory.by_magic)

    if cfg.input.limit is not None:
        # Applied before the resume filter, so --limit names the same slice of
        # the corpus on every invocation rather than "N more each time".
        paths = paths[: cfg.input.limit]

    # Absolute, always. `pdf_path` in results.jsonl is what --resume matches on
    # and what another machine reads as a worklist, and a relative path means
    # something different from a different working directory — a supervisor
    # restarting the job from elsewhere would silently reprocess the whole
    # corpus. Resolving here is the one place both input routes pass through.
    paths = [p.resolve() for p in paths]

    info["selected"] = len(paths)
    return paths, info


def _path_keys(path: str | None) -> set[str]:
    """The strings that could name this same file in results.jsonl.

    ``resolve_inputs`` records absolute paths, so new files match on the string
    alone. The resolved form is kept as a second key so a results.jsonl written
    before that — holding paths relative to whatever directory the run started
    in — still matches when resumed from that same directory. It cannot rescue
    a relative path resumed from elsewhere; nothing can, which is why the paths
    are absolute now.
    """
    if not path:
        return set()
    keys = {str(path)}
    with contextlib.suppress(OSError):
        keys.add(str(Path(path).resolve()))
    return keys


def _previous_stages(summary_path: Path) -> list[str] | None:
    """The stage list the run being resumed was started with, if recorded."""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stages = data.get("config_stages")
    return stages if isinstance(stages, list) else None


class CorruptResultsError(RuntimeError):
    """results.jsonl has damage that cannot be attributed to an interrupted write."""


class LaneConflictError(RuntimeError):
    """Resuming here would skip documents an earlier lane handed to this one."""


class ParserOutputDirError(RuntimeError):
    """The configured sidecar directory cannot be written to."""


def _scan_completed(
    path: Path, on_row: Callable[[dict[str, Any]], None]
) -> tuple[int, int]:
    """Stream an existing results.jsonl. Returns (rows, bytes of intact prefix).

    Rows are handed to *on_row* and dropped, never accumulated: a 218k-document
    results.jsonl is hundreds of megabytes, and resume needs a tally and a set
    of keys, not the corpus.

    A line counts as complete only when it is newline-terminated *and* parses.
    Requiring the newline is what makes the byte count land on a record
    boundary — a final line that is valid JSON but lost its terminator would
    otherwise be counted as intact, and the next append would splice the two
    records into one.

    Damage in the *tail* is an interrupted write, which is expected and
    recoverable. Damage anywhere else is not explicable that way, and this
    raises rather than guessing: JSONL records are framed independently, so a
    bad line in the middle says nothing about the intact rows after it, and
    truncating to the prefix would delete work that was really done.
    """
    n_rows = 0
    good = 0
    bad_at: int | None = None
    trailing = 0
    with path.open("rb") as f:
        for lineno, raw in enumerate(f, start=1):
            ok = raw.endswith(b"\n")
            if ok:
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    ok = False
                else:
                    ok = isinstance(row, dict)
            if not ok:
                if bad_at is None:
                    bad_at = lineno
                trailing += len(raw)
                continue
            if bad_at is not None:
                raise CorruptResultsError(
                    f"{path} 第 {bad_at} 行损坏，但后面还有完好的记录"
                    f"（例如第 {lineno} 行）。中间的坏行无法用“写到一半被中断”解释，"
                    f"自动截断会删掉真的做过的工作。请人工检查后再用 --resume。"
                )
            on_row(row)
            n_rows += 1
            good += len(raw)
    return n_rows, good


def _tally(summary: dict[str, Any], row: dict[str, Any]) -> None:
    """Fold one result row into the summary counters.

    Takes a dict rather than a DocResult so a run being resumed can replay the
    rows it already wrote through the identical arithmetic — otherwise the
    summary would describe only the last leg of a restarted run.
    """
    summary["num_pdfs"] += 1
    if row.get("backend"):
        by_b = summary["by_backend"]
        final = row.get("extract_backend") or row["backend"]
        by_b[final] = by_b.get(final, 0) + 1
    if row.get("stage_b_backend"):
        by_sb = summary["by_stage_b"]
        by_sb[row["stage_b_backend"]] = by_sb.get(row["stage_b_backend"], 0) + 1
    # Extracted means text came back — not merely "was routed". sha256 is set
    # for every routed document now, so it no longer distinguishes anything;
    # extract_backend plus the absence of a skip does.
    if (
        row.get("error_class") is None
        and row.get("skip_reason") is None
        and row.get("extract_backend") is not None
    ):
        summary["num_extracted"] += 1
    if row.get("skip_reason") is not None:
        summary["num_skipped"] += 1
        by_sr = summary["by_skip_reason"]
        by_sr[row["skip_reason"]] = by_sr.get(row["skip_reason"], 0) + 1
        if row["skip_reason"] == "lane-filter":
            # Which lane the document went to, so the caller can tell a normal
            # hand-off (this box filtering out OCR work) from a document this
            # box thinks is already done but was sent here to be OCR'd.
            by_lane = summary.setdefault("by_filtered_backend", {})
            be = row.get("extract_backend") or "unknown"
            by_lane[be] = by_lane.get(be, 0) + 1
    if row.get("quality_score") is not None:
        summary["num_scored"] += 1
        summary["sum_quality"] += row["quality_score"]
    if row.get("error_class") is not None:
        summary["num_errors"] += 1


# ---------------------------------------------------------------- util

def _sha256_of_file(path: Path) -> str:
    """Hash a file in 1 MiB chunks. Same value pdfsys_parser_mupdf computes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Discovery lives in pdfsys_core.discovery so that `pdfsys run`,
# `pdfsys dataset` and pdfsys-bench cannot drift apart about what a PDF is.
# See resolve_inputs above.
