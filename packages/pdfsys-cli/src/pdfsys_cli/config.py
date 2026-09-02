"""YAML-based pipeline configuration.

Loads a YAML config file, merges with CLI overrides, and produces a
:class:`RunConfig` that the runner consumes. Generates example configs
via ``pdfsys init-config``.

Precedence (highest wins): CLI flags > YAML file > built-in defaults.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical stage order — runner always executes in this order regardless
# of the order the user types them.
VALID_STAGES = ("router", "layout", "extract", "quality", "parquet")


@dataclass(slots=True)
class InputConfig:
    pdf_dir: str = ""
    limit: int | None = None
    #: A worklist of PDF paths, one per line, instead of scanning a directory.
    #: This is how a machine processes a slice it was handed — the leftover
    #: documents another box routed to it, or one bucket of a fleet split.
    pdf_list: str | None = None
    #: Re-anchors relative entries in ``pdf_list``. Lets one worklist be read
    #: on a machine that mounted the corpus somewhere else.
    path_root: str | None = None


@dataclass(slots=True)
class OutputConfig:
    dir: str = "./out"
    jsonl: str = "results.jsonl"
    markdown_dir: str | None = None
    cache_dir: str = ".cache"


@dataclass(slots=True)
class RouterCfg:
    ocr_threshold: float = 0.05
    weights: str | None = None


@dataclass(slots=True)
class LayoutCfg:
    model: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    backend: str | None = None  # auto-detect from model, or "yolo" / "pp-doclayoutv3"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    render_dpi: int = 200


@dataclass(slots=True)
class PipelineCfg:
    formula_enable: bool = True
    table_enable: bool = True
    p_lang: str = "ch"


@dataclass(slots=True)
class VlmCfg:
    engine: str = "transformers"     # transformers | mlx-engine | vllm-engine
    enabled: bool = False
    formula_enable: bool = True
    table_enable: bool = True
    p_lang: str = "ch"


@dataclass(slots=True)
class QualityCfg:
    enabled: bool = True
    # Final scoring model: ModernBERT-base fine-tune (4 ordinal classes,
    # 8192-token context). Legacy fallback for comparison runs:
    # HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn
    # (max_tokens 512, max_chars 10_000).
    model: str = "miracleyin/mnbvc-pdf-quality-scorer-modernbert"
    max_tokens: int = 8192
    # Char-level pre-truncation before tokenization. Must scale with
    # max_tokens: 10k chars saturates ~2.5k English tokens.
    max_chars: int = 40_000
    device: str | None = None


@dataclass(slots=True)
class RuntimeCfg:
    omp_threads: int = 1


@dataclass(slots=True)
class ParquetCfg:
    enabled: bool = True
    out: str = "dataset.parquet"
    compression: str = "zstd"
    quality_threshold: float = 2.0
    include_markdown: bool = True


@dataclass(slots=True)
class RunConfig:
    """Fully resolved pipeline configuration."""

    stages: list[str] = field(default_factory=lambda: list(VALID_STAGES))
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    router: RouterCfg = field(default_factory=RouterCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    vlm: VlmCfg = field(default_factory=VlmCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    parquet: ParquetCfg = field(default_factory=ParquetCfg)

    #: Stages an override removed rather than the user. The CLI prints these,
    #: so a stage that was asked for never disappears without a word.
    dropped_stages: list[str] = field(default_factory=list)
    #: Why each of those went, keyed by stage — several flags can drop one.
    drop_reasons: dict[str, str] = field(default_factory=dict)

    #: Which extraction backends this process is willing to run. ``None`` means
    #: all of them. This is how one binary is a CPU lane on one machine and a
    #: GPU lane on another: the CPU box runs ``mupdf`` and records the rest as
    #: someone else's work, the GPU box runs ``pipeline``/``vlm`` and skips what
    #: the CPU box already did.
    extract_backends: list[str] | None = None

    #: Continue a run that already wrote rows: append to results.jsonl and skip
    #: the documents already in it. Without this the runner truncates on every
    #: start, so a crash at hour six of a 218k-document run erases everything it
    #: had done — and, when that file is also the worklist another machine is
    #: waiting on, erases that too.
    resume: bool = False

    # --- derived paths ---

    @property
    def out_dir(self) -> Path:
        return Path(self.output.dir)

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / self.output.jsonl

    @property
    def markdown_path(self) -> Path | None:
        if self.output.markdown_dir is None:
            return None
        return self.out_dir / self.output.markdown_dir

    @property
    def cache_path(self) -> Path:
        return self.out_dir / self.output.cache_dir

    @property
    def parquet_path(self) -> Path:
        return self.out_dir / self.parquet.out

    def has_stage(self, name: str) -> bool:
        return name in self.stages


def _fill_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config file and return a :class:`RunConfig`."""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    stages = raw.get("stages", list(VALID_STAGES))
    _validate_stages(stages)

    backends = _normalize_backends(raw.get("extract_backends"))

    return RunConfig(
        extract_backends=backends,
        stages=_normalize_stages(stages),
        input=_fill_dataclass(InputConfig, raw.get("input")),
        output=_fill_dataclass(OutputConfig, raw.get("output")),
        router=_fill_dataclass(RouterCfg, raw.get("router")),
        layout=_fill_dataclass(LayoutCfg, raw.get("layout")),
        pipeline=_fill_dataclass(PipelineCfg, raw.get("pipeline")),
        vlm=_fill_dataclass(VlmCfg, raw.get("vlm")),
        quality=_fill_dataclass(QualityCfg, raw.get("quality")),
        runtime=_fill_dataclass(RuntimeCfg, raw.get("runtime")),
        parquet=_fill_dataclass(ParquetCfg, raw.get("parquet")),
        resume=bool(raw.get("resume", False)),
    )


def default_config() -> RunConfig:
    """Return a RunConfig with all defaults."""
    return RunConfig()


def apply_cli_overrides(cfg: RunConfig, **overrides: Any) -> RunConfig:
    """Apply CLI flag overrides onto a loaded config. None values are skipped."""
    if overrides.get("stages") is not None:
        raw = overrides["stages"]
        stages = [s.strip() for s in raw.split(",")] if isinstance(raw, str) else raw
        _validate_stages(stages)
        cfg.stages = _normalize_stages(stages)

    if overrides.get("pdf_dir") is not None:
        cfg.input.pdf_dir = str(overrides["pdf_dir"])
    if overrides.get("extract_backends") is not None:
        cfg.extract_backends = _normalize_backends(overrides["extract_backends"])
    if overrides.get("pdf_list") is not None:
        cfg.input.pdf_list = str(overrides["pdf_list"])
    if overrides.get("path_root") is not None:
        cfg.input.path_root = str(overrides["path_root"])
    if overrides.get("resume") is True:
        cfg.resume = True
    if overrides.get("limit") is not None:
        cfg.input.limit = int(overrides["limit"])
    if overrides.get("out_dir") is not None:
        cfg.output.dir = str(overrides["out_dir"])
    if overrides.get("markdown_dir") is not None:
        cfg.output.markdown_dir = str(overrides["markdown_dir"])
    if overrides.get("cache_dir") is not None:
        cfg.output.cache_dir = str(overrides["cache_dir"])
    if overrides.get("ocr_threshold") is not None:
        cfg.router.ocr_threshold = float(overrides["ocr_threshold"])
    if overrides.get("router_weights") is not None:
        cfg.router.weights = str(overrides["router_weights"])
    if overrides.get("vlm_enabled") is True:
        cfg.vlm.enabled = True
    if overrides.get("vlm_engine") is not None:
        cfg.vlm.engine = str(overrides["vlm_engine"])
    if overrides.get("no_quality") is True:
        cfg.quality.enabled = False
        # parquet depends on quality (quality_score feeds kept) — drop both together
        for stage in ("quality", "parquet"):
            _drop_stage(
                cfg,
                stage,
                "--no-quality: the L1 parquet's `kept` column is decided by "
                "quality_score, so without the scorer there is nothing to write",
            )

    if cfg.resume:
        # Parquet cannot be appended to. pq.ParquetWriter truncates on open, so
        # a resumed leg would leave dataset.parquet covering only that leg while
        # results.jsonl covers the whole run — two artifacts of one run
        # disagreeing, with nothing to say which is which.
        _drop_stage(
            cfg,
            "parquet",
            "--resume: Parquet is not appendable, so a resumed leg would leave "
            "dataset.parquet describing only part of the run. Rebuild it with a "
            "final pass without --resume, or use `pdfsys dataset`",
        )
    if overrides.get("quality_model") is not None:
        cfg.quality.model = str(overrides["quality_model"])

    return cfg


def _drop_stage(cfg: RunConfig, stage: str, reason: str) -> None:
    """Remove a stage an override makes impossible, and record why.

    Only stages actually in the plan are recorded — warning about removing
    something never asked for is noise.
    """
    if stage not in cfg.stages:
        return
    cfg.stages.remove(stage)
    cfg.dropped_stages.append(stage)
    cfg.drop_reasons[stage] = reason


#: Backends a lane can be asked to run. ``deferred`` is stage-B declining, not
#: something a process performs, so it is not selectable.
RUNNABLE_BACKENDS = ("mupdf", "pipeline", "vlm")


def _normalize_backends(value: Any) -> list[str] | None:
    """Accept a lane as a comma string or a list, from CLI or YAML alike.

    One normalizer for both entry points, because they were diverging: the CLI
    split ``"mupdf,vlm"`` on commas while YAML fed the same scalar to ``list()``
    and got ``['m','u','p','d','f']``, reporting an unknown backend ``'m'``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        backends = [s for s in (p.strip() for p in value.split(",")) if s]
    elif isinstance(value, (list, tuple)):
        backends = [str(s).strip() for s in value]
    else:
        raise ValueError(
            f"extract_backends must be a list or a comma-separated string, "
            f"got {type(value).__name__}"
        )
    if not backends:
        raise ValueError(
            "extract_backends needs at least one of "
            f"{', '.join(RUNNABLE_BACKENDS)}; an empty lane extracts nothing"
        )
    for b in backends:
        if b not in RUNNABLE_BACKENDS:
            raise ValueError(
                f"Unknown extract backend {b!r}. "
                f"Valid backends: {', '.join(RUNNABLE_BACKENDS)}"
            )
    return backends


def _validate_stages(stages: list[str]) -> None:
    for s in stages:
        if s not in VALID_STAGES:
            raise ValueError(
                f"Unknown stage {s!r}. Valid stages: {', '.join(VALID_STAGES)}"
            )


def _normalize_stages(stages: list[str]) -> list[str]:
    """Sort stages into canonical order and auto-include dependencies.

    Rules:
    - ``extract`` requires ``router``
    - ``layout`` requires ``router``
    - ``quality`` requires ``router`` + ``extract``
    - ``parquet`` requires ``router`` + ``extract`` + ``quality``
    """
    s = set(stages)

    if "extract" in s or "layout" in s or "quality" in s or "parquet" in s:
        s.add("router")
    if "quality" in s or "parquet" in s:
        s.add("extract")
    if "parquet" in s:
        s.add("quality")

    return [stage for stage in VALID_STAGES if stage in s]


# ---------------------------------------------------------------- template

EXAMPLE_CONFIG = textwrap.dedent("""\
    # pdfsys pipeline configuration
    # Docs: see packages/pdfsys-cli/README.md

    # Which stages to run (in order: router → layout → extract → quality → parquet)
    # Omit stages to skip them; dependencies auto-included.
    stages:
      - router
      - layout
      - extract
      - quality
      - parquet

    input:
      pdf_dir: ./data/pdfs          # directory of source PDFs (recursive).
                                    # Matches *.pdf in any case, plus files with
                                    # no extension whose header is %PDF-.
      pdf_list: null                # …or a file of paths, one per line, used
                                    # instead of scanning pdf_dir. This is how a
                                    # machine works on a slice it was handed.
      path_root: null               # resolve relative pdf_list entries against
                                    # this, so one worklist works on a box that
                                    # mounted the corpus elsewhere
      limit: null                   # max PDFs to process; null = no cap

    # Which extraction backends THIS machine runs. Omit (or null) to run all.
    # The CPU box takes [mupdf] and records the rest as another box's work;
    # the GPU box takes [pipeline] (add vlm with `layout` in stages + vlm.enabled).
    # A document filtered out here is reported as skip_reason=lane-filter.
    extract_backends: null

    # Append to an existing results.jsonl and skip what is already in it,
    # instead of truncating. Also settable with --resume.
    resume: false

    output:
      dir: ./out/run_001            # output root directory
      jsonl: results.jsonl          # per-PDF results (relative to dir)
      markdown_dir: markdown        # dump extracted markdown (relative to dir); null = skip
      cache_dir: .cache             # LayoutCache directory (relative to dir)

    router:
      ocr_threshold: 0.05           # P(ocr) above this → needs-ocr path (bench-tuned; keep <0.60)
      weights: null                 # XGBoost weights path; null = bundled default

    layout:
      model: juliozhao/DocLayout-YOLO-DocStructBench
      # model: PaddlePaddle/PP-DocLayoutV3_safetensors  # alternative: RT-DETR based
      backend: null                 # auto-detect from model, or: yolo | pp-doclayoutv3
      conf_threshold: 0.25
      iou_threshold: 0.45
      render_dpi: 200

    pipeline:
      formula_enable: true
      table_enable: true
      p_lang: ch

    vlm:
      engine: transformers          # transformers | mlx-engine | vllm-engine
      enabled: false
      formula_enable: true
      table_enable: true
      p_lang: ch

    quality:
      enabled: true
      model: miracleyin/mnbvc-pdf-quality-scorer-modernbert
      max_tokens: 8192
      max_chars: 40000
      device: null                  # null = auto (cuda if available, else cpu); set "mps" on Apple Silicon

    parquet:
      enabled: true
      out: dataset.parquet          # path relative to output.dir
      compression: zstd             # zstd | snappy | none
      quality_threshold: 2.0        # kept = (no error) AND (quality_score >= this)
      include_markdown: true        # embed full markdown text in the parquet row

    runtime:
      omp_threads: 1                # OMP_NUM_THREADS (prevent deadlocks on macOS)
""")
