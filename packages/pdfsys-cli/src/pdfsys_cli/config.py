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
VALID_STAGES = ("router", "layout", "extract", "quality")


@dataclass(slots=True)
class InputConfig:
    pdf_dir: str = ""
    limit: int | None = None


@dataclass(slots=True)
class OutputConfig:
    dir: str = "./out"
    jsonl: str = "results.jsonl"
    markdown_dir: str | None = None
    cache_dir: str = ".cache"


@dataclass(slots=True)
class RouterCfg:
    ocr_threshold: float = 0.5
    weights: str | None = None


@dataclass(slots=True)
class LayoutCfg:
    model: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    render_dpi: int = 200


@dataclass(slots=True)
class PipelineCfg:
    ocr_engine: str = "rapidocr"
    languages: list[str] = field(default_factory=lambda: ["ch", "en"])
    render_dpi: int = 200


@dataclass(slots=True)
class VlmCfg:
    model: str = "mineru-2.5"
    enabled: bool = False


@dataclass(slots=True)
class QualityCfg:
    enabled: bool = True
    model: str = "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn"
    max_tokens: int = 512
    device: str | None = None


@dataclass(slots=True)
class RuntimeCfg:
    omp_threads: int = 1


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
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    stages = raw.get("stages", list(VALID_STAGES))
    _validate_stages(stages)

    return RunConfig(
        stages=_normalize_stages(stages),
        input=_fill_dataclass(InputConfig, raw.get("input")),
        output=_fill_dataclass(OutputConfig, raw.get("output")),
        router=_fill_dataclass(RouterCfg, raw.get("router")),
        layout=_fill_dataclass(LayoutCfg, raw.get("layout")),
        pipeline=_fill_dataclass(PipelineCfg, raw.get("pipeline")),
        vlm=_fill_dataclass(VlmCfg, raw.get("vlm")),
        quality=_fill_dataclass(QualityCfg, raw.get("quality")),
        runtime=_fill_dataclass(RuntimeCfg, raw.get("runtime")),
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
    if overrides.get("no_quality") is True:
        cfg.quality.enabled = False
        if "quality" in cfg.stages:
            cfg.stages.remove("quality")
    if overrides.get("quality_model") is not None:
        cfg.quality.model = str(overrides["quality_model"])

    return cfg


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
    """
    s = set(stages)

    if "extract" in s or "layout" in s or "quality" in s:
        s.add("router")
    if "quality" in s:
        s.add("extract")

    return [stage for stage in VALID_STAGES if stage in s]


# ---------------------------------------------------------------- template

EXAMPLE_CONFIG = textwrap.dedent("""\
    # pdfsys pipeline configuration
    # Docs: see packages/pdfsys-cli/README.md

    # Which stages to run (in order: router → layout → extract → quality)
    # Omit stages to skip them; dependencies auto-included.
    stages:
      - router
      - layout
      - extract
      - quality

    input:
      pdf_dir: ./data/pdfs          # directory of source PDFs (recursive)
      limit: null                   # max PDFs to process; null = no cap

    output:
      dir: ./out/run_001            # output root directory
      jsonl: results.jsonl          # per-PDF results (relative to dir)
      markdown_dir: markdown        # dump extracted markdown (relative to dir); null = skip
      cache_dir: .cache             # LayoutCache directory (relative to dir)

    router:
      ocr_threshold: 0.5            # P(ocr) above this → needs-ocr path
      weights: null                 # XGBoost weights path; null = bundled default

    layout:
      model: juliozhao/DocLayout-YOLO-DocStructBench
      conf_threshold: 0.25
      iou_threshold: 0.45
      render_dpi: 200

    pipeline:
      ocr_engine: rapidocr          # rapidocr | paddleocr
      languages: [ch, en]
      render_dpi: 200

    vlm:
      model: mineru-2.5
      enabled: false                # set true to enable MinerU VLM lane

    quality:
      enabled: true
      model: HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn
      max_tokens: 512
      device: null                  # null = auto (cuda if available, else cpu)

    runtime:
      omp_threads: 1                # OMP_NUM_THREADS (prevent deadlocks on macOS)
""")
