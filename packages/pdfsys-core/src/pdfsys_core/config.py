"""Runtime configuration dataclasses shared across every pdfsys stage.

Each pipeline package (router, layout-analyser, parser-*) pulls its own
sub-config out of :class:`PdfsysConfig`. The whole tree is JSON / TOML
serializable through the generic :mod:`pdfsys_core.serde` helpers.

Config instances are mutable (``slots=True`` only), so callers can tweak
values after loading. If you need an immutable snapshot, ``serde.to_dict``
+ re-parse gives you one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PathsConfig:
    """Where things live on disk. All fields accept local paths or URIs."""

    input_uri: str = ""            # directory of source PDFs (or s3://... in future)
    cache_root: str = ".cache"     # parent dir for LayoutCache
    output_root: str = "out"       # parser backend outputs


@dataclass(slots=True)
class RouterConfig:
    """Stage-A and stage-B router configuration.

    Stage-A is a cheap feature-based classifier deciding text-ok vs needs-ocr.
    Stage-B reads a cached LayoutDocument and picks pipeline / vlm / deferred.

    ``vlm_enabled`` is the fleet-level gate for the VLM lane: when False,
    complex-content pages are marked :attr:`pdfsys_core.types.Backend.DEFERRED`
    and skipped, letting the pipeline lane run to completion without blocking
    on GPU availability. Flip it to True when a VLM worker pool is online.
    """

    text_ratio_threshold: float = 0.8
    image_area_threshold: float = 0.3
    vlm_enabled: bool = False


@dataclass(slots=True)
class LayoutConfig:
    """Layout analyser model selection and render settings."""

    model_name: str = "pp-doclayoutv3"
    model_version: str = "1.0"
    render_dpi: int = 200

    @property
    def model_tag(self) -> str:
        """Composite tag used as the LayoutCache filename suffix."""
        return f"{self.model_name}@{self.model_version}"


@dataclass(slots=True)
class MupdfConfig:
    """Text-ok backend (parser-mupdf) configuration."""

    max_pages: int | None = None    # None = no cap


@dataclass(slots=True)
class PipelineConfig:
    """OCR-pipeline backend (parser-pipeline) configuration."""

    ocr_engine: str = "rapidocr"                  # rapidocr | paddleocr-classic
    languages: tuple[str, ...] = ("ch", "en")
    render_dpi: int = 200


@dataclass(slots=True)
class VlmConfig:
    """VLM backend (parser-vlm) configuration."""

    model: str = "mineru-2.5"                     # mineru-2.5 | paddleocr-vl
    max_batch_size: int = 4
    render_dpi: int = 300


@dataclass(slots=True)
class RuntimeConfig:
    """Process-level runtime knobs that every stage shares."""

    num_workers: int = 4
    shard_size: int = 1000                        # records per output shard


@dataclass(slots=True)
class PdfsysConfig:
    """Top-level config. Each stage reads only its own sub-field."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    mupdf: MupdfConfig = field(default_factory=MupdfConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def default(cls) -> "PdfsysConfig":
        return cls()
