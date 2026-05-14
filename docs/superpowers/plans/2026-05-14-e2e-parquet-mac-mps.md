# End-to-end PDF → Parquet on Mac (MPS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `pdfsys-cli` to run the full router → layout → extract → quality pipeline on the bundled 150 PDFs and emit a single `dataset.parquet` with a `kept` flag at `quality_score >= 2.0`, using MinerU 2.5 (VLM) and ModernBERT (quality) on Apple MPS.

**Architecture:** Append a `parquet` stage to the existing stage-aware runner; add two small modules (`parquet_writer.py`, `_mineru_config.py`) under `packages/pdfsys-cli/src/pdfsys_cli/`; touch no other workspace packages. Single-shard zstd Parquet output. Per-PDF failures captured as `error_class` / `error_message`, never abort the run.

**Tech Stack:** Python 3.11, pyarrow ≥ 15, magic-pdf (MinerU 2.5), transformers + torch (MPS backend), existing `pdfsys-*` workspace packages.

**Source spec:** `docs/superpowers/specs/2026-05-14-e2e-parquet-mac-mps-design.md` (read this first if anything below is unclear).

**User constraint (overrides default TDD pattern):** Spec §10 explicitly defers unit tests. Tasks below use a "build + integration-verify" pattern: each task ends with an import / `--help` / dataclass roundtrip smoke check, and the full pipeline smoke run is Task 10. Do **not** add unit tests in this plan.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `packages/pdfsys-cli/pyproject.toml` | modify | Add `pyarrow>=15.0,<19.0` dependency |
| `packages/pdfsys-cli/src/pdfsys_cli/config.py` | modify | Add `ParquetCfg`, extend `VlmCfg` / `QualityCfg`, add `parquet` to `VALID_STAGES`, update example template |
| `packages/pdfsys-cli/src/pdfsys_cli/runner.py` | modify | Split `extract_error` into `error_class` / `error_message`; add `kept` + `wall_ms_total` + `markdown` to `DocResult`; thread `ExtractedDoc` through `_process_one`; call `ParquetSink` from `run()`; invoke `_mineru_config.ensure_config` when VLM enabled |
| `packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py` | create | `SCHEMA` constant + `ParquetSink` class (open / write_row / close) |
| `packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py` | create | Idempotent `ensure_config(device_mode: str)` that writes/updates `~/magic-pdf.json` |
| `pdfsys.smoke.yaml` | create | Phase-1 config: 5 PDFs from omnidocbench, MPS, VLM on |
| `pdfsys.full.yaml` | create | Phase-2 config: full 150 PDFs, MPS, VLM on |

---

## Decomposition notes

- Why one `parquet_writer.py` module, not a package: 150 PDFs need ~100 LOC of writer code. Promoting it to `pdfsys-output` per ROADMAP §5.7 is premature — defer until sharding / S3 actually shows up.
- Why pass `ExtractedDoc` through `_process_one` return value (not store markdown on `DocResult`): keeps `DocResult` flat-JSON-serializable for the JSONL writer; avoids round-tripping the markdown through `dataclasses.asdict`. The parquet writer reads markdown directly from the in-memory `ExtractedDoc`.
- Why no `device` field for `LayoutCfg`: spec §9 risks accepts CPU layout (DocLayout-YOLO ONNX) for this iteration. Adding MPS support to PP-DocLayoutV3 requires modifying `pdfsys-layout-analyser`, which §11 of the spec forbids in this iteration. Future spec.

---

## Task 1: Add `pyarrow` dependency

**Files:**
- Modify: `packages/pdfsys-cli/pyproject.toml`

- [ ] **Step 1: Add the dependency line**

In `packages/pdfsys-cli/pyproject.toml`, the `dependencies` list currently ends with `"pyyaml>=6.0",`. Add one line after it:

```toml
dependencies = [
    "pdfsys-core",
    "pdfsys-router",
    "pdfsys-parser-mupdf",
    "pdfsys-layout-analyser",
    "pdfsys-parser-pipeline",
    "pdfsys-parser-vlm",
    "pdfsys-bench",
    "pyyaml>=6.0",
    "pyarrow>=15.0,<19.0",
]
```

- [ ] **Step 2: Sync the workspace**

Run from repo root:

```bash
uv sync
```

Expected: completes successfully; one new package (`pyarrow`) added to the lock.

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "import pyarrow as pa; import pyarrow.parquet as pq; print(pa.__version__)"
```

Expected: prints a 15.x / 16.x / 17.x / 18.x version, no traceback.

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-cli/pyproject.toml uv.lock
git commit -m "feat(cli): add pyarrow dependency for Parquet output"
```

---

## Task 2: Extend `config.py` with `ParquetCfg`, device fields, and stage normalization

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/config.py`

- [ ] **Step 1: Add `ParquetCfg` dataclass after `RuntimeCfg`**

In `config.py`, after the `RuntimeCfg` definition (currently lines 74-76), insert:

```python
@dataclass(slots=True)
class ParquetCfg:
    enabled: bool = True
    out: str = "dataset.parquet"
    compression: str = "zstd"
    quality_threshold: float = 2.0
    include_markdown: bool = True
```

- [ ] **Step 2: Add `device_mode` to `VlmCfg`**

Find `class VlmCfg` (currently lines 60-63). Replace the dataclass body with:

```python
@dataclass(slots=True)
class VlmCfg:
    model: str = "mineru-2.5"
    enabled: bool = False
    device_mode: str = "cpu"  # "mps" on Apple Silicon; written to ~/magic-pdf.json
```

(`QualityCfg.device` already exists as `str | None`, no change needed — the YAML will set it to `"mps"`.)

- [ ] **Step 3: Add `parquet` to `VALID_STAGES`**

Replace the `VALID_STAGES` tuple at the top of the file (currently line 21):

```python
VALID_STAGES = ("router", "layout", "extract", "quality", "parquet")
```

- [ ] **Step 4: Add `parquet` field to `RunConfig` + helper getters**

In `RunConfig` (lines 80-91), add a field after `runtime`:

```python
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
```

Add a derived path getter alongside the others (after `cache_path`):

```python
    @property
    def parquet_path(self) -> Path:
        return self.out_dir / self.parquet.out
```

- [ ] **Step 5: Wire `parquet` into `load_config`**

Update the `load_config` return statement (lines 135-144) to pass the parquet block:

```python
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
        parquet=_fill_dataclass(ParquetCfg, raw.get("parquet")),
    )
```

- [ ] **Step 6: Update stage dependency rules in `_normalize_stages`**

Replace the body of `_normalize_stages` (currently lines 195-210):

```python
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
```

- [ ] **Step 7: Update the `EXAMPLE_CONFIG` template**

Replace the `EXAMPLE_CONFIG` string (currently lines 215-266). The new full template:

```python
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
      # model: PaddlePaddle/PP-DocLayoutV3_safetensors  # alternative: RT-DETR based
      backend: null                 # auto-detect from model, or: yolo | pp-doclayoutv3
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
      device_mode: cpu              # cpu | mps | cuda — written to ~/magic-pdf.json

    quality:
      enabled: true
      model: HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn
      max_tokens: 512
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
```

- [ ] **Step 8: Verify config loads cleanly**

```bash
uv run python -c "
from pdfsys_cli.config import RunConfig, ParquetCfg, default_config, EXAMPLE_CONFIG
import yaml
cfg = default_config()
assert 'parquet' in cfg.stages, cfg.stages
assert cfg.parquet.quality_threshold == 2.0
parsed = yaml.safe_load(EXAMPLE_CONFIG)
assert parsed['parquet']['quality_threshold'] == 2.0
assert parsed['vlm']['device_mode'] == 'cpu'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 9: Verify the existing `init-config` command still works**

```bash
uv run pdfsys init-config | head -20
```

Expected: prints the new YAML template starting with `# pdfsys pipeline configuration`.

- [ ] **Step 10: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/config.py
git commit -m "feat(cli): add ParquetCfg + VlmCfg.device_mode + parquet stage"
```

---

## Task 3: Create `_mineru_config.py`

**Files:**
- Create: `packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py`

- [ ] **Step 1: Write the helper module**

Create `packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py`:

```python
"""Idempotent helper that ensures ~/magic-pdf.json sets the expected device-mode.

magic-pdf (MinerU) reads ~/magic-pdf.json on import to pick its device.
This module writes / updates that file before the VLM parser is first
loaded, so the runner can guarantee MPS (or CPU / CUDA) without asking
the user to hand-edit a JSON file.

Idempotency:
- file missing                  → write a fresh default + chosen device-mode
- file present, mode matches    → no-op
- file present, mode differs    → patch only the `device-mode` key, preserve
                                  every other key the user may have set
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path.home() / "magic-pdf.json"

_DEFAULT_MODELS_DIR = Path.home() / ".cache" / "mineru" / "models"


def ensure_config(device_mode: str) -> Path:
    """Make sure ~/magic-pdf.json has the requested device-mode.

    Parameters
    ----------
    device_mode:
        One of ``"cpu"``, ``"mps"``, ``"cuda"``. Other values are passed
        through verbatim — magic-pdf will reject unknowns at load time.

    Returns
    -------
    Path
        Absolute path to the (now-correct) config file.
    """
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("device-mode") == device_mode:
            return CONFIG_PATH
        existing["device-mode"] = device_mode
        CONFIG_PATH.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return CONFIG_PATH

    _DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    default = {
        "device-mode": device_mode,
        "models-dir": str(_DEFAULT_MODELS_DIR),
        "table-config": {"model": "rapid_table", "enable": True},
        "formula-config": {"enable": True},
    }
    CONFIG_PATH.write_text(
        json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CONFIG_PATH
```

- [ ] **Step 2: Verify idempotency in-place**

```bash
uv run python -c "
from pdfsys_cli._mineru_config import ensure_config, CONFIG_PATH
import json
p = ensure_config('mps')
data = json.loads(p.read_text())
assert data['device-mode'] == 'mps', data
# call again — should be a no-op
ensure_config('mps')
data2 = json.loads(p.read_text())
assert data2 == data, 'idempotent failure'
# switch mode — should patch only one key
ensure_config('cpu')
data3 = json.loads(p.read_text())
assert data3['device-mode'] == 'cpu'
assert data3['models-dir'] == data['models-dir']
# restore mps for the upcoming run
ensure_config('mps')
print('OK')
"
```

Expected: prints `OK` and `~/magic-pdf.json` ends with `"device-mode": "mps"`.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py
git commit -m "feat(cli): add idempotent ~/magic-pdf.json device-mode writer"
```

---

## Task 4: Create `parquet_writer.py`

**Files:**
- Create: `packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py`

- [ ] **Step 1: Write the module**

Create `packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py`:

```python
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
```

- [ ] **Step 2: Verify schema round-trip with a synthetic row**

```bash
uv run python -c "
from pathlib import Path
from pdfsys_cli.parquet_writer import ParquetSink
import pyarrow.parquet as pq

# Fake DocResult-shaped object (duck-typed — sink only reads attrs)
class Fake:
    pdf_path='/x.pdf'; sha256='abc'; backend='MUPDF'; stage_b_backend=None
    ocr_prob=0.03; num_pages=10; is_form=False; garbled_text_ratio=0.0; is_encrypted=False
    layout_model=None; layout_num_regions=None; layout_has_complex=None
    extract_backend='MUPDF'; markdown_chars=1234
    quality_score=2.5; quality_model='m'; error_class=None; error_message=None
    wall_ms_router=10.0; wall_ms_layout=0.0; wall_ms_extract=20.0; wall_ms_quality=30.0
    sha256=None

p = Path('/tmp/_pdfsys_smoke.parquet')
sink = ParquetSink(p)
sink.write_row(Fake(), 'hello world')
sink.close()
t = pq.read_table(p)
d = t.to_pylist()[0]
assert d['markdown'] == 'hello world'
assert d['kept'] is True
assert d['wall_ms_total'] == 60.0
print('OK', t.num_rows, 'rows')
p.unlink()
"
```

Expected: prints `OK 1 rows`.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py
git commit -m "feat(cli): add ParquetSink for end-to-end pipeline output"
```

---

## Task 5: Update `DocResult` in `runner.py`

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py` (lines 22-57)

- [ ] **Step 1: Replace the `DocResult` dataclass**

Locate the existing `DocResult` definition (currently lines 22-57). Replace it entirely with:

```python
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
    # quality
    quality_score: float | None = None
    quality_num_chars: int | None = None
    quality_num_tokens: int | None = None
    quality_model: str | None = None
    # error capture (split from old extract_error)
    error_class: str | None = None  # router | layout | extract_mupdf | extract_pipeline | extract_vlm | quality
    error_message: str | None = None  # f"{type(e).__name__}: {e}", truncated to 500 chars
    # timing
    wall_ms_router: float = 0.0
    wall_ms_layout: float = 0.0
    wall_ms_extract: float = 0.0
    wall_ms_quality: float = 0.0

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
```

(Changes vs. before: removed `extract_error`, added `is_encrypted`, added `error_class` + `error_message`.)

- [ ] **Step 2: Verify the dataclass still imports**

```bash
uv run python -c "
from pdfsys_cli.runner import DocResult
r = DocResult(pdf_path='x', error_class='router', error_message='ValueError: bad')
line = r.to_json_line()
import json
parsed = json.loads(line)
assert parsed['error_class'] == 'router'
assert parsed['error_message'] == 'ValueError: bad'
assert 'extract_error' not in parsed  # confirm old field gone
print('OK')
"
```

Expected: prints `OK`.

Do **not** commit yet — Task 6 fills in the call sites that referenced `extract_error`. The intermediate state will fail wider imports.

---

## Task 6: Migrate stage error capture in `runner.py`

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py` (4 stage helpers, ~5 sites)

- [ ] **Step 1: Add a small error-capture helper above `_process_one`**

In `runner.py`, immediately above the `# ---------------------------------------------------------------- per-pdf` comment (currently line 204), insert:

```python
def _set_error(row: DocResult, error_class: str, exc: BaseException) -> None:
    """Record the first error to hit this row; later errors are dropped."""
    if row.error_class is not None:
        return
    row.error_class = error_class
    msg = f"{type(exc).__name__}: {exc}"
    row.error_message = msg[:500]
```

- [ ] **Step 2: Update `_stage_router` to fill `is_encrypted` and route encrypted/error to `error_class`**

Replace the body of `_stage_router` (currently lines 238-249) with:

```python
def _stage_router(row: DocResult, pdf_path: Path, comps: Components) -> None:
    t0 = time.perf_counter()
    decision = comps.router.classify(pdf_path)
    t1 = time.perf_counter()

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
```

- [ ] **Step 3: Update `_stage_layout` error handling**

Replace the `except` clause inside `_stage_layout` (currently lines 277-279):

```python
    except Exception as e:  # noqa: BLE001
        _set_error(row, "layout", e)
        return None
```

- [ ] **Step 4: Update `_stage_extract` error handling (3 except clauses)**

In `_stage_extract`, find each of the three `except Exception as e:` blocks (lines 304-306, 319-321, 334-336). Replace them respectively with:

```python
        except Exception as e:  # noqa: BLE001 — mupdf
            _set_error(row, "extract_mupdf", e)
            return None
```

```python
        except Exception as e:  # noqa: BLE001 — pipeline
            _set_error(row, "extract_pipeline", e)
            return None
```

```python
        except Exception as e:  # noqa: BLE001 — vlm
            _set_error(row, "extract_vlm", e)
            return None
```

- [ ] **Step 5: Update `_stage_quality` error handling**

Replace the `except` clause inside `_stage_quality` (currently lines 363-364):

```python
    except Exception as e:  # noqa: BLE001
        _set_error(row, "quality", e)
```

- [ ] **Step 6: Drop the obsolete `extract_error` references in the summary**

In `run()`, find the two lines (around lines 183-189) that read:

```python
            if row.extract_error is None and row.sha256 is not None:
                summary["num_extracted"] += 1
            ...
            if row.router_error or row.extract_error:
                summary["num_errors"] += 1
```

Replace with:

```python
            if row.error_class is None and row.sha256 is not None:
                summary["num_extracted"] += 1
            ...
            if row.error_class is not None:
                summary["num_errors"] += 1
```

- [ ] **Step 7: Verify the runner module imports**

```bash
uv run python -c "
from pdfsys_cli.runner import DocResult, _set_error
r = DocResult()
_set_error(r, 'router', ValueError('boom'))
assert r.error_class == 'router'
assert r.error_message == 'ValueError: boom'
# second call should NOT overwrite
_set_error(r, 'quality', RuntimeError('later'))
assert r.error_class == 'router'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 8: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/runner.py
git commit -m "feat(cli): split extract_error into error_class + error_message"
```

---

## Task 7: Thread `ExtractedDoc` through `_process_one` and wire ParquetSink

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py`

- [ ] **Step 1: Add imports at the top of the file**

After the existing `from .config import RunConfig` line (currently line 19), add:

```python
from .config import RunConfig
from .parquet_writer import ParquetSink
```

- [ ] **Step 2: Change `_process_one` return signature**

Locate `_process_one` (currently lines 206-227). Change its return type and tail:

```python
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

    return row, extracted
```

- [ ] **Step 3: Update the `run()` loop to consume the tuple and write parquet**

Replace the body of `run()` from the `with cfg.jsonl_path.open(...)` block (currently lines 169-189) with:

```python
    # ---- optional pre-flight: configure magic-pdf for the requested device ----
    if cfg.has_stage("extract") and cfg.vlm.enabled:
        from ._mineru_config import ensure_config  # noqa: PLC0415

        ensure_config(cfg.vlm.device_mode)

    # ---- optional parquet sink (opened lazily, closed via context) ----
    parquet_sink: ParquetSink | None = None
    if cfg.has_stage("parquet") and cfg.parquet.enabled:
        parquet_sink = ParquetSink(
            path=cfg.parquet_path,
            compression=cfg.parquet.compression,
            quality_threshold=cfg.parquet.quality_threshold,
            include_markdown=cfg.parquet.include_markdown,
        )

    try:
        with cfg.jsonl_path.open("w", encoding="utf-8") as out_f:
            for pdf_path in _iter_pdfs(Path(cfg.input.pdf_dir), cfg.input.limit):
                row, extracted = _process_one(pdf_path, cfg, comps)
                out_f.write(row.to_json_line() + "\n")
                out_f.flush()

                if parquet_sink is not None:
                    md = extracted.markdown if extracted is not None else None
                    parquet_sink.write_row(row, md)

                summary["num_pdfs"] += 1
                if row.backend:
                    by_b = summary["by_backend"]
                    final = row.extract_backend or row.backend
                    by_b[final] = by_b.get(final, 0) + 1
                if row.stage_b_backend:
                    by_sb = summary["by_stage_b"]
                    by_sb[row.stage_b_backend] = by_sb.get(row.stage_b_backend, 0) + 1
                if row.error_class is None and row.sha256 is not None:
                    summary["num_extracted"] += 1
                if row.quality_score is not None:
                    summary["num_scored"] += 1
                    summary["sum_quality"] += row.quality_score
                if row.error_class is not None:
                    summary["num_errors"] += 1
    finally:
        if parquet_sink is not None:
            parquet_sink.close()
            summary["parquet_rows"] = parquet_sink.rows_written
            summary["parquet_path"] = str(cfg.parquet_path)
```

- [ ] **Step 4: Wire `quality.device` through `Components.scorer`**

Locate `Components.scorer` (currently lines 124-133). The constructor already passes `device=self.cfg.quality.device`, so **no change needed** — verify by reading. The YAML will populate `quality.device = "mps"`.

- [ ] **Step 5: Verify the runner module still imports**

```bash
uv run python -c "
from pdfsys_cli.runner import run, DocResult, _process_one
from pdfsys_cli.config import default_config
import inspect
sig = inspect.signature(_process_one)
assert sig.return_annotation == 'tuple[DocResult, Any]', sig.return_annotation
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 6: Verify a minimal MUPDF-only run still works (regression check on the JSONL path)**

```bash
uv run pdfsys run --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
  --out-dir /tmp/_pdfsys_reg \
  --stages router,extract \
  --limit 2 \
  --no-quality
```

Expected: completes; `/tmp/_pdfsys_reg/results.jsonl` has 2 lines. (This run does NOT enable parquet — confirms the new code is backward compatible.)

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/runner.py
git commit -m "feat(cli): wire ParquetSink + thread ExtractedDoc through pipeline"
```

---

## Task 8: Create phase-1 / phase-2 YAML configs

**Files:**
- Create: `pdfsys.smoke.yaml`
- Create: `pdfsys.full.yaml`

- [ ] **Step 1: Write `pdfsys.smoke.yaml`**

```yaml
# Phase-1 smoke test: 5 PDFs, full pipeline, MPS where supported.
# Goal: prove magic-pdf installs, MPS works, parquet writes.

stages:
  - router
  - layout
  - extract
  - quality
  - parquet

input:
  pdf_dir: packages/pdfsys-bench/omnidocbench_100/pdfs
  limit: 5

output:
  dir: ./out/e2e_smoke
  jsonl: results.jsonl
  markdown_dir: markdown
  cache_dir: .cache

router:
  ocr_threshold: 0.5
  weights: null

layout:
  model: juliozhao/DocLayout-YOLO-DocStructBench
  backend: null
  conf_threshold: 0.25
  iou_threshold: 0.45
  render_dpi: 200

pipeline:
  ocr_engine: rapidocr
  languages: [ch, en]
  render_dpi: 200

vlm:
  model: mineru-2.5
  enabled: true
  device_mode: mps

quality:
  enabled: true
  model: HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn
  max_tokens: 512
  device: mps

parquet:
  enabled: true
  out: dataset.parquet
  compression: zstd
  quality_threshold: 2.0
  include_markdown: true

runtime:
  omp_threads: 1
```

- [ ] **Step 2: Write `pdfsys.full.yaml`**

Identical to smoke except `input` and `output`:

```yaml
# Phase-2 full run: all 150 bundled PDFs (omnidocbench_100 + olmocr_bench_50).
# Pipeline + thresholds identical to pdfsys.smoke.yaml.

stages:
  - router
  - layout
  - extract
  - quality
  - parquet

input:
  pdf_dir: packages/pdfsys-bench
  limit: null

output:
  dir: ./out/e2e_full
  jsonl: results.jsonl
  markdown_dir: markdown
  cache_dir: .cache

router:
  ocr_threshold: 0.5
  weights: null

layout:
  model: juliozhao/DocLayout-YOLO-DocStructBench
  backend: null
  conf_threshold: 0.25
  iou_threshold: 0.45
  render_dpi: 200

pipeline:
  ocr_engine: rapidocr
  languages: [ch, en]
  render_dpi: 200

vlm:
  model: mineru-2.5
  enabled: true
  device_mode: mps

quality:
  enabled: true
  model: HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn
  max_tokens: 512
  device: mps

parquet:
  enabled: true
  out: dataset.parquet
  compression: zstd
  quality_threshold: 2.0
  include_markdown: true

runtime:
  omp_threads: 1
```

Note: `input.pdf_dir: packages/pdfsys-bench` is a directory above both `omnidocbench_100/pdfs` and `olmocr_bench_50/pdfs`. The runner uses `rglob("*.pdf")` (see `runner.py:370`) so it picks up both subsets in one pass.

- [ ] **Step 3: Sanity-check config parsing**

```bash
uv run python -c "
from pdfsys_cli.config import load_config
s = load_config('pdfsys.smoke.yaml')
f = load_config('pdfsys.full.yaml')
assert s.vlm.enabled and s.vlm.device_mode == 'mps'
assert f.parquet.quality_threshold == 2.0
assert 'parquet' in s.stages
assert s.input.limit == 5 and f.input.limit is None
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add pdfsys.smoke.yaml pdfsys.full.yaml
git commit -m "feat(cli): add smoke + full phase configs for e2e Mac+MPS run"
```

---

## Task 9: Pre-flight (model downloads, magic-pdf install check)

This task does no code change — it primes the environment so the smoke run doesn't fail on first-fetch issues.

- [ ] **Step 1: Confirm magic-pdf is importable**

```bash
uv run python -c "
import importlib
try:
    importlib.import_module('magic_pdf.data.dataset')
    print('magic-pdf v2 API present')
except ImportError as e:
    try:
        importlib.import_module('magic_pdf.pipe.OCRPipe')
        print('magic-pdf v1 API present')
    except ImportError as e2:
        raise SystemExit(f'magic-pdf NOT importable: v2={e} v1={e2}')
"
```

Expected: prints either `magic-pdf v2 API present` or `magic-pdf v1 API present`.

If it fails: `uv add --package pdfsys-parser-vlm magic-pdf` (or check the installed version with `uv pip list | grep -i magic`). Stop and report — don't proceed.

- [ ] **Step 2: Confirm MPS is available**

```bash
uv run python -c "
import torch
assert torch.backends.mps.is_available(), 'MPS not available'
assert torch.backends.mps.is_built(), 'MPS not built'
print('MPS OK, PyTorch', torch.__version__)
"
```

Expected: prints `MPS OK, PyTorch <version>`.

If it fails: this is a Mac-without-Apple-Silicon or a torch build issue. Drop `device: mps` to `device: cpu` (or `null`) in both YAMLs and continue — the spec accepts CPU as fallback for both quality and VLM (just slower).

- [ ] **Step 3: Write magic-pdf config**

```bash
uv run python -c "from pdfsys_cli._mineru_config import ensure_config; print(ensure_config('mps'))"
```

Expected: prints `/Users/<you>/magic-pdf.json`.

- [ ] **Step 4: Optional — pre-warm MinerU models**

MinerU downloads ~3 GB on first run; this can take 30+ minutes on a fresh box. If you want to download outside the run, the magic-pdf docs suggest:

```bash
uv run python -c "
# Triggers the model loader by attempting a tiny analysis.
# Skip if you're fine downloading during the smoke run itself.
from pathlib import Path
import importlib
# Just check the loader path exists; actual download happens on first inference.
m = importlib.import_module('magic_pdf.model.doc_analyze_by_custom_model')
print('doc_analyze loader path:', m.__file__)
"
```

Expected: prints the loader path. No download triggered here — that happens during the smoke run.

- [ ] **Step 5: No commit** (this task is purely environment validation).

---

## Task 10: Smoke run (5 PDFs)

- [ ] **Step 1: Clear any stale output**

```bash
rm -rf out/e2e_smoke
```

- [ ] **Step 2: Run the smoke config**

```bash
uv run pdfsys run -c pdfsys.smoke.yaml 2>&1 | tee out/_smoke.log
```

(The `2>&1 | tee` captures stderr too — MinerU is noisy on first run; we want the full log.)

Expected (rough):
- exit 0
- log shows `Stage A …` per PDF, plus MinerU model download messages on first run
- `out/e2e_smoke/results.jsonl` has 5 lines
- `out/e2e_smoke/dataset.parquet` exists
- `out/e2e_smoke/markdown/*.md` exists (one per successfully extracted PDF)

Wall time: ~10-30 minutes on a fresh box (model download dominates the first run).

- [ ] **Step 3: Verify with DuckDB / pandas**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('out/e2e_smoke/dataset.parquet')
print(df[['pdf_path','backend','extract_backend','quality_score','error_class','kept']].to_string(index=False))
print()
print('row count:', len(df))
print('backend dist:', dict(df['extract_backend'].fillna('NULL').value_counts()))
print('error dist:', dict(df['error_class'].fillna('OK').value_counts()))
print('kept rows:', int(df['kept'].sum()))
"
```

Expected:
- row count: 5
- at least one of (`MUPDF`, `PIPELINE`, `VLM`) appears in the backend dist
- error dist may show some failures (especially `extract_vlm` on first MinerU run) — log them; not a blocker if at least one row has `error_class IS NULL`
- `kept rows >= 0`

- [ ] **Step 4: Confirm magic-pdf config matches**

```bash
uv run python -c "
import json, pathlib
p = pathlib.Path.home() / 'magic-pdf.json'
data = json.loads(p.read_text())
assert data['device-mode'] == 'mps', data
print('device-mode:', data['device-mode'])
"
```

Expected: prints `device-mode: mps`.

- [ ] **Step 5: Diagnostic snapshot (do not commit logs)**

If errors appeared, capture a short note for later analysis:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('out/e2e_smoke/dataset.parquet')
errs = df[df['error_class'].notnull()][['pdf_path','error_class','error_message']]
print(errs.to_string(index=False) if len(errs) else 'no errors')
"
```

- [ ] **Step 6: Decide go/no-go for Task 11**

Pass criteria (smoke):
- exit 0
- at least 1 row has `error_class IS NULL`
- `dataset.parquet` opens in pandas
- if any `error_class = "layout"` due to MPS errors → STOP and investigate before running phase 2 (these will multiply by 30× on full)

If pass → proceed to Task 11. If fail → diagnose, fix, rerun smoke before phase 2.

- [ ] **Step 7: No code commit; commit only if Task 8/9 found a config issue you patched.**

---

## Task 11: Full run (150 PDFs)

- [ ] **Step 1: Clear stale output**

```bash
rm -rf out/e2e_full
```

- [ ] **Step 2: Kick off the full run in the background**

This may take 1-4 hours.

```bash
uv run pdfsys run -c pdfsys.full.yaml > out/_full.log 2>&1 &
echo $! > /tmp/_pdfsys_full.pid
```

Optionally monitor with:

```bash
tail -f out/_full.log
```

- [ ] **Step 3: When complete, verify counts**

```bash
wait $(cat /tmp/_pdfsys_full.pid) 2>/dev/null
echo "Exit: $?"
ls -lh out/e2e_full/
wc -l out/e2e_full/results.jsonl
```

Expected: exit 0, `results.jsonl` has 150 lines, `dataset.parquet` is present (likely 5-30 MB).

- [ ] **Step 4: Run all 4 acceptance queries from spec §8**

```bash
uv run python <<'PY'
import pandas as pd
df = pd.read_parquet('out/e2e_full/dataset.parquet')

print("== Coverage ==")
print("rows:", len(df), "(expect 150)")

print("\n== Backend mix ==")
print(df['extract_backend'].fillna('NULL').value_counts())

print("\n== Quality / errors ==")
print("kept_rows:", int(df['kept'].sum()))
print("errors:", int(df['error_class'].notnull().sum()))
print("avg quality:", df['quality_score'].mean())

print("\n== Error breakdown ==")
print(df[df['error_class'].notnull()]['error_class'].value_counts())
PY
```

Pass criteria (full):
- 150 rows
- at least 100 rows have `error_class IS NULL` (errors < 50, soft threshold from spec §8 — 20% tolerance for first Mac/MinerU run)
- `kept_rows > 0`
- MUPDF still dominates; PIPELINE > 0; VLM > 0 (proves MinerU fired)

- [ ] **Step 5: Append a post-run note to the spec**

Open `docs/superpowers/specs/2026-05-14-e2e-parquet-mac-mps-design.md` and append a `## 13 · Post-run note (YYYY-MM-DD)` section with:

- exit code + wall time
- backend distribution table
- error_class distribution table
- 2-3 sentences on what failure modes dominated (informs next iteration)

This is the visibility deliverable the spec promises in §8.

- [ ] **Step 6: Commit the post-run note + the spec update**

```bash
git add docs/superpowers/specs/2026-05-14-e2e-parquet-mac-mps-design.md
git commit -m "docs(spec): append phase-2 post-run results"
```

- [ ] **Step 7: Optional — commit the output summary (NOT the parquet itself)**

```bash
# only the JSON summary, not the multi-MB parquet or markdown dump
git add -- out/e2e_full/*.summary.json 2>/dev/null || true
git status
# decide based on git status whether to commit; out/ may be gitignored
```

---

## Self-review notes

- **Spec coverage:** §3.1 routing strategy is documentation-only (not a code change). §4.1 ParquetSink → Task 4. §4.2 runner integration + error split → Tasks 5, 6, 7. §4.3 config + YAML → Task 2 + Task 8. §4.4 magic-pdf MPS config → Task 3 + Task 7 wiring. §4.5 pyarrow → Task 1. §5 schema → Task 4 SCHEMA constant. §6 two-phase → Tasks 10, 11. §7 error model → Task 6 (6 buckets). §8 acceptance → Task 11 step 4. §9 risks → Task 9 (preflight) catches them. §11 files touched → matches file map.
- **Placeholders:** none. No "TBD", no "add error handling", no "similar to Task X".
- **Type consistency:** `ParquetSink.write_row(row, markdown)` matches `_process_one`'s tuple return shape; `DocResult.error_class` / `error_message` names used consistently across Tasks 5, 6, 7.
- **One soft assumption:** Task 7 step 6 regression check uses `--stages router,extract` flags. Verify those flags exist in `__main__.py` before running — if not, fall back to a YAML-only invocation. (`apply_cli_overrides` in config.py shows `stages` is supported via CLI.)
