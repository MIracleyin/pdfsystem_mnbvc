# Mineru Parsers Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch `pdfsys-parser-pipeline` and `pdfsys-parser-vlm` to symmetric end-to-end wrappers around `mineru.cli.common.do_parse(backend=...)`. Delete RapidOCR and the region-based VLM merge code. No backwards compatibility.

**Architecture:** Both ocr-needing parsers expose `parser.extract(pdf_path) -> ExtractedDoc`. Internally each calls `mineru.cli.common.do_parse(...)` with `backend="pipeline"` or `backend="vlm-<engine>"`. Mineru does its own layout + OCR / VLM internally; the layout-analyser stays only for Stage-B routing decisions and no longer feeds parsers.

**Tech Stack:** Python 3.11+ (workspace pins), `mineru[pipeline]` + `mineru[vlm]` (extras pull torch / transformers / opencv-python / etc.), stdlib only for tests, pytest, ruff. `uv` for venv + lockfile.

**Source spec:** `docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md`

**Conventions in this codebase:**
- `from __future__ import annotations` at the top of every Python file.
- `@dataclass(slots=True)` for data containers.
- Module-private helpers `_snake_case`.
- Tests in `tests/<area>/test_<module>.py`; plain pytest functions, no test classes.
- One commit per task; format `feat(<scope>): <one-line>` / `fix(<scope>): ...` / `refactor(<scope>): ...`.
- Run: `uv run pytest tests/<path> -v`, `uv run ruff check packages/<pkg>`.
- Direct commits to `main` are the project convention (verified in `git log`).

---

## File Structure (target state)

**New files:**

```
tests/parsers/
├── __init__.py
├── test_pipeline_parser.py     # Tier A unit tests, mocked do_parse
├── test_vlm_parser.py          # Tier A unit tests, mocked do_parse
└── integration/
    ├── __init__.py
    ├── test_pipeline_integration.py   # Tier B, gated by MINERU_INTEGRATION=1
    └── test_vlm_integration.py        # Tier B, gated
```

**Replaced files:**

```
packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/
├── __init__.py            # MODIFIED — remove RapidOcrEngine etc. exports
├── extract.py             # REWRITTEN — thin mineru wrapper, ~100 LOC
└── ocr_engine.py          # DELETED — RapidOCR/PaddleOCR abstraction

packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/
├── __init__.py            # MODIFIED — remove extract_doc, extract_doc_from_layout
└── extract.py             # REWRITTEN — thin mineru wrapper, ~120 LOC
```

**Modified files:**

```
packages/pdfsys-core/src/pdfsys_core/config.py   # PipelineConfig + VlmConfig field rename
packages/pdfsys-parser-pipeline/pyproject.toml   # drop rapidocr, add mineru[pipeline]
packages/pdfsys-parser-vlm/pyproject.toml        # mineru → mineru[vlm], drop accelerate
packages/pdfsys-bench/src/pdfsys_bench/loop.py   # call-site updates (~3 spots)
packages/pdfsys-bench/src/pdfsys_bench/__main__.py  # --vlm-engine flag
packages/pdfsys-cli/src/pdfsys_cli/config.py     # PipelineCfg + VlmCfg YAML mirror
packages/pdfsys-cli/src/pdfsys_cli/runner.py     # PipelineConfig + VlmConfig assembly
docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md  # §15 post-build
```

---

## Task 1: Env hygiene + mineru import smoke

Verify `mineru.cli.common.do_parse` imports cleanly post-`uv sync`. This is the first gate — everything downstream depends on it. The earlier session had a `cv2` AttributeError because `opencv-python` and `opencv-python-headless` both got installed; `uv` re-installed `opencv-python-headless` automatically after a manual uninstall. The plan must explicitly verify the import chain works, and document the recovery procedure if it breaks.

**Files:** none modified — this task is purely verification. Output goes into commit message.

- [ ] **Step 1: Run `uv sync` to ensure the venv matches the lockfile**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv sync
```

Expected: completes without errors. If errors mention `opencv`, proceed to Step 2.

- [ ] **Step 2: Verify cv2 + mineru imports**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run python - <<'PY'
import cv2
assert cv2.__file__ is not None, "cv2 module body is empty (opencv conflict)"
assert hasattr(cv2, "INTER_NEAREST"), "cv2.INTER_NEAREST missing (broken install)"
print(f"cv2 OK: {cv2.__file__} v{cv2.__version__}")

from mineru.cli.common import do_parse, prepare_env
print(f"mineru.cli.common.do_parse OK: {do_parse.__module__}")
print(f"mineru.cli.common.prepare_env OK: {prepare_env.__module__}")

from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
print(f"mineru.backend.pipeline OK: {pipeline_doc_analyze.__module__}")

from mineru.backend.vlm.vlm_analyze import ModelSingleton
print(f"mineru.backend.vlm OK")

print("All imports succeeded.")
PY
```

Expected output ends with `All imports succeeded.`

- [ ] **Step 3 (only if Step 2 fails with cv2 error): Recovery procedure**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv pip uninstall opencv-python-headless
uv pip install --force-reinstall 'opencv-python>=4.11.0.86'
```

Re-run Step 2. If it still fails, STOP and report BLOCKED — a different transitive dep is at fault.

- [ ] **Step 4: Record the working state in a tiny smoke file**

Create `tests/parsers/__init__.py` (empty marker file for the new package):

```bash
mkdir -p tests/parsers
: > tests/parsers/__init__.py
```

And create `tests/parsers/test_mineru_smoke.py` to lock the import contract in CI:

```python
"""Locks the mineru import surface used by both parser packages.

If these imports break, parsers can't work. Fast (no model load).
"""

from __future__ import annotations


def test_mineru_do_parse_importable() -> None:
    from mineru.cli.common import do_parse, prepare_env
    assert callable(do_parse)
    assert callable(prepare_env)


def test_mineru_backend_modules_importable() -> None:
    """Both pipeline and vlm backend modules import without cv2 errors."""
    from mineru.backend.pipeline import pipeline_analyze  # noqa: F401
    from mineru.backend.vlm import vlm_analyze  # noqa: F401


def test_cv2_module_body_loaded() -> None:
    """Guard against opencv-python / opencv-python-headless conflict."""
    import cv2
    assert cv2.__file__ is not None
    assert hasattr(cv2, "INTER_NEAREST")
```

- [ ] **Step 5: Run the smoke tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_mineru_smoke.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check tests/parsers/
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add tests/parsers/__init__.py tests/parsers/test_mineru_smoke.py
git commit -m "test(parsers): lock mineru + cv2 import surface for downstream tasks"
```

---

## Task 2: Migrate `PipelineConfig` + `VlmConfig` schemas in `pdfsys-core`

Replace the RapidOCR-flavored fields with mineru-flavored ones. This is a breaking change to the dataclass schemas; the next tasks update all callers. No tests need to fail in this task — we add one new test that pins the new shape.

**Files:**
- Modify: `packages/pdfsys-core/src/pdfsys_core/config.py`

- [ ] **Step 1: Read the current state**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
sed -n '65,82p' packages/pdfsys-core/src/pdfsys_core/config.py
```

Confirm the file currently has `class PipelineConfig` with `ocr_engine`, `languages`, `render_dpi` and `class VlmConfig` with `model`, `max_batch_size`, `render_dpi`.

- [ ] **Step 2: Replace `PipelineConfig` and `VlmConfig`**

In `packages/pdfsys-core/src/pdfsys_core/config.py`, replace the existing two dataclasses with:

```python
@dataclass(slots=True)
class PipelineConfig:
    """Mineru pipeline-mode backend (parser-pipeline) configuration.

    All fields map directly to ``mineru.cli.common.do_parse`` kwargs.
    """

    formula_enable: bool = True                   # do_parse(formula_enable=...)
    table_enable: bool = True                     # do_parse(table_enable=...)
    p_lang: str = "ch"                            # do_parse(p_lang_list=[p_lang])
    output_dir: Path | None = None                # None = use tmpdir, delete after


@dataclass(slots=True)
class VlmConfig:
    """Mineru VLM-mode backend (parser-vlm) configuration.

    ``engine`` is appended to ``vlm-`` and passed to do_parse(backend=...).
    Available engines (per mineru 3.x): ``transformers`` (default, portable),
    ``mlx-engine`` (Apple Silicon), ``vllm-engine`` (NVIDIA GPU).
    """

    engine: str = "transformers"                  # transformers | mlx-engine | vllm-engine
    formula_enable: bool = True
    table_enable: bool = True
    p_lang: str = "ch"
    output_dir: Path | None = None
```

Add `from pathlib import Path` at the top of the file if not already present.

- [ ] **Step 3: Add a unit test pinning the new shape**

Create `tests/core/__init__.py` (if not present):

```bash
mkdir -p tests/core
[ -f tests/core/__init__.py ] || : > tests/core/__init__.py
```

Create `tests/core/test_parser_configs.py`:

```python
"""Pin the new shape of PipelineConfig + VlmConfig (mineru-flavored)."""

from __future__ import annotations

from pathlib import Path

from pdfsys_core import PipelineConfig, VlmConfig


def test_pipeline_config_defaults() -> None:
    c = PipelineConfig()
    assert c.formula_enable is True
    assert c.table_enable is True
    assert c.p_lang == "ch"
    assert c.output_dir is None


def test_pipeline_config_override() -> None:
    c = PipelineConfig(
        formula_enable=False, table_enable=False, p_lang="en",
        output_dir=Path("/tmp/x"),
    )
    assert c.formula_enable is False
    assert c.p_lang == "en"
    assert c.output_dir == Path("/tmp/x")


def test_vlm_config_defaults() -> None:
    c = VlmConfig()
    assert c.engine == "transformers"
    assert c.formula_enable is True
    assert c.p_lang == "ch"
    assert c.output_dir is None


def test_vlm_config_engine_override() -> None:
    c = VlmConfig(engine="mlx-engine")
    assert c.engine == "mlx-engine"
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/core/test_parser_configs.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full test suite to catch broken callers**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/ -v 2>&1 | tail -10
```

Expected: a small number of failures in modules that still reference the old field names — `pdfsys-cli/runner.py` and any old test of `pdfsys-parser-pipeline` / `pdfsys-parser-vlm`. Note them; subsequent tasks fix them.

It is OK to commit this task with downstream failures — those callers are updated in Tasks 3, 6, 7. The release-gate tests (`tests/bench/`, `tests/architecture/`) must NOT break — verify:

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/bench/ tests/architecture/ -v 2>&1 | tail -5
```

Expected: all green.

- [ ] **Step 6: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/pdfsys-core/src/pdfsys_core/config.py tests/core/test_parser_configs.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-core/src/pdfsys_core/config.py \
        tests/core/__init__.py \
        tests/core/test_parser_configs.py
git commit -m "refactor(core): PipelineConfig + VlmConfig → mineru do_parse field names"
```

---

## Task 3: Parser dependency swap

Replace RapidOCR with `mineru[pipeline]` in parser-pipeline, switch parser-vlm to `mineru[vlm]`. Verify `uv sync` resolves cleanly.

**Files:**
- Modify: `packages/pdfsys-parser-pipeline/pyproject.toml`
- Modify: `packages/pdfsys-parser-vlm/pyproject.toml`

- [ ] **Step 1: Update parser-pipeline pyproject.toml**

Replace `packages/pdfsys-parser-pipeline/pyproject.toml` dependency block (lines ~9–14) with:

```toml
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru[pipeline]>=3.1,<4.0",
    "numpy>=1.24",
    "Pillow>=10.0",
]
```

Also update the `description = "..."` line at the top to:

```toml
description = "OCR pipeline backend: thin wrapper around mineru.cli.common.do_parse(backend='pipeline')."
```

- [ ] **Step 2: Update parser-vlm pyproject.toml**

Replace `packages/pdfsys-parser-vlm/pyproject.toml` dependency block with:

```toml
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru[vlm]>=3.1,<4.0",
    "numpy>=1.24",
    "Pillow>=10.0",
]
```

Also update the `description = "..."` line:

```toml
description = "VLM backend: thin wrapper around mineru.cli.common.do_parse(backend='vlm-<engine>')."
```

Note: explicit `accelerate>=1.0` is dropped — it comes via `mineru[vlm]`'s `accelerate>=1.5.1` extra.

- [ ] **Step 3: Sync the workspace**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv sync
```

Expected: completes without errors. mineru extras may pull additional packages but should not produce conflicts.

- [ ] **Step 4: Verify the mineru smoke tests still pass**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_mineru_smoke.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Confirm rapidocr is no longer a direct workspace dep**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
grep -rn "rapidocr\|RapidOCR" packages/*/pyproject.toml || echo "no direct rapidocr refs"
```

Expected: `no direct rapidocr refs`. (The package may still be in `uv.lock` as a transitive — that's OK; we only care that no workspace package declares it.)

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-parser-pipeline/pyproject.toml \
        packages/pdfsys-parser-vlm/pyproject.toml \
        uv.lock
git commit -m "deps(parsers): swap rapidocr → mineru[pipeline]; mineru → mineru[vlm]"
```

(Include `uv.lock` only if `uv sync` modified it. Verify with `git status` first.)

---

## Task 4: Rewrite `pdfsys-parser-pipeline` as a mineru pipeline-mode wrapper

Delete `ocr_engine.py`, replace `extract.py` with a thin `do_parse(backend="pipeline")` wrapper, update `__init__.py` exports, add Tier A unit tests.

**Files:**
- Replace: `packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/extract.py`
- Delete: `packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/ocr_engine.py`
- Modify: `packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/__init__.py`
- Create: `tests/parsers/test_pipeline_parser.py`

- [ ] **Step 1: Write the failing unit tests first**

Create `tests/parsers/test_pipeline_parser.py`:

```python
"""Tier A: PipelineParser unit tests with mocked mineru.cli.common.do_parse.

These tests must NEVER load real mineru models. The mock writes a known
.md + sidecars to the output_dir so the parser's read-back logic is
exercised end-to-end without touching the network or disk-cache models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, PipelineConfig
from pdfsys_parser_pipeline import PipelineParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_do_parse(expected_md: str, expected_middle: dict, expected_content: list):
    """Returns a side_effect that writes mineru-shaped outputs to the dir
    the parser would pass in."""
    def _side_effect(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                     backend, **kwargs):
        # mineru lays files at <output_dir>/<pdf_name>/<parse_method>/<pdf_name>.md
        parse_method = "auto"
        for name in pdf_file_names:
            md_dir = Path(output_dir) / name / parse_method
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "images").mkdir(exist_ok=True)
            (md_dir / f"{name}.md").write_text(expected_md, encoding="utf-8")
            (md_dir / f"{name}_middle.json").write_text(
                json.dumps(expected_middle), encoding="utf-8"
            )
            (md_dir / f"{name}_content_list.json").write_text(
                json.dumps(expected_content), encoding="utf-8"
            )
    return _side_effect


def test_extract_returns_doc_with_markdown(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()

    fake = _fake_do_parse("# Hello\n\nWorld.\n", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake) as m:
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "out"))
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.PIPELINE
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# Hello\n\nWorld.\n"

    # mineru received the right backend argument
    assert m.call_count == 1
    _, kwargs = m.call_args
    assert kwargs["backend"] == "pipeline"
    assert kwargs["p_lang_list"] == ["ch"]
    assert kwargs["formula_enable"] is True
    assert kwargs["table_enable"] is True


def test_extract_records_sidecar_paths(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    fake = _fake_do_parse("md", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake):
        parser = PipelineParser(PipelineConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "pipeline"
    assert doc.stats["middle_json_path"] is not None
    assert doc.stats["content_list_path"] is not None
    # Paths are relative to output_dir
    middle_full = out_dir / doc.stats["middle_json_path"]
    assert middle_full.exists()


def test_extract_tmpdir_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("# X", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake):
        parser = PipelineParser(PipelineConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.markdown == "# X"
    # Sidecar paths are null when no persistent output_dir
    assert doc.stats["middle_json_path"] is None
    assert doc.stats["content_list_path"] is None


def test_extract_uses_config_p_lang(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake) as m:
        parser = PipelineParser(PipelineConfig(p_lang="en", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["p_lang_list"] == ["en"]


def test_extract_propagates_do_parse_errors(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    def _raise(*a, **kw):
        raise RuntimeError("simulated mineru failure")

    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=_raise):
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "o"))
        with pytest.raises(RuntimeError, match="simulated mineru failure"):
            parser.extract(pdf)


def test_extract_raises_when_markdown_missing(tmp_path: Path) -> None:
    """If mineru returns without writing a .md, surface a clear error."""
    pdf = _make_pdf(tmp_path)

    def _do_nothing(*a, **kw):
        pass  # mineru wrote nothing

    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=_do_nothing):
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "o"))
        with pytest.raises(FileNotFoundError, match="markdown"):
            parser.extract(pdf)
```

- [ ] **Step 2: Run the failing tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_pipeline_parser.py -v
```

Expected: all 6 fail with ImportError or AttributeError (`PipelineParser.extract` doesn't yet match the new signature).

- [ ] **Step 3: Delete `ocr_engine.py`**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
rm packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/ocr_engine.py
```

- [ ] **Step 4: Replace `extract.py`**

Overwrite `packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/extract.py` with:

```python
"""Mineru-backed pipeline parser.

Thin wrapper around ``mineru.cli.common.do_parse(backend="pipeline")``.
Mineru handles layout analysis + OCR + post-processing end-to-end; this
module only marshals input PDFs in and reads markdown + sidecars out.

See ``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from mineru.cli.common import do_parse
from pdfsys_core import Backend, ExtractedDoc, PipelineConfig

_PARSE_METHOD = "auto"  # mineru pipeline mode default


class PipelineParser:
    """Mineru pipeline-mode parser. Stateless; mineru manages model caching."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def extract(self, pdf_path: Path) -> ExtractedDoc:
        """Extract markdown from ``pdf_path`` via mineru pipeline mode.

        Writes to ``config.output_dir/<sha>/<parse_method>/`` if set,
        otherwise a tmpdir that is cleaned up before returning.
        """
        pdf_path = Path(pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()

        if self.config.output_dir is not None:
            output_root = Path(self.config.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            return self._run(sha, pdf_bytes, output_root, persistent=True)

        with tempfile.TemporaryDirectory(prefix="pdfsys-mineru-pipeline-") as td:
            return self._run(sha, pdf_bytes, Path(td), persistent=False)

    def _run(
        self,
        sha: str,
        pdf_bytes: bytes,
        output_root: Path,
        *,
        persistent: bool,
    ) -> ExtractedDoc:
        do_parse(
            output_dir=str(output_root),
            pdf_file_names=[sha],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[self.config.p_lang],
            backend="pipeline",
            parse_method=_PARSE_METHOD,
            formula_enable=self.config.formula_enable,
            table_enable=self.config.table_enable,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            image_analysis=True,
        )

        md_dir = output_root / sha / _PARSE_METHOD
        md_path = md_dir / f"{sha}.md"
        if not md_path.exists():
            # Defensive fallback: glob, in case mineru changes layout
            candidates = list(md_dir.glob("*.md")) if md_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(
                    f"mineru did not produce a markdown file under {md_dir} (sha={sha})"
                )
            md_path = candidates[0]

        markdown = md_path.read_text(encoding="utf-8")

        stats: dict[str, Any] = {
            "mineru_backend": "pipeline",
            "mineru_version": _mineru_version(),
            "middle_json_path": _rel_or_none(
                md_dir / f"{sha}_middle.json", output_root, persistent
            ),
            "content_list_path": _rel_or_none(
                md_dir / f"{sha}_content_list.json", output_root, persistent
            ),
        }

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.PIPELINE,
            segments=(),
            markdown=markdown,
            stats=stats,
        )


def _rel_or_none(path: Path, root: Path, persistent: bool) -> str | None:
    """Return path relative to root if it exists AND output is persistent.

    For tmpdir runs, all paths vanish on cleanup so we record None.
    """
    if not persistent:
        return None
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mineru_version() -> str | None:
    try:
        import mineru  # noqa: PLC0415
        return getattr(mineru, "__version__", None)
    except ImportError:
        return None
```

- [ ] **Step 5: Update `__init__.py`**

Overwrite `packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/__init__.py` with:

```python
"""pdfsys-parser-pipeline — mineru pipeline-mode wrapper.

Thin shim over ``mineru.cli.common.do_parse(backend="pipeline")``. The
old RapidOCR / region-OCR pipeline was deleted in the mineru migration
(2026-05-22, spec: docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md).
"""

from __future__ import annotations

from .extract import PipelineParser

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PipelineParser",
]
```

- [ ] **Step 6: Run the unit tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_pipeline_parser.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Verify no surviving rapidocr references in this package**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
grep -rn "rapidocr\|RapidOCR\|ocr_engine\|OcrEngine" packages/pdfsys-parser-pipeline/ || echo "clean"
```

Expected: `clean`.

- [ ] **Step 8: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/pdfsys-parser-pipeline/ tests/parsers/test_pipeline_parser.py
```

Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add packages/pdfsys-parser-pipeline/src/ tests/parsers/test_pipeline_parser.py
git commit -m "refactor(parser-pipeline): replace RapidOCR with mineru pipeline-mode wrapper"
```

Verify the commit includes the deletion of `ocr_engine.py`:

```bash
git show --stat HEAD | head -10
```

Expected: shows `delete mode` for `ocr_engine.py`.

---

## Task 5: Rewrite `pdfsys-parser-vlm` as a mineru VLM-mode wrapper

Replace `extract.py` with a thin `do_parse(backend="vlm-<engine>")` wrapper, update `__init__.py` exports, add Tier A unit tests. Same structure as Task 4 but with a configurable engine string.

**Files:**
- Replace: `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py`
- Modify: `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/__init__.py`
- Create: `tests/parsers/test_vlm_parser.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/parsers/test_vlm_parser.py`:

```python
"""Tier A: VlmParser unit tests with mocked mineru.cli.common.do_parse.

These tests must NEVER load real mineru VLM weights (~7GB).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, VlmConfig
from pdfsys_parser_vlm import VlmParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_do_parse(expected_md: str):
    """Returns a side_effect writing mineru-shaped outputs."""
    def _side_effect(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                     backend, **kwargs):
        parse_method = "auto"
        for name in pdf_file_names:
            md_dir = Path(output_dir) / name / parse_method
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "images").mkdir(exist_ok=True)
            (md_dir / f"{name}.md").write_text(expected_md, encoding="utf-8")
            (md_dir / f"{name}_middle.json").write_text(
                json.dumps({"pages": []}), encoding="utf-8"
            )
            (md_dir / f"{name}_content_list.json").write_text(
                json.dumps([]), encoding="utf-8"
            )
    return _side_effect


def test_vlm_extract_returns_doc(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()

    fake = _fake_do_parse("# VLM Output\n\n$$E=mc^2$$\n")
    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "out"))
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.VLM
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# VLM Output\n\n$$E=mc^2$$\n"

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-transformers"


def test_vlm_extract_with_mlx_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(engine="mlx-engine", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-mlx-engine"


def test_vlm_extract_with_vllm_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(engine="vllm-engine", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-vllm-engine"


def test_vlm_extract_records_sidecars(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=fake):
        parser = VlmParser(VlmConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "vlm-transformers"
    assert doc.stats["middle_json_path"] is not None
    assert doc.stats["content_list_path"] is not None


def test_vlm_extract_tmpdir_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("# Y")
    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=fake):
        parser = VlmParser(VlmConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.markdown == "# Y"
    assert doc.stats["middle_json_path"] is None


def test_vlm_extract_propagates_errors(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    def _raise(*a, **kw):
        raise RuntimeError("simulated vlm failure")

    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=_raise):
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "o"))
        with pytest.raises(RuntimeError, match="simulated vlm failure"):
            parser.extract(pdf)


def test_vlm_extract_raises_when_markdown_missing(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    def _do_nothing(*a, **kw):
        pass

    with patch("pdfsys_parser_vlm.extract.do_parse", side_effect=_do_nothing):
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "o"))
        with pytest.raises(FileNotFoundError, match="markdown"):
            parser.extract(pdf)
```

- [ ] **Step 2: Run the failing tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_vlm_parser.py -v
```

Expected: all 7 fail (current VlmParser has a different signature: `extract_complex_pages`).

- [ ] **Step 3: Replace `extract.py`**

Overwrite `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` with:

```python
"""Mineru-backed VLM parser.

Thin wrapper around ``mineru.cli.common.do_parse(backend="vlm-<engine>")``.
Mineru handles layout analysis + per-region VLM extraction + markdown
assembly end-to-end; this module only marshals input PDFs in and reads
markdown + sidecars out.

See ``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from mineru.cli.common import do_parse
from pdfsys_core import Backend, ExtractedDoc, VlmConfig

_PARSE_METHOD = "auto"


class VlmParser:
    """Mineru VLM-mode parser. Stateless; mineru manages model caching."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()

    def extract(self, pdf_path: Path) -> ExtractedDoc:
        """Extract markdown from ``pdf_path`` via mineru VLM mode.

        Writes to ``config.output_dir/<sha>/<parse_method>/`` if set,
        otherwise a tmpdir that is cleaned up before returning.
        """
        pdf_path = Path(pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()

        if self.config.output_dir is not None:
            output_root = Path(self.config.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            return self._run(sha, pdf_bytes, output_root, persistent=True)

        with tempfile.TemporaryDirectory(prefix="pdfsys-mineru-vlm-") as td:
            return self._run(sha, pdf_bytes, Path(td), persistent=False)

    def _run(
        self,
        sha: str,
        pdf_bytes: bytes,
        output_root: Path,
        *,
        persistent: bool,
    ) -> ExtractedDoc:
        backend = f"vlm-{self.config.engine}"
        do_parse(
            output_dir=str(output_root),
            pdf_file_names=[sha],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[self.config.p_lang],
            backend=backend,
            parse_method=_PARSE_METHOD,
            formula_enable=self.config.formula_enable,
            table_enable=self.config.table_enable,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            image_analysis=True,
        )

        md_dir = output_root / sha / _PARSE_METHOD
        md_path = md_dir / f"{sha}.md"
        if not md_path.exists():
            candidates = list(md_dir.glob("*.md")) if md_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(
                    f"mineru did not produce a markdown file under {md_dir} (sha={sha})"
                )
            md_path = candidates[0]

        markdown = md_path.read_text(encoding="utf-8")

        stats: dict[str, Any] = {
            "mineru_backend": backend,
            "mineru_version": _mineru_version(),
            "middle_json_path": _rel_or_none(
                md_dir / f"{sha}_middle.json", output_root, persistent
            ),
            "content_list_path": _rel_or_none(
                md_dir / f"{sha}_content_list.json", output_root, persistent
            ),
        }

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=(),
            markdown=markdown,
            stats=stats,
        )


def _rel_or_none(path: Path, root: Path, persistent: bool) -> str | None:
    if not persistent:
        return None
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mineru_version() -> str | None:
    try:
        import mineru  # noqa: PLC0415
        return getattr(mineru, "__version__", None)
    except ImportError:
        return None
```

- [ ] **Step 4: Update `__init__.py`**

Overwrite `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/__init__.py` with:

```python
"""pdfsys-parser-vlm — mineru VLM-mode wrapper.

Thin shim over ``mineru.cli.common.do_parse(backend="vlm-<engine>")``.
The old region-based ModelSingleton path was deleted in the mineru
migration (2026-05-22).
"""

from __future__ import annotations

from .extract import VlmParser

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "VlmParser",
]
```

- [ ] **Step 5: Run the unit tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/test_vlm_parser.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Verify no surviving region-based / ModelSingleton references**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
grep -rn "extract_complex_pages\|ModelSingleton\|merge_segments_to_markdown\|_run_vlm_per_region" packages/pdfsys-parser-vlm/ || echo "clean"
```

Expected: `clean`.

- [ ] **Step 7: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/pdfsys-parser-vlm/ tests/parsers/test_vlm_parser.py
```

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add packages/pdfsys-parser-vlm/src/ tests/parsers/test_vlm_parser.py
git commit -m "refactor(parser-vlm): replace region-based ModelSingleton with mineru VLM-mode wrapper"
```

---

## Task 6: Wire bench loop + add `--vlm-engine` flag

Update `pdfsys-bench/loop.py` to call the new `parser.extract(pdf_path)` signature. Add a `--vlm-engine` CLI flag. Verify the bench suite still passes.

**Files:**
- Modify: `packages/pdfsys-bench/src/pdfsys_bench/loop.py`
- Modify: `packages/pdfsys-bench/src/pdfsys_bench/__main__.py`

- [ ] **Step 1: Update `loop.py` pipeline call site**

Locate the existing pipeline-extract block in `loop.py` (around line 310–321):

```python
    # -- Pipeline extraction ---------------------------------------------------
    if stage_b.backend == Backend.PIPELINE and pipeline_parser is not None:
        try:
            t4 = time.perf_counter()
            extracted = pipeline_parser.extract(
                pdf_path, layout, sha256=layout.sha256
            )
```

Replace the `extracted = pipeline_parser.extract(...)` call with the new signature (no `layout`, no `sha256`):

```python
            extracted = pipeline_parser.extract(pdf_path)
```

(Find the line via `grep -n "pipeline_parser.extract(" packages/pdfsys-bench/src/pdfsys_bench/loop.py` if line numbers differ.)

- [ ] **Step 2: Update `loop.py` VLM call site**

Find the existing VLM-extract block:

```python
            extracted = vlm_parser.extract_complex_pages(
                pdf_path, layout, sha256=layout.sha256
            )
```

Replace with:

```python
            extracted = vlm_parser.extract(pdf_path)
```

- [ ] **Step 3: Update `loop.py` cascade VLM closure**

Find (around line 435–437):

```python
        def _vlm_extract(p: Path) -> Any:
            layout = _ensure_layout(p)
            return vlm_parser.extract_complex_pages(p, layout, sha256=layout.sha256)
```

Replace with:

```python
        def _vlm_extract(p: Path) -> Any:
            return vlm_parser.extract(p)
```

The `_ensure_layout` call is removed because mineru does its own layout. Layout-analyser is still called earlier in the cascade for Stage-B routing, and the closure cache it builds (`layout_holder`) is harmless to keep — but the VLM closure no longer touches it.

- [ ] **Step 4: Update `loop.py` cascade pipeline closure**

Find:

```python
    def _pipeline_extract(p: Path) -> Any:
        layout = _ensure_layout(p)
        return pipeline_parser.extract(p, layout, sha256=layout.sha256)
```

Replace with:

```python
    def _pipeline_extract(p: Path) -> Any:
        return pipeline_parser.extract(p)
```

- [ ] **Step 5: Plumb VlmConfig.engine from CLI flag**

In `packages/pdfsys-bench/src/pdfsys_bench/loop.py`, find the `vlm_parser` construction site (around line 156–159):

```python
        if vlm_enabled:
            from pdfsys_parser_vlm import VlmParser
            from pdfsys_core import VlmConfig
            vlm_parser = VlmParser()
```

Replace with:

```python
        if vlm_enabled:
            from pdfsys_parser_vlm import VlmParser  # noqa: PLC0415
            from pdfsys_core import VlmConfig  # noqa: PLC0415
            vlm_parser = VlmParser(VlmConfig(engine=vlm_engine))
```

The function signature for `run_loop` needs `vlm_engine: str = "transformers"` added. Find the `def run_loop(` signature and add the parameter right after `vlm_enabled: bool = False`:

```python
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
    full_pipeline: bool = False,
    cache_dir: str | Path | None = None,
    vlm_enabled: bool = False,
    vlm_engine: str = "transformers",
    cascade: bool = False,
    cascade_skip_mupdf_threshold: float = 0.9,
) -> dict[str, Any]:
```

(The exact existing signature may differ; just insert `vlm_engine` after `vlm_enabled`.)

Also update the `_run_one_cascade` function's signature in the same way and forward `vlm_engine` when constructing VLM in the cascade path. Find any other place that calls `VlmParser()` with no args and pass the engine through.

- [ ] **Step 6: Add `--vlm-engine` CLI flag**

In `packages/pdfsys-bench/src/pdfsys_bench/__main__.py`, find the existing `--vlm` flag (around line 93). Just below it, add:

```python
    p.add_argument(
        "--vlm-engine",
        choices=("transformers", "mlx-engine", "vllm-engine"),
        default="transformers",
        help="Mineru VLM inference engine. Default transformers is portable; "
             "mlx-engine is faster on Apple Silicon; vllm-engine needs NVIDIA GPU.",
    )
```

In `main()`, pass it through:

```python
    summary = run_loop(
        ...
        vlm_enabled=args.vlm_enabled,
        vlm_engine=args.vlm_engine,
        cascade=args.cascade,
        ...
    )
```

(Insert `vlm_engine=args.vlm_engine,` right after `vlm_enabled=args.vlm_enabled,`.)

- [ ] **Step 7: Run the bench unit tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/bench/ tests/architecture/ tests/core/ tests/parsers/ -v
```

Expected: all green. The cascade and quality_rules tests should be unaffected; release-gate suite likewise.

- [ ] **Step 8: CLI smoke**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run python -m pdfsys_bench --help | grep -A1 vlm-engine
```

Expected: `--vlm-engine {transformers,mlx-engine,vllm-engine}` appears in the help output.

- [ ] **Step 9: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/pdfsys-bench/
```

Expected: `All checks passed!`

- [ ] **Step 10: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/loop.py \
        packages/pdfsys-bench/src/pdfsys_bench/__main__.py
git commit -m "feat(bench): wire mineru parsers + --vlm-engine flag"
```

---

## Task 7: Update `pdfsys-cli` config + runner

The CLI has its own YAML-loadable mirror (`PipelineCfg`, `VlmCfg`) and an assembly site in `runner.py` that constructs the canonical `PipelineConfig` and `VlmConfig` with the old field names. Update both.

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/config.py`
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py`

- [ ] **Step 1: Update `PipelineCfg` and `VlmCfg` in `config.py`**

Find the existing dataclasses (around lines 53–65) and replace:

```python
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
```

Also remove the YAML defaults in the documentation comment block at lines ~265–275 (search for `ocr_engine: rapidocr` and `device_mode: cpu` and replace with the new field names).

Locate the block (it lives inside a multi-line docstring or comment showing example YAML):

```yaml
    pipeline:
      ocr_engine: rapidocr          # rapidocr | paddleocr
      languages: [ch, en]
      render_dpi: 200
```

Replace with:

```yaml
    pipeline:
      formula_enable: true
      table_enable: true
      p_lang: ch
```

And replace any equivalent `vlm:` block with:

```yaml
    vlm:
      engine: transformers          # transformers | mlx-engine | vllm-engine
      enabled: false
      formula_enable: true
      table_enable: true
      p_lang: ch
```

- [ ] **Step 2: Update `runner.py` assembly site**

In `packages/pdfsys-cli/src/pdfsys_cli/runner.py`, find the `pipeline_parser` property (around line 107–118):

```python
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
```

Replace with:

```python
    @property
    def pipeline_parser(self) -> Any:
        if self._pipeline is None:
            from pdfsys_parser_pipeline import PipelineParser  # noqa: PLC0415
            from pdfsys_core import PipelineConfig  # noqa: PLC0415

            pc = PipelineConfig(
                formula_enable=self.cfg.pipeline.formula_enable,
                table_enable=self.cfg.pipeline.table_enable,
                p_lang=self.cfg.pipeline.p_lang,
            )
            self._pipeline = PipelineParser(config=pc)
        return self._pipeline
```

Find the `vlm_parser` property (around line 120–128):

```python
    @property
    def vlm_parser(self) -> Any:
        if self._vlm is None:
            from pdfsys_parser_vlm import VlmParser  # noqa: PLC0415
            from pdfsys_core import VlmConfig  # noqa: PLC0415

            vc = VlmConfig(model=self.cfg.vlm.model)
            self._vlm = VlmParser(config=vc)
        return self._vlm
```

Replace with:

```python
    @property
    def vlm_parser(self) -> Any:
        if self._vlm is None:
            from pdfsys_parser_vlm import VlmParser  # noqa: PLC0415
            from pdfsys_core import VlmConfig  # noqa: PLC0415

            vc = VlmConfig(
                engine=self.cfg.vlm.engine,
                formula_enable=self.cfg.vlm.formula_enable,
                table_enable=self.cfg.vlm.table_enable,
                p_lang=self.cfg.vlm.p_lang,
            )
            self._vlm = VlmParser(config=vc)
        return self._vlm
```

- [ ] **Step 3: Verify all callers compile**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run python -c "
from pdfsys_cli.runner import PipelineRunner  # or whatever the entry class is
from pdfsys_cli.config import PdfsysCfg
print('imports OK')
" 2>&1 | head -5
```

Expected: prints `imports OK`. If `pdfsys_cli` doesn't have a `PipelineRunner`, swap in the actual class name (find via `grep -nE "^class " packages/pdfsys-cli/src/pdfsys_cli/runner.py`).

- [ ] **Step 4: Run all tests**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/ -v 2>&1 | tail -10
```

Expected: all green.

- [ ] **Step 5: Verify no lingering old field references**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
grep -rn "ocr_engine\|languages.*list\|max_batch_size\|cfg\.vlm\.model" \
    packages/pdfsys-cli/src/ packages/pdfsys-bench/src/ packages/pdfsys-core/src/ \
    | grep -v __pycache__ || echo "clean"
```

Expected: `clean`. (Any remaining references are deal-breakers.)

- [ ] **Step 6: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/pdfsys-cli/
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/
git commit -m "refactor(cli): align YAML schema + runner with mineru parser configs"
```

---

## Task 8: Tier B integration tests (gated, optional)

These tests actually invoke mineru. They require model weights cached locally (mineru downloads ~3GB for pipeline, ~7GB for VLM on first call). Gate by env var `MINERU_INTEGRATION=1` so CI without weights doesn't trigger downloads.

**Files:**
- Create: `tests/parsers/integration/__init__.py`
- Create: `tests/parsers/integration/test_pipeline_integration.py`
- Create: `tests/parsers/integration/test_vlm_integration.py`

- [ ] **Step 1: Locate a small real PDF for testing**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
ls packages/pdfsys-bench/omnidocbench_100/pdfs | head -3
```

Pick one (e.g. the first listed). Note its filename for the test fixtures.

- [ ] **Step 2: Create the gating marker package**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
mkdir -p tests/parsers/integration
: > tests/parsers/integration/__init__.py
```

- [ ] **Step 3: Create `test_pipeline_integration.py`**

Create `tests/parsers/integration/test_pipeline_integration.py`:

```python
"""Tier B: real mineru pipeline-mode invocation. Slow, gated.

Runs only when MINERU_INTEGRATION=1. Downloads ~3GB of weights on
first run; subsequent runs use the local cache. Run via:

    MINERU_INTEGRATION=1 uv run pytest tests/parsers/integration/ -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_GATE = os.environ.get("MINERU_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not _GATE, reason="set MINERU_INTEGRATION=1 to run mineru integration tests"
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PDF_DIR = _REPO_ROOT / "packages" / "pdfsys-bench" / "omnidocbench_100" / "pdfs"


def _first_pdf() -> Path:
    pdfs = sorted(p for p in _PDF_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        pytest.skip(f"no PDFs found under {_PDF_DIR}")
    return pdfs[0]


def test_pipeline_real_extraction(tmp_path: Path) -> None:
    from pdfsys_core import Backend, PipelineConfig
    from pdfsys_parser_pipeline import PipelineParser

    pdf = _first_pdf()
    parser = PipelineParser(PipelineConfig(output_dir=tmp_path))
    doc = parser.extract(pdf)

    assert doc.backend == Backend.PIPELINE
    assert doc.sha256
    assert len(doc.markdown.strip()) > 0, "expected non-empty markdown"
    assert doc.stats["mineru_backend"] == "pipeline"
    assert doc.stats["middle_json_path"] is not None
```

- [ ] **Step 4: Create `test_vlm_integration.py`**

Create `tests/parsers/integration/test_vlm_integration.py`:

```python
"""Tier B: real mineru VLM-mode invocation. Slow, gated.

Runs only when MINERU_INTEGRATION=1. Downloads ~7GB on first run.
Defaults to engine=transformers (most portable). Override via
MINERU_VLM_ENGINE=mlx-engine on Apple Silicon for speed.

    MINERU_INTEGRATION=1 uv run pytest tests/parsers/integration/test_vlm_integration.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_GATE = os.environ.get("MINERU_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not _GATE, reason="set MINERU_INTEGRATION=1 to run mineru integration tests"
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PDF_DIR = _REPO_ROOT / "packages" / "pdfsys-bench" / "omnidocbench_100" / "pdfs"
_ENGINE = os.environ.get("MINERU_VLM_ENGINE", "transformers")


def _first_pdf() -> Path:
    pdfs = sorted(p for p in _PDF_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        pytest.skip(f"no PDFs found under {_PDF_DIR}")
    return pdfs[0]


def test_vlm_real_extraction(tmp_path: Path) -> None:
    from pdfsys_core import Backend, VlmConfig
    from pdfsys_parser_vlm import VlmParser

    pdf = _first_pdf()
    parser = VlmParser(VlmConfig(engine=_ENGINE, output_dir=tmp_path))
    doc = parser.extract(pdf)

    assert doc.backend == Backend.VLM
    assert doc.sha256
    assert len(doc.markdown.strip()) > 0
    assert doc.stats["mineru_backend"] == f"vlm-{_ENGINE}"
```

- [ ] **Step 5: Verify the gate works (no env var → skipped)**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/parsers/integration/ -v 2>&1 | tail -10
```

Expected: 2 tests SKIPPED (gate inactive).

- [ ] **Step 6: (Optional) Run the integration tests if you have model weights**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
MINERU_INTEGRATION=1 uv run pytest tests/parsers/integration/test_pipeline_integration.py -v -s 2>&1 | tail -20
```

Expected: 1 passed (after potentially-long model download on first run).

If you skip this step (no GPU / no weight cache), note it explicitly in the commit message.

- [ ] **Step 7: Lint**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check tests/parsers/integration/
```

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add tests/parsers/integration/
git commit -m "test(parsers): Tier B mineru integration tests gated by MINERU_INTEGRATION=1"
```

---

## Task 9: E2E bench smoke + spec §15 post-build note

Run the bench loop with `--cascade --vlm` on a small slice of OmniDocBench and verify mineru's VLM mode actually fires for at least one row. Update the spec with the post-build note documenting commits + observed behavior.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md` (append §15)

- [ ] **Step 1: Run a 5-row bench with `--cascade --vlm`**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run python -m pdfsys_bench \
    --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
    --out out/bench_mineru_smoke.jsonl \
    --cascade --vlm \
    --vlm-engine transformers \
    --limit 5 \
    --no-quality
```

Expected:
- Completes without errors.
- May take several minutes if weights download.
- Writes 5 rows to `out/bench_mineru_smoke.jsonl`.

If weight download fails or the run takes longer than 30 minutes on a 5-row slice, capture the failure and commit Task 9 with **status DONE_WITH_CONCERNS** documenting the issue in §15. The migration code itself is still landed.

- [ ] **Step 2: Inspect the backend distribution**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
python3 - <<'PY'
import collections, json
counts_backend = collections.Counter()
counts_stages = collections.Counter()
mineru_backends = collections.Counter()
for line in open("out/bench_mineru_smoke.jsonl"):
    r = json.loads(line)
    counts_backend[r.get("backend")] += 1
    for att in (r.get("cascade_attempts") or []):
        counts_stages[att.get("stage")] += 1
    mb = r.get("extract_stats", {}).get("mineru_backend")
    if mb:
        mineru_backends[mb] += 1
print("backend:        ", dict(counts_backend))
print("cascade_stages: ", dict(counts_stages))
print("mineru_backends:", dict(mineru_backends))
PY
```

Expected: at least one of:
- `mineru_backends` contains `"pipeline"` or `"vlm-transformers"` (proves mineru actually ran).
- `backend` shows `"vlm"` for at least one row when `--vlm` is enabled (proves the silent-fallback bug from the spec is gone for cascade mode).

Record the actual output for §15.

- [ ] **Step 3: Append §15 to the spec**

Append the following block to the end of `docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md`:

```markdown

## 15. Post-build note

Implementation landed across 9 tasks. Plan: `docs/superpowers/plans/2026-05-22-mineru-parsers-migration.md`.

### Commits (in order)

```
Task 1 — env smoke
  <SHA> test(parsers): lock mineru + cv2 import surface for downstream tasks

Task 2 — config schema
  <SHA> refactor(core): PipelineConfig + VlmConfig → mineru do_parse field names

Task 3 — deps swap
  <SHA> deps(parsers): swap rapidocr → mineru[pipeline]; mineru → mineru[vlm]

Task 4 — parser-pipeline rewrite
  <SHA> refactor(parser-pipeline): replace RapidOCR with mineru pipeline-mode wrapper

Task 5 — parser-vlm rewrite
  <SHA> refactor(parser-vlm): replace region-based ModelSingleton with mineru VLM-mode wrapper

Task 6 — bench loop wiring
  <SHA> feat(bench): wire mineru parsers + --vlm-engine flag

Task 7 — cli alignment
  <SHA> refactor(cli): align YAML schema + runner with mineru parser configs

Task 8 — integration tests
  <SHA> test(parsers): Tier B mineru integration tests gated by MINERU_INTEGRATION=1
```

### End-to-end smoke

```
uv run python -m pdfsys_bench \
    --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
    --out out/bench_mineru_smoke.jsonl \
    --cascade --vlm --vlm-engine transformers \
    --limit 5 --no-quality
```

Result (fill in actuals):
- `backend = <dict>`
- `cascade_stages = <dict>`
- `mineru_backends = <dict>`

### Known follow-ups

- **opencv-python-headless reinstall.** The workspace still ends up with `opencv-python-headless` after `uv sync` (some transitive pulls it). Manual `uv pip uninstall opencv-python-headless` and reinstall of `opencv-python` is the current recovery. A proper fix is a workspace-level pin or a constraint file.
- **Spec #2 (HTTP services + Docker)** is the next plan. The parsers as in-process modules with their own `mineru[pipeline]` and `mineru[vlm]` extras coexist in one venv today; Spec #2 splits them into separate venvs / containers so cv2 + torch + transformers conflicts can't happen across parsers.
- **Stage-B `vlm` decisions still fall back to pipeline when `--vlm` is off.** Behavior preserved from before the migration. Spec #2 (or a separate small fix) should turn this into an explicit warning rather than silent degrade.
- **viz "Per-region extraction" card is now empty.** Mineru's middle.json contains the data; a future spec can render it back into the detail card.

### Test surface

```
tests/parsers/test_mineru_smoke.py       3 tests  (cv2 + mineru imports)
tests/parsers/test_pipeline_parser.py    6 tests  (Tier A, mocked do_parse)
tests/parsers/test_vlm_parser.py         7 tests  (Tier A, mocked do_parse)
tests/parsers/integration/               2 tests  (Tier B, gated)
tests/core/test_parser_configs.py        4 tests  (config dataclasses)
```

All Tier A + smoke tests pass via `uv run pytest tests/parsers/ tests/core/ -v`.
```

Fill in the `<SHA>` placeholders by running:

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
git log --oneline -10
```

And fill in the actual numbers from Step 2's output.

- [ ] **Step 4: Final test sweep**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run pytest tests/ -v 2>&1 | tail -5
```

Expected: all green (excluding Tier B integration tests which are gated).

- [ ] **Step 5: Lint sweep**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
uv run ruff check packages/ tests/
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit the spec note**

```bash
git add docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md
git commit -m "docs(spec): mineru parsers migration post-build note (§15)"
```

- [ ] **Step 7: Clean up the smoke artifact**

```bash
cd /Users/yinz/Codes/mnbvc/pdfsystem_mnbvc
rm -f out/bench_mineru_smoke.jsonl out/bench_mineru_smoke.summary.json
```

---

## Self-review notes

**Spec coverage:**
- §2 Goal 1 (parser-pipeline → mineru pipeline mode) → Task 4.
- §2 Goal 2 (parser-vlm → mineru VLM with engine flag) → Tasks 5, 6.
- §2 Goal 3 (symmetric `extract(pdf_path) -> ExtractedDoc`) → Tasks 4, 5.
- §2 Goal 4 (keep middle.json + content_list.json sidecars) → Tasks 4, 5 (`_rel_or_none`).
- §2 Goal 5 (bench `--cascade --vlm` actually hits mineru) → Task 9 E2E smoke.
- §2 Goal 6 (delete RapidOCR, region-based code) → Tasks 4, 5 (file deletions explicit).
- §5 Parser interface (`parser.extract(pdf_path) -> ExtractedDoc`) → Tasks 4, 5.
- §5 `stats` keys (`mineru_backend`, `mineru_version`, `middle_json_path`, `content_list_path`) → Tasks 4, 5.
- §5 sha256 computed inside parser → Tasks 4, 5 (`hashlib.sha256(pdf_path.read_bytes()).hexdigest()`).
- §5 output_dir semantics (None = tmpdir + delete, set = persist) → Tasks 4, 5 (`_run(..., persistent=...)`).
- §6 mineru entry-point usage (do_parse with specific kwargs) → Tasks 4, 5 (exact kwargs listed).
- §7 PipelineConfig / VlmConfig schemas → Task 2.
- §8 Bench loop integration → Task 6.
- §9 Removed code (ocr_engine.py, RapidOCR, accelerate dep) → Tasks 3, 4, 5.
- §10 Dependency cleanup → Task 3.
- §11 Testing strategy (Tier A mocked + Tier B gated) → Tasks 4, 5, 8.
- §12 Risks (cv2 install, model download in CI, p_lang default) → Tasks 1, 8 (cv2 + integration gate).
- §13 Acceptance criteria — every checkbox maps to a task step.

**Type / signature consistency:**
- `PipelineParser.extract(pdf_path: Path) -> ExtractedDoc` — Tasks 4, 6, 7 all agree.
- `VlmParser.extract(pdf_path: Path) -> ExtractedDoc` — Tasks 5, 6, 7 all agree.
- `VlmConfig.engine: str` — Tasks 2, 5, 6, 7 all use the same field name.
- `PipelineConfig` fields `formula_enable / table_enable / p_lang / output_dir` — Tasks 2, 4, 7 consistent.
- `ExtractedDoc.stats` keys `mineru_backend / mineru_version / middle_json_path / content_list_path` — Tasks 4, 5 emit; spec §5 documents.

**Placeholder scan:** none — every step has actual code, commands, or assertions. §15 in Task 9 has explicit `<SHA>` and `<dict>` placeholders that are FILLED IN at execution time per explicit instruction.
