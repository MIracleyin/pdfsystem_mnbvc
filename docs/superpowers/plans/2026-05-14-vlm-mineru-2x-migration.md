# VLM Engine Migration (magic-pdf 1.x → mineru 3.x) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap `pdfsys-parser-vlm`'s backend from broken `magic-pdf 1.0.1` (with paddle + 9 hidden deps + 3 site-packages patches) to `mineru >= 3.1`, re-enabling formula + table recognition while preserving the `VlmParser` public contract so runner/ParquetSink/Stage-B all stay untouched.

**Architecture:** Spike-first. Step 2 verifies mineru's actual Python API on a single PDF before committing to a rewrite. Steps 3-7 do the dependency swap + module rename + extract.py rewrite. Step 8-10 re-run the 10 VLM PDFs from the previous iteration and verify all 5 acceptance gates from spec §7.

**Tech Stack:** Python 3.11, mineru >= 3.1 (replacing magic-pdf), torch + MPS, existing `pdfsys-core` / `pdfsys-layout-analyser` / `pdfsys-cli` packages.

**Source spec:** `docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md` — read first if anything is unclear.

**User constraint (overrides default TDD pattern):** No unit tests, per precedent set in the previous iteration. Verification is exclusively via `import` checks, single-PDF smoke runs, and the 5 SQL acceptance queries in §7.

**Phasing note:** Task 2 is a discovery spike. Its output (a notes file at `out/mineru_spike_notes.md`) directly informs Task 7's rewrite. The controller should review the spike notes between Tasks 2 and 7. If subagent-driven: dispatch a fresh implementer for Task 7 with the spike notes pasted into the prompt.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `out/mineru_spike_notes.md` | create (spike output) | Verified mineru API surface, MPS path, content-type map, env vars, model download notes |
| `packages/pdfsys-parser-vlm/pyproject.toml` | modify | Replace `magic-pdf>=1.0` with `mineru>=3.1,<4.0` |
| `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` | rewrite | New ~250-LOC implementation using mineru API. Same public class/methods, internal engine swapped. |
| `packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py` | rename → `_vlm_config.py` | Module renamed; function `ensure_config` renamed `ensure_mineru_env`; writes mineru env vars + relocates old `~/magic-pdf.json` → `.bak` |
| `packages/pdfsys-cli/src/pdfsys_cli/runner.py` | modify (1 line) | Import + call-site rename for `_vlm_config.ensure_mineru_env` |
| `docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md` | modify (§12) | Append post-run note |

---

## Decomposition notes

- The spike (Task 2) is treated as a separate task because its output is consumed by Task 7. Subagents reading Task 7 in isolation MUST be given `out/mineru_spike_notes.md` contents in their prompt — they cannot infer API from `mineru 3.x` documentation that may not exist or be incomplete.
- Rename of `_mineru_config.py` is done as a small Task 5 rather than folded into the rewrite, because it crosses both `pdfsys-parser-vlm` and `pdfsys-cli`. Isolating it keeps the rewrite focused.
- The previous full-run parquet (`out/e2e_full/dataset.parquet`) is **not modified** during this plan. The new VLM rows go into a separate `out/vlm_mineru3_retry/dataset.parquet`. Merging back into `e2e_full` is deferred — out of scope.

---

## Task 1: Snapshot current env state (rollback baseline)

**Files:**
- Create: `out/env_snapshot_pre_mineru_migration.txt`

- [ ] **Step 1: Capture currently installed packages**

```bash
.venv/bin/python -c "import importlib.metadata as m; \
  print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" \
  > out/env_snapshot_pre_mineru_migration.txt
wc -l out/env_snapshot_pre_mineru_migration.txt
```

Expected: ~150-200 lines printed to file.

- [ ] **Step 2: Capture the magic-pdf site-packages patch fingerprint**

```bash
echo "=== modified magic_pdf files (file sizes/mtimes) ===" >> out/env_snapshot_pre_mineru_migration.txt
find .venv/lib/python*/site-packages/magic_pdf -type f -newer .venv/lib/python*/site-packages/magic_pdf/__init__.py 2>/dev/null \
  -exec stat -f "%N %z %Sm" {} \; \
  >> out/env_snapshot_pre_mineru_migration.txt || echo "(no magic_pdf dir or no patched files)" >> out/env_snapshot_pre_mineru_migration.txt
tail -10 out/env_snapshot_pre_mineru_migration.txt
```

- [ ] **Step 3: Snapshot existing `~/magic-pdf.json`**

```bash
echo "=== ~/magic-pdf.json content ===" >> out/env_snapshot_pre_mineru_migration.txt
cat ~/magic-pdf.json >> out/env_snapshot_pre_mineru_migration.txt 2>/dev/null || \
  echo "(no ~/magic-pdf.json)" >> out/env_snapshot_pre_mineru_migration.txt
```

- [ ] **Step 4: Commit the snapshot to git (out/ should be untracked but the snapshot is one-time documentation)**

```bash
# Force-add the snapshot even if out/ is gitignored
git add -f out/env_snapshot_pre_mineru_migration.txt
git commit -m "chore: snapshot env state before mineru migration"
```

If `out/` isn't gitignored (verify with `cat .gitignore`), drop the `-f`.

---

## Task 2: Spike — verify mineru 3.x API on one PDF

**Files:**
- Create: `out/mineru_spike_notes.md`
- Create (throwaway): `/tmp/mineru_spike.py`

This task does NOT modify any package code. It discovers the API surface that Task 7 will commit to.

- [ ] **Step 1: Install bare mineru into the existing venv**

```bash
uv pip install --index https://pypi.org/simple "mineru>=3.1,<4.0"
```

Expected: ~30 packages downloaded/installed. Note: this co-exists with magic-pdf during the spike (no uninstall yet — that's Task 3). They share some torch/transformers deps.

If `uv pip install` errors on dep conflicts (likely with `paddlepaddle`, `albumentations`, `opencv-python` version mismatches), record the error in the spike notes (Step 8 below), STOP and report BLOCKED.

- [ ] **Step 2: Inspect mineru's module layout**

```bash
.venv/bin/python -c "
import mineru, pkgutil
print('mineru.__version__:', getattr(mineru, '__version__', 'unknown'))
print('mineru.__file__:', mineru.__file__)
print()
print('submodules:')
for m in pkgutil.iter_modules(mineru.__path__):
    print(' -', m.name)
"
```

Record the output.

- [ ] **Step 3: Find the public VLM entry point**

mineru 3.x is undergoing a refactor; the entry point name may differ. Try in order:

```bash
.venv/bin/python -c "
candidates = [
    'mineru.cli.common',
    'mineru.backend.pipeline.pipeline_analyze',
    'mineru.backend.vlm.vlm_analyze',
    'mineru.backend.vlm',
    'mineru.api',
]
import importlib
for c in candidates:
    try:
        m = importlib.import_module(c)
        attrs = [a for a in dir(m) if not a.startswith('_')]
        print(f'  {c}: OK ({len(attrs)} public attrs) — {attrs[:8]}')
    except ImportError as e:
        print(f'  {c}: FAIL ({e})')
"
```

Record the OK candidates. At least one of these will be the entry point. If all fail, search alternatives:

```bash
grep -rn "def.*analyze\|def parse_pdf\|class.*Analyzer\|class.*Parser" \
  .venv/lib/python*/site-packages/mineru/ \
  --include="*.py" | head -30
```

Record the most-likely-public entry-points.

- [ ] **Step 4: Read the mineru CLI to find the canonical "analyze a PDF" call sequence**

```bash
.venv/bin/python -c "
import mineru.cli
import pkgutil, os
for m in pkgutil.iter_modules(mineru.cli.__path__):
    p = os.path.join(mineru.cli.__path__[0], m.name + '.py')
    if os.path.exists(p):
        print('===', m.name, '===')
        with open(p) as f:
            print(f.read()[:2000])
        print()
" 2>&1 | head -200
```

Record observed function names and call shapes. The CLI is the most reliable source-of-truth for "how to invoke mineru programmatically".

- [ ] **Step 5: Set up MPS + ModelScope env**

```bash
export MINERU_DEVICE_MODE=mps
export MINERU_MODEL_SOURCE=modelscope  # or 'huggingface' if MS fails
```

(These env var names are best-guess; verify in Step 3/4's output. If different, record the correct names in spike notes.)

- [ ] **Step 6: Run mineru on one known-VLM PDF**

Use a PDF that the previous iteration's Stage-B routed to VLM. The 10 are listed in `out/e2e_full/dataset.parquet` with `extract_backend = 'vlm'`. Pick one:

```bash
.venv/bin/python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet('out/e2e_full/dataset.parquet')
vlm = df[df['extract_backend'] == 'vlm']['pdf_path'].iloc[0]
# Find the actual source path (the parquet may have /tmp/... from the previous merge)
name = Path(vlm).name
for cand in Path('packages/pdfsys-bench').rglob(name):
    print(cand)
    break
"
```

Save that path. Then write the throwaway spike script `/tmp/mineru_spike.py` using whatever API Steps 3-4 surfaced. Example shape (adapt to actual mineru API):

```python
# /tmp/mineru_spike.py — throwaway. Adapt to discovered API.
import os
os.environ.setdefault("MINERU_DEVICE_MODE", "mps")
os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")

# Replace these imports with what Step 3/4 discovered:
from mineru.cli.common import do_parse  # placeholder — verify!

PDF = "PATH_FROM_PREVIOUS_STEP"
with open(PDF, "rb") as f:
    pdf_bytes = f.read()

# Replace with discovered call shape:
result = do_parse(pdf_bytes)  # placeholder — verify signature!
print("result keys:", list(result.keys()) if isinstance(result, dict) else type(result))
print("result repr (truncated):", repr(result)[:500])
```

Run it. **First-time model download will be ~3-4 GB** — be patient.

```bash
.venv/bin/python /tmp/mineru_spike.py 2>&1 | tee /tmp/mineru_spike.log
```

- [ ] **Step 7: Map mineru's content types to `pdfsys_core.RegionType`**

Whatever data structure Step 6 returned, identify the field names for:
- text content
- table content (HTML / LaTeX / markdown form?)
- formula content (LaTeX?)
- image references
- page index
- bbox

These map to our existing `_MINERU_TYPE_MAP` in `extract.py:48-63`. The current map is for magic-pdf 1.x's content-list format. mineru 3.x's may differ — record the diff.

- [ ] **Step 8: Write `out/mineru_spike_notes.md`**

This file is consumed by Task 7. It must answer:

```markdown
# mineru 3.x spike notes — 2026-05-14

## Installed version
mineru==X.Y.Z

## Module layout
(output of Step 2)

## Public entry point
The verified import is: `from mineru.X.Y import Z`
Call signature: `Z(pdf_bytes: bytes, ...) -> ...`

## Env vars
- `MINERU_DEVICE_MODE`: accepted values: cpu / mps / cuda
- `MINERU_MODEL_SOURCE`: accepted values: ...
- (other relevant ones)

## Output shape
`result = ...` — show actual return type and key fields.

## Content-type → RegionType map
| mineru type | RegionType | notes |
|---|---|---|
| ... | TEXT | |
| ... | TABLE | content lives in field `html` / `latex` |
| ... | FORMULA | content lives in field `latex` |

## MPS path
Worked / didn't work. If didn't, what was the error? Did CPU fallback work?

## Model cache size after first run
~3-4 GB at `~/.cache/...` (record actual path)

## Quirks / open questions
- ...
```

- [ ] **Step 9: Commit the spike notes**

```bash
git add -f out/mineru_spike_notes.md
git commit -m "docs(spike): mineru 3.x API surface notes"
```

Do NOT commit `/tmp/mineru_spike.py` — it's throwaway.

---

## Task 3: Uninstall magic-pdf + its forced deps

**Files:** none (env mutation only).

- [ ] **Step 1: Uninstall in one batch**

```bash
uv pip uninstall \
  magic-pdf \
  paddlepaddle \
  paddleocr \
  unimernet \
  detectron2 \
  ultralytics \
  openai \
  pycocotools \
  rapid-table \
  struct-eqtable
```

Some of these may not be installed (e.g. detectron2 was built from source by the fix subagent — uv may not know about it). Ignore "not installed" errors.

- [ ] **Step 2: Verify magic-pdf is gone**

```bash
ls .venv/lib/python*/site-packages/magic_pdf/ 2>&1 | head -3
```

Expected: `No such file or directory`.

If the directory still exists, force-remove:

```bash
rm -rf .venv/lib/python*/site-packages/magic_pdf
rm -rf .venv/lib/python*/site-packages/magic_pdf-*.dist-info
```

- [ ] **Step 3: Verify mineru still imports cleanly after removal**

```bash
.venv/bin/python -c "import mineru; print('mineru still works:', mineru.__file__)"
```

Expected: prints the mineru file path. If this fails, mineru shared a dep with magic-pdf that got removed — install it explicitly and continue.

- [ ] **Step 4: Verify the rest of the pipeline still imports**

```bash
.venv/bin/python -c "
import pdfsys_core, pdfsys_router, pdfsys_layout_analyser
import pdfsys_parser_mupdf, pdfsys_parser_pipeline
import pdfsys_bench, pdfsys_cli
print('all packages OK')
"
```

Expected: prints `all packages OK`. **NOTE:** `pdfsys_parser_vlm` will still try to import `magic_pdf` lazily from inside `_invoke_magic_pdf_v1` etc — that's fine for now; Task 7 deletes that code.

- [ ] **Step 5: No commit** — env state only.

---

## Task 4: Update `pdfsys-parser-vlm/pyproject.toml`

**Files:**
- Modify: `packages/pdfsys-parser-vlm/pyproject.toml`

- [ ] **Step 1: Replace the `magic-pdf` line with `mineru`**

Open `packages/pdfsys-parser-vlm/pyproject.toml`. The current `dependencies` list contains `"magic-pdf>=1.0",`. Replace that single line with:

```toml
    "mineru>=3.1,<4.0",
```

The full dependencies block should now read (other lines unchanged):

```toml
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru>=3.1,<4.0",
    "numpy>=1.24",
    "Pillow>=10.0",
]
```

- [ ] **Step 2: Re-resolve the workspace**

```bash
uv pip install --index https://pypi.org/simple -e packages/pdfsys-parser-vlm
```

Expected: completes; no new package installs (mineru already installed in Task 2). Confirms the new pyproject is internally consistent.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-parser-vlm/pyproject.toml
git commit -m "feat(parser-vlm): swap magic-pdf for mineru>=3.1"
```

---

## Task 5: Rename `_mineru_config.py` → `_vlm_config.py` + adapt to mineru

**Files:**
- Rename + rewrite: `packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py` → `_vlm_config.py`
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py` (1 import + 1 call-site)

- [ ] **Step 1: Create the new `_vlm_config.py` with mineru-aware logic**

Path: `packages/pdfsys-cli/src/pdfsys_cli/_vlm_config.py`

```python
"""VLM engine environment setup.

mineru 3.x reads device-mode and model-source from environment
variables (verified in the migration spike — see
out/mineru_spike_notes.md). This module sets those env vars before the
VLM parser is first imported, and relocates the legacy
~/magic-pdf.json out of the way so it can't interfere.

Idempotency:
- env vars: always set to the requested value (env vars are session-local).
- ~/magic-pdf.json: moved to ~/magic-pdf.json.bak on first call, only
  if both (a) the source exists and (b) the .bak doesn't already exist.
"""

from __future__ import annotations

import os
from pathlib import Path

# Verified in spike: mineru 3.x honours these env vars.
# If the spike notes contradict, update these constants.
ENV_DEVICE_MODE = "MINERU_DEVICE_MODE"
ENV_MODEL_SOURCE = "MINERU_MODEL_SOURCE"

LEGACY_CONFIG_PATH = Path.home() / "magic-pdf.json"
LEGACY_BACKUP_PATH = Path.home() / "magic-pdf.json.bak"


def ensure_mineru_env(device_mode: str, model_source: str = "modelscope") -> None:
    """Configure mineru's environment.

    Parameters
    ----------
    device_mode:
        One of ``"cpu"``, ``"mps"``, ``"cuda"``. Forwarded to mineru as
        ``MINERU_DEVICE_MODE``. Other values are passed through verbatim.
    model_source:
        ``"modelscope"`` (default, China-friendly) or ``"huggingface"``.
        Forwarded as ``MINERU_MODEL_SOURCE``.
    """
    os.environ[ENV_DEVICE_MODE] = device_mode
    os.environ[ENV_MODEL_SOURCE] = model_source

    # Move legacy magic-pdf config out of the way (once).
    if LEGACY_CONFIG_PATH.exists() and not LEGACY_BACKUP_PATH.exists():
        LEGACY_CONFIG_PATH.rename(LEGACY_BACKUP_PATH)
```

- [ ] **Step 2: Delete the old `_mineru_config.py`**

```bash
git rm packages/pdfsys-cli/src/pdfsys_cli/_mineru_config.py
```

- [ ] **Step 3: Update `runner.py` import + call**

In `packages/pdfsys-cli/src/pdfsys_cli/runner.py`, find the block (currently around lines 174–177):

```python
    if cfg.has_stage("extract") and cfg.vlm.enabled:
        from ._mineru_config import ensure_config  # noqa: PLC0415

        ensure_config(cfg.vlm.device_mode)
```

Replace with:

```python
    if cfg.has_stage("extract") and cfg.vlm.enabled:
        from ._vlm_config import ensure_mineru_env  # noqa: PLC0415

        ensure_mineru_env(cfg.vlm.device_mode)
```

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -c "
from pdfsys_cli._vlm_config import ensure_mineru_env, LEGACY_CONFIG_PATH, LEGACY_BACKUP_PATH
import os
ensure_mineru_env('mps')
assert os.environ.get('MINERU_DEVICE_MODE') == 'mps'
assert os.environ.get('MINERU_MODEL_SOURCE') == 'modelscope'
# Legacy file should have been moved
if LEGACY_BACKUP_PATH.exists():
    print('legacy file moved to:', LEGACY_BACKUP_PATH)
else:
    print('no legacy file existed (already migrated)')
print('OK')
"
```

Expected: prints `OK` plus the legacy-file status line. Also confirm the old module is gone:

```bash
.venv/bin/python -c "
try:
    from pdfsys_cli._mineru_config import ensure_config
    raise SystemExit('FAIL: old module still imports')
except ImportError:
    print('OK: old module gone')
"
```

- [ ] **Step 5: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/_vlm_config.py \
        packages/pdfsys-cli/src/pdfsys_cli/runner.py
git commit -m "refactor(cli): rename _mineru_config → _vlm_config; switch to mineru env vars"
```

---

## Task 6: Move legacy `~/magic-pdf.json` aside, restore if it ever needs to come back

This is purely operational. The Task 5 module does it automatically on first call, but explicitly performing it now keeps the spike-output verifiable.

- [ ] **Step 1: Move it**

```bash
.venv/bin/python -c "
from pdfsys_cli._vlm_config import ensure_mineru_env
ensure_mineru_env('mps')
"
ls -la ~/magic-pdf.json* 2>&1 || true
```

Expected output (or similar):
```
ls: /Users/.../magic-pdf.json: No such file or directory
-rw-------  1 user  group  227 ... /Users/.../magic-pdf.json.bak
```

- [ ] **Step 2: No commit** — operational only.

---

## Task 7: Rewrite `pdfsys-parser-vlm/extract.py` against mineru 3.x

**Files:**
- Rewrite: `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py`

**Required input:** `out/mineru_spike_notes.md` (from Task 2). The implementer subagent for this task MUST be given the spike notes contents in their prompt — they cannot infer mineru's API from training data.

- [ ] **Step 1: Read the spike notes**

```bash
cat out/mineru_spike_notes.md
```

Pin in mind:
- the verified import path
- the call signature for "process a PDF"
- the content-type → RegionType map
- the env vars required
- whether the call returns markdown directly or requires assembly from segments

- [ ] **Step 2: Read the current `extract.py` to understand the contract**

```bash
.venv/bin/python -c "
import inspect
from pdfsys_parser_vlm import extract
print(inspect.getsource(extract.VlmParser))
" 2>&1 | head -100
```

Note the three public methods that MUST be preserved:
- `extract(pdf_path, sha256=None) -> ExtractedDoc`
- `extract_bytes(pdf_bytes, sha256=None) -> ExtractedDoc`
- `extract_complex_pages(pdf_path, layout, sha256=None) -> ExtractedDoc`

Plus the module-level convenience functions `extract_doc` and `extract_doc_from_layout`.

- [ ] **Step 3: Rewrite `extract.py`**

Path: `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py`

The new shape — adapt the `_run_mineru` internals using the spike-verified API:

```python
"""VLM extraction backend using mineru 3.x.

Replaces magic-pdf 1.x (removed 2026-05-14). The :class:`VlmParser`
public contract is unchanged — only the model package underneath
changes. The two-path fallback (mineru_v2 + magic_pdf_v1) that lived
here previously is gone; there is one engine.

Heavy dependencies (``mineru``, ``torch``) are imported lazily inside
the methods that need them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    BBox,
    ExtractedDoc,
    LayoutDocument,
    RegionType,
    Segment,
    VlmConfig,
    merge_segments_to_markdown,
)

# Mapping from mineru's content-type strings to our RegionType enum.
# Verified against the spike output (out/mineru_spike_notes.md).
# Update the right-hand-side keys if the spike shows different type names.
_MINERU_TYPE_MAP: dict[str, RegionType] = {
    "text": RegionType.TEXT,
    "title": RegionType.TEXT,
    "equation": RegionType.FORMULA,
    "inline_equation": RegionType.FORMULA,
    "interline_equation": RegionType.FORMULA,
    "table": RegionType.TABLE,
    "image": RegionType.IMAGE,
    "figure": RegionType.IMAGE,
    "figure_caption": RegionType.TEXT,
    "table_caption": RegionType.TEXT,
    "table_footnote": RegionType.TEXT,
    "header": RegionType.TEXT,
    "footer": RegionType.TEXT,
    "reference": RegionType.TEXT,
}


class VlmParser:
    """mineru-based VLM extraction parser."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()

    # ------------------------------------------------------------------ public

    def extract(
        self, pdf_path: str | Path, sha256: str | None = None
    ) -> ExtractedDoc:
        """Process an entire PDF through mineru and return ExtractedDoc."""
        path = Path(pdf_path)
        sha = sha256 or _sha256_of_file(path)
        with path.open("rb") as f:
            pdf_bytes = f.read()
        return self._run_mineru(pdf_bytes, sha, complex_pages=None)

    def extract_bytes(
        self, pdf_bytes: bytes, sha256: str | None = None
    ) -> ExtractedDoc:
        """Same as :meth:`extract`, but from an in-memory buffer."""
        sha = sha256 or hashlib.sha256(pdf_bytes).hexdigest()
        return self._run_mineru(pdf_bytes, sha, complex_pages=None)

    def extract_complex_pages(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Process only pages flagged as having complex content.

        Reads the :class:`LayoutDocument` to identify pages with TABLE
        or FORMULA regions, runs mineru on the full PDF, then filters
        segments to those pages. Simple pages are skipped (they should
        be handled by the pipeline parser).
        """
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)

        complex_pages = {
            lp.index
            for lp in layout.pages
            if any(r.type in (RegionType.TABLE, RegionType.FORMULA) for r in lp.regions)
        }

        if not complex_pages:
            return ExtractedDoc(
                sha256=sha,
                backend=Backend.VLM,
                segments=(),
                markdown="",
                stats={"complex_pages": 0, "reason": "no_complex_content"},
            )

        with path.open("rb") as f:
            pdf_bytes = f.read()

        return self._run_mineru(pdf_bytes, sha, complex_pages=complex_pages)

    # --------------------------------------------------------------- internal

    def _run_mineru(
        self, pdf_bytes: bytes, sha256: str, complex_pages: set[int] | None
    ) -> ExtractedDoc:
        """Run mineru's VLM pipeline on raw PDF bytes.

        If ``complex_pages`` is given, the returned segments are
        filtered to those page indices.
        """
        content_list, md_content, stats = self._invoke_mineru(pdf_bytes)

        segments = self._content_list_to_segments(content_list)
        if complex_pages is not None:
            segments = [s for s in segments if s.page_index in complex_pages]
            stats["complex_pages"] = len(complex_pages)
            stats["complex_page_indices"] = sorted(complex_pages)

        seg_tuple = tuple(segments)

        markdown = merge_segments_to_markdown(seg_tuple)
        if not markdown.strip() and md_content:
            markdown = md_content.strip() + "\n"

        stats["char_count"] = len(markdown)
        stats["segment_count"] = len(seg_tuple)
        stats["vlm_engine"] = "mineru-3.x"
        stats["vlm_model"] = self.config.model

        return ExtractedDoc(
            sha256=sha256,
            backend=Backend.VLM,
            segments=seg_tuple,
            markdown=markdown,
            stats=stats,
        )

    def _invoke_mineru(
        self, pdf_bytes: bytes
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """Call mineru's VLM entry point and return (content_list, markdown, stats).

        The exact import and call signature MUST be taken from the
        spike output (out/mineru_spike_notes.md). The placeholders
        below are the most likely candidates; verify and adjust.
        """
        # *** ADAPT THIS BLOCK FROM out/mineru_spike_notes.md ***
        #
        # Example based on PyPI's `mineru` 3.x docs at time of writing
        # (may need adjustment per spike):
        #
        from mineru.cli.common import do_parse  # noqa: PLC0415 — adapt per spike

        with __import__("tempfile").TemporaryDirectory(prefix="pdfsys_vlm_") as tmpdir:
            # do_parse signature is taken from mineru.cli.common; adapt freely.
            result = do_parse(
                output_dir=tmpdir,
                pdf_file_names=["doc"],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                backend="vlm-transformers",  # vlm-transformers | pipeline — adapt per spike
                # device_mode and model_source come from env vars set by _vlm_config
            )

            content_list, md_content = self._extract_from_result(result, tmpdir)

        stats: dict[str, Any] = {
            "api": "mineru_v3",
        }
        return content_list, md_content, stats

    def _extract_from_result(
        self, result: Any, tmpdir: str
    ) -> tuple[list[dict[str, Any]], str]:
        """Pull (content_list, markdown) out of mineru's result.

        mineru 3.x typically writes its artifacts to ``tmpdir`` as
        files (`*_content_list.json` and `*.md`). Read them back.
        Adapt per spike if it returns in-memory objects directly.
        """
        import json
        from pathlib import Path as _Path

        td = _Path(tmpdir)
        content_list: list[dict[str, Any]] = []
        md_content = ""

        # Try the common file names mineru emits.
        for candidate in td.rglob("*_content_list.json"):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    content_list = [item for item in data if isinstance(item, dict)]
                break
            except (json.JSONDecodeError, OSError):
                continue

        for candidate in td.rglob("*.md"):
            try:
                md_content = candidate.read_text(encoding="utf-8")
                break
            except OSError:
                continue

        return content_list, md_content

    def _content_list_to_segments(
        self, content_list: list[dict[str, Any]]
    ) -> list[Segment]:
        """Map mineru's content_list items to our Segment format."""
        segments: list[Segment] = []

        for item in content_list:
            item_type = item.get("type", "text")
            region_type = _MINERU_TYPE_MAP.get(item_type, RegionType.TEXT)

            content = ""
            if region_type == RegionType.IMAGE:
                content = item.get("img_path", "") or "[image]"
            elif region_type == RegionType.TABLE:
                content = (
                    item.get("html", "")
                    or item.get("latex", "")
                    or item.get("text", "")
                    or item.get("md", "")
                )
            elif region_type == RegionType.FORMULA:
                content = (
                    item.get("latex", "")
                    or item.get("text", "")
                    or item.get("md", "")
                )
            else:
                content = item.get("text", "") or item.get("md", "")

            if not content:
                continue

            bbox = None
            raw_bbox = item.get("bbox")
            page_w = item.get("page_width", 0)
            page_h = item.get("page_height", 0)
            if raw_bbox and len(raw_bbox) == 4 and page_w > 0 and page_h > 0:
                nx0 = max(0.0, min(1.0, raw_bbox[0] / page_w))
                ny0 = max(0.0, min(1.0, raw_bbox[1] / page_h))
                nx1 = max(0.0, min(1.0, raw_bbox[2] / page_w))
                ny1 = max(0.0, min(1.0, raw_bbox[3] / page_h))
                if nx1 > nx0 and ny1 > ny0:
                    try:
                        bbox = BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
                    except ValueError:
                        bbox = None

            page_index = item.get("page_idx", item.get("page_index", 0))

            segments.append(
                Segment(
                    index=len(segments),
                    backend=Backend.VLM,
                    page_index=int(page_index),
                    type=region_type,
                    content=content.strip(),
                    bbox=bbox,
                    source_region_id=None,
                )
            )

        return segments


# ---------------------------------------------------------------- convenience

def extract_doc(
    pdf_path: str | Path,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: create a VLM parser, extract, return."""
    parser = VlmParser(config=config)
    return parser.extract(pdf_path, sha256=sha256)


def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: extract only complex pages identified by layout."""
    parser = VlmParser(config=config)
    return parser.extract_complex_pages(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Verify module imports cleanly**

```bash
.venv/bin/python -c "
from pdfsys_parser_vlm import VlmParser, extract_doc, extract_doc_from_layout
import pdfsys_parser_vlm.extract as ex
assert hasattr(ex.VlmParser, 'extract')
assert hasattr(ex.VlmParser, 'extract_bytes')
assert hasattr(ex.VlmParser, 'extract_complex_pages')
# Confirm no magic_pdf references remain in the rewritten module:
import inspect
src = inspect.getsource(ex)
assert 'magic_pdf' not in src, 'magic_pdf reference still in module'
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 5: One-PDF smoke against the rewritten extract**

```bash
.venv/bin/python <<'PY'
from pdfsys_cli._vlm_config import ensure_mineru_env
ensure_mineru_env('mps')

from pathlib import Path
from pdfsys_parser_vlm import VlmParser
from pdfsys_core import VlmConfig

# Pick one of the previously-failed VLM PDFs
import pandas as pd
df = pd.read_parquet('out/e2e_full/dataset.parquet')
vlm = df[df['extract_backend'] == 'vlm']['pdf_path'].iloc[0]
name = Path(vlm).name
src = next(Path('packages/pdfsys-bench').rglob(name))
print(f'testing on: {src}')

parser = VlmParser(config=VlmConfig(model='mineru-2.5'))
doc = parser.extract(src)
print(f'segments: {len(doc.segments)}')
print(f'markdown chars: {len(doc.markdown)}')
print(f'backend: {doc.backend}')
print(f'stats keys: {list(doc.stats.keys())}')
print('--- first 500 chars of markdown ---')
print(doc.markdown[:500])
PY
```

Expected:
- `segments > 0`
- `markdown chars >= 200`
- `backend = Backend.VLM` (or `vlm`)
- some recognizable text/formula/table content in the first 500 chars

If `markdown chars < 200` or content is garbled, the spike's API mapping is wrong — go back to `out/mineru_spike_notes.md`, fix the call signature in `_invoke_mineru`, and retry.

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py
git commit -m "feat(parser-vlm): rewrite extract.py against mineru 3.x API"
```

---

## Task 8: Re-stage the 10 VLM PDFs (reproduce previous fixture)

**Files:** none committed; staging in `/tmp`.

- [ ] **Step 1: Re-create the staging dir**

```bash
.venv/bin/python <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_parquet('out/e2e_full/dataset.parquet')
failed_or_vlm = df[df['extract_backend'] == 'vlm']['pdf_path'].tolist()
print('VLM rows from previous run:', len(failed_or_vlm))

dst = Path('/tmp/vlm_retry_pdfs')
dst.mkdir(exist_ok=True)
# Clear any old symlinks
for old in dst.glob('*.pdf'):
    old.unlink()

repo_bench = Path('packages/pdfsys-bench')
linked = 0
for pdf_path in failed_or_vlm:
    name = Path(pdf_path).name
    candidates = list(repo_bench.rglob(name))
    if not candidates:
        print(f'WARN: could not find source for {name}')
        continue
    target = dst / name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(candidates[0].resolve())
    linked += 1

print(f'linked: {linked} PDFs into {dst}')
PY
ls -la /tmp/vlm_retry_pdfs/
```

Expected: `linked: 10 PDFs into /tmp/vlm_retry_pdfs`.

- [ ] **Step 2: No commit** — fixture staging only.

---

## Task 9: Run the 10-PDF retry with the new engine

**Files:** none committed.

- [ ] **Step 1: Clear stale output and kick off the run**

```bash
rm -rf out/vlm_mineru3_retry
.venv/bin/pdfsys run -c pdfsys.smoke.yaml \
  --pdf-dir /tmp/vlm_retry_pdfs \
  --out-dir out/vlm_mineru3_retry \
  --limit 100 \
  2>&1 | tee out/_vlm_mineru3_retry.log
```

(Note: `--limit 100` overrides the YAML's `limit: 5` — there are only 10 PDFs in the fixture, so all will be processed.)

Wall time: **5-30 minutes**. First call downloads mineru's models (~3-4 GB) if not cached.

- [ ] **Step 2: Inspect output**

```bash
ls -la out/vlm_mineru3_retry/
wc -l out/vlm_mineru3_retry/results.jsonl
```

Expected: `dataset.parquet`, `results.jsonl` (10 lines), `markdown/` dir, `results.summary.json`.

- [ ] **Step 3: No commit yet** — verification happens in Task 10.

---

## Task 10: Verify all 5 acceptance gates from spec §7

**Files:** none committed (results-only verification).

- [ ] **Step 1: Run all 5 acceptance queries**

```bash
.venv/bin/python <<'PY'
import pandas as pd
df = pd.read_parquet('out/vlm_mineru3_retry/dataset.parquet')
print('=== acceptance gates ===\n')

# Gate 1: row count
g1 = len(df) == 10
print(f'[Gate 1] row count = {len(df)} (expect 10): {"PASS" if g1 else "FAIL"}')

# Gate 2: zero errors
errs = df['error_class'].notnull().sum()
g2 = errs == 0
print(f'[Gate 2] error_class IS NULL for all = {10 - errs}/10: {"PASS" if g2 else "FAIL"}')
if not g2:
    print('  error breakdown:')
    print(df[df['error_class'].notnull()][['pdf_path','error_class','error_message']].to_string(index=False))

# Gate 3: all VLM
vlm_count = (df['extract_backend'] == 'vlm').sum()
g3 = vlm_count == 10
print(f'[Gate 3] extract_backend == "vlm" = {vlm_count}/10: {"PASS" if g3 else "FAIL"}')

# Gate 4: markdown_chars >= 200 (key anti-regression gate)
mdc_ge200 = (df['markdown_chars'] >= 200).sum()
g4 = mdc_ge200 == 10
print(f'[Gate 4] markdown_chars >= 200 = {mdc_ge200}/10: {"PASS" if g4 else "FAIL"}')
if not g4:
    print('  short rows:')
    print(df[df['markdown_chars'] < 200][['pdf_path','markdown_chars']].to_string(index=False))

# Gate 5: at least one row with formula or table markup
has_formula = df['markdown'].str.contains(r'\\\\|\$', regex=True, na=False)
has_table = df['markdown'].str.contains(r'<table', regex=True, na=False)
g5_count = (has_formula | has_table).sum()
g5 = g5_count >= 1
print(f'[Gate 5] >= 1 row with LaTeX or HTML table = {g5_count}/10: {"PASS" if g5 else "FAIL"}')

print()
print('=== summary ===')
all_pass = all([g1, g2, g3, g4, g5])
print('OVERALL:', 'PASS' if all_pass else 'FAIL')
if not all_pass:
    print('\nFAILED — see specific gate failures above. Do not proceed to full re-run.')
PY
```

Expected: all 5 gates print `PASS`, overall `PASS`.

- [ ] **Step 2: Snapshot the markdown_chars distribution**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('out/vlm_mineru3_retry/dataset.parquet')
print(df['markdown_chars'].describe())
print('quality_score:', df['quality_score'].describe())
print('kept:', int(df['kept'].sum()))
"
```

Save this output for Task 11.

- [ ] **Step 3: If any gate fails, STOP and report.**

Do not proceed to Task 11. The implementation needs another spike pass; record the failures and dispatch back to Task 7.

- [ ] **Step 4: If all gates pass, no commit yet** — Task 11 writes the post-run note + commits.

---

## Task 11: Post-run note + final commit

**Files:**
- Modify: `docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md`

- [ ] **Step 1: Append §12 post-run note**

Edit the spec file. Locate the `## 12 · Post-run note (to be filled in)` line and replace it with:

```markdown
## 12 · Post-run note · 2026-05-14

### Engine
- mineru version installed: `mineru==X.Y.Z` (from `.venv/bin/python -c "import mineru; print(mineru.__version__)"`)
- Extras used: bare / `[core]` / other — (record what Task 2 spike decided)
- Active device: mps / cpu / cuda — (record what actually fired)
- Model cache size at completion: `~/...` ≈ X GB

### 10-PDF retry stats
| Gate | Value | Result |
|---|---|---|
| Row count | 10 | PASS |
| Errors | 0 | PASS |
| extract_backend == vlm | 10 / 10 | PASS |
| markdown_chars >= 200 | 10 / 10 | PASS |
| Rows with formula or table markup | N / 10 | PASS |
| kept (quality_score >= 2.0) | N / 10 |  |
| markdown_chars distribution (min / median / max) | A / B / C |  |
| Wall time | T seconds |  |

### Failure modes observed (record any, even if non-blocking)
- ...

### Cleanup confirmation
- `magic-pdf` no longer in `uv pip list`: confirmed / not confirmed
- `.venv/lib/python*/site-packages/magic_pdf/` removed: confirmed / not confirmed
- `~/magic-pdf.json` moved to `.bak`: confirmed

### Open follow-ups
- (Optional) Re-run full 150-PDF pipeline (`pdfsys.full.yaml`) to verify no regression on the 140 non-VLM rows.
- (If VLM dep changed, e.g. needed `[core]`) Document in `pdfsys-parser-vlm/pyproject.toml` and update extras.
```

Replace each `X`/`Y`/`Z`/`A`/`B`/`C`/`N`/`T` with the actual values observed in Tasks 9 and 10.

- [ ] **Step 2: Verify the spec file is well-formed**

```bash
head -50 docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md
tail -50 docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md
```

Confirm the §12 placeholder is replaced.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md
git commit -m "docs(spec): append mineru 3.x migration post-run note"
```

---

## Self-review notes

**Spec coverage:**
- Spec §1 goal (5 acceptance criteria) → Task 10 gate 1-5
- Spec §3 dep churn → Tasks 3 (uninstall) + 4 (pyproject) + 2 (install mineru)
- Spec §4 spike-first → Task 2
- Spec §5 extract.py rewrite → Task 7
- Spec §6 cleanup → Tasks 3 + 5 + 6
- Spec §7 acceptance queries → Task 10
- Spec §8 optional full re-run → noted in Task 11 §3 post-run open follow-ups (NOT a required task; user can trigger separately)
- Spec §10 files touched → matches §"File map" above
- Spec §11 definition of done → Task 11 §1 captures all required confirmations

**Placeholders:** The text `*** ADAPT THIS BLOCK FROM out/mineru_spike_notes.md ***` in Task 7 step 3 is an explicit instruction to the implementer to substitute the spike-verified API; it is not a plan placeholder — it's a contract between Task 2's output and Task 7's input. Highlighted so the implementer subagent must address it, not skip it. The `X`/`Y`/`Z`/`N`/`T` placeholders in Task 11's post-run note template are intentional fill-ins to be filled with real numbers, not plan-time TBDs.

**Type consistency:** `ensure_mineru_env(device_mode, model_source)` signature in Task 5 matches the call site update in the same task. `VlmParser` public methods in Task 7 match the contract documented at the top of the task and consumed by `runner.py` (which is not modified for the VLM rewrite — only for the import rename in Task 5).

**One soft assumption:** Task 7's example `mineru.cli.common.do_parse` call is a best-guess based on PyPI 3.1.12 module structure. If the spike (Task 2) discovers a different entry point, Task 7's implementer must substitute it — and the example code in Step 3's code block treats this as a known adaptation point, not a defect.
