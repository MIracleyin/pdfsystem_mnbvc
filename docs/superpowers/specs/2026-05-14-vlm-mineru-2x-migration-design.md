# VLM Engine Migration: magic-pdf 1.x → mineru 3.x — Design

**Date:** 2026-05-14
**Builds on:** `docs/superpowers/specs/2026-05-14-e2e-parquet-mac-mps-design.md` (post-run note §13)
**Scope:** Swap the VLM engine in `pdfsys-parser-vlm` from the legacy `magic-pdf 1.0.1` package to the new `mineru >= 3.1` package, eliminating the paddle/detectron2/site-packages-patch mess introduced during the previous run, and producing usable Parquet rows on the 10 VLM-routed PDFs (formula+table re-enabled, MPS retained).
**Out of scope:** layout analyser changes, Stage-A/B router logic, ParquetSink, CLI runner structure, smoke/full YAML configs, VLM batching/OOM retry.

---

## 1 · Goal

Re-run the same 10 VLM-routed PDFs from the previous full run (`out/e2e_full/dataset.parquet` rows where `stage_b_backend = 'vlm'`) and produce:

1. row count = 10 (no extraction-time drop)
2. `error_class IS NULL` for all 10
3. `extract_backend = 'vlm'` for all 10
4. `markdown_chars >= 200` for all 10 (acceptance gate added to prevent the previous "22-char degraded" regression)
5. At least one row whose `markdown` column contains either an inline LaTeX formula (`\`/`$` patterns) or an HTML table (`<table>`) — proves formula/table sub-models are firing

Pipeline contract unchanged: `VlmParser.extract_complex_pages(path, layout)` still returns an `ExtractedDoc` with the same schema. Runner, ParquetSink, Stage-B, layout analyser all untouched.

## 2 · Non-goals

- Keep magic-pdf 1.x as a parallel backend.
- Layout improvements (PP-DocLayoutV3 stays on its current path).
- VLM batching, dynamic batch size, or OOM retry.
- Switching layout from `juliozhao/DocLayout-YOLO-DocStructBench` to `PP-DocLayoutV3`. (PP-DocLayoutV3 is already a configurable backend; user can swap at YAML level if they want.)

## 3 · What changes vs. previous spec

| Aspect | Previous (magic-pdf 1.x) | This iteration (mineru 3.x) |
|---|---|---|
| PyPI package | `magic-pdf>=1.0` | `mineru>=3.1,<4.0` |
| Python import root | `magic_pdf.*` | `mineru.*` |
| Hidden runtime deps | paddlepaddle, detectron2 (built from source), unimernet, paddleocr, ultralytics, openai, pycocotools, rapid-table, struct-eqtable | mineru-vl-utils, modelscope; `[core]` extra adds accelerate / albumentations as needed |
| Config file | `~/magic-pdf.json` | env vars (`MINERU_DEVICE_MODE`, `MINERU_MODEL_SOURCE`) and/or mineru's own config path |
| MPS support | broken (paddle has no MPS path; v1 API gone) | first-class — mineru has documented MPS device-mode |
| Site-packages patches | 3 files patched in-place during the previous fix-up | none (clean install) |
| formula+table | disabled in `~/magic-pdf.json` to get past missing deps | enabled |

## 4 · Implementation strategy: spike-first, no premature commitment to `[core]`

Two-stage install to avoid the previous "install 10 packages then patch site-packages" trap:

### Stage A: bare `mineru` smoke

1. `uv pip uninstall magic-pdf paddlepaddle paddleocr unimernet detectron2 ultralytics openai pycocotools rapid-table struct-eqtable` (keep `transformers`, `timm` — used elsewhere)
2. `uv add --package pdfsys-parser-vlm "mineru>=3.1,<4.0"` (no extras)
3. Spike script (~30 LOC, throwaway): import the smallest VLM entry point of mineru, run on **one** known-VLM PDF (`/tmp/vlm_retry_pdfs/4_pg433.pdf` from the previous retry), confirm it returns structured content.
4. Outcome:
   - **Bare install works** → proceed to §5 (rewrite `extract.py`).
   - **Bare install missing deps** → upgrade to `mineru[core]` (or specific extras), re-spike, document what bare-vs-core covers.
   - **mineru fundamentally broken on Mac MPS** → fall back to `device_mode=cpu`, record in §13 of this spec, decide whether to abort.

### Stage B: rewrite `extract.py` against verified API

Once the spike proves mineru works on one PDF, port the public surface:

```python
class VlmParser:
    def __init__(self, config: VlmConfig | None = None) -> None: ...

    def extract(self, pdf_path, sha256: str | None = None) -> ExtractedDoc: ...
    def extract_bytes(self, pdf_bytes, sha256: str | None = None) -> ExtractedDoc: ...
    def extract_complex_pages(
        self, pdf_path, layout: LayoutDocument, sha256: str | None = None,
    ) -> ExtractedDoc: ...
```

The internal `_invoke_mineru_v2` / `_invoke_magic_pdf_v1` two-path fallback is **deleted** — there's only one engine now, and ImportError from mineru means "broken install" and should not be swallowed.

The `_MINERU_TYPE_MAP` mapping (mineru's content types → our `RegionType`) is preserved; mineru 3.x's content types may differ from 1.x's — verify during the spike and update the map.

### Stage C: integration

- `_mineru_config.py` → rename to `_vlm_config.py` (less version-bound name); export `ensure_mineru_env(device_mode: str)`:
  - sets `MINERU_DEVICE_MODE` env var (or whatever mineru 3.x reads)
  - moves `~/magic-pdf.json` to `~/magic-pdf.json.bak` (idempotent — only if exists and not already moved)
  - returns the active device mode
- `runner.py:174–177` — one line change to the import: `from ._mineru_config import ensure_config` → `from ._vlm_config import ensure_mineru_env`; rename the call site.
- `pdfsys.example.yaml` / `pdfsys.smoke.yaml` / `pdfsys.full.yaml` — no schema change. The `vlm.model: mineru-2.5` value stays as a descriptive label (mineru-2.5 is the model version inside mineru 3.x package).

## 5 · `extract.py` rewrite

~250-350 LOC depending on mineru 3.x API verbosity. Structure:

```python
"""VLM extraction backend using mineru 3.x.

Replaces magic-pdf 1.x (removed 2026-05-14). API contract on
:class:`VlmParser` is preserved — only the internal call into the
model package changes.
"""

from __future__ import annotations
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend, BBox, ExtractedDoc, LayoutDocument, RegionType, Segment,
    VlmConfig, merge_segments_to_markdown,
)

# Verified against mineru 3.x during spike; update if API moves.
_MINERU_TYPE_MAP: dict[str, RegionType] = { ... }


class VlmParser:
    def __init__(self, config: VlmConfig | None = None) -> None: ...

    def extract(self, pdf_path, sha256=None) -> ExtractedDoc:
        path = Path(pdf_path)
        sha = sha256 or _sha256_of_file(path)
        with path.open("rb") as f:
            pdf_bytes = f.read()
        return self._run_mineru(pdf_bytes, sha, complex_pages=None)

    def extract_complex_pages(self, pdf_path, layout, sha256=None) -> ExtractedDoc:
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)
        complex_pages = {
            lp.index for lp in layout.pages
            if any(r.type in (RegionType.TABLE, RegionType.FORMULA) for r in lp.regions)
        }
        if not complex_pages:
            return ExtractedDoc(
                sha256=sha, backend=Backend.VLM, segments=(), markdown="",
                stats={"complex_pages": 0, "reason": "no_complex_content"},
            )
        with path.open("rb") as f:
            pdf_bytes = f.read()
        return self._run_mineru(pdf_bytes, sha, complex_pages=complex_pages)

    def _run_mineru(self, pdf_bytes, sha256, complex_pages):
        # Calls mineru's verified VLM entry point (TBD by spike).
        # Filters to complex_pages if provided.
        # Maps mineru content types via _MINERU_TYPE_MAP.
        # Returns ExtractedDoc with backend=VLM.
        ...
```

The two `_invoke_*` private methods from the old code are gone. There is one engine.

## 6 · Cleanup of previous-iteration leftovers

When applying this spec:

1. **Uninstall packages** (list in §3 row "Hidden runtime deps") in one `uv pip uninstall` invocation.
2. **Remove site-packages patches** — these vanish automatically when `magic-pdf` is uninstalled (the patched files were inside the magic-pdf package tree). Verify by `ls .venv/lib/python3.12/site-packages/magic_pdf/` returns "no such directory".
3. **`~/magic-pdf.json`** — the new `_vlm_config.py::ensure_mineru_env()` renames it to `.bak` on first call. Manual cleanup not required.
4. **`out/e2e_full/dataset.parquet.bak`** — left from previous retry merge; delete during the new full run setup.

## 7 · Acceptance (re-run the 10 VLM PDFs only)

Use the same `/tmp/vlm_retry_pdfs/` fixture from the previous iteration (10 symlinks to PDFs that Stage-B routes to VLM):

```bash
# Reproduce the staging dir from the previous retry
.venv/bin/python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet('out/e2e_full/dataset.parquet')
vlm = df[df['extract_backend'] == 'vlm']['pdf_path']
dst = Path('/tmp/vlm_retry_pdfs'); dst.mkdir(exist_ok=True)
for p in vlm:
    # original-source path lookup: the parquet may have /tmp paths from the merge
    name = Path(p).name
    for cand in Path('packages/pdfsys-bench').rglob(name):
        target = dst / name
        if target.exists() or target.is_symlink(): target.unlink()
        target.symlink_to(cand.resolve()); break
print('linked:', len(list(dst.glob('*.pdf'))))
"

# Run
rm -rf out/vlm_mineru3_retry
.venv/bin/pdfsys run -c pdfsys.smoke.yaml \
  --pdf-dir /tmp/vlm_retry_pdfs \
  --out-dir out/vlm_mineru3_retry \
  --limit 100
```

### Pass criteria

```sql
SELECT count(*) FROM 'out/vlm_mineru3_retry/dataset.parquet';
-- expected: 10

SELECT count(*) FROM ... WHERE error_class IS NULL;
-- expected: 10  (zero errors — magic-pdf brittleness gone)

SELECT count(*) FROM ... WHERE extract_backend = 'vlm';
-- expected: 10  (all routed to VLM and all succeeded)

SELECT count(*) FROM ... WHERE markdown_chars >= 200;
-- expected: 10  (new acceptance gate; prevents the 22-char regression)

SELECT count(*) FROM ... WHERE
  markdown LIKE '%\\\\%' OR markdown LIKE '%$%' OR markdown LIKE '%<table%';
-- expected: >= 1  (formula/table sub-models actually fired)
```

If gate 4 or 5 fails, the spike from §4 didn't actually verify VLM functionality and the implementation is degraded again — STOP and triage.

## 8 · Full re-run (optional, after retry passes)

If the 10-PDF retry passes all five gates, optionally re-run the whole 150 (`pdfsys.full.yaml`). Pass criteria become:

- 150 rows
- `errors == 0` (vlm engine no longer brittle)
- `kept_rows >= 35` (no regression vs. previous run)
- 104 mupdf / 36 pipeline / 10 vlm distribution preserved
- VLM rows now have `markdown_chars` distribution similar to other backends (no more 22-char outliers)

This full re-run is **not blocking** — the 10-PDF retry is the contract.

## 9 · Risks

| Risk | Mitigation |
|---|---|
| mineru 3.x API doesn't expose a "use my own layout, just process these pages" entry | Spike (§4 Stage A) exposes this in 30 LOC before committing. If only whole-doc API exists, run mineru on the whole PDF and filter by `page_index` in segments (existing pattern). |
| `[core]` extras pull more dep churn on Mac | Stay on bare `mineru` if possible. Upgrade to `[core]` only with a verified spike each time. |
| MPS path in mineru 3.x is undocumented or buggy | Fall back to `device_mode=cpu`. Acceptance allows CPU. Note in post-run §13 if needed. |
| ModelScope download blocked on user's network | Set `MINERU_MODEL_SOURCE=huggingface` env var (mineru honours this — verify in spike). |
| Old `~/magic-pdf.json` interferes with mineru config | `ensure_mineru_env` renames it to `.bak` on first call. |
| `pdfsys-parser-vlm/extract.py` rewrite breaks runner contract (return type, exceptions) | Spec §1 explicitly preserves `VlmParser.extract_complex_pages` signature and `ExtractedDoc` contract. No new exception types — keep `error_class = "extract_vlm"` semantics. |

## 10 · Files touched

```
packages/pdfsys-parser-vlm/
├── pyproject.toml          (modified: magic-pdf → mineru, extras decided by §4 Stage A spike)
└── src/pdfsys_parser_vlm/
    ├── extract.py          (REWRITTEN ~250-350 LOC)
    └── __init__.py         (unchanged — same public exports)

packages/pdfsys-cli/src/pdfsys_cli/
├── _mineru_config.py       (RENAMED → _vlm_config.py + adapted to mineru env)
└── runner.py               (1-line import path update + 1 call-site rename)

docs/superpowers/specs/
└── 2026-05-14-vlm-mineru-2x-migration-design.md    (this file)
```

Untouched: `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-bench`, `pdfsys-cli/config.py`, `pdfsys-cli/parquet_writer.py`, all YAML configs.

## 11 · Definition of done

- 10-PDF retry (`out/vlm_mineru3_retry/dataset.parquet`) passes all 5 acceptance queries in §7.
- `magic-pdf` and its forced deps no longer appear in `uv pip list`.
- No files patched under `.venv/lib/.../magic_pdf/` (because the directory itself is gone).
- Post-run note appended to this spec (§12) capturing:
  - mineru version installed
  - whether bare or `[core]` extras were needed
  - actual MPS path used (mps vs cpu fallback)
  - wall time + markdown_chars distribution
  - any environment quirks for the next implementer

## 12 · Post-run note (to be filled in)

(Section reserved — implementer fills after Task N completion.)
