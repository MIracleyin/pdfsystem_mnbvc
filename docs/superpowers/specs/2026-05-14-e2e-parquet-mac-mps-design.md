# End-to-end PDF → Parquet on Mac (MPS) — Design

**Date:** 2026-05-14
**Scope:** Wire the existing 4 stages (router → layout → extract → quality) into a single end-to-end run that emits a Parquet dataset, with VLM (MinerU 2.5) enabled and MPS acceleration where supported. Target: the bundled 150 PDFs (`omnidocbench_100` + `olmocr_bench_50`).
**Out of scope:** unit tests, sharding, Stage-3 post-processing, Stage-4 PII/MinHash dedup, router perf optimisation. Those follow in separate specs.

---

## 1 · Goal

Run a single command on a Mac (Apple Silicon) that:

1. classifies each PDF (Stage-A router, XGBoost)
2. analyses layout for needs-OCR docs (PP-DocLayoutV3 on MPS)
3. decides Stage-B backend (PIPELINE vs VLM via `decide()`)
4. extracts via MuPDF / RapidOCR / **MinerU 2.5 on MPS** as appropriate
5. scores quality (ModernBERT on MPS)
6. writes one `dataset.parquet` containing all 150 rows + a `kept` flag for `quality_score >= 2.0`

Pipeline must not abort on per-PDF failures; every failure mode is captured in `error_class` / `error_message` columns. After the run, a single DuckDB / pandas query returns the high-quality subset.

## 2 · Non-goals

- Multiple Parquet shards (single shard is fine for 150 rows).
- Tests, CI, parity harness (deliberately deferred — see ROADMAP P0).
- PII detection, MinHash dedup, reading-order post-processing.
- Performance optimisation beyond what MPS gives us out of the box.

## 3 · Existing pipeline (unchanged)

`packages/pdfsys-cli/src/pdfsys_cli/runner.py:144-201` already implements:

```
router → layout → extract → quality   → JSONL + per-PDF markdown files
```

with lazy component loading (`Components` class), Stage-B routing via `decide()`, MUPDF / PIPELINE / VLM branch dispatch, and per-stage try/except. **We will not rewrite this**; we will append a `parquet` stage and refine error capture.

### 3.1 Routing strategy (as implemented)

The router is two-stage. Stage-A looks at PyMuPDF features only — it never sees layout. Stage-B looks at layout regions only — it never sees XGBoost output.

```
Stage-A (classifier.py:191-201) — XGBoost over 124 features
   ocr_prob < 0.5                  → MUPDF
   ocr_prob ≥ 0.5 && vlm_off       → PIPELINE
   ocr_prob ≥ 0.5 && vlm_on        → DEFERRED          ← waits for Stage-B

Stage-B (decider.py:57-71) — reads LayoutDocument
   has_complex_content = False     → PIPELINE
   has_complex_content = True  && vlm_on  → VLM
   has_complex_content = True  && vlm_off → DEFERRED

where has_complex_content := any region.type ∈ {TABLE, FORMULA}
```

Two subtleties worth pinning down because the obvious mental model is wrong:

- **MUPDF is chosen by Stage-A on `ocr_prob` alone — it does NOT check for tables / formulas / images.** A clean-text PDF with insets / figures still goes MUPDF; images are simply dropped from the output. The 124 features include `garbled_text_ratio`, `is_form`, `creator_or_producer_is_known_scanner` (matches `epson`/`canon`/`kofax`/... in PDF metadata), per-page font coverage and image-area ratio. The XGBoost weights are a verbatim port of FinePDFs `ocr_predictor.py`.
- **`has_complex_content` only counts TABLE and FORMULA — IMAGE does NOT trigger VLM.** A scanned page with text + photos (no tables, no formulas) goes PIPELINE (RapidOCR), not VLM. VLM is reserved for documents whose structural understanding actually requires it.

Concrete scenarios:

| Document | Stage-A | Stage-B | Backend |
|---|---|---|---|
| LaTeX paper (embedded fonts, has figures) | `ocr_prob ≈ 0.03` | — | MUPDF |
| Word-exported Chinese report (embedded fonts) | `ocr_prob ≈ 0.05` | — | MUPDF |
| Scanned notes (raster, no formulas/tables) | `ocr_prob ≈ 0.7` | `has_complex=False` | PIPELINE |
| Scanned newspaper (raster + photos, no tables) | `ocr_prob ≈ 0.8` | `has_complex=False` | PIPELINE |
| Scanned financial report (raster + tables) | `ocr_prob ≈ 0.6` | `has_complex=True` | VLM |
| Scanned math textbook (raster + formulas) | `ocr_prob ≈ 0.7` | `has_complex=True` | VLM |
| Encrypted / corrupt / 0-page PDF | error | — | DEFERRED + `error` |

This explains the historical `bench_omnidoc100.summary.json` distribution (70 MUPDF / 30 PIPELINE / 0 VLM): 70 PDFs scored low `ocr_prob`, 30 needed OCR but had neither tables nor formulas, and VLM was disabled anyway. Once VLM is on in this run, the same 30 may split — pages with detected tables/formulas peel off into VLM.

## 4 · Changes

### 4.1 New module: `pdfsys_cli/parquet_writer.py`

~100 LOC. Single responsibility: take an iterable of `DocResult` + the markdown text from the in-memory `ExtractedDoc`, append rows to a `pyarrow.parquet.ParquetWriter`, finalise on close.

Public API:

```python
class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, compression: str = "zstd"): ...
    def write_row(self, row: DocResult, markdown: str | None) -> None: ...
    def close(self) -> None: ...
```

- `markdown=None` when extraction failed or produced no text → stored as null in the column (not empty string), so downstream `WHERE markdown IS NOT NULL` works as expected.
- Schema construction lives in `parquet_writer.SCHEMA` (module constant) so consumers can introspect.

### 4.2 Runner integration: `runner.py`

- Add `cfg.has_stage("parquet")` branch at top of `run()` — opens a `ParquetSink` and closes on exit.
- In `_process_one`, after `_stage_quality`, call `sink.write_row(row, markdown)` if the parquet stage is active.
- Replace the current single `extract_error: str` field on `DocResult` with two fields:
  - `error_class: str | None` — coarse bucket (`router | layout | extract_mupdf | extract_pipeline | extract_vlm | quality`)
  - `error_message: str | None` — the original exception text
  Migrate the four existing `row.extract_error = f"X_failed: {e}"` sites accordingly.
- Add `kept: bool` to `DocResult`, computed at write time (not stored on JSONL).
- Add `wall_ms_total` to `DocResult`, computed as sum of stage timings.

### 4.3 Config: `pdfsys_cli/config.py` + `pdfsys.example.yaml`

New `parquet:` block:

```yaml
parquet:
  enabled: true
  out: dataset.parquet           # path relative to output.dir
  compression: zstd              # zstd | snappy | none
  quality_threshold: 2.0
  include_markdown: true         # full markdown text in column
```

Extend existing blocks:

```yaml
layout:
  backend: pp_doclayoutv3        # use transformers backend (MPS-capable)
  device: mps                    # new — passed to LayoutAnalyser

vlm:
  enabled: true                  # was false
  device_mode: mps               # new — written to ~/magic-pdf.json

quality:
  device: mps                    # was null (auto)
```

Append `parquet` to the canonical stage order:

```python
CANONICAL_STAGES = ("router", "layout", "extract", "quality", "parquet")
```

### 4.4 magic-pdf MPS configuration

MinerU reads `~/magic-pdf.json` on import. When `vlm.enabled` and `vlm.device_mode = mps`, the runner writes/updates this file before the first VLM call (idempotent — only writes if missing or `device-mode` differs).

Minimum required content (the helper expands `~` to the absolute home path before writing, since magic-pdf does not):

```json
{
  "device-mode": "mps",
  "models-dir": "/Users/<user>/.cache/mineru/models",
  "table-config": { "model": "rapid_table", "enable": true },
  "formula-config": { "enable": true }
}
```

Helper: `pdfsys_cli/_mineru_config.py::ensure_config(device_mode: str)`. Behaviour:
- if `~/magic-pdf.json` missing → write full default
- if present and `device-mode` already matches → no-op
- if present and `device-mode` differs → update only that key, preserve everything else

### 4.5 Dependency

`packages/pdfsys-cli/pyproject.toml`:

```toml
"pyarrow>=15.0,<19.0"
```

No other new deps. magic-pdf, doclayout-yolo, transformers already in the workspace.

## 5 · Parquet schema (final)

| column | type | nullable | source |
|---|---|---|---|
| `pdf_path` | string | no | input |
| `sha256` | string | yes | layout / extract |
| `backend` | string | no | router (`MUPDF \| PIPELINE \| VLM \| DEFERRED`) |
| `stage_b_backend` | string | yes | decider |
| `ocr_prob` | float64 | yes | router |
| `num_pages` | int32 | no | router |
| `is_form` | bool | no | router |
| `garbled_text_ratio` | float64 | no | router |
| `is_encrypted` | bool | no | router |
| `layout_model` | string | yes | layout |
| `layout_num_regions` | int32 | yes | layout |
| `layout_has_complex` | bool | yes | layout |
| `extract_backend` | string | yes | parser actually run |
| `markdown` | string | yes | parser (null if extraction failed) |
| `markdown_chars` | int64 | no | parser |
| `quality_score` | float64 | yes | scorer |
| `quality_model` | string | yes | scorer |
| `error_class` | string | yes | first failing stage |
| `error_message` | string | yes | exception text |
| `kept` | bool | no | computed |
| `wall_ms_total` | float64 | no | sum of stage timings |

`kept = (error_class is None) AND (quality_score is not None) AND (quality_score >= cfg.parquet.quality_threshold)`.

## 6 · Two-phase run

Both phase configs start from a single source of truth — the updated `pdfsys.example.yaml` — and only override `input.limit` / `input.pdf_dir` / `output.dir`. Everything else (router threshold, layout backend, VLM/quality device, parquet block) is identical.

### Phase 1 · smoke (5 PDFs, ~10 minutes)

Config `pdfsys.smoke.yaml`:
- `input.pdf_dir: packages/pdfsys-bench/omnidocbench_100/pdfs`
- `input.limit: 5`
- `output.dir: ./out/e2e_smoke`

Pass criteria:
- run exits 0
- `dataset.parquet` opens in pandas with 5 rows
- at least one row has `extract_backend = "vlm"` (proves MinerU actually fired)
- no `error_class = "layout"` from MPS device errors
- `~/magic-pdf.json` updated with `"device-mode": "mps"`

If smoke fails, fix and rerun smoke before phase 2.

### Phase 2 · full (150 PDFs, est. 1–4 h)

Config `pdfsys.full.yaml`:
- `input.pdf_dir: packages/pdfsys-bench`  (recursive, picks up both subsets)
- `input.limit: null`
- `output.dir: ./out/e2e_full`

Pass criteria — see §8.

## 7 · Error model

Six `error_class` buckets, mutually exclusive. First stage to fail wins; downstream stages are skipped.

| `error_class` | Triggered by | Typical cause |
|---|---|---|
| `router` | router open / classify exception | encrypted, corrupt, 0-page PDF |
| `layout` | LayoutAnalyser exception | MPS OOM, transformers model load |
| `extract_mupdf` | parser-mupdf exception | malformed text stream |
| `extract_pipeline` | parser-pipeline exception | RapidOCR / image crop failure |
| `extract_vlm` | parser-vlm exception | MinerU OOM, missing models, timeout |
| `quality` | OcrQualityScorer exception | ModernBERT load / MPS issue |

`error_message` is `f"{type(e).__name__}: {e}"`, truncated to 500 chars.

No retries in scope. A future PR can add `error_class = "extract_vlm_oom"` + half-batch retry — out of scope here.

## 8 · Phase-2 acceptance

After the full run, the following queries must all succeed:

```sql
-- Coverage: every input file accounted for
SELECT count(*) FROM 'out/e2e_full/dataset.parquet';   -- expect 150

-- Backend mix sanity
SELECT extract_backend, count(*)
FROM 'out/e2e_full/dataset.parquet'
GROUP BY extract_backend;
-- expect MUPDF dominant, PIPELINE + VLM both > 0, DEFERRED small

-- Quality distribution
SELECT
  count(*) FILTER (WHERE kept) AS kept_rows,
  count(*) FILTER (WHERE error_class IS NOT NULL) AS errors,
  avg(quality_score) FILTER (WHERE quality_score IS NOT NULL) AS avg_q
FROM 'out/e2e_full/dataset.parquet';
-- kept_rows > 0 (proves the kept logic fires)
-- errors < 30: heuristic 20% tolerance — MinerU on Mac/MPS is unproven,
-- this is the threshold above which we stop and triage rather than ship.
-- A higher rate is informative, not fatal.

-- Error mix — should be inspectable, not catastrophic
SELECT error_class, count(*)
FROM 'out/e2e_full/dataset.parquet'
WHERE error_class IS NOT NULL
GROUP BY error_class;
```

If `error_class = "extract_vlm"` is the majority of failures, that's expected on first Mac run and feeds the next iteration (model download issues, magic-pdf version mismatch). It is **not** a blocker for accepting this spec — visibility into the failure modes is itself a deliverable.

## 9 · Risks & mitigations

| Risk | Mitigation |
|---|---|
| `magic-pdf` not installed / wrong version on Mac | Phase 1 smoke catches it on 5 PDFs, not 150 |
| MinerU model download (~3 GB) blocks first run | Document `MODELSCOPE_CACHE` / `HF_ENDPOINT` env vars in the runbook; pre-warm before smoke |
| MPS OOM on large pages | Failures bucketed as `error_class=extract_vlm`, run continues. Retry logic deferred. |
| PP-DocLayoutV3 MPS path broken in current transformers | Fallback: set `layout.backend: doclayout_yolo` (ONNX, CPU). Spec accepts CPU layout. |
| pyarrow version conflict | Pin `>=15.0,<19.0`; if conflict surfaces, downgrade is one line. |
| 150 PDFs estimated 1-4 h on Mac CPU/MPS | Acceptable for first run. Sharding / batching reserved for next iteration. |

## 10 · Out of this spec (explicit)

The following are intentionally **not** in scope and will be separate specs:

- P0 engineering basics: unit tests, parity harness, ruff/mypy config (ROADMAP §3)
- Stage-3 post-processing (reading-order, paragraph merge, unicode norm)
- Stage-4 quality/PII/MinHash dedup beyond the single ModernBERT score
- Multi-shard Parquet, S3/OSS output
- VLM batching / OOM retry
- structlog / Prometheus metrics

## 11 · Files touched

```
packages/pdfsys-cli/src/pdfsys_cli/
├── runner.py             (modified: add parquet stage, split error fields)
├── config.py             (modified: add ParquetConfig + device fields)
├── parquet_writer.py     (new ~100 LOC)
└── _mineru_config.py     (new ~40 LOC)

packages/pdfsys-cli/pyproject.toml   (modified: add pyarrow)
pdfsys.example.yaml                  (modified: parquet block + device fields)
pdfsys.smoke.yaml                    (new — phase 1 config)
pdfsys.full.yaml                     (new — phase 2 config)
```

No changes to `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-parser-vlm`, `pdfsys-bench`.

## 12 · Definition of done

- Phase 1 smoke run green on this Mac.
- Phase 2 full run completes; `dataset.parquet` exists at `out/e2e_full/`.
- `kept` column > 0 rows; error breakdown documented in a short post-run note appended to this spec.
- All four §8 acceptance queries return sensible numbers.
- No regression in existing `pdfsys run` flows that don't enable the `parquet` stage (backward compatible — old YAMLs still work).

## 13 · Post-run note · 2026-05-14

### Phase-2 stats

| Metric | Value |
|---|---|
| Wall time | 183 s for 150 PDFs (~1.2 s/PDF avg) |
| Exit | 0 |
| Parquet size | 1.5 MB (zstd) |
| Rows | 150 / 150 (100 % coverage) |
| `error_class` after fix-up | 0 |
| `kept = True` rows | 35 |
| Avg `quality_score` (non-null) | 1.418 |

### Backend distribution (after VLM retry merge)

| Stage-A `backend` | Stage-B `stage_b_backend` | `extract_backend` | Count |
|---|---|---|---|
| `mupdf` | — | `mupdf` | 104 |
| `pipeline` | `pipeline` | `pipeline` | 36 |
| `pipeline` | `vlm` | `vlm` | 10 |

This matches the historical OmniDocBench-100 70/30 mupdf/pipeline split, plus the 10 olmocr scanned-with-tables PDFs that newly fired Stage-B `vlm`.

### Failure modes encountered (and resolutions)

| Symptom | Root cause | Fix applied |
|---|---|---|
| All 10 VLM rows failed with `ModuleNotFoundError: No module named 'magic_pdf.pipe'` | `magic_pdf 1.0.1` calls `import paddle` in `doc_analyze_by_custom_model`; `paddlepaddle` was not installed. `pdfsys-parser-vlm.extract.py:194` caught the ImportError and fell back to v1's `magic_pdf.pipe.OCRPipe` — which doesn't exist in modern magic-pdf. | Installed `paddlepaddle`, `openai`, `ultralytics`, `pycocotools`, `detectron2` (from source), `timm`, `unimernet`, `paddleocr`, `rapid-table`, `struct-eqtable`, then **patched three magic-pdf site-packages files** to work with current transformers (>= 4.48). |
| First VLM extraction produced very low quality (`markdown_chars` 22, garbled formulas like `Let  = V/- 1`) | Fix subagent disabled `formula-config.enable=false` and `table-config.enable=false` in `~/magic-pdf.json` because formula/table sub-models had additional environment issues. | Tracked as known gap — VLM rows currently behave like a degraded PIPELINE pass. |
| `mupdf` raster page color-space warnings (`syntax error: could not parse color space (7 0 R)`) on 4 PDFs | Cosmetic PyMuPDF warning; extraction succeeded. | Ignored — not an error path. |

### Honest limitations of this iteration

1. **VLM is wired but not delivering value.** All 10 VLM rows succeeded only after disabling table+formula recognition — the two features that justified the VLM lane in the first place. Practically, these rows currently behave like a slower PIPELINE pass. Treating them as VLM-equivalent for downstream consumption is incorrect until table+formula are re-enabled.
2. **`paddlepaddle` + 9 other packages are installed in the venv but not in any `pyproject.toml`.** The next time someone runs `uv sync` from a clean state, VLM will break the same way. Adding these deps cleanly (or migrating to `mineru` 2.x which has a different module layout) is a follow-up.
3. **Site-packages patches to `magic_pdf/model/sub_modules/...` are not version-controlled.** `uv sync --reinstall` will undo them. Either fork magic-pdf or contribute upstream; either way, out of scope for this spec.
4. **MPS was configured but not measured.** The MinerU sub-models that exercise MPS (formula, table) were the ones disabled. The only torch-on-MPS path actually used was the ModernBERT quality scorer.

### Acceptance queries (§8) — actuals

```
rows                        : 150  (expected 150)        ✅
backend mix                 : mupdf 104 / pipeline 36 / vlm 10 — VLM > 0 ✅
kept_rows                   : 35   (> 0)                  ✅
errors                      : 0    (< 30 threshold)       ✅
error mix                   : (empty after retry)         ✅
```

§8's acceptance threshold (errors < 30) was met. The VLM-quality concern is a separate axis not covered by §8 — flagging it here so the next spec author knows to define a `markdown_chars_min` or "VLM produced ≥ N tokens of structured content" gate.

### Next iteration candidates (priority order)

1. **Fix the VLM env properly** — either pin `magic-pdf` to a version that doesn't need paddle, or migrate to the `mineru` 2.x package that supersedes it. Then re-enable formula+table and re-run the 10 VLM PDFs.
2. **Remove the `except ImportError: pass` in `pdfsys-parser-vlm/extract.py:194`** — it masked the paddle-missing error as a v2-API issue, causing the misleading "No module named 'magic_pdf.pipe'" message that took diagnostics to unwind. Surface ImportError straight to `_set_error("extract_vlm", e)`.
3. **Add a `markdown_chars` floor to the `kept` logic** so degraded extractions (22 chars) don't slip through as `kept=True` (one row currently does).
4. **Reproducibility:** record `pip freeze` output as a lockfile artifact for the VLM stack; right now the working venv is uncommittable.
