# Release Gate — Layer 4 + Calibration (v1)

**Date:** 2026-05-22
**Status:** Design — awaiting plan
**Source brief:** `ocr_quality.md` (untracked, sections §3 / §4 / §7 / §12)
**Predecessor:** `5ac2ec4 feat(bench): release-gate cascade — Layer-1 hard rules + escalation`

## 1. Context

`ocr_quality.md` proposes a 4-layer Release Gate that turns raw PDF extraction output into
`publish | review | reject` decisions for a publishable dataset. Current state of the
implementation:

- **Layer 1** (deterministic hard rules) is done — `quality_rules.py` + `cascade.py` ship 5
  blockers and a cascade-with-early-exit engine wired into `loop.py`.
- **Layer 2** (text-only OCR quality scorer) is done — ModernBERT regression in `quality.py`,
  invoked per-doc in the loop.
- **Layer 3** (visual verifier / consensus) — **not started, deferred to a future spec.**
- **Layer 4** (calibrated document-level release decision + manifest schema + calibration
  protocol) — **the subject of this spec.**

Separately, an LLM-judge OCR-quality scorer (`quality_llm.py`, mimo-v2.5-pro) was built but
is explicitly **out** of the internal pipeline. Its only role is external / annotation-time
review (see §7).

## 2. Goals

1. Take the existing bench JSONL (cascade + BERT scores) and emit a per-PDF
   `release_manifest.jsonl` with a `publish | review | reject` decision, a quality grade,
   blockers, reasons, and traceability metadata (`scorer_version`, `threshold_profile`).
2. Make thresholds editable and versioned without re-running extraction — calibrate, edit
   TOML, re-run the gate, compare.
3. Build a v0 calibration set on the existing 150 OmniDocBench samples and use it to fit
   the first `default-v1` threshold profile.
4. Provide a separate offline LLM-review tool that consumes the manifest and writes back
   `quality_*_llm` fields, controlled by a `--llm-scope all|review` flag.

## 3. Non-Goals

- **Layer 3** (visual verifier / consensus) — separate spec.
- **Page-level scoring** (`page_quality_p05`, `page_quality_min`, `bad_page_ratio`) — schema
  reserves the keys with `null` for v1; populated when Layer 3 lands.
- **Folding the LLM score into the decision rule.** Architecturally forbidden in v1: LLM is
  external review only.
- **Promoting Layer-1 hard-rule thresholds into the profile.** They stay as code defaults in
  `quality_rules.py`; cascade runs inside the loop and shouldn't take a runtime config path.
- **Building the cross-source calibration set** (500–1000 diverse PDFs). The v0 set is the
  existing 150 OmniDocBench samples; the larger set is a follow-up project.

## 4. Architecture

```
bench loop (existing — unchanged)
  ├─ Layer 1: cascade + hard rules  → cascade_decision: publish|reject + audit trail
  ├─ Layer 2: ModernBERT scorer     → quality_score (float, 0–3)
  └─ writes bench JSONL (one row per PDF)
              │
              ▼
release_gate.py (new — downstream, no loop changes)
  • reads bench JSONL
  • loads threshold profile (TOML)
  • applies decide(row, profile)
  • writes release_manifest.jsonl
              │
              ▼
[optional] llm_review.py (new — offline, never called from the loop)
  • reads manifest
  • --llm-scope all     → score every row (benchmark phase, default for v1)
  • --llm-scope review  → score only rows with decision == "review" (production phase)
  • patches manifest in place with quality_*_llm fields
```

**Key invariant:** `release_gate.py` and `llm_review.py` never import anything from
`pdfsys_bench.loop` or `pdfsys_bench.cascade`. They only consume the JSONL output.
Iterating thresholds or LLM rules does not require re-running extraction.

## 5. Manifest Schema

One JSON object per line in `release_manifest.jsonl`. The full v1 shape:

```json
{
  "doc_id": "9f8e7d6c5b4a...",
  "decision": "publish",
  "doc_quality_score": 2.31,
  "doc_quality_grade": "good",
  "blockers": {
    "empty_output": false,
    "too_short": false,
    "high_replacement_chars": false,
    "high_garbage_chars": false,
    "repetition_loop": false
  },
  "reasons": [
    "doc_quality_score=2.31 >= t_publish=2.00",
    "no Layer-1 blockers triggered"
  ],
  "cascade_final_stage": "mupdf",
  "page_quality_p05": null,
  "page_quality_min": null,
  "bad_page_ratio": null,
  "visual_alignment_score": null,
  "consensus_score": null,
  "scorer_version": "release-gate-v0.1",
  "threshold_profile": "default-v1@1.0.0",
  "quality_score_llm": 2.0,
  "quality_reason_llm": "Markdown is clean; minor inconsistent heading levels.",
  "quality_model_llm": "mimo-v2.5-pro"
}
```

**Field semantics:**

| Field | Source | v1 |
|-------|--------|----|
| `doc_id` | sha256 from bench JSONL | required |
| `decision` | `release_gate.decide()` | required, one of `publish / review / reject` |
| `doc_quality_score` | Layer-2 BERT (from bench JSONL) | required, may be `null` if `--no-quality` |
| `doc_quality_grade` | mapped from `doc_quality_score` via profile | required (or `null` if score is `null`) |
| `blockers` | Layer-1 from cascade audit trail (final stage) | required; **only the 5 we actually compute** |
| `reasons` | auto-generated from triggered blockers + grade | required; human-readable strings |
| `cascade_final_stage` | from `cascade_final_stage` in bench JSONL | optional (`null` in non-cascade runs) |
| `page_quality_*` / `bad_page_ratio` | Layer 3 / page-level | always `null` in v1 |
| `visual_alignment_score` / `consensus_score` | Layer 3 | always `null` in v1 |
| `scorer_version` | constant in `release_gate.py` | required (e.g. `"release-gate-v0.1"`) |
| `threshold_profile` | `<name>@<version>` from the loaded TOML | required |
| `quality_*_llm` | from `llm_review.py` if it has been run | optional (`null` until reviewed) |

**Grade mapping** (default, override-able via profile):

| Score range | Grade |
|-------------|-------|
| `>= 2.5` | `excellent` |
| `1.5 ≤ score < 2.5` | `good` |
| `0.5 ≤ score < 1.5` | `fair` |
| `< 0.5` | `poor` |

## 6. Decision Logic

```python
def decide(row: BenchRow, profile: ThresholdProfile) -> tuple[str, list[str]]:
    """Return (decision, reasons).

    Order matters: Layer-1 blockers veto regardless of score.
    Score-only decisions sit between t_publish and t_reject.
    """
    reasons: list[str] = []

    triggered = [name for name, hit in row.blockers.items() if hit]
    if triggered:
        reasons.append(f"Layer-1 blockers triggered: {triggered}")
        return "reject", reasons

    score = row.doc_quality_score
    if score is None:
        reasons.append("doc_quality_score missing — falling to review")
        return "review", reasons

    if score >= profile.t_publish:
        reasons.append(f"doc_quality_score={score:.2f} >= t_publish={profile.t_publish:.2f}")
        reasons.append("no Layer-1 blockers triggered")
        return "publish", reasons

    if score < profile.t_reject:
        reasons.append(f"doc_quality_score={score:.2f} < t_reject={profile.t_reject:.2f}")
        return "reject", reasons

    reasons.append(
        f"doc_quality_score={score:.2f} in grey band "
        f"[{profile.t_reject:.2f}, {profile.t_publish:.2f}) — needs review"
    )
    return "review", reasons
```

## 7. Threshold Profile

**Location:** `packages/pdfsys-bench/calibration/profiles/default-v1.toml`

**Format:** TOML (matches the rest of the codebase).

```toml
# packages/pdfsys-bench/calibration/profiles/default-v1.toml
name = "default-v1"
version = "1.0.0"
created_at = "2026-05-22"
description = "v0 calibration on OmniDocBench 150 samples"

[grade_boundaries]
# Lower bounds, inclusive. score >= excellent => "excellent", etc.
excellent = 2.5
good = 1.5
fair = 0.5
# Anything below `fair` is "poor".

[decision]
t_publish = 2.0   # score >= t_publish AND no blockers => publish
t_reject = 0.5    # score < t_reject => reject (or any blocker => reject)
# t_reject <= score < t_publish => review

[blockers]
# All Layer-1 blockers count by default. To ignore a noisy blocker in this
# profile, list its name here. Use sparingly — the safer default is to fix
# the blocker itself in quality_rules.py.
disable = []
```

**Profile identifier in the manifest:** `<name>@<version>`, e.g. `"default-v1@1.0.0"`.
Bump `version` when you re-fit thresholds; keep `name` for the policy family.

**Loader contract:** `release_gate.load_profile(path: Path) -> ThresholdProfile`. Raises on
missing keys. Validates `t_reject < t_publish`. Validates grade boundaries are monotonic.

## 8. Calibration Set v0

**Location:**
```
packages/pdfsys-bench/calibration/
├── README.md          # protocol + how to re-fit thresholds
├── labels.jsonl       # human-curated ground truth (this section)
└── profiles/
    └── default-v1.toml
```

**Source samples:** the 150 OmniDocBench rows already in `out/viz_final`. No new extraction.

**Label schema** (one JSON object per line in `labels.jsonl`):

```json
{
  "doc_id": "9f8e7d6c5b4a...",
  "doc_quality": 2,                          // 0–3, FinePDFs rubric
  "doc_publishable": true,                   // ground truth for the gate
  "severity": "none",                        // none|minor|major|critical (set when doc_publishable=false)
  "issue_flags": [],                         // multi-select from a fixed vocab (below)
  "note": "",                                // free text
  "labeled_by": "miracleyin",
  "labeled_at": "2026-05-22T10:30:00",
  "source": "human",                         // human | llm_draft (for filtering)
  "draft_score_llm": 2.0,                    // optional: LLM seed score that was reviewed
  "draft_reason_llm": "..."                  // optional: LLM reason that was reviewed
}
```

**`issue_flags` vocabulary (v1, only the things we can actually detect or label):**

- `garbage_text` — random characters, broken encoding
- `repetition` — long runs of the same line/paragraph
- `encoding_issue` — replacement chars, mojibake
- `missing_content` — large gaps relative to the page image
- `broken_table` — table structure destroyed
- `reading_order` — paragraphs/columns reordered

**Labeling workflow:**

1. Pre-fill `labels.jsonl` with one row per doc, populated with `draft_score_llm` and
   `draft_reason_llm` from `llm_review.py --llm-scope all` (see §9). `source` is set to
   `"llm_draft"` for these seed rows. Each draft row is `doc_publishable: null` until a
   human reviews it.
2. Human review happens in the viz site (recommended path, §11) OR by editing
   `labels.jsonl` directly with a text editor (fallback for v1 if the viz UI isn't built
   yet). Either way, the human sets `source: "human"`, fills `doc_quality` /
   `doc_publishable` / `severity` / `issue_flags`, and appends a new row with the same
   `doc_id`.
3. **Latest-wins read** — the loader scans the file forward and keeps the last row per
   `doc_id`, so human rows override LLM drafts whenever both exist. No in-place rewrites
   needed; the file is append-only.

**Fitting `default-v1`:** once a meaningful fraction (≥ 50 / 150) is human-labeled, run a
fitting script that searches `t_publish` and `t_reject` to optimize:

```
maximize  count(decision == "publish" AND doc_publishable == true)
subject to
  false_publish_rate <= 0.05   # publishing labeled-unpublishable docs
  review_rate         <= 0.30  # fraction routed to review
```

Print the chosen thresholds, the resulting confusion matrix, and write them into
`default-v1.toml`. Bump the profile `version` whenever thresholds change.

## 9. LLM External Review

**Module:** `packages/pdfsys-bench/src/pdfsys_bench/llm_review.py`.

**Hard rule:** `llm_review.py` MUST NOT be imported by `loop.py`, `cascade.py`, or
`release_gate.py`. The LLM score does not enter the decision function. The only place LLM
output lives is the optional `quality_*_llm` columns in the manifest, for human reviewers
and the calibration drafting workflow.

**CLI:**

```
uv run python -m pdfsys_bench.llm_review \
    --manifest out/release_manifest.jsonl \
    --markdown-dir out/viz_final/markdown \
    --llm-scope all|review \
    --workers 6 \
    --resume
```

**Scope semantics:**

- `--llm-scope all` (v1 default, benchmark phase): score every row, regardless of decision.
  Cost is real but acceptable while we're calibrating; gives auditable data on whether the
  `publish` bucket has escapees.
- `--llm-scope review` (production phase): score only rows with `decision == "review"`.
  Becomes the default once the threshold profile is trusted; reviewers see LLM hints only
  for the rows they actually inspect.

**Resume:** appends to a checkpoint file (`<manifest>.llm.jsonl`) per row, same pattern as
the existing `score_with_llm.py` script. Already-scored `doc_id`s skip on re-run.

**Calibration-time seeding:** running `llm_review.py --llm-scope all` against the bench
output produces the `draft_score_llm` / `draft_reason_llm` fields that pre-populate
`calibration/labels.jsonl`.

## 10. CLI Surface

```
# 1. Bench loop (unchanged):
uv run python -m pdfsys_bench --pdf-dir ... --out out/bench_full.jsonl --cascade

# 2. Release gate (new):
uv run python -m pdfsys_bench.release_gate \
    --bench-jsonl out/bench_full.jsonl \
    --out out/release_manifest.jsonl \
    --profile packages/pdfsys-bench/calibration/profiles/default-v1.toml

# 3. Offline LLM review (new, optional):
uv run python -m pdfsys_bench.llm_review \
    --manifest out/release_manifest.jsonl \
    --markdown-dir out/viz_final/markdown \
    --llm-scope all

# 4. (Calibration helper, optional in v1):
uv run python -m pdfsys_bench.fit_profile \
    --bench-jsonl out/bench_full.jsonl \
    --labels packages/pdfsys-bench/calibration/labels.jsonl \
    --out packages/pdfsys-bench/calibration/profiles/default-v1.toml
```

## 11. New Code

| Path | Purpose |
|------|---------|
| `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py` | `decide()`, `load_profile()`, `run_gate()`, `__main__` CLI |
| `packages/pdfsys-bench/src/pdfsys_bench/llm_review.py` | offline LLM scope-controlled scorer + `__main__` |
| `packages/pdfsys-bench/src/pdfsys_bench/fit_profile.py` | (calibration helper) threshold search over `labels.jsonl` |
| `packages/pdfsys-bench/calibration/profiles/default-v1.toml` | initial profile (placeholder thresholds until fit) |
| `packages/pdfsys-bench/calibration/README.md` | calibration protocol + how to add labels + how to re-fit |
| `packages/pdfsys-bench/calibration/labels.jsonl` | initially empty / LLM-drafted |
| `tests/bench/test_release_gate.py` | decision logic, profile loading, edge cases (missing score, all blockers) |
| `tests/bench/test_profile_loader.py` | TOML schema validation, monotonicity checks |
| `tests/bench/test_llm_review.py` | scope filtering, resume behavior (with mocked client) |

**Viz changes** (recommended for v1; **not strictly required** since labels.jsonl can be
hand-edited as a fallback — see §8 step 2):

- Add a labeling control to the detail card (0–3 radio, publishable toggle, issue flag
  checkboxes, note field, save button).
- Add `POST /api/label` to `viz_server.py` that appends a new row to
  `calibration/labels.jsonl` (same append-only pattern as `badcases.jsonl`).
- Add a `decision` column to the table once `release_manifest.jsonl` is wired in.

The plan can ship the viz UI as a separate task; if it slips, manual JSONL editing is
acceptable for the v0 calibration pass over 150 rows.

## 12. Risks & Open Questions

- **Threshold over-fitting on 150 samples.** v0 profile will be brittle. Document this in
  the calibration README; treat `default-v1` as provisional until the larger set lands.
- **`review` bucket capacity.** If the grey band is too wide, the LLM-review job dominates
  cost. Profile `review_rate <= 0.30` constraint mitigates this; revisit if it's still too
  high in practice.
- **`doc_quality_score = null` rows** (when bench is run with `--no-quality`). Currently
  routed to `review` — this is the safe default but means `--no-quality` runs produce a
  100%-review manifest. Acceptable for v1.
- **Labels JSONL append-and-replay.** The latest-wins read-forward strategy is fine for v1
  scale; revisit if labelers churn or labels grow past a few thousand.
- **No multi-annotator support in v1.** Single-annotator labels in `labels.jsonl`. If we
  add multi-annotator later, the schema needs a list of label records per doc.

## 13. Acceptance Criteria

- [ ] `release_gate.py` reads a bench JSONL and writes a valid `release_manifest.jsonl`
  conforming to §5 — including for runs without `--cascade` (no `cascade_final_stage`) and
  for `--no-quality` runs (score=null → review).
- [ ] `default-v1.toml` exists with placeholder thresholds; `load_profile` rejects malformed
  TOML, missing keys, and non-monotonic boundaries.
- [ ] `llm_review.py` honors `--llm-scope`, resumes correctly, and only patches
  `quality_*_llm` fields without altering `decision`.
- [ ] Calibration directory + README exists; LLM-drafted `labels.jsonl` seeds populated
  from `out/viz_llm`.
- [ ] Unit tests cover: blocker-veto, t_publish boundary, t_reject boundary, grey-band,
  missing score, profile load failures, scope filter.
- [ ] On the existing 150-sample bench run, the gate produces a manifest with non-trivial
  distribution across all three decisions (i.e. not 100% one bucket) on `default-v1`'s
  initial (hand-set) thresholds; the calibration loop is expected to tighten them.

## 14. Out of Scope (re-stated for clarity)

- Layer 3 (visual verifier / consensus).
- Page-level scoring.
- Cross-source calibration set (500–1000 PDFs).
- Folding LLM score into the decision rule.
- Promoting Layer-1 thresholds into the profile.

## 15. Post-build note (2026-05-22)

Implementation landed across 9 tasks. Plan: `docs/superpowers/plans/2026-05-22-release-gate-layer4.md`.

### Commits (in order)

```
Task 1 — TOML profile loader + default-v1.toml
  d993759 feat(bench): release-gate TOML profile loader + default-v1 profile
  5b98488 fix(bench): release-gate loader — immutable grades + strict disable + grade-key validation
  c996c88 chore: add ruff to dev dependency group

Task 2 — decide() pure logic
  a8efcef feat(bench): release-gate decide() — 3-state decision with blocker veto
  132d888 fix(bench): release-gate — boundary tests + post_init + sorted blockers

Task 3 — run_gate() + CLI
  b2519b1 feat(bench): release-gate run_gate() + CLI → release_manifest.jsonl
  8cfe50c fix(bench): release-gate run_gate — atomic write + edge tests + JSON summary

Task 4 — Calibration directory scaffold
  a5f65dc docs(bench): release-gate calibration directory + README

Task 5 — llm_review.py + scope/resume
  54a69be feat(bench): offline llm_review with --llm-scope + resume checkpoint
  04728d5 fix(bench): llm_review — atomic .tmp cleanup + parse_error in manifest + per-row exception + boundary test

Task 6 — fit_profile.py
  accdf89 feat(bench): release-gate fit_profile — grid search t_publish/t_reject
  ac96558 fix(bench): fit_profile — human-only latest-wins + TOML escape

Task 7 — viz_server POST /api/label
  2f86448 feat(viz): POST /api/label + GET /api/labels for calibration
  2e0e2b2 fix(viz): label endpoint — dedupe issue_flags + doc README sync step + docstrings

Task 8 — Viz UI labeling form
  b4f778b feat(viz): calibration label form on detail card (POST /api/label)
```

### End-to-end smoke (OmniDocBench 100)

```
uv run python -m pdfsys_bench.release_gate \
    --bench-jsonl out/bench_omnidoc100.jsonl \
    --out out/release_manifest_smoke.jsonl \
    --profile packages/pdfsys-bench/calibration/profiles/default-v1.toml
```

Output:
- `num_rows = 100`
- `by_decision = {"review": 74, "publish": 24, "reject": 2}`
- `grade_distribution = {"good": 46, "fair": 19, "excellent": 3, "poor": 2, null: 30}`

Notes:
- The `default-v1@0.1.0` profile uses hand-set placeholder thresholds.
  Distribution skew is expected pre-calibration. Re-fit via
  `fit_profile.py` once ≥ 50 / 100 rows are human-labeled.

### LLM external review smoke (5 rows)

```
uv run python -m pdfsys_bench.llm_review \
    --manifest out/release_manifest_smoke_5.jsonl \
    --markdown-dir out/viz_final/markdown \
    --llm-scope all \
    --workers 2
```

Result: `num_scored=5 / no failures — all 5 rows scored successfully via mimo-v2.5-pro; quality_score_llm populated (e.g. 1.0), quality_parse_error_llm=null`

### Schema deviation from §5

Task 5's review surfaced an information gap: a row whose LLM score had a
parse error was indistinguishable from a row that had never been scored.
The fix extended both `_NULL_LLM_FIELDS` in `release_gate.py` and
`_PATCH_FIELDS` in `llm_review.py` to include
`quality_parse_error_llm`. Every manifest row now carries this 4th LLM
field (`null` until reviewed).

The schema example in §5 of this spec was written before that fix —
update the §5 manifest example to include `quality_parse_error_llm`
when this spec is next revised.

### Architectural boundary now machine-enforced

`tests/architecture/test_boundary.py` gained
`test_llm_review_not_imported_by_decision_modules`: an AST scan that
fails the build if `loop.py`, `cascade.py`, or `release_gate.py` ever
imports `llm_review`. This codifies the invariant from §9.

### Known follow-ups

- `default-v1@0.1.0` thresholds are hand-set; re-fit via `fit_profile.py`
  once `calibration/labels.jsonl` has enough human labels. Bump to `0.2.0`.
- The viz server writes `<bundle>/labels.jsonl` (bundle-local, matching
  `badcases.jsonl`). Operators must sync to
  `packages/pdfsys-bench/calibration/labels.jsonl` before fitting —
  documented in `calibration/README.md` step 2.5.
- Layer 3 (visual verifier / consensus) is the next spec on the critical
  path for catching fluent-hallucination escapes that BERT alone misses.

### Test surface added

- `tests/bench/test_release_gate.py` — 26 tests (loader / decide / run_gate / boundary / null score / atomic write / CLI).
- `tests/bench/test_llm_review.py` — 7 tests (scope filter, resume, missing markdown, scorer exception).
- `tests/bench/test_fit_profile.py` — 5 tests (evaluate, fit, no-feasible, human-only, TOML escape).
- `tests/architecture/test_boundary.py` — +1 (`llm_review` not imported by decision modules).

All pass via `uv run pytest tests/bench/ tests/architecture/ -v`.
