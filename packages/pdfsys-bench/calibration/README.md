# Release-gate calibration

Ground-truth labels and threshold profiles for the Layer-4 release gate.

Spec: `docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md`.

## Layout

- `profiles/<name>.toml` — threshold profiles consumed by `release_gate.py`.
- `labels.jsonl` — append-only JSONL; one row per labeling decision.
  Latest-wins on `doc_id` when re-read.

## Label schema

```json
{
  "doc_id": "<sha256>",
  "doc_quality": 2,
  "doc_publishable": true,
  "severity": "none",
  "issue_flags": [],
  "note": "",
  "labeled_by": "<user>",
  "labeled_at": "<iso8601>",
  "source": "human",
  "draft_score_llm": 2.0,
  "draft_reason_llm": "..."
}
```

- `doc_quality`: 0–3 (FinePDFs rubric).
- `doc_publishable`: `true | false | null`. `null` for LLM-draft rows
  awaiting human review.
- `severity`: `none | minor | major | critical` (set when `doc_publishable=false`).
- `issue_flags`: subset of `["garbage_text", "repetition", "encoding_issue",
  "missing_content", "broken_table", "reading_order"]`.
- `source`: `human | llm_draft` — humans override drafts on the same `doc_id`.

## Workflow

1. Seed drafts:
   ```
   uv run python -m pdfsys_bench.llm_review \
       --manifest out/release_manifest.jsonl \
       --markdown-dir out/viz_final/markdown \
       --llm-scope all
   ```
   Then export to `labels.jsonl` with `source=llm_draft` and
   `doc_publishable=null`.

2. Human review in the viz site (recommended) or by appending rows to
   `labels.jsonl` directly. New rows for the same `doc_id` override
   prior ones.

3. Once ≥ 50 / 150 rows are human-labeled, re-fit:
   ```
   uv run python -m pdfsys_bench.fit_profile \
       --bench-jsonl out/bench_full.jsonl \
       --labels packages/pdfsys-bench/calibration/labels.jsonl \
       --out packages/pdfsys-bench/calibration/profiles/default-v1.toml
   ```

## Re-fitting guidance

Bump `version` (e.g. `0.1.0` → `0.2.0`) whenever thresholds change.
Profile identifiers (`<name>@<version>`) are written into every manifest
row's `threshold_profile` field so historical decisions stay traceable.
