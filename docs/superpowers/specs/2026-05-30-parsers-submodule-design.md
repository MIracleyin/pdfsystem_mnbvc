# Parsers Submodule + System Release Manifest (Spec #1)

**Date:** 2026-05-30
**Status:** Design — awaiting plan
**Predecessor:** `1a07a98 docs(ocr-quality): §13 metrics governance` (sets the meta-principle of component isolation)
**Successor (out of scope):** Spec #2 — `quality-scorer` as independent submodule with sealed holdout

## 1. Context

`docs/ocr-quality-strategy.md §13` establishes the rule: any component whose output is used as an optimization target (quality scorer first, but parsers are a second-order case) must be **versioned independently from the pipeline that consumes it**, otherwise pipeline-level optimization will silently game the consumer's internal signals (Goodhart). The mechanism we picked is:

1. Component lives in its own git repo, owns its own release cadence.
2. Main repo (`pdfsys`) mounts it as a git submodule.
3. A `system_release.toml` at main-repo root pins each submodule to a specific **commit SHA + human-readable tag** for every system release. A `pdfsys` system release is exactly the cartesian product of pinned component versions plus the main-repo tree.

This spec ships the **first** extraction: `pdfsys-parsers` (the four document-understanding packages — `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-parser-vlm`, `pdfsys-layout-analyser` — plus the `mineru-api` HTTP boundary they already share). It is intentionally the **easier** of the two planned extractions because the HTTP boundary is already in place (commits `3847b6f` and `375c9a2`), so no new abstraction is needed — only repo split + tooling.

`quality-scorer` extraction (Spec #2) reuses the `system_release.toml` machinery introduced here. Doing the parsers first is process risk reduction.

## 2. Goals

1. New git repo `MIracleyin/pdfsys-parsers` containing the four parser packages, the `mineru-api` wrapper code, and a frozen JSON schema for `ExtractedDoc` (mirrored from `pdfsys-core`).
2. Main repo `pdfsys` mounts that repo as a submodule at `external/parsers/`. The old in-tree paths `packages/pdfsys-parser-*` and `packages/pdfsys-layout-analyser/` are **removed from the main repo** (their history moves to the new repo).
3. Workspace `pyproject.toml` in the main repo updated to point its path-deps at the submodule paths. `uv sync` in the main repo Just Works after `git submodule update --init`.
4. New `system_release.toml` at main-repo root with the documented schema (§5). Format machine-readable so a future CI step can verify drift.
5. New `pdfsys release` CLI subcommand with three modes:
   - `pdfsys release status` — diff between `system_release.toml` and current submodule HEADs.
   - `pdfsys release lock` — read submodule HEADs, write/update `system_release.toml`, print a release-notes-friendly diff.
   - `pdfsys release verify` — exit non-zero if any submodule HEAD does not match its pinned commit; used in CI before publishing artifacts.
6. New `scripts/bootstrap.sh` that handles `git submodule update --init --recursive` + `uv sync` + sanity-checks; documented in README.
7. CI updated to (a) `git submodule update --init`, (b) `pdfsys release verify`, (c) run the existing 150-PDF bench and confirm `kept` and `avg_quality` are unchanged from the last in-tree run (`out/e2e_full_mineru3_regional`: `kept=34/150`, `avg_quality=1.4198`).
8. `release_manifest.jsonl` entries gain a `system_version` field (the value of `system.version` in `system_release.toml`); already-present `scorer_version` and `threshold_profile` stay alongside.

## 3. Non-Goals

- **No `quality-scorer` extraction.** Spec #2 covers it. This spec only adds the `system_release.toml` schema with a `quality-scorer` entry **declared but pointing at the in-tree commit** as a placeholder, so the schema is forward-compatible.
- **No public release.** New repo is private, owned by `MIracleyin`, until parsers stabilize. README at the new repo's root explains the relationship.
- **No interface changes.** `Parser.extract(pdf_path) -> ExtractedDoc` stays exactly as it is. No new flags, no API rev.
- **No bench/viz extraction.** They live in main repo; they are the measuring instruments, not components, and must evolve in lockstep with the gate/router.
- **No per-parser sub-submodules.** `pdfsys-parser-mupdf`, `-pipeline`, `-vlm`, `-layout-analyser` all live in **one** repo, advanced together with a single version. Per-parser repos would be over-decomposition for v1 — revisit only if mupdf's release cadence diverges sharply from VLM's.
- **No history rewrite of submodule contents.** We do `git mv`-equivalent extraction with `git filter-repo` to preserve per-file history in the new repo. Pre-cut commits in the main repo still reference deleted paths — that's fine for archaeology, blame on those paths just stops at the cut commit.
- **No backwards-compatible import paths.** If someone has `import pdfsys_parser_vlm` from outside the workspace, after `uv sync` it still works because the package name is unchanged. Inside the workspace, path-deps change, that's all.

## 4. Architecture (after this spec)

```
pdfsys (main repo)
├── packages/
│   ├── pdfsys-core/        # shared types, stays in main
│   ├── pdfsys-router/      # stays
│   ├── pdfsys-cli/         # stays — adds `release` subcommand
│   ├── pdfsys-bench/       # stays — calibration/benchmarks
│   └── pdfsys-quality/     # current scorer — stays (Spec #2 will extract)
│
├── external/
│   └── parsers/            # ← git submodule, points to pdfsys-parsers@<sha>
│       ├── packages/
│       │   ├── pdfsys-parser-mupdf/
│       │   ├── pdfsys-parser-pipeline/
│       │   ├── pdfsys-parser-vlm/
│       │   └── pdfsys-layout-analyser/
│       ├── schema/
│       │   └── extracted_doc.v1.json     # mirror of pdfsys-core's ExtractedDoc
│       ├── scripts/
│       └── README.md
│
├── system_release.toml     # ← pins component commits for the release
├── .gitmodules
├── scripts/bootstrap.sh
└── docs/...
```

**Workspace `pyproject.toml` change (main repo):**

```toml
# Before
[tool.uv.workspace]
members = [
  "packages/pdfsys-core",
  "packages/pdfsys-parser-mupdf",          # ← removed
  "packages/pdfsys-parser-pipeline",        # ← removed
  "packages/pdfsys-parser-vlm",             # ← removed
  "packages/pdfsys-layout-analyser",        # ← removed
  ...
]

# After
[tool.uv.workspace]
members = [
  "packages/pdfsys-core",
  "external/parsers/packages/pdfsys-parser-mupdf",
  "external/parsers/packages/pdfsys-parser-pipeline",
  "external/parsers/packages/pdfsys-parser-vlm",
  "external/parsers/packages/pdfsys-layout-analyser",
  ...
]
```

**Cross-repo dependency direction:**

`pdfsys-parsers` repo depends on `pdfsys-core`. We resolve this by **vendoring the read-only subset of `pdfsys-core` types into `pdfsys-parsers/schema/`** as a JSON schema file (`extracted_doc.v1.json`) + a small Python dataclass mirror. The parsers do **not** depend on main-repo `pdfsys-core` as a Python package. This keeps the submodule's dep graph one-way: main repo → submodule, never back.

Why JSON schema + dataclass mirror, not just JSON: existing parser code constructs `ExtractedDoc(...)` directly. Easier to mirror the dataclass than refactor every call site to `dict`. The mirror is generated from the JSON schema by a small script committed to both repos, so they cannot drift silently.

Why we don't share `pdfsys-core` directly: that would couple the two repos' release cadences, which is the exact thing this spec is trying to avoid. If `pdfsys-core` evolves, parsers stay pinned to the old schema until parsers explicitly re-vendor the new version. That's the desired semantics — schema-breaking changes become an explicit cross-repo PR.

## 5. `system_release.toml` schema

```toml
# Top of pdfsys main repo.
# Owned by docs/superpowers/specs/2026-05-30-parsers-submodule-design.md (§5).
# Update via `pdfsys release lock`. Diff via `pdfsys release status`.
# CI runs `pdfsys release verify` and fails if submodule HEAD != pinned commit.

[system]
version = "0.4.0"             # SemVer for the *system as a whole*
released_at = "2026-05-30"
released_by = "miracleyin"
notes = "First release pinning parsers as external submodule."

[components.parsers]
repo = "git@github.com:MIracleyin/pdfsys-parsers.git"
path = "external/parsers"
commit = "0000000000000000000000000000000000000000"   # 40-char SHA
tag = "v0.1.0"
schema_version = "extracted_doc.v1"

[components.quality-scorer]
# Spec #2 will move this to an external repo. Until then this points at
# the in-tree commit of packages/pdfsys-quality/ at the time of release.
repo = "in-tree:packages/pdfsys-quality"
path = "packages/pdfsys-quality"
commit = "1a07a987a3f8b1c2e4d5f6a7b8c9d0e1f2a3b4c5"
tag = "in-tree-0.1.0"

[runtime]
# Externally-tracked dependencies whose versions matter for reproducibility.
# Not a substitute for the lockfile — these are the "soft" pins for narrative.
mineru = "3.x"
python = "3.11+"
```

**Schema rules:**

- `system.version` is SemVer. Bump rules: any component pin change bumps minor; breaking config or output schema bumps major; doc-only fixes bump patch.
- `components.<name>.commit` is a 40-character lowercase hex SHA. CI rejects abbreviated SHAs.
- `components.<name>.tag` is human-readable, must exist in the component repo (verified by `release verify`).
- `components.<name>.schema_version` is the wire format version that component speaks. Parsers' `extracted_doc.v1` means the parsers commit-pinned here produces v1 ExtractedDoc JSON.
- `[runtime]` section is informational only. Lockfiles (uv.lock) are still ground truth for Python deps.
- A `[components.X]` entry with `repo = "in-tree:..."` is a transitional pin used during incremental extraction. CI accepts it but `release lock` emits a warning so we don't forget to migrate.

## 6. CLI surface (`pdfsys release`)

```
$ pdfsys release status
✓ system.version            : 0.4.0
✓ released_at               : 2026-05-30
─ components.parsers
  pinned commit             : 0000000…  (tag v0.1.0)
  external/parsers HEAD     : 0000000…
  status                    : up-to-date
─ components.quality-scorer
  pinned commit             : 1a07a98…  (tag in-tree-0.1.0)
  status                    : in-tree (Spec #2 pending)

$ pdfsys release lock
Reading submodule HEADs…
  external/parsers           HEAD=abcdef1  pinned=0000000   CHANGED
Updating system_release.toml…
  components.parsers.commit  → abcdef1
  components.parsers.tag     → (please set manually after `git tag` in the component repo)
WARNING: 1 in-tree component(s) present: quality-scorer
Wrote system_release.toml. Diff:
  components.parsers.commit: 0000000 → abcdef1

$ pdfsys release verify
✓ external/parsers HEAD matches pin
✓ tag v0.1.0 exists in external/parsers
✓ all in-tree components present at pinned commit
PASS
```

Exit codes: `verify` returns 0 on PASS, 1 on any mismatch. `status` always returns 0. `lock` returns 0 on successful write, 1 if working tree dirty.

## 7. Acceptance criteria

1. `git clone --recurse-submodules` on a fresh machine + `bash scripts/bootstrap.sh` produces a working venv with no manual steps.
2. `uv run pytest` passes the full main-repo test suite (parsers tests now run from the submodule path).
3. `uv run pdfsys release verify` PASSes.
4. Full 150-PDF bench (`uv run pdfsys-bench --cascade --vlm` against `omnidocbench_100 + olmocr_bench_50`) finishes within 10% wall-time of the last in-tree run (`491s` baseline from `out/e2e_full_mineru3_regional/`), with `kept = 34/150` and `avg_quality = 1.4198 ± 0.001`. Exact-match on `kept` is required; quality may drift within rounding because of dataclass ↔ JSON round-tripping.
5. `release_manifest.jsonl` produced by the release-gate contains a `system_version` field on every row, matching `system.version` in the active `system_release.toml`.
6. README updated: clone instructions, `bootstrap.sh` workflow, `system_release.toml` explanation, link back to this spec.
7. `docs/superpowers/specs/<this-file>` gets a §15 post-build note recording the bench numbers and any rough edges hit during the extraction.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `git filter-repo` history extraction loses file-level history or commits | Dry-run on a scratch clone first; verify `git log --follow packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` in the new repo shows all expected commits. |
| Workspace path-dep churn breaks `uv sync` on contributor machines | Bootstrap script gates on `[ -d external/parsers/packages ]` and prints a clear error message asking the user to run `git submodule update --init`. |
| `pdfsys-core` types drift between repos | Mirror is generated from a JSON schema file checked into both repos; a `pre-commit` hook diffs them and fails the commit if out-of-sync. |
| Contributors forget to update `system_release.toml` after bumping submodule | CI runs `release verify`; PR blocks if pin is stale. |
| Submodule access requires SSH key on CI runners | Use deploy keys; documented in CI setup. New repo starts private, so HTTPS clone needs token regardless. |
| Submodule extraction makes `git bisect` across the cut painful | Document the cut commit SHA in `docs/reports/2026-W22.md` and link from README. Bisects spanning the cut require both repos' working trees, which is expected. |

## 9. Open questions (to resolve in plan)

1. Exact name and layout of `pdfsys-parsers` repo (`packages/<x>/` vs flat).
2. Whether mineru's own version (`mineru[pipeline]>=3.1,<4.0`) should appear in `system_release.toml` `[runtime]` section or stay implicit in lockfile only.
3. Whether `pdfsys release lock` should auto-write a git tag in the submodule, or require the user to tag manually first (current draft: manual tag, lock reads it).
4. Whether `external/parsers/` or `vendor/parsers/` is the better mount point (Go convention vs. generic). Current draft: `external/`.
5. Whether the JSON schema mirror script lives in main repo or submodule (current draft: both, kept in sync by hook).
