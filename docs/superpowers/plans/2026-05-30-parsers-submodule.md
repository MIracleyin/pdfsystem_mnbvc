# Parsers Submodule — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `pdfsys-parser-{mupdf,pipeline,vlm}` and `pdfsys-layout-analyser` into a separate git repo `MIracleyin/pdfsys-parsers`, mount as a submodule at `external/parsers/`, introduce `system_release.toml` + `pdfsys release` CLI + bootstrap script. No interface change, no behavioral change. Bench parity required.

**Architecture:** see `docs/superpowers/specs/2026-05-30-parsers-submodule-design.md`.

**Tech Stack:** `git filter-repo` (history-preserving extraction), `uv` workspaces (path-deps repointed), Python 3.11+ stdlib + `tomllib` for the release CLI, `httpx` (already a dep) for the mineru-api boundary that already exists. New deps: none.

**Source spec:** `docs/superpowers/specs/2026-05-30-parsers-submodule-design.md`

**Conventions in this codebase:**
- `from __future__ import annotations` at the top of every Python file.
- `@dataclass(slots=True)` for data containers.
- Module-private helpers `_snake_case`.
- Tests in `tests/<area>/test_<module>.py`; plain pytest functions.
- One commit per task; `feat(<scope>): ...` / `fix(<scope>): ...` / `refactor(<scope>): ...` / `chore(<scope>): ...`.
- Run: `uv run pytest tests/<path> -v`, `uv run ruff check packages/<pkg>`.
- Direct commits to `main` are project convention.

---

## File Structure (target state)

**New (main repo):**

```
system_release.toml
.gitmodules
external/                    # mount point — empty until `git submodule update --init`
scripts/bootstrap.sh
packages/pdfsys-cli/src/pdfsys_cli/release.py     # `pdfsys release {status,lock,verify}`
tests/release/
├── __init__.py
├── test_toml_schema.py      # parse, validate, round-trip
├── test_status.py           # diff submodule HEAD vs pin
├── test_lock.py             # write pin from HEAD
└── test_verify.py           # CI guard
```

**New (parsers repo, `MIracleyin/pdfsys-parsers`):**

```
README.md
pyproject.toml               # workspace root
schema/
├── extracted_doc.v1.json    # JSON schema mirror of pdfsys-core's ExtractedDoc
└── generate_dataclass.py    # produces _schema_mirror.py from above
packages/
├── pdfsys-parser-mupdf/     # moved verbatim with git filter-repo
├── pdfsys-parser-pipeline/
├── pdfsys-parser-vlm/
├── pdfsys-layout-analyser/
└── pdfsys-schema-mirror/    # tiny pkg exposing the generated dataclass
scripts/
└── ci_test.sh
.github/workflows/
└── ci.yml
```

**Removed from main repo (after extraction):**

```
packages/pdfsys-parser-mupdf/        # gone — lives in submodule now
packages/pdfsys-parser-pipeline/
packages/pdfsys-parser-vlm/
packages/pdfsys-layout-analyser/
```

**Modified (main repo):**

```
pyproject.toml                       # workspace.members → external/parsers/packages/*
README.md                            # clone + bootstrap section
.gitignore                           # add external/parsers/.venv etc. if any
docs/architecture/LAYERS.md          # update layer diagram
docs/PRD.md                          # one-line update to mention component split
.github/workflows/ci.yml             # `git submodule update --init` + `pdfsys release verify`
```

---

## Task list

### Phase 0 — Schema lock (no extraction yet, reversible)

- [ ] **Task 0.1: Freeze `ExtractedDoc` JSON schema as v1.**
  - File: `docs/schema/extracted_doc.v1.json` (new).
  - Generate by introspecting `pdfsys_core.types.ExtractedDoc` (or wherever the dataclass lives — verify before writing).
  - Write `tests/schema/test_extracted_doc_v1.py` that loads a real `ExtractedDoc` instance from a fixture, dumps to JSON, validates against the schema, and round-trips back. Failing test first.
  - Commit: `chore(schema): freeze ExtractedDoc v1 JSON schema`.

- [ ] **Task 0.2: Add `_schema_mirror.py` generation script.**
  - File: `docs/schema/generate_dataclass.py` (new).
  - Reads the JSON schema, writes a `@dataclass(slots=True)` mirror. The generated file is checked in (`docs/schema/_extracted_doc_v1_mirror.py`).
  - Test: regenerating must produce a byte-identical file (deterministic codegen). Add `tests/schema/test_codegen_stable.py`.
  - Commit: `chore(schema): mirror generator + golden test`.

### Phase 1 — `system_release.toml` + `pdfsys release` CLI (no submodule yet)

- [ ] **Task 1.1: TOML schema parser + dataclass.**
  - File: `packages/pdfsys-cli/src/pdfsys_cli/release.py` (new).
  - `@dataclass(slots=True) SystemRelease`, `Component`, `Runtime`. Use `tomllib`. Validate SHA is 40 hex chars; validate `t_publish` is absent (this is not a gate profile, don't confuse readers); validate every component has either a real `repo` URL or `repo = "in-tree:<path>"`.
  - Test: `tests/release/test_toml_schema.py` — parse a fixture TOML, assert parsed fields; assert short-SHA rejected; assert missing required key rejected.
  - Commit: `feat(cli): SystemRelease TOML parser + dataclass`.

- [ ] **Task 1.2: `pdfsys release status` (read-only).**
  - Reads `system_release.toml` + walks `external/<path>` for each component, shells out to `git rev-parse HEAD` for the real SHA. Prints aligned diff to stdout. Exit 0.
  - In-tree components (`repo = "in-tree:..."`) show "in-tree" status, no diff.
  - Test: `tests/release/test_status.py` with two fixtures — pin matches, pin drifted. Capture stdout, assert exact strings.
  - Commit: `feat(cli): pdfsys release status`.

- [ ] **Task 1.3: `pdfsys release lock` (mutating).**
  - Reads current submodule HEADs, writes `system_release.toml` in-place preserving comments. Use `tomlkit` (new dep — add to `pdfsys-cli`) for comment-preserving round-trip; if avoiding the dep is feasible, hand-roll a minimal in-place rewriter. **Decision in plan execution: try `tomlkit` first.**
  - Refuses to write if `git status --porcelain` is non-empty (working tree dirty).
  - Test: `tests/release/test_lock.py` with a tmp-dir fixture mimicking a 2-component repo.
  - Commit: `feat(cli): pdfsys release lock`.

- [ ] **Task 1.4: `pdfsys release verify` (CI guard).**
  - Same diff logic as `status` but exits 1 on any mismatch; prints a single-line summary first, then full diff.
  - Test: `tests/release/test_verify.py` — exit code 0 on match, 1 on mismatch, with a clear stderr message.
  - Commit: `feat(cli): pdfsys release verify`.

- [ ] **Task 1.5: Initial `system_release.toml` checked in (all in-tree pins).**
  - File: `system_release.toml` at repo root.
  - `[system] version = "0.4.0-pre"`, every component currently `in-tree:<path>`.
  - Run `pdfsys release verify` — must PASS against the working tree.
  - Commit: `chore(release): initial system_release.toml — all in-tree pins`.

### Phase 2 — Extract `pdfsys-parsers` to its own repo

These tasks happen in **two repos**. Use a scratch directory for the extraction; do not push anything until Phase 2 is end-to-end green.

- [ ] **Task 2.1: Scratch-clone main repo + dry-run `git filter-repo`.**
  - Outside the working repo, clone the main repo into a scratch dir.
  - Run `git filter-repo --path packages/pdfsys-parser-mupdf --path packages/pdfsys-parser-pipeline --path packages/pdfsys-parser-vlm --path packages/pdfsys-layout-analyser`.
  - Manually verify: `git log --oneline -- packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` in the scratch shows all expected commits (the rewrite at `fd8b8d8`, the mineru migration commits, etc.).
  - If verification passes, this becomes the seed for the new repo.
  - **No commit in the main repo for this task — it's investigative.**

- [ ] **Task 2.2: Create `MIracleyin/pdfsys-parsers` GitHub repo, push extracted history.**
  - **Manual step requiring user**: `gh repo create MIracleyin/pdfsys-parsers --private --description "..."` (user runs this, not the agent).
  - Push the scratch repo to it as `main`.
  - Add `LICENSE`, top-level `README.md` (copy a stub from `docs/architecture/LAYERS.md` describing the layer).
  - Tag `v0.1.0` on the initial commit after push.
  - **No commit in the main repo for this task.**

- [ ] **Task 2.3: Add the vendor schema mirror to `pdfsys-parsers`.**
  - In the parsers repo: copy `docs/schema/extracted_doc.v1.json` to `schema/extracted_doc.v1.json`. Add a CI step in the parsers repo that fails if the file diverges from the upstream one (compare via curl of raw GitHub URL pinned to a SHA, or a manual sync workflow).
  - Add `packages/pdfsys-schema-mirror/` containing the generated dataclass + its `pyproject.toml`.
  - Repoint parser packages' `dependencies = ["pdfsys-core", ...]` to `dependencies = ["pdfsys-schema-mirror", ...]`. Verify imports compile.
  - **Commit in the parsers repo:** `feat(schema): vendor pdfsys-schema-mirror`.

- [ ] **Task 2.4: Add the submodule in the main repo.**
  - In main repo: `git submodule add git@github.com:MIracleyin/pdfsys-parsers.git external/parsers`.
  - Remove old paths: `git rm -r packages/pdfsys-parser-mupdf packages/pdfsys-parser-pipeline packages/pdfsys-parser-vlm packages/pdfsys-layout-analyser`.
  - Update root `pyproject.toml` `[tool.uv.workspace] members = [...]` to point at `external/parsers/packages/*`.
  - `uv lock` to regenerate lockfile against the new paths.
  - `uv run pytest` — full suite must pass.
  - Commit: `refactor(parsers): extract to external submodule — pdfsys-parsers@v0.1.0`.

- [ ] **Task 2.5: Update `system_release.toml` — bump parsers from in-tree to submodule.**
  - Run `pdfsys release lock`. Verify the diff shows only `components.parsers` changing.
  - Bump `[system] version = "0.4.0"` (drop `-pre`).
  - Run `pdfsys release verify` — must PASS.
  - Commit: `chore(release): pin parsers@v0.1.0 — system 0.4.0`.

### Phase 3 — Tooling, docs, CI

- [ ] **Task 3.1: `scripts/bootstrap.sh`.**
  - Idempotent. Steps: `git submodule update --init --recursive`, sanity-check `external/parsers/packages` exists, `uv sync`, `uv run pdfsys release verify`.
  - Bash test: source it in a CI-style fresh clone, no errors, no Python output before final "ready" message.
  - Test: `tests/scripts/test_bootstrap.sh` (shell-based, gated by `CI=1`).
  - Commit: `chore(scripts): bootstrap.sh + submodule init`.

- [ ] **Task 3.2: README clone + bootstrap section.**
  - Update top-level `README.md`: add "Quickstart" section showing `git clone --recurse-submodules` + `bash scripts/bootstrap.sh` + first command. Link to `docs/superpowers/specs/2026-05-30-parsers-submodule-design.md` for full context.
  - Commit: `docs(readme): submodule clone + bootstrap quickstart`.

- [ ] **Task 3.3: `docs/architecture/LAYERS.md` update.**
  - Add a "Component versioning" subsection: list each component, its repo, its pin source. Reference `system_release.toml` schema in §5 of the spec.
  - Commit: `docs(architecture): component versioning + submodule layer`.

- [ ] **Task 3.4: CI workflow.**
  - File: `.github/workflows/ci.yml` (new or update).
  - Steps: checkout with `submodules: recursive`, `uv sync`, `uv run pdfsys release verify`, `uv run ruff check`, `uv run pytest`. Bench is NOT in CI (too slow); a separate scheduled workflow runs the 150-PDF bench weekly.
  - Commit: `ci: submodule init + release verify + tests`.

### Phase 4 — Acceptance

- [ ] **Task 4.1: 150-PDF bench parity.**
  - Run `uv run pdfsys-bench --cascade --vlm` against `omnidocbench_100 + olmocr_bench_50`.
  - Compare `out/<new>/results.summary.json` to baseline (`out/e2e_full_mineru3_regional/results.summary.json`: `wall_seconds=490.80`, `num_pdfs=150`, `num_errors=0`, `avg_quality=1.4198`).
  - Acceptance: `kept` identical, `avg_quality` within ±0.001, `wall_seconds` within ±10%.
  - Record exact numbers in §15 post-build note (Task 4.3).

- [ ] **Task 4.2: Fresh-clone smoke.**
  - On a clean directory, `git clone --recurse-submodules git@github.com:MIracleyin/pdfsystem_mnbvc.git` + `bash scripts/bootstrap.sh` + `uv run pdfsys --help`. Zero manual intervention.
  - Document any rough edges in §15.

- [ ] **Task 4.3: Spec §15 post-build note.**
  - Append a `## 15. Post-build note (YYYY-MM-DD)` section to the spec capturing: bench numbers, total wall-clock for the extraction, any unexpected churn (lockfile, import paths, schema mirror sync), and any open follow-ups.
  - Commit: `docs(spec): parsers-submodule §15 post-build`.

---

## Critical-path dependency graph

```
0.1 schema freeze ──┐
0.2 mirror codegen ─┤
                    ├──► 2.3 vendor mirror in parsers repo
                    │
1.1 TOML parser ────┼──► 1.5 initial TOML ──┐
1.2 status ─────────┤                       │
1.3 lock ───────────┤                       ├──► 2.5 bump pin ──► 3.4 CI ──► 4.1 bench parity ──► 4.3 post-build
1.4 verify ─────────┘                       │
                                            │
2.1 dry-run filter-repo ──► 2.2 new repo ──► 2.4 add submodule ──┘
                                                                  │
                                            3.1 bootstrap ────────┤
                                            3.2 README ───────────┤
                                            3.3 LAYERS ───────────┘
```

Phase 0 + Phase 1 are independent of each other and of Phase 2.1; do them in parallel if dispatching subagents. Phase 2.4 is the no-going-back commit on the main repo; everything before it is reversible. Phase 4 is gating.

## Rollback

If Phase 4.1 fails (bench parity broken), revert Task 2.5 + Task 2.4 commits, leave the parsers repo alone (still useful for the future), and file a follow-up issue. The submodule remains uninitialized; the main repo continues to use in-tree paths as if nothing happened.
