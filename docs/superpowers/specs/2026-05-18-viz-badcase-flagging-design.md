# Viz Bad-Case Flagging — Design

**Date:** 2026-05-18
**Builds on:** `docs/superpowers/specs/2026-05-16-pipeline-viz-html-design.md` + `docs/superpowers/specs/2026-05-16-vlm-region-based-design.md`
**Scope:** Add a human-in-the-loop bad-case flagging feature to the viz UI. User browses pipeline output, clicks "flag as bad case" on a row, fills in stage + tags + note, and the flag persists to `out/viz_<run>/badcases.jsonl`. Filter shows only flagged rows. Export downloads the full JSONL for downstream model-improvement work.

---

## 1 · Goal

Let one person (the user) accumulate a curated list of "this output is wrong" labels across multiple pipeline runs, persisted to a `badcases.jsonl` file that can be committed to the repo and consumed later for training-data curation. The flagging UX is integrated into the existing viz detail page; no separate annotation tool, no context-switching.

Concrete acceptance: after the user flags ~10 PDFs across stages, kills the server, restarts the server, re-opens viz, and all 10 flags are still visible (🔴 in dashboard + populated form in detail view).

## 2 · Non-goals

- Multi-user collaboration / conflict resolution (single-machine use).
- Undo / redo / version history.
- Authentication (localhost-only server).
- A `pdfsys badcase-export` CLI to turn flags into training data (separate future spec).
- Backend persistence to anything other than a local JSONL file.

## 3 · Architecture

```
┌─────────────────────────────────────────────────────────┐
│  out/viz_<run>/                                          │
│  ├── index.html         (SPA — fetches data + POSTs)     │
│  ├── viz_data.json      (per-row pipeline metadata)      │
│  ├── badcases.jsonl     (NEW — persistent flags)         │
│  ├── viz_server.py      (NEW — micro Python server)      │
│  └── serve.sh           (MODIFIED — invokes viz_server)  │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │   POST /api/badcase
                          │   DELETE /api/badcase/<sha>
                          │   GET /api/badcases
                          │   GET /* (static fallthrough)
                          │
                  browser at http://localhost:8765/
```

`viz_server.py` is a single-file ~150-LOC standard-library Python server using `http.server.BaseHTTPRequestHandler`. No FastAPI, no Flask. Same port as the previous static server (8765).

`badcases.jsonl` lives in the bundle dir (next to `viz_data.json`). It is **append-only on disk** — every save appends a new line; the browser-side dedup-by-sha-keep-latest happens at load. This makes the file safe under concurrent writes (rare in practice; user is single) and gives an implicit change log.

## 4 · `badcases.jsonl` schema

One JSON object per line:

```json
{
  "sha256": "054c70a2ffddeb5b0235b8025c9aef86f8046efca82be3c10998bef1c6145359",
  "is_bad": true,
  "stage": "extract",
  "tags": ["handwriting", "rapidocr_fail"],
  "note": "RapidOCR 把手写笔记打成 [image] 占位",
  "flagged_at": "2026-05-18T16:30:12+08:00",
  "flagged_by": "yinz"
}
```

Field semantics:

| Field | Type | Required | Notes |
|---|---|---|---|
| `sha256` | string | yes | Row key. |
| `is_bad` | bool | yes | Always `true` for now; `false` reserved for "reviewed and confirmed not bad". |
| `stage` | enum string | yes | One of `router | layout | extract | quality | overall`. |
| `tags` | list[str] | yes (can be empty) | Free-form. Frontend lets user add new tags inline. |
| `note` | string | yes (can be empty) | Free-form, 0-1000 chars. |
| `flagged_at` | ISO-8601 string | yes | Server fills this on POST (ignores client value). |
| `flagged_by` | string | yes | Server fills from `--user` flag or `$USER`. |

Server **rejects** a POST if `sha256` or `is_bad` is missing. Other fields default to empty.

### Delete semantics

DELETE writes `{"sha256": ..., "is_bad": false, ..., "flagged_at": ..., "flagged_by": ...}` — an explicit "un-flag" record. Frontend dedup by sha (latest wins) makes this clear the flag.

## 5 · `viz_server.py` API

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/` | — | `index.html` |
| GET | `/<file>` | — | static file from bundle dir |
| GET | `/api/badcases` | — | `200 {"badcases": [<latest-per-sha>...]}` |
| POST | `/api/badcase` | `{sha256, stage, tags, note}` | `200 {<full record with server-filled flagged_at/by>}` |
| DELETE | `/api/badcase/<sha>` | — | `200 {sha256, is_bad:false, flagged_at, flagged_by}` |

Implementation:

- `BaseHTTPRequestHandler` subclass with `do_GET` / `do_POST` / `do_DELETE`.
- Static path routing: anything not starting with `/api/` → static file (read from `Path(__file__).parent` aka the bundle dir).
- Atomic JSONL append: open `badcases.jsonl` in append mode with `fcntl.flock(LOCK_EX)`, write `json.dumps(...) + "\n"`, fsync, unlock.
- `--port 8765 --user <name>` CLI flags; user defaults to `$USER` env var then `"anon"`.
- Server prints a startup banner: `pdfsys viz server: http://localhost:8765/ (user=yinz, badcases=out/viz/badcases.jsonl)`.

## 6 · `viz.py` (CLI) changes

- New step in `main()` after `_copy_template_and_assets`: copy `viz_server.py` template to `<out-dir>/viz_server.py`.
- Modify `_write_serve_script` to write:

  ```bash
  #!/usr/bin/env bash
  cd "$(dirname "$0")"
  python3 viz_server.py --port "${PORT:-8765}" --user "${USER:-anon}"
  ```

- **Do not** overwrite an existing `badcases.jsonl` if it's present in the out-dir (so `pdfsys visualize` re-runs preserve user's flags).

## 7 · `index.html` changes

### 7.1 Dashboard

- New column **🚩** as the leftmost cell: `🔴` if row is in badcases (with `is_bad=true`), else empty. Tooltip on the cell shows the note + stage.
- New filter: `☐ 🔴 bad only` — when checked, hides rows that are not flagged.
- Top-bar stats: existing `"X rows · Y errors · Z kept"` extended to `"... · F flagged"`.
- Top-bar buttons: existing `← back` plus a new `📥 Export` button that downloads the current in-memory badcases as JSONL.

### 7.2 Detail view (new card after Quality)

```
┌─ Bad case flag ─────────────────────────────────────────────┐
│ [🚩 Flag as bad case]   (toggles based on current state)    │
│                                                              │
│ Stage where it broke:                                        │
│   ○ router  ○ layout  ● extract  ○ quality  ○ overall       │
│                                                              │
│ Tags:  [handwriting ×] [rapidocr_fail ×] [+ Add tag…]       │
│                                                              │
│ Note:                                                        │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ RapidOCR 把手写笔记打成 [image] 占位                    │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ flagged 2026-05-18 16:30:12 by yinz                          │
│   [💾 Save]  [🗑 Remove]                                     │
└──────────────────────────────────────────────────────────────┘
```

Behavior:

- Initial state: button shows "🚩 Flag as bad case" and the form fields are empty/disabled.
- Click button: form enables, defaults `stage = "overall"`, tags empty, note empty.
- After Save: POST `/api/badcase`, on 200 update in-memory `BADCASES[sha] = record`, re-render dashboard 🚩 column, swap card to show "flagged at … by …" + Remove button.
- Click Remove: DELETE `/api/badcase/<sha>`, on 200 remove from in-memory, restore initial state.
- Form errors (network down etc.): inline red message, don't lose user input.

### 7.3 Boot-time data load

`fetch('viz_data.json')` and `fetch('/api/badcases')` in parallel. Both must succeed. Build `BADCASES = Object.fromEntries(badcases.map(b => [b.sha256, b]))` indexed by sha. Pass to rendering functions.

## 8 · Workflow end-to-end

```bash
# 1. Generate viz bundle (CLI now also copies viz_server.py).
pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional

# 2. Start the new server.
bash out/viz_regional/serve.sh
# OR explicitly:
python3 out/viz_regional/viz_server.py --port 8765 --user yinz

# 3. Browse http://localhost:8765/, flag bad cases.
# 4. badcases.jsonl auto-persists to out/viz_regional/badcases.jsonl.

# 5. (Optional) Commit the file to track curation.
git add -f out/viz_regional/badcases.jsonl
git commit -m "data: 12 bad cases flagged for 2026-05-18 review"

# 6. (Optional) Export from UI: click Export → download JSONL.

# 7. Re-run pipeline + viz preserves flags:
pdfsys run ... -o out/e2e_full_mineru4
pdfsys visualize -r out/e2e_full_mineru4 -o out/viz_regional
#   ↑ existing badcases.jsonl in out/viz_regional/ is preserved.
```

## 9 · Risks

| Risk | Mitigation |
|---|---|
| Concurrent writes corrupt JSONL | `fcntl.flock(LOCK_EX)` + atomic append + fsync per write |
| Port 8765 in use at startup | Server prints clear error with `lsof -ti:8765 \| xargs kill` hint |
| User edits jsonl by hand and corrupts it | Server's load path tolerates malformed lines (logs warning, skips line) |
| jsonl grows without bound (many un-flag/re-flag cycles) | Acceptable — file size is tiny per record (~300 bytes). Future: add `pdfsys badcase-compact` to rewrite latest-per-sha only |
| User starts viz_server.py outside the bundle dir | Server resolves `Path(__file__).parent` as bundle root, so it works from any cwd |

## 10 · Files touched

```
NEW:
packages/pdfsys-bench/viz/viz_server.py       ~150 LOC (template, copied into each bundle)
docs/superpowers/specs/2026-05-18-viz-badcase-flagging-design.md   this file

MODIFIED:
packages/pdfsys-cli/src/pdfsys_cli/viz.py     copy viz_server.py + don't overwrite badcases.jsonl + new serve.sh
packages/pdfsys-bench/viz/index.html          dashboard 🚩 column + filter + export button + detail-view card
```

Untouched: `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-parser-vlm`, `pdfsys-bench/annotation/*`, `pdfsys-cli/runner.py`, `pdfsys-cli/parquet_writer.py`, all YAMLs.

## 11 · Definition of done

- `bash out/viz_regional/serve.sh` boots the new server (visible in `lsof -i:8765`).
- Flag 5 rows via UI (different stages), kill server, restart, all 5 still 🔴.
- Filter "🔴 bad only" reduces dashboard to exactly 5 rows.
- Export downloads a 5-line JSONL.
- Delete one flag via UI, refresh, only 4 remain.
- `out/viz_regional/badcases.jsonl` is committable text (UTF-8, newline-terminated).
- Re-running `pdfsys visualize -r ... -o out/viz_regional` preserves `badcases.jsonl`.

## 12 · Post-build note · 2026-05-18

### Server

- `viz_server.py` total LOC: **228** (stdlib only — `http.server`, `fcntl`, `json`, `argparse`).
- Zero new dependencies in any `pyproject.toml`.
- Port 8765 default; `--user` CLI flag wired (POST/DELETE records reflect `flagged_by: yinz`).
- Atomic JSONL append via `fcntl.flock(LOCK_EX)` + `os.fsync` — verified on macOS Darwin 25.4.
- 8 isolated unit smoke tests from Task 1 all PASS (GET static, GET empty, POST, GET-after-POST, POST-without-sha → 400, POST-invalid-stage → 400, DELETE, jsonl-on-disk has 2 lines, path traversal blocked).

### Acceptance walkthrough — all PASS

| # | Check | Result |
|---|---|---|
| 1 | Server boots; `GET /api/badcases` returns empty | PASS |
| 2 | Flag 5 PDFs (one per stage: router / layout / extract / quality / overall) | 5/5 HTTP 200 |
| 3 | `GET /api/badcases` returns 5 flagged | PASS |
| 4 | `badcases.jsonl` on disk = 5 lines | PASS |
| 5 | Kill server, restart, all 5 still present | PASS |
| 6 | DELETE one flag → 4 active | PASS |
| 7 | Re-run `pdfsys visualize` preserves `badcases.jsonl` (size before/after match) | PASS (`[viz] preserved existing badcases.jsonl (1379 bytes)`) |
| 8 | Server still serves the 4 flagged records after re-run | PASS |

After all 8 server-API checks: 5 commits (`ba8f8d1` server, `5926d65` CLI, `00a41ff` dashboard, `7976f55` detail card, this commit).

### Browser checks (manual, deferred to user)

The following bullets from §11 require a human in front of the browser to verify; the server-side equivalents are passed above:
- 🚩 column displays 🔴 for flagged rows (verified via DOM in dev console: `BADCASES` populated correctly).
- 🔴 bad only filter narrows table to flagged rows (verified by reading code path: `if (badOnly && !BADCASES[r.sha256]) return false;`).
- Detail card UI: stage radio + tags + note + Save/Remove (rendered by `badcaseCardHtml`, wired by `attachBadcaseHandlers`).
- 📥 Export downloads `badcases-YYYY-MM-DD.jsonl` (via Blob + URL.createObjectURL).

### Notable observations

1. **Server emits a useful startup banner**: prints URL, bundle dir, current user, jsonl path + active flag count. Made the acceptance walkthrough easy to spot.
2. **JSONL grows on every save / delete** (append-only). After 6 operations (5 POST + 1 DELETE) the file has 6 lines. For our use scale (≤ low thousands of flags per project) this is fine; future `pdfsys badcase-compact` could rewrite latest-per-sha if needed.
3. **`flagged_at` ISO string includes timezone offset** (`+08:00`), so lexicographic sort = chronological sort. The browser's "latest wins" dedup is correct without parsing.

### Open follow-ups

- `pdfsys badcase-export` CLI to turn `badcases.jsonl` into a structured training-data file (probably JSONL keyed by sha → markdown + flag).
- Multi-bundle aggregation: combine `badcases.jsonl` across multiple `out/viz_<runN>/` directories by sha for unified curation.
- The current "🚩 Flag as bad case" + "🗑 Remove" UX is binary (`is_bad=true` vs deleted). A future iteration could add an explicit "✅ Reviewed (not a bad case)" path that writes `is_bad=false` records — useful when triaging large sets of suspects.
- Browser auto-refresh of `/api/badcases` when bundle is shared and multiple reviewers are editing concurrently. Not needed for solo use.
