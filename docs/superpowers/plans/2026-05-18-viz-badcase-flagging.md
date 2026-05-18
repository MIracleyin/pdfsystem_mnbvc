# Viz Bad-Case Flagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a human-in-the-loop bad-case flagging feature to viz: 🚩 column + filter on dashboard, Bad case flag card in detail view, persistent JSONL via a micro Python server replacing `python -m http.server`.

**Architecture:** New stdlib server `viz_server.py` (~150 LOC) handles 3 API endpoints (POST/DELETE/GET `/api/badcase[s]`) on top of static file serving. Flags persist to `out/viz_<run>/badcases.jsonl` (append-only, latest-per-sha dedup at browser load). CLI copies the server template into each bundle and rewrites `serve.sh`. HTML SPA gets a new card + column + filter + Export button.

**Tech Stack:** Python 3.11 stdlib (`http.server.BaseHTTPRequestHandler`, `fcntl`, `json`), vanilla JS (existing pattern), no new deps.

**Source spec:** `docs/superpowers/specs/2026-05-18-viz-badcase-flagging-design.md` — read §4 for jsonl schema, §5 for API, §7 for UX.

**User constraint:** No unit tests, consistent with prior iterations. Verification is by curl + manual browser flag/unflag/restart cycle.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `packages/pdfsys-bench/viz/viz_server.py` | create | Stdlib HTTP server: static + 3 API endpoints + atomic JSONL writes |
| `packages/pdfsys-cli/src/pdfsys_cli/viz.py` | modify | Copy viz_server.py into bundle, preserve existing badcases.jsonl, rewrite serve.sh |
| `packages/pdfsys-bench/viz/index.html` | modify | Dashboard 🚩 column + "bad only" filter + Export button + detail Bad-case card; boot loads /api/badcases in parallel with viz_data.json |

Untouched: pdfsys-core, router, layout, parsers, runner, parquet_writer, configs.

---

## Task 1: Build `viz_server.py` micro server

**Files:**
- Create: `packages/pdfsys-bench/viz/viz_server.py`

This is a stdlib-only Python server. ~180 LOC including docstrings + CLI.

- [ ] **Step 1: Write the file**

Create `packages/pdfsys-bench/viz/viz_server.py`:

```python
"""pdfsys viz server — static file serving + bad-case flagging API.

Replaces `python -m http.server` for the viz bundle. Adds three API
endpoints backed by an append-only badcases.jsonl in the bundle dir.

API:
    GET  /api/badcases             → {"badcases": [<latest-per-sha>...]}
    POST /api/badcase              ← {sha256, stage, tags, note}
                                   → {<full record with server-filled
                                     flagged_at/by, is_bad: true>}
    DELETE /api/badcase/<sha256>   → {sha256, is_bad: false, ...}

Anything else falls through to static file serving from the bundle dir
(resolved as the directory containing this script).

Run:
    python3 viz_server.py --port 8765 --user yinz
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
BADCASES_PATH = BUNDLE_DIR / "badcases.jsonl"
VALID_STAGES = ("router", "layout", "extract", "quality", "overall")

# --user defaults to $USER env var, falling back to "anon". Settable via CLI.
USER = os.environ.get("USER", "anon")


# ---------------------------------------------------------------- jsonl io

def _load_badcases() -> list[dict]:
    """Load all jsonl records, return latest per sha256."""
    if not BADCASES_PATH.exists():
        return []
    latest: dict[str, dict] = {}
    with BADCASES_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[viz_server] WARN: line {line_no} not valid JSON, skipping", file=sys.stderr)
                continue
            sha = rec.get("sha256")
            if not sha:
                continue
            prev = latest.get(sha)
            if prev is None or rec.get("flagged_at", "") >= prev.get("flagged_at", ""):
                latest[sha] = rec
    return list(latest.values())


def _append_badcase(rec: dict) -> None:
    """Append a record to badcases.jsonl with an exclusive flock + fsync."""
    BADCASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with BADCASES_PATH.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- handler

class VizHandler(BaseHTTPRequestHandler):

    # Silence default per-request log lines (server already prints a banner).
    def log_message(self, format: str, *args) -> None:
        pass

    # -------------------- helpers --------------------

    def _send_json(self, status: HTTPStatus, body: dict | list) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status: HTTPStatus, msg: str) -> None:
        payload = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _serve_static(self) -> None:
        # Resolve the requested path under BUNDLE_DIR; default to index.html.
        url_path = self.path.split("?", 1)[0]
        if url_path in ("", "/"):
            url_path = "/index.html"
        # Strip leading "/", join, resolve. Prevent escape via ".." traversal.
        rel = url_path.lstrip("/")
        target = (BUNDLE_DIR / rel).resolve()
        try:
            target.relative_to(BUNDLE_DIR)
        except ValueError:
            self._send_text(HTTPStatus.FORBIDDEN, "path traversal denied")
            return
        if not target.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, f"not found: {url_path}")
            return
        data = target.read_bytes()
        # Minimal content-type guess; browsers are tolerant for the file types we ship.
        suffix = target.suffix.lower()
        ct = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".md": "text/markdown; charset=utf-8",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".woff2": "font/woff2",
        }.get(suffix, "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -------------------- routes --------------------

    def do_GET(self) -> None:
        if self.path == "/api/badcases":
            self._send_json(HTTPStatus.OK, {"badcases": _load_badcases()})
            return
        self._serve_static()

    def do_POST(self) -> None:
        if self.path != "/api/badcase":
            self._send_text(HTTPStatus.NOT_FOUND, "no such endpoint")
            return
        body = self._read_json_body()
        if not isinstance(body, dict):
            self._send_text(HTTPStatus.BAD_REQUEST, "body must be JSON object")
            return
        sha = body.get("sha256")
        if not isinstance(sha, str) or not sha:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing sha256")
            return
        stage = body.get("stage", "overall")
        if stage not in VALID_STAGES:
            self._send_text(HTTPStatus.BAD_REQUEST,
                            f"stage must be one of {VALID_STAGES}")
            return
        tags = body.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            self._send_text(HTTPStatus.BAD_REQUEST, "tags must be list[str]")
            return
        note = body.get("note", "")
        if not isinstance(note, str):
            self._send_text(HTTPStatus.BAD_REQUEST, "note must be str")
            return
        rec = {
            "sha256": sha,
            "is_bad": True,
            "stage": stage,
            "tags": tags,
            "note": note[:1000],
            "flagged_at": datetime.datetime.now(datetime.timezone.utc)
                          .astimezone().isoformat(timespec="seconds"),
            "flagged_by": USER,
        }
        _append_badcase(rec)
        self._send_json(HTTPStatus.OK, rec)

    def do_DELETE(self) -> None:
        if not self.path.startswith("/api/badcase/"):
            self._send_text(HTTPStatus.NOT_FOUND, "no such endpoint")
            return
        sha = self.path[len("/api/badcase/"):].split("?", 1)[0]
        if not sha:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing sha")
            return
        rec = {
            "sha256": sha,
            "is_bad": False,
            "stage": "overall",
            "tags": [],
            "note": "",
            "flagged_at": datetime.datetime.now(datetime.timezone.utc)
                          .astimezone().isoformat(timespec="seconds"),
            "flagged_by": USER,
        }
        _append_badcase(rec)
        self._send_json(HTTPStatus.OK, rec)


# ---------------------------------------------------------------- main

def main() -> int:
    global USER
    parser = argparse.ArgumentParser(prog="pdfsys viz_server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user", type=str, default=USER,
                        help="written into flagged_by on POST/DELETE")
    args = parser.parse_args()
    USER = args.user

    server = ThreadingHTTPServer(("127.0.0.1", args.port), VizHandler)
    n_existing = len(_load_badcases())
    print(f"pdfsys viz server: http://localhost:{args.port}/")
    print(f"  bundle dir: {BUNDLE_DIR}")
    print(f"  user (flagged_by): {USER}")
    print(f"  badcases.jsonl: {BADCASES_PATH} ({n_existing} active flags)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz_server] interrupted, shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke the server in isolation**

Start the server in background pointing at a temp bundle dir:

```bash
TMP_BUNDLE=$(mktemp -d)
cp packages/pdfsys-bench/viz/viz_server.py "$TMP_BUNDLE/"
echo '<html>test</html>' > "$TMP_BUNDLE/index.html"

cd "$TMP_BUNDLE"
python3 viz_server.py --port 8765 --user testuser > /tmp/_viz_smoke.log 2>&1 &
SERVER_PID=$!
sleep 1

# GET / (static)
curl -fsS http://localhost:8765/ -o /dev/null -w "GET /            : %{http_code}\n"

# GET /api/badcases (empty)
curl -fsS http://localhost:8765/api/badcases -w "\nGET /api/badcases: %{http_code}\n"

# POST /api/badcase
curl -fsS -X POST http://localhost:8765/api/badcase \
  -H 'Content-Type: application/json' \
  -d '{"sha256":"abc123","stage":"extract","tags":["test"],"note":"smoke"}' \
  -w "\nPOST            : %{http_code}\n"

# GET /api/badcases (one record now)
curl -fsS http://localhost:8765/api/badcases -w "\nGET (after POST): %{http_code}\n"

# DELETE
curl -fsS -X DELETE http://localhost:8765/api/badcase/abc123 \
  -w "\nDELETE          : %{http_code}\n"

# Final state
curl -fsS http://localhost:8765/api/badcases

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
cd - > /dev/null
```

Expected:
- `GET /` → 200
- `GET /api/badcases` (initial) → `{"badcases": []}`
- POST → 200 with full record including `flagged_at` + `flagged_by: testuser`
- `GET /api/badcases` (after POST) → 1 record with `is_bad: true`
- DELETE → 200 with `is_bad: false`
- Final GET → empty `badcases` array (because latest record is is_bad=false; client filters)

**Note:** the server returns ALL records via latest-per-sha dedup, including is_bad=false. The browser filters `is_bad === true` for display. So the final GET will actually return the is_bad=false record. Fix the expected — final GET returns `[{"sha256":"abc123","is_bad":false,...}]`.

If any step fails, fix the bug before committing.

- [ ] **Step 3: Verify badcases.jsonl content**

```bash
cat "$TMP_BUNDLE/badcases.jsonl"
```

Expected: 2 lines — one is_bad=true, one is_bad=false. Both valid JSON.

```bash
rm -rf "$TMP_BUNDLE"
```

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-bench/viz/viz_server.py
git commit -m "feat(viz): stdlib micro server with badcases API"
```

---

## Task 2: Update `viz.py` CLI to ship the server + preserve badcases

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/viz.py`

- [ ] **Step 1: Update `_copy_template_and_assets` to also copy viz_server.py**

Locate the existing `_copy_template_and_assets` function in `packages/pdfsys-cli/src/pdfsys_cli/viz.py`. Change its body to copy `viz_server.py` alongside `index.html` and `assets/`:

```python
def _copy_template_and_assets(template_dir: Path, out_dir: Path) -> None:
    src_html = template_dir / "index.html"
    if not src_html.exists():
        print(f"[viz] ERROR: template {src_html} not found", file=sys.stderr)
        sys.exit(1)
    shutil.copyfile(src_html, out_dir / "index.html")

    src_assets = template_dir / "assets"
    dst_assets = out_dir / "assets"
    if dst_assets.exists():
        shutil.rmtree(dst_assets)
    shutil.copytree(src_assets, dst_assets)

    src_server = template_dir / "viz_server.py"
    if src_server.exists():
        shutil.copyfile(src_server, out_dir / "viz_server.py")
        os.chmod(out_dir / "viz_server.py", 0o755)

    print("[viz] copied template + assets + viz_server.py")
```

- [ ] **Step 2: Update `_write_serve_script` to invoke viz_server.py**

Replace the body of `_write_serve_script`:

```python
def _write_serve_script(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "# Local server for the pdfsys viz bundle.\n"
        "# Serves static files + bad-case flagging API.\n"
        "# Open http://localhost:8765/ in a browser.\n"
        "cd \"$(dirname \"$0\")\"\n"
        "python3 viz_server.py --port \"${PORT:-8765}\" "
        "--user \"${USER:-anon}\"\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
```

- [ ] **Step 3: Preserve existing badcases.jsonl on viz regeneration**

In `viz.py`'s `main()`, locate the line:

```python
    out_dir.mkdir(parents=True, exist_ok=True)
```

Insert AFTER it (before any subsequent writes that might overwrite content):

```python
    # Preserve any existing badcases.jsonl across viz regenerations.
    # User's flags are accumulated state, not derived data — viz.py never
    # generates this file, so just leave it alone if present.
    existing_badcases = out_dir / "badcases.jsonl"
    badcases_preserved = existing_badcases.exists()
```

After `_write_serve_script(...)` is called, before the final print statement, add:

```python
    if badcases_preserved:
        print(f"[viz] preserved existing badcases.jsonl ({existing_badcases.stat().st_size} bytes)")
```

- [ ] **Step 4: Verify the CLI**

```bash
rm -rf /tmp/_viz_t2
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o /tmp/_viz_t2 2>&1 | tail -10
echo "--- bundle contents ---"
ls /tmp/_viz_t2/
echo "--- serve.sh content ---"
cat /tmp/_viz_t2/serve.sh
echo "--- viz_server.py exists ---"
ls -la /tmp/_viz_t2/viz_server.py
```

Expected:
- Bundle includes `viz_server.py` (executable bit set)
- `serve.sh` invokes `viz_server.py` not `python -m http.server`
- Existing `badcases.jsonl` would be preserved (test this in next step)

- [ ] **Step 5: Verify badcases.jsonl preservation**

```bash
echo '{"sha256":"abc","is_bad":true,"stage":"extract","tags":[],"note":"test","flagged_at":"2026-05-18T10:00:00+08:00","flagged_by":"test"}' > /tmp/_viz_t2/badcases.jsonl

# Re-run viz CLI on the same out-dir
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o /tmp/_viz_t2 2>&1 | tail -5

# badcases.jsonl should still be there
cat /tmp/_viz_t2/badcases.jsonl
```

Expected: the original test line is still in the file. Print line confirms `[viz] preserved existing badcases.jsonl (N bytes)`.

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/viz.py
git commit -m "feat(cli): ship viz_server.py into bundle, preserve badcases.jsonl across re-runs"
```

---

## Task 3: Update `index.html` — boot loads /api/badcases + dashboard 🚩 column + filter + Export

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html`

Three changes in this task. Splitting into multiple steps but one commit.

### 3.1 Boot: load /api/badcases in parallel + global BADCASES state

- [ ] **Step 1: Add BADCASES global + update boot fetch**

In `packages/pdfsys-bench/viz/index.html`, find the boot block at the bottom (the `try { ... } catch(e) { ... }` inside the main `<script type="module">`). It currently does:

```javascript
  try {
    const [d, p] = await Promise.all([
      fetch('viz_data.json').then(r => r.json()),
      fetch('previews.json').then(r => r.json()).catch(() => ({})),
    ]);
    DATA = d; PREVIEWS = p;
```

Replace with:

```javascript
  try {
    const [d, p, bc] = await Promise.all([
      fetch('viz_data.json').then(r => r.json()),
      fetch('previews.json').then(r => r.json()).catch(() => ({})),
      fetch('/api/badcases').then(r => r.ok ? r.json() : {badcases: []})
                            .catch(() => ({badcases: []})),
    ]);
    DATA = d; PREVIEWS = p;
    // Index badcases by sha for O(1) lookup. Filter is_bad=true (server may
    // include unflagged records — the latest-per-sha already happened server-side).
    BADCASES = Object.fromEntries(
      (bc.badcases || []).filter(b => b.is_bad).map(b => [b.sha256, b])
    );
```

Also add `let BADCASES = {};` at the top alongside the other global state declarations (`let DATA = null; let PREVIEWS = null; let SORT = ...`).

### 3.2 Dashboard: 🚩 column + "bad only" filter + Export button + badge count

- [ ] **Step 2: Add the 🚩 column at the start of `COLS`**

Find the `const COLS = [` array. Prepend a new entry at the start:

```javascript
  const COLS = [
    { key: 'flag', label: '🚩', tip: 'Bad case flag. Click row → flag in detail view.',
      render: r => {
        const bc = BADCASES[r.sha256];
        if (!bc) return '';
        const tooltip = `${bc.stage}${bc.note ? ': ' + bc.note.slice(0, 80) : ''}`;
        return `<span title="${tooltip.replace(/"/g, '&quot;')}">🔴</span>`;
      } },
    { key: 'pdf_basename', label: 'file', tip: 'PDF filename (without dir).' },
    // ... rest of existing columns unchanged ...
```

- [ ] **Step 3: Add "bad only" filter checkbox to the HTML filter-bar**

Find the `<div class="filter-bar">` element. Add a checkbox at the start (before `kept only`):

```html
  <div class="filter-bar">
    <label><input type="checkbox" id="f-bad"> 🔴 bad only</label>
    <label><input type="checkbox" id="f-kept"> kept only</label>
    <!-- ... rest unchanged ... -->
```

- [ ] **Step 4: Wire the new filter in `renderTable()` and the event listener**

Find `renderTable()`. After the line `const kept = document.getElementById('f-kept').checked;`, add:

```javascript
    const badOnly = document.getElementById('f-bad').checked;
```

Then in the filter chain (`let rows = DATA.rows.filter(r => { ... })`), add an early bail-out as the first check:

```javascript
      if (badOnly && !BADCASES[r.sha256]) return false;
```

Find the event-listener block at the bottom of `<script>`:

```javascript
  ['f-kept', 'f-error', 'f-backend', 'f-source'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderTable));
```

Add `'f-bad'` to that list:

```javascript
  ['f-bad', 'f-kept', 'f-error', 'f-backend', 'f-source'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderTable));
```

- [ ] **Step 5: Add F-flagged count to top-bar meta + Export button**

Find the `#meta-summary` text update in `renderDashboard()`:

```javascript
    document.getElementById('meta-summary').innerHTML =
      `<strong>${DATA.run_meta.total_rows}</strong> rows · `
      + `<strong>${DATA.run_meta.errors}</strong> errors · `
      + `<strong>${DATA.run_meta.kept}</strong> kept · `
      + `wall ${(DATA.run_meta.wall_seconds || 0).toFixed(0)}s · `
      + `avg quality ${(DATA.run_meta.avg_quality || 0).toFixed(2)}`;
```

Replace with:

```javascript
    const flaggedCount = Object.keys(BADCASES).length;
    document.getElementById('meta-summary').innerHTML =
      `<strong>${DATA.run_meta.total_rows}</strong> rows · `
      + `<strong>${DATA.run_meta.errors}</strong> errors · `
      + `<strong>${DATA.run_meta.kept}</strong> kept · `
      + `<strong>${flaggedCount}</strong> flagged · `
      + `wall ${(DATA.run_meta.wall_seconds || 0).toFixed(0)}s · `
      + `avg quality ${(DATA.run_meta.avg_quality || 0).toFixed(2)}`;
```

In the topbar HTML, add the Export button next to `back-btn`:

```html
<div id="topbar">
  <h1>pdfsys viz</h1>
  <span class="meta" id="meta-summary">loading…</span>
  <button id="export-btn">📥 Export</button>
  <button id="back-btn">← back</button>
</div>
```

Add CSS for `#export-btn` matching `#back-btn` style (right after the `#back-btn` style block):

```css
  #export-btn { padding: 4px 10px; background: var(--surface2);
                border: 1px solid var(--border); color: var(--text);
                border-radius: 3px; cursor: pointer; font-size: 12px; }
  #export-btn:hover { background: #1a4a80; }
```

And the click handler (next to the `back-btn` handler):

```javascript
  document.getElementById('export-btn').addEventListener('click', () => {
    const records = Object.values(BADCASES);
    const jsonl = records.map(r => JSON.stringify(r)).join('\n') + '\n';
    const blob = new Blob([jsonl], {type: 'application/x-ndjson'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `badcases-${new Date().toISOString().slice(0,10)}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  });
```

### 3.3 Verify dashboard changes (without detail card yet)

- [ ] **Step 6: Regenerate viz + start server**

```bash
rm -rf out/viz_regional
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional 2>&1 | tail -5
lsof -ti:8765 2>/dev/null | xargs -r kill 2>/dev/null
cd out/viz_regional && nohup python3 viz_server.py --port 8765 --user yinz > /tmp/_viz_server.log 2>&1 &
echo $! > /tmp/_viz_server.pid
cd -
sleep 1
curl -fsS http://localhost:8765/api/badcases -w "\napi: %{http_code}\n"
curl -fsS http://localhost:8765/ -o /dev/null -w "static: %{http_code}\n"
```

Expected: api returns `{"badcases": []}` 200, static returns 200.

- [ ] **Step 7: Inject a test badcase + verify dashboard renders 🔴**

```bash
SAMPLE_SHA=$(.venv/bin/python -c "import json; print(json.load(open('out/viz_regional/viz_data.json'))['rows'][0]['sha256'])")
curl -fsS -X POST http://localhost:8765/api/badcase \
  -H 'Content-Type: application/json' \
  -d "{\"sha256\":\"$SAMPLE_SHA\",\"stage\":\"overall\",\"tags\":[\"smoke\"],\"note\":\"test\"}"
echo ""
echo "Now open http://localhost:8765/ — the first row should have 🔴; 'bad only' filter shows just 1."
```

Manually verify in browser. Then clean up:

```bash
curl -fsS -X DELETE "http://localhost:8765/api/badcase/$SAMPLE_SHA"
echo ""
```

- [ ] **Step 8: Commit (dashboard half only — detail card comes in Task 4)**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): dashboard 🚩 column + bad-only filter + Export button + /api/badcases boot fetch"
```

---

## Task 4: Detail view — Bad case flag card

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html`

- [ ] **Step 1: Add helper functions for badcase form rendering + API calls**

Inside the main `<script type="module">`, near the existing helper functions (`badge`, `fmt`, etc.), add:

```javascript
  // ---------- badcase helpers ----------
  async function saveBadcase(sha, stage, tags, note) {
    const res = await fetch('/api/badcase', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sha256: sha, stage, tags, note}),
    });
    if (!res.ok) throw new Error(`save failed: HTTP ${res.status} ${await res.text()}`);
    return await res.json();
  }

  async function deleteBadcase(sha) {
    const res = await fetch(`/api/badcase/${encodeURIComponent(sha)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error(`delete failed: HTTP ${res.status} ${await res.text()}`);
    return await res.json();
  }

  function badcaseCardHtml(row) {
    const bc = BADCASES[row.sha256];
    const stages = ['router', 'layout', 'extract', 'quality', 'overall'];
    const currentStage = bc?.stage || 'overall';
    const currentTags = bc?.tags || [];
    const currentNote = bc?.note || '';
    const isFlagged = !!bc;

    const stageRadios = stages.map(s => `
      <label style="margin-right:10px;font-size:11px;cursor:pointer">
        <input type="radio" name="bc-stage" value="${s}" ${s === currentStage ? 'checked' : ''}>
        ${s}
      </label>`).join('');

    const tagChips = currentTags.map(t => `
      <span class="bc-chip" data-tag="${t.replace(/"/g, '&quot;')}">
        ${t}
        <button class="bc-chip-x" data-tag="${t.replace(/"/g, '&quot;')}" title="remove">×</button>
      </span>`).join('');

    const flaggedMeta = isFlagged
      ? `<div style="color:var(--dim);font-size:11px;margin-top:8px">
           flagged ${bc.flagged_at} by ${bc.flagged_by || 'anon'}
         </div>`
      : '';

    return `
      <div class="stage-card" style="border-left-color:${isFlagged ? 'var(--red)' : 'var(--yellow)'}">
        <h3 style="color:${isFlagged ? 'var(--red)' : 'var(--yellow)'}">
          ${isFlagged ? '🔴 Bad case flag' : '🚩 Bad case flag'}
        </h3>
        <div class="kv"><span class="k">Stage where it broke</span>
          <span class="v" id="bc-stage-row">${stageRadios}</span></div>
        <div class="kv"><span class="k">Tags</span>
          <span class="v">
            <span id="bc-tags-host">${tagChips}</span>
            <input type="text" id="bc-tag-input" placeholder="add tag…"
                   style="margin-left:6px;padding:2px 6px;border:1px solid #444;
                          background:#111;color:var(--text);font-size:11px;width:120px;border-radius:3px">
          </span></div>
        <div class="kv"><span class="k">Note</span>
          <span class="v" style="width:100%">
            <textarea id="bc-note" rows="3"
                      style="width:100%;background:#111;color:var(--text);border:1px solid #444;
                             padding:6px;font-size:12px;font-family:inherit;border-radius:3px;resize:vertical"
                      placeholder="what's wrong with this row?">${currentNote.replace(/</g, '&lt;')}</textarea>
          </span></div>
        ${flaggedMeta}
        <div id="bc-actions" style="margin-top:8px;display:flex;gap:6px">
          <button id="bc-save" style="padding:4px 12px;background:var(--green);color:#fff;
                                       border:none;border-radius:3px;cursor:pointer;font-size:12px">
            💾 Save
          </button>
          ${isFlagged ? `<button id="bc-remove" style="padding:4px 12px;background:var(--red);color:#fff;
                                                         border:none;border-radius:3px;cursor:pointer;font-size:12px">
            🗑 Remove
          </button>` : ''}
          <span id="bc-status" style="margin-left:auto;font-size:11px;color:var(--dim)"></span>
        </div>
      </div>`;
  }

  function attachBadcaseHandlers(row) {
    // Tag chip removal.
    const tagsHost = document.getElementById('bc-tags-host');
    tagsHost.addEventListener('click', (ev) => {
      const btn = ev.target.closest('.bc-chip-x');
      if (!btn) return;
      btn.parentElement.remove();
    });

    // Tag input — Enter to add.
    document.getElementById('bc-tag-input').addEventListener('keydown', (ev) => {
      if (ev.key !== 'Enter') return;
      ev.preventDefault();
      const v = ev.target.value.trim();
      if (!v) return;
      const chip = document.createElement('span');
      chip.className = 'bc-chip';
      chip.dataset.tag = v;
      chip.innerHTML = `${v} <button class="bc-chip-x" data-tag="${v.replace(/"/g, '&quot;')}" title="remove">×</button>`;
      tagsHost.appendChild(chip);
      ev.target.value = '';
    });

    // Save.
    document.getElementById('bc-save').addEventListener('click', async () => {
      const stage = document.querySelector('input[name="bc-stage"]:checked')?.value || 'overall';
      const tags = Array.from(document.querySelectorAll('#bc-tags-host .bc-chip'))
                        .map(c => c.dataset.tag);
      const note = document.getElementById('bc-note').value;
      const status = document.getElementById('bc-status');
      status.textContent = 'saving…'; status.style.color = 'var(--dim)';
      try {
        const rec = await saveBadcase(row.sha256, stage, tags, note);
        BADCASES[row.sha256] = rec;
        // Refresh detail card AND dashboard 🚩.
        rerenderBadcaseCard(row);
        renderTable();
        // Refresh top-bar flagged count.
        renderDashboard();
        status.textContent = 'saved'; status.style.color = 'var(--green)';
      } catch (e) {
        status.textContent = e.message; status.style.color = 'var(--red)';
      }
    });

    // Remove (only present when already flagged).
    const removeBtn = document.getElementById('bc-remove');
    if (removeBtn) {
      removeBtn.addEventListener('click', async () => {
        const status = document.getElementById('bc-status');
        status.textContent = 'removing…'; status.style.color = 'var(--dim)';
        try {
          await deleteBadcase(row.sha256);
          delete BADCASES[row.sha256];
          rerenderBadcaseCard(row);
          renderTable();
          renderDashboard();
          status.textContent = 'removed'; status.style.color = 'var(--green)';
        } catch (e) {
          status.textContent = e.message; status.style.color = 'var(--red)';
        }
      });
    }
  }

  function rerenderBadcaseCard(row) {
    // Replace the existing card in-place. The card lives in #detail-stages
    // somewhere; we identify it by its h3 text. Simpler: re-render all stage
    // cards by calling renderStages.
    renderStages(row);
  }
```

- [ ] **Step 2: Add CSS for `.bc-chip`**

In the `<style>` block, add (anywhere among the existing class rules):

```css
  .bc-chip { display: inline-flex; align-items: center; gap: 4px;
             background: rgba(255,193,7,0.15); color: var(--yellow);
             padding: 2px 6px 2px 8px; border-radius: 10px;
             margin-right: 4px; font-size: 11px; }
  .bc-chip-x { background: transparent; border: none; color: var(--yellow);
               cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1; }
  .bc-chip-x:hover { color: var(--red); }
```

- [ ] **Step 3: Insert the badcase card into `renderStages()`**

Find `async function renderStages(row) {`. After the Quality card push (before the optional error card), insert the badcase card. Locate:

```javascript
    parts.push(`
      <div class="stage-card"><h3>Quality</h3>
        ...
      </div>`);

    if (row.error_class) {
      parts.push(`
        <div class="stage-card" style="border-left-color:var(--red)"><h3 style="color:var(--red)">Error</h3>
        ...
```

Between those two `push` calls, insert:

```javascript
    parts.push(badcaseCardHtml(row));
```

Then at the end of `renderStages(row)`, AFTER the existing `el.innerHTML = parts.join('');` line (and after the markdown loading block), append:

```javascript
    // Wire up badcase form handlers — must run after innerHTML is set.
    attachBadcaseHandlers(row);
```

- [ ] **Step 4: Verify HTML balance**

```bash
.venv/bin/python -c "
html = open('packages/pdfsys-bench/viz/index.html').read()
import re
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
b_open = sum(s.count('{') for s in scripts)
b_close = sum(s.count('}') for s in scripts)
p_open = sum(s.count('(') for s in scripts)
p_close = sum(s.count(')') for s in scripts)
print(f'braces: {b_open}/{b_close}, parens: {p_open}/{p_close}, lines: {html.count(chr(10))+1}')
assert b_open == b_close and p_open == p_close, 'unbalanced'
print('balanced')
print('badcaseCardHtml defined:', 'function badcaseCardHtml' in html)
print('attachBadcaseHandlers defined:', 'function attachBadcaseHandlers' in html)
"
```

Expected: balanced + both functions defined.

- [ ] **Step 5: Smoke test in browser**

Restart server with new bundle:

```bash
rm -rf out/viz_regional
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional 2>&1 | tail -3
lsof -ti:8765 2>/dev/null | xargs -r kill 2>/dev/null
cd out/viz_regional && nohup python3 viz_server.py --port 8765 --user yinz > /tmp/_viz_server.log 2>&1 &
echo $! > /tmp/_viz_server.pid
cd -
sleep 1
curl -fsS http://localhost:8765/ -o /dev/null -w "static: %{http_code}\n"
```

Manual browser checks:
- Dashboard loads, 🚩 column visible (all empty initially).
- Click any row → detail view shows "🚩 Bad case flag" card (yellow border).
- Pick a stage radio, type a tag + Enter, type a note, click Save.
- Status changes to "saved" (green).
- Dashboard → back → that row now has 🔴.
- Toggle "🔴 bad only" filter → only that row shows.
- Re-open the row → card shows red border + "🔴 Bad case flag" + populated fields + 🗑 Remove button.
- Click Remove → card flips back to yellow.
- Click 📥 Export → downloads `badcases-<date>.jsonl`.

- [ ] **Step 6: Kill server + restart + verify persistence**

```bash
SAMPLE_SHA=$(.venv/bin/python -c "import json; print(json.load(open('out/viz_regional/viz_data.json'))['rows'][1]['sha256'])")
curl -fsS -X POST http://localhost:8765/api/badcase \
  -H 'Content-Type: application/json' \
  -d "{\"sha256\":\"$SAMPLE_SHA\",\"stage\":\"extract\",\"tags\":[\"persistence-test\"],\"note\":\"survive restart?\"}" \
  -w "\npost: %{http_code}\n"

lsof -ti:8765 | xargs kill
sleep 1

cd out/viz_regional && nohup python3 viz_server.py --port 8765 --user yinz > /tmp/_viz_server.log 2>&1 &
echo $! > /tmp/_viz_server.pid
cd -
sleep 1

curl -fsS http://localhost:8765/api/badcases | .venv/bin/python -m json.tool
```

Expected: GET returns 1 badcase with the persistence-test tag, is_bad=true.

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): bad-case flag detail card with stage/tags/note + persistence"
```

---

## Task 5: Final acceptance + post-build note

**Files:**
- Modify: `docs/superpowers/specs/2026-05-18-viz-badcase-flagging-design.md` (§12)

- [ ] **Step 1: Run the full §11 definition-of-done checklist manually**

Server should already be running from Task 4. Clear any existing badcases:

```bash
rm -f out/viz_regional/badcases.jsonl
lsof -ti:8765 | xargs -r kill
cd out/viz_regional && nohup python3 viz_server.py --port 8765 --user yinz > /tmp/_viz_server.log 2>&1 &
echo $! > /tmp/_viz_server.pid
cd -
sleep 1
```

Then in browser:
1. Flag 5 different rows (each with a different `stage` value). Save each.
2. Verify dashboard shows 5 🔴.
3. Toggle "🔴 bad only" filter — exactly 5 rows visible.
4. Click 📥 Export — downloads 5-line jsonl.
5. Kill server (`lsof -ti:8765 | xargs kill`).
6. Restart server (`cd out/viz_regional && python3 viz_server.py --port 8765 --user yinz &`).
7. Reload browser — still 5 🔴.
8. Delete 1 flag via UI → dashboard shows 4 🔴.
9. Reload → 4 🔴 persists.
10. `cat out/viz_regional/badcases.jsonl` — text is human-readable, UTF-8.
11. Re-run `.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional`. File preserved.
12. Reload browser → 4 🔴 still there.

- [ ] **Step 2: Append §12 post-build note to spec**

Edit `docs/superpowers/specs/2026-05-18-viz-badcase-flagging-design.md`. Replace `## 12 · Post-build note (to be filled in)` with:

```markdown
## 12 · Post-build note · 2026-05-18

### Server
- `viz_server.py` total LOC: <N>
- No new dependencies (stdlib only).
- Port 8765 default; --user CLI flag works (verified `flagged_by` matches).
- atomic write via fcntl.flock + os.fsync — verified by hand on macOS 25.4.

### Acceptance walk-through (§11 checklist)
- ✅ 5 flags created across 5 stages
- ✅ 🚩 column shows 🔴 for those rows
- ✅ "🔴 bad only" reduces dashboard to 5 rows
- ✅ 📥 Export downloads 5-line JSONL (size ≈ <N> bytes)
- ✅ Server kill + restart preserves all 5
- ✅ Delete brings count to 4 across restart
- ✅ Re-running `pdfsys visualize -r ... -o out/viz_regional` preserves badcases.jsonl

### Notable observations
- (any surprises during build — leave blank if none)

### Open follow-ups
- `pdfsys badcase-export` CLI to turn badcases.jsonl into structured training data.
- Optionally add `is_bad: false` ("reviewed not bad") UX path; current Remove just blanks the record.
- Multi-bundle aggregation: combine badcases.jsonl across runs by sha → unified curation set.
```

Fill in `<N>` placeholders with actual values.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-18-viz-badcase-flagging-design.md
git commit -m "docs(spec): viz bad-case flagging post-build note"
```

---

## Self-review notes

**Spec coverage:**
- §3 architecture (server + bundle) → Tasks 1 + 2
- §4 jsonl schema → Task 1 (server validates & writes the schema)
- §5 server API (3 endpoints + static fallthrough) → Task 1
- §6 viz.py CLI changes (copy server, preserve jsonl, rewrite serve.sh) → Task 2
- §7 index.html changes (dashboard column / filter / Export, detail card) → Tasks 3 + 4
- §8 workflow → covered implicitly by Task 5 acceptance
- §11 definition of done → Task 5 step 1 walks each bullet

**Placeholders:** §12 has `<N>` fill-in template — explicit instruction to replace with actuals. No other "TBD" / "implement later" anywhere.

**Type consistency:**
- `BADCASES` keyed by `sha256` (string) → list of records (one per sha after browser-side dedup).
- Server returns `{badcases: [...]}` shape — matches the boot fetch + the `Object.fromEntries(bc.badcases.filter...)` consumer.
- `stage` enum values used identically in Task 1 (`VALID_STAGES`) and Task 4 (`stages = ['router', ...]`).
- POST request body shape `{sha256, stage, tags, note}` — sent by `saveBadcase()` in Task 4, accepted by `do_POST()` in Task 1.

**One soft assumption:** Browser fetch `/api/badcases` is from the same origin as the page, so no CORS handling needed. If the user opens `index.html` via `file://` instead of through the server, the fetch will fail and BADCASES stays empty — UI still works for browsing but flagging is disabled. Acceptable per spec (server is required for this feature).
