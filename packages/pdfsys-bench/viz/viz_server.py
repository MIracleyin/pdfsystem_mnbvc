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

USER = os.environ.get("USER", "anon")


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


class VizHandler(BaseHTTPRequestHandler):

    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, status: HTTPStatus, body) -> None:
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
        url_path = self.path.split("?", 1)[0]
        if url_path in ("", "/"):
            url_path = "/index.html"
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
            self._send_text(HTTPStatus.BAD_REQUEST, f"stage must be one of {VALID_STAGES}")
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
            "flagged_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
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
            "flagged_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "flagged_by": USER,
        }
        _append_badcase(rec)
        self._send_json(HTTPStatus.OK, rec)


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
