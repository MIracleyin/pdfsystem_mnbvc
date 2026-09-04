"""End-to-end check of the split pipeline, on a corpus small enough to be free.

Validating the CPU/GPU split by hand means a run, a worklist, a second run, a
scoring pass and two packaging passes — six commands whose failure modes are
mostly "produced nothing and said nothing". Doing that against a real corpus
costs minutes and a GPU; doing it against 50 real PDFs still costs enough that
nobody does it every time.

So this generates a corpus of eight tiny PDFs — a few born-digital, a few
image-only, plus the shapes that have actually broken things here (an uppercase
suffix, no suffix at all, an encrypted file, a byte-identical duplicate) — and
runs the whole four-phase flow over it. With no URLs it stands up in-process
stubs for mineru-api and the quality scorer, so it needs no GPU, no model
weights and no network, and finishes in seconds. Point ``--mineru-url`` and
``--quality-url`` at real services and the same check validates a real
deployment.

It asserts the things that have gone wrong before: that every document lands in
exactly one lane, that the handed-over ones are packaged by the lane that
extracted them, that the merged dataset validates, and that no document appears
in two shards.
"""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

__all__ = ["SmokeResult", "build_corpus", "cmd_smoke", "run_smoke"]


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def build_corpus(root: Path) -> dict[str, Path]:
    """Write the smallest corpus that still exercises every lane and edge case.

    Each entry is here because it broke something: the uppercase suffix and the
    extensionless file were invisible to the old glob, the encrypted file used
    to abort a whole packaging run, and the duplicate is what the shard's
    primary key is most easily violated by.
    """
    import pymupdf

    root.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}

    def _text_pdf(path: Path, seed: str) -> Path:
        doc = pymupdf.open()
        page = doc.new_page()
        for i in range(28):
            page.insert_text(
                (60, 60 + i * 22),
                f"{seed} line {i}: ordinary body text with a real text layer.",
                fontsize=11,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path)
        doc.close()
        return path

    def _scan_pdf(path: Path, seed: str) -> Path:
        """Real words, rendered to pixels, with no text layer left behind.

        A page of grey bars also routes to OCR — the router only sees the
        absent text layer — but a real MinerU correctly returns nothing for it,
        and the check would then be measuring the stub rather than the service.
        So the glyphs have to actually be there.
        """
        src = pymupdf.open()
        page = src.new_page(width=420, height=560)
        for i, line in enumerate([
            f"SCANNED {seed.upper()}",
            "",
            "This page has no text layer.",
            "The words exist only as pixels,",
            "which is what OCR is for.",
        ]):
            page.insert_text((40, 60 + i * 30), line, fontsize=15)
        pix = page.get_pixmap(dpi=110, colorspace=pymupdf.csGRAY)
        src.close()

        doc = pymupdf.open()
        doc.new_page(width=420, height=560).insert_image(
            pymupdf.Rect(0, 0, 420, 560), pixmap=pix
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path, deflate=True)
        doc.close()
        return path

    made["text_a"] = _text_pdf(root / "a.pdf", "alpha")
    made["text_b"] = _text_pdf(root / "nested" / "b.pdf", "beta")
    made["text_upper"] = _text_pdf(root / "UPPER.PDF", "gamma")
    made["text_noext"] = _text_pdf(root / "extensionless", "delta")
    made["scan_a"] = _scan_pdf(root / "scan-a.pdf", "epsilon")
    made["scan_b"] = _scan_pdf(root / "nested" / "scan-b.pdf", "zeta")

    # A byte-identical copy under another name: one doc_id, two paths.
    made["duplicate"] = root / "copy-of-a.pdf"
    made["duplicate"].write_bytes(made["text_a"].read_bytes())

    # Encrypted: the router cannot open it. Certain to exist at corpus scale.
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 60), "locked", fontsize=12)
    doc.save(
        root / "locked.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="o",
        user_pw="u",
    )
    doc.close()
    made["encrypted"] = root / "locked.pdf"

    (root / "notes.txt").write_text("not a pdf at all", encoding="utf-8")
    made["decoy"] = root / "notes.txt"
    return made


# ---------------------------------------------------------------------------
# in-process stand-ins for the GPU services
# ---------------------------------------------------------------------------


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _json_handler(post: Any, get: Any) -> type[BaseHTTPRequestHandler]:
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # keep the output readable
            pass

        def _send(self, obj: Any) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(get())

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            self._send(post(self.rfile.read(n), self.headers.get("Content-Type", "")))

    return _H


SMOKE_MODEL = "smoke/stub-scorer"


@contextmanager
def stub_services(mineru_url: str | None, quality_url: str | None):
    """Yield (mineru_url, quality_url), standing up stubs for whichever is None."""
    servers: list[HTTPServer] = []
    try:
        if mineru_url is None:
            def _parse(body: bytes, ctype: str) -> dict:
                return {
                    "status": "completed",
                    "version": "smoke-stub",
                    "results": {"f.pdf": {
                        "md_content": "# Scanned page\n\nRecognised body text.",
                        "middle_json": {"backend": "pipeline",
                                        "pdf_info": [{"page_size": [612.0, 792.0]}]},
                        "content_list": [
                            {"type": "text", "text": "Scanned page", "page_idx": 0},
                            {"type": "text", "text": "Recognised body text.",
                             "page_idx": 0},
                        ],
                        "images": {},
                    }},
                }

            server, mineru_url = _serve(_json_handler(_parse, lambda: {"ok": True}))
            servers.append(server)

        if quality_url is None:
            def _score(body: bytes, ctype: str) -> dict:
                text = json.loads(body)["text"]
                return {"score": round(min(3.0, len(text) / 60), 3),
                        "num_chars": len(text), "num_tokens": len(text) // 4,
                        "model": SMOKE_MODEL}

            server, quality_url = _serve(
                _json_handler(_score, lambda: {"ok": True, "model": SMOKE_MODEL})
            )
            servers.append(server)

        yield mineru_url, quality_url
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    steps: list[tuple[str, bool, str]] = field(default_factory=list)
    #: Everything the sub-commands printed, kept for when something fails.
    chatter: str = ""

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.steps)

    def report(self) -> str:
        return "\n".join(
            f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else "")
            for name, ok, detail in self.steps
        )


def run_smoke(
    workdir: Path,
    *,
    mineru_url: str | None = None,
    quality_url: str | None = None,
    keep: bool = False,
    verbose: bool = False,
) -> SmokeResult:
    """Run all four phases over a generated corpus. Returns what held and what did not."""
    import contextlib
    import io
    import os

    from .__main__ import main as _main
    from .dataset_validate import validate_shard

    res = SmokeResult()
    workdir = Path(workdir)
    corpus = workdir / "corpus"
    build_corpus(corpus)

    # Six sub-commands' worth of normal chatter would bury the eleven lines that
    # matter. It is kept, and printed when something fails.
    chatter = io.StringIO()

    def main(argv: list[str]) -> int:
        if verbose:
            return _main(argv)
        with (
            contextlib.redirect_stdout(chatter),
            contextlib.redirect_stderr(chatter),
        ):
            return _main(argv)

    with stub_services(mineru_url, quality_url) as (m_url, q_url):
        env = dict(os.environ)
        os.environ.update(
            MINERU_PIPELINE_URL=m_url, MINERU_VLM_URL=m_url, QUALITY_URL=q_url,
            NO_PROXY="*", no_proxy="*",
        )
        try:
            _phases(
                workdir, corpus, res, validate_shard, main,
                expect_model=None if quality_url else SMOKE_MODEL,
            )
        finally:
            os.environ.clear()
            os.environ.update(env)

    res.chatter = chatter.getvalue()
    if not keep:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    return res


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8", errors="surrogatepass").splitlines()
        if x.strip()
    ]


def _phases(
    workdir: Path, corpus: Path, res: SmokeResult, validate_shard, main,
    expect_model: str | None = None,
) -> None:
    p1, p2, ds = workdir / "p1", workdir / "p2", workdir / "dataset"

    # -- Phase 1: CPU lane -------------------------------------------------
    rc = main([
        "run", "--pdf-dir", str(corpus), "--out-dir", str(p1),
        "--stages", "router,extract", "--extract-backends", "mupdf",
        "--markdown-dir", "markdown", "--ocr-threshold", "0.05",
    ])
    rows1 = _rows(p1 / "results.jsonl") if (p1 / "results.jsonl").exists() else []
    mine = [r for r in rows1 if r["extract_backend"] == "mupdf" and not r["skip_reason"]
            and not r["error_class"]]
    handed = [r for r in rows1 if r["skip_reason"] == "lane-filter"]
    res.record(
        "phase 1 (CPU lane)", rc == 0 and bool(mine) and bool(handed),
        f"{len(rows1)} discovered, {len(mine)} extracted, {len(handed)} handed over",
    )
    # The uppercase and extensionless files exist only to prove discovery.
    names = {Path(r["pdf_path"]).name for r in rows1}
    res.record(
        "discovery finds .PDF and extensionless",
        {"UPPER.PDF", "extensionless"} <= names,
        f"missing {sorted({'UPPER.PDF', 'extensionless'} - names)}" if
        not {"UPPER.PDF", "extensionless"} <= names else "",
    )
    res.record(
        "the decoy .txt is not a document", "notes.txt" not in names,
    )

    # -- handoff -----------------------------------------------------------
    gpu_list, cpu_list = workdir / "gpu_lane.txt", workdir / "cpu_lane.txt"
    gpu_list.write_text("".join(r["pdf_path"] + "\n" for r in handed), encoding="utf-8")
    cpu_list.write_text("".join(r["pdf_path"] + "\n" for r in mine), encoding="utf-8")

    # -- Phase 2: GPU lane -------------------------------------------------
    rc = main([
        "run", "--pdf-list", str(gpu_list), "--out-dir", str(p2),
        "--stages", "router,extract", "--extract-backends", "pipeline",
        "--parser-output-dir", str(p2 / "mineru"), "--markdown-dir", "markdown",
        "--ocr-threshold", "0.05",
    ])
    rows2 = _rows(p2 / "results.jsonl") if (p2 / "results.jsonl").exists() else []
    got = [r for r in rows2 if r["extract_backend"] == "pipeline" and not r["error_class"]]
    stranded = [r for r in rows2 if r["skip_reason"] == "lane-filter"]
    res.record(
        "phase 2 (GPU lane)", rc == 0 and len(got) == len(handed) and not stranded,
        f"{len(got)}/{len(handed)} extracted"
        + (f", {len(stranded)} stranded — thresholds disagree" if stranded else ""),
    )
    sidecars = list((p2 / "mineru").glob("*/*_content_list.json"))
    res.record(
        # `0 == 0` is not a pass: with nothing extracted there is nothing to
        # have persisted, and the check would be vacuously green.
        "MinerU sidecars persisted", bool(got) and len(sidecars) == len(got),
        f"{len(sidecars)} content lists for {len(got)} extractions",
    )

    # -- Phase 3: score both lanes ----------------------------------------
    scored_ok = True
    for name, out_dir in (("CPU", p1), ("GPU", p2)):
        rc = main([
            "score", "--results", str(out_dir / "results.jsonl"),
            "--markdown-dir", str(out_dir / "markdown"),
            "--out", str(out_dir / "results.scored.jsonl"),
            # Only assert a model name when we know it: a real server serves a
            # real model, and demanding the stub's would fail every time.
            *(("--model", expect_model) if expect_model else ()),
        ])
        n = sum(1 for r in _rows(out_dir / "results.scored.jsonl")
                if r.get("quality_score") is not None) if rc == 0 else 0
        scored_ok = scored_ok and rc == 0 and n > 0
        res.record(f"phase 3 (score {name} lane)", rc == 0 and n > 0, f"{n} scored")

    # -- Phase 4: package each lane ---------------------------------------
    rc_cpu = main([
        "dataset", "--from-pdf-list", str(cpu_list), "--to", str(ds),
        "--shard", "cpu-00", "--images", "none",
        "--meta", str(p1 / "results.scored.jsonl"),
    ])
    rc_gpu = main([
        "dataset", "--from-mineru", str(p2 / "mineru"), "--to", str(ds),
        "--shard", "gpu-00", "--images", "none",
        "--meta", str(p2 / "results.scored.jsonl"),
    ])
    res.record("phase 4 (package both lanes)", rc_cpu == 0 and rc_gpu == 0)

    # -- the guard, and the contract --------------------------------------
    rc_bad = main([
        "dataset", "--from-pdf-dir", str(corpus), "--to", str(workdir / "bad"),
        "--shard", "x", "--images", "none",
        "--meta", str(p1 / "results.scored.jsonl"),
    ])
    res.record(
        "scanning the whole corpus is refused", rc_bad == 1,
        "" if rc_bad == 1 else "it packaged the other lane's documents",
    )

    report = validate_shard(ds) if (ds / "pages").exists() else None
    res.record(
        "merged dataset validates", bool(report and report.ok),
        "" if report and report.ok else
        "; ".join(str(f) for f in (report.findings[:3] if report else [])),
    )

    if report:
        import pyarrow.parquet as pq

        ids = [r["doc_id"] for p in sorted((ds / "pages").glob("*.parquet"))
               for r in pq.read_table(p).to_pylist()]
        res.record(
            "no document in two shards", len(ids) == len(set(ids)),
            f"{len(set(ids))} distinct of {len(ids)} rows",
        )


def cmd_smoke(args: Any) -> int:
    """CLI entry point for ``pdfsys smoke``."""
    import tempfile

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="pdfsys-smoke-"))
    keep = bool(args.workdir)

    where = []
    if args.mineru_url:
        where.append(f"mineru={args.mineru_url}")
    if args.quality_url:
        where.append(f"quality={args.quality_url}")
    print(f"[pdfsys smoke] {' '.join(where) if where else 'in-process stubs'}")
    print(f"[pdfsys smoke] workdir: {workdir}")

    result = run_smoke(
        workdir, mineru_url=args.mineru_url, quality_url=args.quality_url,
        keep=keep, verbose=args.verbose,
    )
    print(result.report())

    if result.ok:
        print("[pdfsys smoke] all checks passed")
        return 0
    failed = sum(1 for _, ok, _ in result.steps if not ok)
    if result.chatter:
        print("\n[pdfsys smoke] what the commands said:", file=sys.stderr)
        print(result.chatter, file=sys.stderr)
    print(f"[pdfsys smoke] {failed} check(s) failed", file=sys.stderr)
    return 1
