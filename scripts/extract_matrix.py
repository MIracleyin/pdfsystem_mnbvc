"""Run each PDF through all three parser backends (mupdf, pipeline, vlm).

Produces a "parser × PDF" matrix dataset for the downstream quality-
score teammate: each PDF gets up to three candidate Markdown
extractions (one per backend), and the consumer can label which one is
best or score each on absolute quality.

This is NOT a benchmark — there's no cascade, no router, no quality
scoring. Just the raw extraction matrix.

Output:
  <out>             JSONL, one row per (PDF, parser) tuple
  <markdown_dir>/   <sha256>__<parser>.md per successful extraction

Usage:
  uv run python scripts/extract_matrix.py \\
      --pdf-dir /tmp/bench-150 \\
      --out out/annotation-set/results.jsonl \\
      --markdown-dir out/annotation-set/markdown \\
      --vlm-engine mlx-engine

Time on Apple Silicon (mlx engine), 150 PDFs:
  mupdf:    ~2 s     (in-process PyMuPDF)
  pipeline: ~10 min  (mineru-api subprocess + per-PDF parse)
  vlm:      ~20 min  (mineru-api MLX subprocess + per-PDF parse)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from pdfsys_parser_mupdf import extract_doc as mupdf_extract
from pdfsys_parser_pipeline import PipelineParser
from pdfsys_parser_vlm import VlmParser
from pdfsys_types import PipelineConfig, VlmConfig

_LOG = logging.getLogger("extract_matrix")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_one(
    parser_name: str,
    fn: Callable,
    pdf: Path,
    md_dir: Path,
    sha_fallback: str,
) -> dict:
    """Run one parser on one PDF; return the row dict, write markdown file."""
    t0 = time.monotonic()
    err: str | None = None
    md = ""
    sha = sha_fallback
    char_count = 0
    try:
        doc = fn(pdf)
        sha = doc.sha256
        md = doc.markdown
        char_count = len(md)
    except Exception as e:  # noqa: BLE001 — we want every parser-side failure captured
        err = f"{type(e).__name__}: {e}"

    wall_ms = (time.monotonic() - t0) * 1000.0
    markdown_file: str | None = None
    if md:
        out = md_dir / f"{sha}__{parser_name}.md"
        out.write_text(md, encoding="utf-8")
        markdown_file = out.name

    return {
        "parser": parser_name,
        "char_count": char_count,
        "wall_ms": wall_ms,
        "error": err,
        "markdown_file": markdown_file,
        "sha256": sha,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="extract_matrix")
    p.add_argument("--pdf-dir", type=Path, required=True,
                   help="Directory of PDFs (rglob).")
    p.add_argument("--out", type=Path, required=True,
                   help="JSONL output path.")
    p.add_argument("--markdown-dir", type=Path, required=True,
                   help="Directory to dump <sha256>__<parser>.md per "
                        "successful extraction.")
    p.add_argument("--vlm-engine", default="mlx-engine",
                   choices=["transformers", "mlx-engine", "vllm-engine"],
                   help="Mineru VLM backend. Default mlx-engine (Apple Silicon).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of PDFs processed (for smoke tests).")
    p.add_argument("--skip-mupdf", action="store_true")
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in args.pdf_dir.rglob("*.pdf") if p.is_file())
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"error: no *.pdf found under {args.pdf_dir}", file=sys.stderr)
        return 1
    print(f"[matrix] {len(pdfs)} PDFs × "
          f"{sum(not s for s in (args.skip_mupdf, args.skip_pipeline, args.skip_vlm))} "
          f"parsers", flush=True)

    pipeline = None if args.skip_pipeline else PipelineParser(PipelineConfig())
    vlm = None
    if not args.skip_vlm:
        # mineru 3.4+ accepts only the short backend names: pipeline,
        # vlm-engine, hybrid-engine, vlm-http-client, hybrid-http-client.
        # The VlmParser prepends "vlm-" to config.engine before posting
        # to /file_parse, so we have to translate the user-facing
        # "vllm-engine" / "transformers" choices to the short form
        # "engine" → posted as "vlm-engine". The "mlx-engine" case is
        # different on Apple Silicon: pre-3.4 mineru still uses
        # "vlm-mlx-engine" there; we keep that path until that mineru
        # version forces an update.
        engine_for_config = "engine" if args.vlm_engine == "vllm-engine" else args.vlm_engine
        vlm = VlmParser(VlmConfig(engine=engine_for_config))

    t_total = time.time()
    n_rows = 0
    try:
        with args.out.open("w", encoding="utf-8") as f:
            for i, pdf in enumerate(pdfs, 1):
                sha = _sha256_file(pdf)
                t_pdf = time.time()
                base = {
                    "file_id": sha,
                    "filename": pdf.name,
                    "pdf_path": str(pdf),
                }

                if not args.skip_mupdf:
                    row = _run_one("mupdf", mupdf_extract, pdf, args.markdown_dir, sha)
                    f.write(json.dumps({**base, **row}, ensure_ascii=False) + "\n")
                    f.flush()
                    n_rows += 1
                if pipeline is not None:
                    row = _run_one("pipeline", pipeline.extract, pdf, args.markdown_dir, sha)
                    f.write(json.dumps({**base, **row}, ensure_ascii=False) + "\n")
                    f.flush()
                    n_rows += 1
                if vlm is not None:
                    row = _run_one("vlm", vlm.extract, pdf, args.markdown_dir, sha)
                    f.write(json.dumps({**base, **row}, ensure_ascii=False) + "\n")
                    f.flush()
                    n_rows += 1

                pdf_wall = time.time() - t_pdf
                print(f"[matrix] {i}/{len(pdfs)} {sha[:8]} "
                      f"({pdf_wall:.1f}s) {pdf.name[:60]}", flush=True)
    finally:
        # Always tear down the mineru-api subprocesses we spawned.
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception as e:  # noqa: BLE001
                _LOG.warning("pipeline.close() raised: %s", e)
        if vlm is not None:
            try:
                vlm.close()
            except Exception as e:  # noqa: BLE001
                _LOG.warning("vlm.close() raised: %s", e)

    wall = time.time() - t_total
    print(
        f"[matrix] done — {len(pdfs)} PDFs, {n_rows} rows, "
        f"{wall:.1f}s ({wall/len(pdfs):.1f}s/pdf avg)",
        flush=True,
    )
    print(f"[matrix] jsonl:    {args.out}", flush=True)
    print(f"[matrix] markdown: {args.markdown_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
