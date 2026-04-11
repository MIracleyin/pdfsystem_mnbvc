"""pdfsys-bench CLI — run the MVP closed loop on a directory of PDFs.

Usage::

    python -m pdfsys_bench \\
        --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \\
        --out out/bench_omnidoc100.jsonl \\
        --limit 20

Flags exposed here are intentionally minimal — anything more is the job
of a proper runner package. This CLI is meant for smoke-testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loop import run_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdfsys-bench", description="Run the MVP pdfsys closed loop.")
    p.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory of PDFs to process (recursive).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path (one line per PDF).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of PDFs processed. Default: no cap.",
    )
    p.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip the ModernBERT quality scorer (fast smoke test).",
    )
    p.add_argument(
        "--quality-model",
        default="HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn",
        help="HuggingFace repo id for the quality scorer.",
    )
    p.add_argument(
        "--router-weights",
        type=Path,
        default=None,
        help="Path to xgb_classifier.ubj. Defaults to the package's bundled path.",
    )
    p.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help="Optional directory to dump per-PDF extracted markdown.",
    )
    p.add_argument(
        "--ocr-threshold",
        type=float,
        default=0.5,
        help="P(ocr) threshold above which a PDF is routed off the text-ok path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_loop(
        pdf_dir=args.pdf_dir,
        out_path=args.out,
        limit=args.limit,
        score_quality=not args.no_quality,
        router_weights=args.router_weights,
        quality_model=args.quality_model,
        markdown_dir=args.markdown_dir,
        ocr_threshold=args.ocr_threshold,
    )

    print(f"[pdfsys-bench] processed {summary['num_pdfs']} PDFs in {summary['wall_seconds']:.1f}s")
    print(f"[pdfsys-bench] by_backend: {summary['by_backend']}")
    print(f"[pdfsys-bench] extracted={summary['num_extracted']} scored={summary['num_scored']} errors={summary['num_errors']}")
    if summary.get("avg_quality") is not None:
        print(f"[pdfsys-bench] avg_quality={summary['avg_quality']:.3f}")
    print(f"[pdfsys-bench] jsonl: {summary['out_path']}")
    print(f"[pdfsys-bench] summary: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
