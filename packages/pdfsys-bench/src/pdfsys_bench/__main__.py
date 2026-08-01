"""pdfsys-bench CLI — run the closed-loop pipeline on a directory of PDFs.

Usage::

    # MVP mode (mupdf only):
    python -m pdfsys_bench \\
        --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \\
        --out out/bench_omnidoc100.jsonl

    # Full pipeline (layout + OCR + optional VLM):
    python -m pdfsys_bench \\
        --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \\
        --out out/bench_full.jsonl \\
        --full-pipeline \\
        --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loop import run_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys-bench",
        description="Run the pdfsys closed-loop pipeline.",
    )
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
        default="miracleyin/mnbvc-pdf-quality-scorer-modernbert",
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
    # --- Full pipeline flags ---
    p.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Enable the full pipeline: layout analyser → Stage-B → pipeline/VLM parser.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="LayoutCache directory. Defaults to <out-dir>/.cache.",
    )
    p.add_argument(
        "--vlm",
        action="store_true",
        dest="vlm_enabled",
        help="Enable the VLM lane (MinerU) for complex-content pages.",
    )
    p.add_argument(
        "--vlm-engine",
        choices=("transformers", "mlx-engine", "vllm-engine"),
        default="transformers",
        help="Mineru VLM inference engine. Default transformers is portable; "
             "mlx-engine is faster on Apple Silicon; vllm-engine needs NVIDIA GPU.",
    )
    # --- Cascade flags ---
    p.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "Use the quality-driven cascade (mupdf → pipeline → vlm) instead "
            "of Stage-B routing. Implies --full-pipeline. Cheapest parser is "
            "tried first; output is gated by Layer-1 hard rules; the next "
            "parser is only invoked if the gate rejects."
        ),
    )
    p.add_argument(
        "--cascade-skip-mupdf-threshold",
        type=float,
        default=0.9,
        help=(
            "In cascade mode, skip the MuPDF attempt if Stage-A's ocr_prob "
            "is at or above this value (router is confident the PDF needs "
            "OCR). Default 0.9."
        ),
    )
    p.add_argument(
        "--cascade-skip-pipeline",
        action="store_true",
        help=(
            "Skip the mineru-pipeline stage in cascade (mupdf → vlm only). "
            "Use on platforms where mineru's pipeline mode hangs (e.g., "
            "macOS without CUDA). Requires --vlm to be useful."
        ),
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
        full_pipeline=args.full_pipeline,
        cache_dir=args.cache_dir,
        vlm_enabled=args.vlm_enabled,
        vlm_engine=args.vlm_engine,
        cascade=args.cascade,
        cascade_skip_mupdf_threshold=args.cascade_skip_mupdf_threshold,
        cascade_skip_pipeline=args.cascade_skip_pipeline,
    )

    print(f"\n[pdfsys-bench] processed {summary['num_pdfs']} PDFs in {summary['wall_seconds']:.1f}s")
    print(f"[pdfsys-bench] by_backend: {summary['by_backend']}")
    if summary.get("by_stage_b"):
        print(f"[pdfsys-bench] stage_b:    {summary['by_stage_b']}")
    if summary.get("by_cascade_decision"):
        print(f"[pdfsys-bench] cascade:    {summary['by_cascade_decision']}")
    print(f"[pdfsys-bench] extracted={summary['num_extracted']} scored={summary['num_scored']} errors={summary['num_errors']}")
    if summary.get("avg_quality") is not None:
        print(f"[pdfsys-bench] avg_quality={summary['avg_quality']:.3f}")
    print(f"[pdfsys-bench] jsonl:    {summary['out_path']}")
    print(f"[pdfsys-bench] summary:  {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
