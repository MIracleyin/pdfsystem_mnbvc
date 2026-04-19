"""pdfsys CLI — unified entry point for the pdfsys pipeline.

Usage::

    # Generate example config
    pdfsys init-config > pdfsys.yaml

    # Run full pipeline
    pdfsys run -c pdfsys.yaml

    # Run specific stages
    pdfsys run -c pdfsys.yaml --stages router
    pdfsys run -c pdfsys.yaml --stages router,layout,extract

    # Override config from CLI
    pdfsys run -c pdfsys.yaml --pdf-dir ./other --limit 10

    # Quick run without config file
    pdfsys run --pdf-dir ./data/pdfs --out-dir ./out --stages router,extract

    # Launch annotation UI
    pdfsys annotate
    pdfsys annotate --port 9000
    pdfsys annotate --import annotations_2026-04-18.json
"""

from __future__ import annotations

import argparse
import sys

from .config import EXAMPLE_CONFIG, VALID_STAGES, RunConfig, apply_cli_overrides, default_config, load_config
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="pdfsys",
        description="Unified CLI for the pdfsys PDF processing pipeline.",
    )
    sub = top.add_subparsers(dest="command", help="Available commands")

    # ---- init-config ----
    sub.add_parser(
        "init-config",
        help="Print an example YAML config to stdout.",
    )

    # ---- run ----
    p = sub.add_parser("run", help="Run the pipeline.")
    p.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=None,
        help=f"Comma-separated stages to run: {','.join(VALID_STAGES)}",
    )
    p.add_argument("--pdf-dir", type=str, default=None, help="Input PDF directory.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--limit", type=int, default=None, help="Max PDFs to process.")
    p.add_argument("--markdown-dir", type=str, default=None, help="Dump markdown here.")
    p.add_argument("--cache-dir", type=str, default=None, help="LayoutCache directory.")
    p.add_argument("--ocr-threshold", type=float, default=None, help="P(ocr) threshold.")
    p.add_argument("--router-weights", type=str, default=None, help="XGBoost weights path.")
    p.add_argument("--vlm", action="store_true", dest="vlm_enabled", default=None, help="Enable VLM lane.")
    p.add_argument("--no-quality", action="store_true", default=False, help="Skip quality scoring.")
    p.add_argument("--quality-model", type=str, default=None, help="HuggingFace quality model.")

    # ---- annotate ----
    a = sub.add_parser("annotate", help="Launch the PDF annotation UI in browser.")
    a.add_argument("--port", type=int, default=8234, help="HTTP server port (default: 8234).")
    a.add_argument("--bench-dir", type=str, default=None, help="Path to pdfsys-bench package.")
    a.add_argument(
        "--import", type=str, default=None, dest="import_file",
        help="Import annotations from an exported JSON file into metadata.json.",
    )

    return top


def cmd_init_config() -> int:
    print(EXAMPLE_CONFIG, end="")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Load config: YAML file → defaults → CLI overrides.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    cfg = apply_cli_overrides(
        cfg,
        stages=args.stages,
        pdf_dir=args.pdf_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        markdown_dir=args.markdown_dir,
        cache_dir=args.cache_dir,
        ocr_threshold=args.ocr_threshold,
        router_weights=args.router_weights,
        vlm_enabled=args.vlm_enabled,
        no_quality=args.no_quality,
        quality_model=args.quality_model,
    )

    if not cfg.input.pdf_dir:
        print("Error: --pdf-dir is required (or set input.pdf_dir in config).", file=sys.stderr)
        return 1

    # Print run plan.
    print(f"[pdfsys] stages:  {' → '.join(cfg.stages)}")
    print(f"[pdfsys] input:   {cfg.input.pdf_dir}" + (f" (limit {cfg.input.limit})" if cfg.input.limit else ""))
    print(f"[pdfsys] output:  {cfg.jsonl_path}")
    if cfg.markdown_path:
        print(f"[pdfsys] markdown: {cfg.markdown_path}")
    if cfg.has_stage("layout"):
        print(f"[pdfsys] layout:  {cfg.layout.model}")
    if cfg.has_stage("extract") and cfg.vlm.enabled:
        print(f"[pdfsys] vlm:     {cfg.vlm.model} (enabled)")
    print()

    # Run pipeline.
    summary = run(cfg)

    # Print summary.
    print()
    print(f"[pdfsys] processed {summary['num_pdfs']} PDFs in {summary['wall_seconds']:.1f}s")
    print(f"[pdfsys] backends:  {summary['by_backend']}")
    if summary.get("by_stage_b"):
        print(f"[pdfsys] stage-b:   {summary['by_stage_b']}")
    print(f"[pdfsys] extracted={summary['num_extracted']} scored={summary['num_scored']} errors={summary['num_errors']}")
    if summary.get("avg_quality") is not None:
        print(f"[pdfsys] avg_quality={summary['avg_quality']:.3f}")
    print(f"[pdfsys] jsonl:     {cfg.jsonl_path}")
    print(f"[pdfsys] summary:   {summary.get('summary_path', '')}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .annotate import _find_bench_dir, import_annotations, serve

    bench_dir = Path(args.bench_dir) if args.bench_dir else _find_bench_dir()
    if bench_dir is None:
        print(
            "Error: cannot find pdfsys-bench directory. "
            "Use --bench-dir to specify it.",
            file=sys.stderr,
        )
        return 1

    metadata_path = bench_dir / "annotation" / "metadata.json"

    # Import mode.
    if args.import_file:
        import_path = Path(args.import_file)
        if not import_path.exists():
            print(f"Error: file not found: {import_path}", file=sys.stderr)
            return 1
        total = import_annotations(metadata_path, import_path)
        print(f"[pdfsys annotate] imported → {total} annotated PDFs in metadata.json")
        return 0

    # Server mode.
    serve(bench_dir, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-config":
        return cmd_init_config()
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "annotate":
        return cmd_annotate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
