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

    # Pack a run's MinerU output into a pdfsys.doc/v1 Parquet shard
    pdfsys dataset --from-mineru ./out --to ./dataset/v1 --meta ./out/results.jsonl
"""

from __future__ import annotations

import argparse
import sys

from .config import (
    EXAMPLE_CONFIG,
    VALID_STAGES,
    apply_cli_overrides,
    default_config,
    load_config,
)
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
    p.add_argument(
        "--vlm-engine",
        choices=("transformers", "mlx-engine", "vllm-engine"),
        default=None,
        help="Mineru VLM inference engine. mlx-engine is much faster on "
             "Apple Silicon; vllm-engine needs NVIDIA GPU; transformers is "
             "the portable default.",
    )
    p.add_argument("--no-quality", action="store_true", default=False, help="Skip quality scoring.")
    p.add_argument("--quality-model", type=str, default=None, help="HuggingFace quality model.")

    # ---- visualize ----
    v = sub.add_parser(
        "visualize",
        help="Build a static HTML+JSON viz bundle from a run directory.",
    )
    v.add_argument("-r", "--run-dir", required=True,
                   help="Pipeline run directory (must contain dataset.parquet).")
    v.add_argument("-o", "--out-dir", default=None,
                   help="Output directory (default: <run-dir>/viz).")
    v.add_argument("--preview-source", default=None,
                   help="Path to previews.json (default: bundled annotation/previews.json).")

    # ---- annotate ----
    a = sub.add_parser("annotate", help="Launch the PDF annotation UI in browser.")
    a.add_argument("--port", type=int, default=8234, help="HTTP server port (default: 8234).")
    a.add_argument("--bench-dir", type=str, default=None, help="Path to pdfsys-bench package.")
    a.add_argument(
        "--import", type=str, default=None, dest="import_file",
        help="Import annotations from an exported JSON file into metadata.json.",
    )

    # ---- dataset ----
    d = sub.add_parser(
        "dataset",
        help="Pack pipeline output into a pdfsys.doc/v1 Parquet shard.",
    )
    d.add_argument("--from-mineru", required=True, dest="from_mineru",
                   help="Run directory containing MinerU *_content_list.json outputs.")
    d.add_argument("--to", required=True, dest="to_dir",
                   help="Output dataset directory (documents/ + images/ are created inside).")
    d.add_argument("--shard", default="shard-00000", help="Shard name (default: shard-00000).")
    d.add_argument("--meta", default=None,
                   help="results.jsonl from the same run; joins quality/router columns by sha256.")
    d.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "none"))
    d.add_argument("--embed-images", action="store_true", default=False,
                   help="Inline image bytes into documents/ as well (self-contained but larger).")
    d.add_argument("--pairs", action="store_true", default=False,
                   help="Also write the materialized image-text pair view to pairs/.")
    d.add_argument("--no-mentions", action="store_true", default=False,
                   help="Skip figure-mention linking (faster; leaves blocks.mentions empty).")
    d.add_argument("--no-text", action="store_true", default=False,
                   help="Leave the rendered `text` column null; derive it from blocks instead. "
                        "Saves ~40%% of the documents file on image-free corpora.")

    # ---- release ----
    r = sub.add_parser("release", help="Manage system_release.toml component pins.")
    r_sub = r.add_subparsers(dest="release_command", help="Release subcommand")
    # Stash the release sub-parser's print_help so main() can show the
    # right help when 'pdfsys release' is invoked without a subcommand.
    r.set_defaults(_release_help=r.print_help)

    r_status = r_sub.add_parser("status", help="Show pin vs HEAD for each component.")
    r_status.add_argument(
        "--config", "-c", type=str, default="system_release.toml",
        help="Path to system_release.toml (default: ./system_release.toml).",
    )

    r_lock = r_sub.add_parser("lock", help="Update system_release.toml from submodule HEADs.")
    r_lock.add_argument(
        "--config", "-c", type=str, default="system_release.toml",
        help="Path to system_release.toml (default: ./system_release.toml).",
    )

    r_verify = r_sub.add_parser("verify", help="Verify pinned commits match resolved HEADs (CI guard).")
    r_verify.add_argument(
        "--config", "-c", type=str, default="system_release.toml",
        help="Path to system_release.toml (default: ./system_release.toml).",
    )

    return top


def cmd_init_config() -> int:
    print(EXAMPLE_CONFIG, end="")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Load config: YAML file → defaults → CLI overrides.
    cfg = load_config(args.config) if args.config else default_config()

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
        vlm_engine=args.vlm_engine,
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
        print(f"[pdfsys] vlm:     {cfg.vlm.engine} (enabled)")
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


def cmd_dataset(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from .dataset_build import build_from_mineru_dir, iter_mineru_dirs
    from .dataset_writer import DatasetWriter, pairs_table

    src = Path(args.from_mineru)
    if not src.is_dir():
        print(f"Error: not a directory: {src}", file=sys.stderr)
        return 1

    meta = _load_run_meta(Path(args.meta)) if args.meta else {}
    doc_dirs = list(iter_mineru_dirs(src))
    if not doc_dirs:
        print(f"Error: no *_content_list.json found under {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.to_dir)
    print(f"[pdfsys dataset] {len(doc_dirs)} documents → {out_dir}")

    docs = []
    failures = 0
    with DatasetWriter(
        out_dir,
        shard=args.shard,
        compression=args.compression,
        embed_images=args.embed_images,
        include_text=not args.no_text,
    ) as writer:
        for doc_dir in doc_dirs:
            try:
                doc, blobs = build_from_mineru_dir(
                    doc_dir,
                    link_figure_mentions=not args.no_mentions,
                )
            except Exception as e:  # one bad document must not kill the shard
                failures += 1
                print(f"  ! {doc_dir}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            doc = _apply_run_meta(doc, meta.get(doc.id))
            writer.write(doc, blobs)
            if args.pairs:
                docs.append(doc)
        n_docs, n_images = writer.docs_written, writer.images_written

    print(f"[pdfsys dataset] documents={n_docs} images={n_images} failed={failures}")

    if args.pairs:
        table = pairs_table(docs)
        (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            out_dir / "pairs" / f"{args.shard}.parquet",
            compression=args.compression,
        )
        print(f"[pdfsys dataset] pairs={table.num_rows}")

    # Machine-readable shard descriptor, mirroring what a manifest would carry.
    (out_dir / f"{args.shard}.meta.json").write_text(
        json.dumps(
            {
                "schema": "pdfsys.doc/1",
                "shard": args.shard,
                "documents": n_docs,
                "images": n_images,
                "failed": failures,
                "source": str(src),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def _load_run_meta(path) -> dict:
    """Index a run's results.jsonl by sha256 so dataset rows inherit its columns."""
    import json

    out: dict = {}
    if not path.exists():
        print(f"Warning: --meta file not found: {path}", file=sys.stderr)
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = row.get("sha256")
            if sha:
                out[sha] = row
    return out


def _apply_run_meta(doc, row):
    import dataclasses

    if not row:
        return doc
    return dataclasses.replace(
        doc,
        source_uri=doc.source_uri or (row.get("pdf_path") or ""),
        backend=row.get("extract_backend") or doc.backend,
        n_pages=doc.n_pages or int(row.get("num_pages") or 0),
        quality_score=row.get("quality_score"),
        quality_model=row.get("quality_model") or "",
        router_ocr_prob=row.get("ocr_prob"),
    )


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
    elif args.command == "visualize":
        from . import viz
        return viz.main([
            "-r", args.run_dir,
            *(("-o", args.out_dir) if args.out_dir else ()),
            *(("--preview-source", args.preview_source) if args.preview_source else ()),
        ])
    elif args.command == "dataset":
        return cmd_dataset(args)
    elif args.command == "annotate":
        return cmd_annotate(args)
    elif args.command == "release":
        from . import release as release_mod
        if args.release_command == "status":
            return release_mod.cmd_status(args)
        elif args.release_command == "lock":
            return release_mod.cmd_lock(args)
        elif args.release_command == "verify":
            return release_mod.cmd_verify(args)
        # No subcommand → show release help, NOT top-level help.
        args._release_help()
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
