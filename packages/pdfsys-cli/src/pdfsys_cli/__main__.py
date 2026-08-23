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

    # Pack a run's MinerU output into a pdfsys.page/v2 Parquet shard
    pdfsys dataset --from-mineru ./out --to ./dataset/v2 --meta ./out/results.jsonl
    pdfsys dataset --from-mineru ./out --to ./dataset/v2 --images pages --pdf-dir ./data/pdfs

    # Re-emit it in the MNBVC multimodal block format
    pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/chinaxiv_0.parquet
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
        help="Pack pipeline output into a pdfsys.page/v2 Parquet shard.",
    )
    d.add_argument("--from-mineru", required=True, dest="from_mineru",
                   help="Run directory containing MinerU *_content_list.json outputs.")
    d.add_argument("--to", required=True, dest="to_dir",
                   help="Output dataset directory (pages/ + images/ are created inside).")
    d.add_argument("--shard", default="shard-00000", help="Shard name (default: shard-00000).")
    d.add_argument("--meta", default=None,
                   help="results.jsonl from the same run; joins quality/router columns by sha256.")
    d.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "none"))
    d.add_argument("--pairs", action="store_true", default=False,
                   help="Also write the materialized image-text pair view to pairs/.")
    d.add_argument("--no-mentions", action="store_true", default=False,
                   help="Skip figure-mention linking (faster; leaves blocks.mentions empty).")
    d.add_argument("--no-blocks", action="store_true", default=False,
                   help="Write the model-derived `blocks` column null. `text` still carries "
                        "the image interleaving; you lose bboxes, captions and block types.")
    d.add_argument("--images", default="crops", choices=("crops", "pages", "none"),
                   help="How image pixels are stored. `crops` (default): only the cropped "
                        "figures, ~90 KiB/page. `pages`: only full-page rasters, with figures "
                        "addressed by bbox and cut out on read, ~311 KiB/page — needs "
                        "--pdf-dir. `none`: no pixels at all. They are mutually exclusive "
                        "because MinerU's crops are already sub-rectangles of a 200-dpi page "
                        "render, so keeping both stores the same pixels twice.")
    d.add_argument("--pdf-dir", default=None,
                   help="Directory of source PDFs, required by --images pages. Matched to "
                        "documents by sha256.")
    d.add_argument("--on-duplicate", default="best", choices=("best", "error"),
                   help="同一份 PDF 有多个 backend 产物时怎么办。`best`（默认）按 "
                        "vlm > pipeline > mupdf 择一并打印丢弃了哪些；`error` 直接报错。"
                        "不能都留 —— (doc_id, page_index) 是主键。")
    d.add_argument("--render-dpi", type=int, default=200,
                   help="DPI for page rasters (default: 200, matching the resolution MinerU "
                        "crops at, so a derived crop is pixel-equivalent to a stored one).")

    # ---- dataset-validate ----
    dv = sub.add_parser(
        "dataset-validate",
        help="Check a pdfsys.page/v2 shard against the format contract.",
    )
    dv.add_argument("--shard", required=True, help="Dataset directory (contains pages/).")
    dv.add_argument("--no-hash", action="store_true", default=False,
                    help="Skip re-hashing every blob (faster, but content addressing "
                         "goes unverified).")

    # ---- mnbvc-export ----
    m = sub.add_parser(
        "mnbvc-export",
        help="Export a pdfsys.page/v2 shard to the MNBVC multimodal block format.",
    )
    m.add_argument("--from-shard", required=True, dest="from_shard",
                   help="A pdfsys.page/v2 dataset directory (contains pages/).")
    m.add_argument("--to", required=True, dest="out_path",
                   help="Output .parquet path.")
    m.add_argument("--dialect", default="v2", choices=("v2", "legacy"),
                   help="`v2` (default): what mm_template_mnbvc writes today — declared "
                        "schema, media columns as struct<bytes, path> so HuggingFace can "
                        "decode them, content-based md5, integer 页ID. `legacy`: what it "
                        "wrote before PR #4 (media base64-encoded into a string column), "
                        "for consumers still reading pre-merge shards. See "
                        "docs/schema/mnbvc-mm-compat.md.")
    m.add_argument("--block-type", default="image-text-pair",
                   help="Value for the 块类型 column (default: image-text-pair).")
    m.add_argument("--date", default=None,
                   help="Value for the 时间 column, YYYYMMDD (default: today).")
    m.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "none"))

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

    from .dataset_build import (
        build_from_mineru_dir,
        iter_mineru_dirs,
        render_page_images,
        select_documents,
    )
    from .dataset_writer import DatasetWriter, pairs_table

    src = Path(args.from_mineru)
    if not src.is_dir():
        print(f"Error: not a directory: {src}", file=sys.stderr)
        return 1

    want_rasters = args.images == "pages"
    pdf_index: dict[str, Path] = {}
    if want_rasters:
        if not args.pdf_dir:
            print(
                "Error: --images pages requires --pdf-dir (figures are cut out of the "
                "page raster, so the raster has to exist).",
                file=sys.stderr,
            )
            return 1
        pdf_index = _index_pdfs_by_sha256(Path(args.pdf_dir))
        print(f"[pdfsys dataset] indexed {len(pdf_index)} source PDFs for rasterisation")

    meta = _load_run_meta(Path(args.meta)) if args.meta else {}
    doc_dirs = list(iter_mineru_dirs(src))
    if not doc_dirs:
        print(f"Error: no *_content_list.json found under {src}", file=sys.stderr)
        return 1

    doc_dirs, dropped = select_documents(doc_dirs)
    if dropped:
        if args.on_duplicate == "error":
            for path, doc_id, why in dropped:
                print(f"Error: {doc_id[:12]}… 有多份产物: {path} — {why}", file=sys.stderr)
            print(
                "Error: (doc_id, page_index) 是主键，重复会写出违约的 shard。"
                "用 --on-duplicate best 择优，或先清理输入。",
                file=sys.stderr,
            )
            return 1
        for path, doc_id, why in dropped:
            print(f"  - 跳过 {path}: {why}", file=sys.stderr)

    out_dir = Path(args.to_dir)
    print(f"[pdfsys dataset] {len(doc_dirs)} documents → {out_dir}"
          + (f"（去重丢弃 {len(dropped)} 份）" if dropped else ""))

    docs: list = []
    failures = 0
    missing_pdfs = 0
    with DatasetWriter(
        out_dir,
        shard=args.shard,
        compression=args.compression,
        include_blocks=not args.no_blocks,
    ) as writer:
        for doc_dir in doc_dirs:
            try:
                pages, blobs = build_from_mineru_dir(
                    doc_dir,
                    link_figure_mentions=not args.no_mentions,
                    images=args.images,
                )
            except Exception as e:  # one bad document must not kill the shard
                failures += 1
                print(f"  ! {doc_dir}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            pages = tuple(
                _apply_run_meta(p, meta.get(p.doc_id)) for p in pages
            )

            rasters: list = []
            if want_rasters:
                pdf_path = pdf_index.get(pages[0].doc_id) if pages else None
                if pdf_path is None:
                    missing_pdfs += 1
                else:
                    rasters = render_page_images(
                        pdf_path, pages, dpi=args.render_dpi
                    )
                    # render_page_images returns pages stamped with the raster
                    # id; those are the rows that must be written.
                    pages = tuple(p for p, _ in rasters)

            writer.write(pages, blobs, [(p, b) for p, b in rasters])
            if args.pairs:
                docs.append(pages)

        stats = (
            writer.docs_written,
            writer.pages_written,
            writer.images_written,
            writer.page_images_written,
        )

    n_docs, n_pages, n_images, n_page_images = stats
    print(
        f"[pdfsys dataset] documents={n_docs} pages={n_pages} "
        f"images={n_images} page_images={n_page_images} failed={failures}"
    )
    if missing_pdfs:
        print(
            f"[pdfsys dataset] warning: {missing_pdfs} documents had no matching "
            f"PDF under --pdf-dir; their page_image_id is null",
            file=sys.stderr,
        )

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
                "schema": "pdfsys.page/2",
                "shard": args.shard,
                "documents": n_docs,
                "pages": n_pages,
                "n_images": n_images,
                "n_page_images": n_page_images,
                "images_mode": args.images,
                "render_dpi": args.render_dpi if want_rasters else None,
                "has_blocks": not args.no_blocks,
                "failed": failures,
                "source": str(src),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def _index_pdfs_by_sha256(pdf_dir) -> dict:
    """Map sha256 -> path for every PDF under ``pdf_dir``.

    Documents are identified by content hash throughout the pipeline, so this
    is the only reliable way to pair a MinerU output directory back to the PDF
    it came from; filenames are not stable across ingest.
    """
    import hashlib
    from pathlib import Path

    index: dict = {}
    for path in sorted(Path(pdf_dir).rglob("*.pdf")):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as e:
            print(f"Warning: cannot read {path}: {e}", file=sys.stderr)
            continue
        index.setdefault(digest, path)
    return index


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


def _apply_run_meta(page, row):
    """Copy a run's per-document telemetry onto a page row.

    The pipeline scores quality per document today, so it lands in
    ``doc_quality_score``; the page-level ``quality_score`` column stays null
    until a per-page scorer fills it. Conflating the two would misreport a
    whole book's score as every page's.
    """
    import dataclasses

    if not row:
        return page
    return dataclasses.replace(
        page,
        source_uri=page.source_uri or (row.get("pdf_path") or ""),
        extractor=row.get("extract_backend") or page.extractor,
        doc_n_pages=page.doc_n_pages or int(row.get("num_pages") or 0),
        doc_quality_score=row.get("quality_score"),
        quality_model=row.get("quality_model") or "",
        router_ocr_prob=row.get("ocr_prob"),
    )


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .dataset_validate import validate_shard

    report = validate_shard(Path(args.shard), verify_hashes=not args.no_hash)

    print(f"[pdfsys dataset-validate] {args.shard}")
    for line in report.findings:
        print(str(line), file=sys.stderr if line.severity == "error" else sys.stdout)
    print("  统计: " + json.dumps(report.stats, ensure_ascii=False))
    if report.ok:
        print(f"  ✓ 通过（{report.n_warnings} 条提示）")
        return 0
    print(f"  ✗ {report.n_errors} 个错误 / {report.n_warnings} 条提示", file=sys.stderr)
    return 1


def cmd_mnbvc_export(args: argparse.Namespace) -> int:
    from datetime import date
    from pathlib import Path

    from .mnbvc_export import export_shard

    shard = Path(args.from_shard)
    if not (shard / "pages").is_dir():
        print(f"Error: {shard} is not a pdfsys.page/v2 shard (no pages/).", file=sys.stderr)
        return 1

    stats = export_shard(
        shard,
        Path(args.out_path),
        dialect=args.dialect,
        timestamp=args.date or date.today().strftime("%Y%m%d"),
        block_type=args.block_type,
        compression=args.compression,
    )
    print(f"[pdfsys mnbvc-export] dialect={args.dialect} blocks={stats['blocks']} -> {args.out_path}")
    if stats["pages_without_image"]:
        print(
            f"[pdfsys mnbvc-export] warning: {stats['pages_without_image']} pages had no "
            f"raster, so their 图片 is null. The MNBVC image-text-pair block is "
            f"(page image, page text) — build the shard with `--images pages` to fill it.",
            file=sys.stderr,
        )
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
    elif args.command == "visualize":
        from . import viz
        return viz.main([
            "-r", args.run_dir,
            *(("-o", args.out_dir) if args.out_dir else ()),
            *(("--preview-source", args.preview_source) if args.preview_source else ()),
        ])
    elif args.command == "dataset":
        return cmd_dataset(args)
    elif args.command == "dataset-validate":
        return cmd_dataset_validate(args)
    elif args.command == "mnbvc-export":
        return cmd_mnbvc_export(args)
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
