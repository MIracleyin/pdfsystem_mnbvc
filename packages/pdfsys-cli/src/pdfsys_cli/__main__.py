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

    # Quick run without config file. --extract-backends names which backends
    # THIS machine runs; without it, OCR-routed documents go to MinerU.
    pdfsys run --pdf-dir ./data/pdfs --out-dir ./out \
               --stages router,extract --extract-backends mupdf

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
    p.add_argument("--pdf-list", type=str, default=None,
                   help="File of PDF paths, one per line, processed instead of "
                        "scanning --pdf-dir. This is how a box works on a slice it "
                        "was handed — the documents another machine routed to it, or "
                        "one bucket of a fleet split (`split -n l/8` on the list). "
                        "Order is preserved; duplicates and missing files are "
                        "reported, not silently dropped.")
    p.add_argument("--path-root", type=str, default=None,
                   help="Directory that relative entries in --pdf-list are resolved "
                        "against. Lets one worklist be read on a machine that mounted "
                        "the corpus somewhere else. Absolute entries are left alone.")
    p.add_argument("--extract-backends", type=str, default=None,
                   help="Comma-separated backends THIS machine runs: mupdf, "
                        "pipeline, vlm. Default: all of them. The CPU box takes "
                        "`mupdf` and records the OCR-bound documents as another "
                        "box's work (skip_reason=lane-filter); the GPU box takes "
                        "`pipeline` and skips what the CPU box already did. Use "
                        "the same --ocr-threshold on both, or a document can be "
                        "filtered out of BOTH lanes — that shows up as a nonzero "
                        "lane-filter count on the GPU box.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Append to an existing results.jsonl and skip the documents "
                        "already in it, instead of truncating it. The summary is "
                        "recomputed over the whole file, so it describes the run and "
                        "not just this leg of it.")
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
    p.add_argument("--parser-output-dir", type=str, default=None,
                   help="Where MinerU's sidecars are kept: <dir>/<sha256>/ with "
                        "_middle.json, _content_list.json and images/. This is the "
                        "only durable copy — mineru-api's own output tree is "
                        "garbage-collected — and it is exactly what "
                        "`pdfsys dataset --from-mineru` reads. Without it the "
                        "markdown survives but nothing packageable does.")
    p.add_argument("--no-parser-images", action="store_true", default=False,
                   help="Do not ask the parser for figure crops. Worth it when the "
                        "shard will use --images pages or none, where the crops are "
                        "downloaded and then thrown away (~90 KiB/page over the "
                        "wire, which on a remote mineru-api is the wire).")
    p.add_argument("--no-quality", action="store_true", default=False, help="Skip quality scoring.")
    p.add_argument("--quality-model", type=str, default=None, help="HuggingFace quality model.")

    # ---- score ----
    sc = sub.add_parser(
        "score",
        help="Quality-score a finished run's markdown, without re-extracting.",
    )
    sc.add_argument("--results", required=True,
                    help="results.jsonl from a run. Rows are matched to markdown "
                         "by sha256 and written back with the quality columns "
                         "filled; every input row reaches the output.")
    sc.add_argument("--markdown-dir", required=True,
                    help="Directory of <sha256>.md, i.e. the run's --markdown-dir.")
    sc.add_argument("--out", required=True, help="Where to write the scored jsonl.")
    sc.add_argument("--model", default=None,
                    help="The model the scorer is expected to be serving. Checked "
                         "against GET /health before any work: two lanes scored by "
                         "two different models put two scales in one column, and "
                         "nothing in the data would say so.")
    sc.add_argument("--workers", type=int, default=4,
                    help="Concurrent requests (default: 4). The server holds one "
                         "model and scores one document per request, so past a few "
                         "workers you are filling its socket queue.")
    sc.add_argument("--max-chars", type=int, default=40_000,
                    help="Clip each document before sending (default: 40000, which "
                         "is where the server truncates anyway — so the difference "
                         "never crosses the wire).")
    sc.add_argument("--resume", action="store_true", default=False,
                    help="Continue from the checkpoint left by an interrupted run "
                         "instead of re-scoring what it already did.")
    sc.add_argument("--rescore", action="store_true", default=False,
                    help="Score rows that already carry a quality_score, e.g. after "
                         "changing the model.")
    sc.add_argument("--overwrite", action="store_true", default=False,
                    help="Replace an existing --out rather than refusing.")

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
    d_src = d.add_mutually_exclusive_group(required=True)
    d_src.add_argument("--from-mineru", dest="from_mineru",
                       help="Run directory containing MinerU *_content_list.json outputs "
                            "(the pipeline / vlm lanes).")
    d_src.add_argument("--from-pdf-dir", dest="from_pdf_dir",
                       help="Directory of source PDFs, packaged through the mupdf fast "
                            "lane. This is the route for text-ok documents the router "
                            "never sent to MinerU — they leave no *_content_list.json "
                            "behind. Extraction is re-run here (~10ms/page) because a "
                            "run persists only merged markdown, with no page boundaries.")
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
    d.add_argument("--images", default=None, choices=("crops", "pages", "none"),
                   help="How image pixels are stored. `crops`: only the cropped figures, "
                        "~90 KiB/page. `pages`: only full-page rasters, with figures "
                        "addressed by bbox and cut out on read, ~311 KiB/page — needs "
                        "--pdf-dir on the MinerU lane. `none`: no pixels at all. They are "
                        "mutually exclusive because MinerU's crops are already "
                        "sub-rectangles of a 200-dpi page render, so keeping both stores "
                        "the same pixels twice. Defaults to `crops` for --from-mineru and "
                        "`pages` for --from-pdf-dir, which has no crops to store.")
    d.add_argument("--pdf-dir", default=None,
                   help="Directory of source PDFs, required by --images pages on the "
                        "MinerU lane. Matched to documents by sha256. Not needed with "
                        "--from-pdf-dir, which already knows where the PDF is.")
    d.add_argument("--on-duplicate", default="best", choices=("best", "error"),
                   help="同一份 PDF 有多个 backend 产物时怎么办。`best`（默认）按 "
                        "vlm > pipeline > mupdf 择一并打印丢弃了哪些；`error` 直接报错。"
                        "不能都留 —— (doc_id, page_index) 是主键。")
    d.add_argument("--render-dpi", type=int, default=200,
                   help="DPI for page rasters (default: 200, matching the resolution MinerU "
                        "crops at, so a derived crop is pixel-equivalent to a stored one).")
    d.add_argument("--allow-missing-crops", action="store_true", default=False,
                   help="Proceed with --images crops even when the MinerU output "
                        "has no images/ directories. Normally that means the run "
                        "used --no-parser-images and the crops were never fetched, "
                        "so the shard would come out with zero images rather than "
                        "the figures it advertises.")
    d.add_argument("--overwrite", action="store_true", default=False,
                   help="Replace pages/<shard>.parquet if it already exists. Without this "
                        "a name collision is an error, because several lanes write their "
                        "shards into one dataset directory and the writer truncates.")

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
    from pathlib import Path

    # Load config: YAML file → defaults → CLI overrides. Both raise ValueError
    # on the same class of typo (an unknown stage or backend), so both are
    # inside the handler — a traceback from a config file reads like a crash.
    try:
        cfg = load_config(args.config) if args.config else default_config()
        cfg = apply_cli_overrides(
            cfg,
            stages=args.stages,
            pdf_dir=args.pdf_dir,
            pdf_list=args.pdf_list,
            path_root=args.path_root,
            extract_backends=args.extract_backends,
            resume=args.resume,
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
            parser_output_dir=args.parser_output_dir,
            no_parser_images=args.no_parser_images,
        )
    except ValueError as e:
        # Unknown --stages or --extract-backends value. A traceback here reads
        # like a crash rather than a typo.
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not cfg.input.pdf_dir and not cfg.input.pdf_list:
        print(
            "Error: --pdf-dir or --pdf-list is required "
            "(or set input.pdf_dir / input.pdf_list in config).",
            file=sys.stderr,
        )
        return 1
    if cfg.input.pdf_list and not Path(cfg.input.pdf_list).is_file():
        print(f"Error: --pdf-list not found: {cfg.input.pdf_list}", file=sys.stderr)
        return 1
    if cfg.input.pdf_list and cfg.input.pdf_dir:
        print(
            f"[pdfsys] note: --pdf-list wins; --pdf-dir {cfg.input.pdf_dir} is ignored",
            file=sys.stderr,
        )
    if cfg.input.path_root and not cfg.input.pdf_list:
        print(
            "[pdfsys] note: --path-root only applies to --pdf-list; ignoring it",
            file=sys.stderr,
        )

    for stage in cfg.dropped_stages:
        # A stage vanishing between what was asked for and what runs is the
        # kind of thing you only notice three hours later, when the output
        # you expected isn't there.
        print(
            f"[pdfsys] warning: dropped stage {stage!r} — "
            f"{cfg.drop_reasons.get(stage, 'no reason recorded')}.",
            file=sys.stderr,
        )
    if "parquet" in cfg.dropped_stages and cfg.parquet_path.exists():
        print(
            f"[pdfsys] warning: {cfg.parquet_path} is left over from an earlier "
            f"leg and now covers only part of this run — treat it as stale.",
            file=sys.stderr,
        )

    # Print run plan.
    print(f"[pdfsys] stages:  {' → '.join(cfg.stages)}")
    source = cfg.input.pdf_list or cfg.input.pdf_dir
    print(
        f"[pdfsys] input:   {source}"
        + (f" (rooted at {cfg.input.path_root})" if cfg.input.path_root else "")
        + (f" (limit {cfg.input.limit})" if cfg.input.limit else "")
    )
    # Per backend, because pipeline and vlm are configured independently in
    # YAML — reading only `pipeline` gives a VLM lane a false all-clear.
    from .runner import parser_output_dirs

    sidecar_dirs = parser_output_dirs(cfg)
    unset = sorted(b for b, d in sidecar_dirs.items() if not d)
    if unset:
        # mineru-api writes its own copy under a task-uuid directory and
        # garbage-collects it, and the containerised deployment mounts no
        # volume for it. So without --parser-output-dir the sidecars are gone
        # by the time anyone goes to package them, and `pdfsys dataset
        # --from-mineru` reports an empty directory rather than a lost one.
        print(
            f"[pdfsys] warning: no output_dir for {', '.join(unset)}, so MinerU's "
            f"middle.json, content_list.json and figure crops are discarded as "
            f"they arrive. The markdown survives; nothing `pdfsys dataset "
            f"--from-mineru` can read does. Set --parser-output-dir.",
            file=sys.stderr,
        )
    for backend, where in sorted(sidecar_dirs.items()):
        if where:
            print(f"[pdfsys] sidecars: {where} ({backend})")

    if cfg.extract_backends is not None:
        print(f"[pdfsys] lane:    {', '.join(cfg.extract_backends)}")
        if "vlm" in cfg.extract_backends and not cfg.vlm.enabled:
            # Only stage-B ever says "vlm", and it only says it when
            # vlm_enabled is set. Without that the lane is empty by
            # construction — knowable before a single PDF is opened.
            print(
                "[pdfsys] warning: lane includes `vlm` but vlm.enabled is false, "
                "so stage-B can never route anything to it. Add --vlm (and the "
                "`layout` stage, which is what produces the stage-B decision).",
                file=sys.stderr,
            )
    if cfg.resume:
        print("[pdfsys] resume:  appending to an existing results.jsonl")
    print(f"[pdfsys] output:  {cfg.jsonl_path}")
    if cfg.markdown_path:
        print(f"[pdfsys] markdown: {cfg.markdown_path}")
    if cfg.has_stage("layout"):
        print(f"[pdfsys] layout:  {cfg.layout.model}")
    if cfg.has_stage("extract") and cfg.vlm.enabled:
        print(f"[pdfsys] vlm:     {cfg.vlm.engine} (enabled)")
    print()

    # Run pipeline.
    from .runner import (
        CorruptResultsError,
        LaneConflictError,
        ParserOutputDirError,
    )

    try:
        summary = run(cfg)
    except (CorruptResultsError, LaneConflictError, ParserOutputDirError) as e:
        # Both are raised before any document is touched, so nothing in the
        # out-dir has been written by this leg.
        print(f"[pdfsys] error: {e}", file=sys.stderr)
        return 1

    # Print summary.
    print()
    disc = summary.get("discovery", {})
    if disc.get("by_magic"):
        print(
            f"[pdfsys] discovered: {disc['by_suffix']} by extension, "
            f"{disc['by_magic']} extensionless (recognised by %PDF- header)"
        )
    if disc.get("missing"):
        print(
            f"[pdfsys] warning: {disc['missing']}/{disc['entries']} listed paths "
            f"do not exist, e.g. {disc.get('missing_examples', [])[:3]}",
            file=sys.stderr,
        )
    if disc.get("duplicates"):
        print(
            f"[pdfsys] warning: {disc['duplicates']}/{disc['entries']} listed paths "
            f"are repeats, processed once, e.g. "
            f"{disc.get('duplicate_examples', [])[:3]}",
            file=sys.stderr,
        )
    if summary.get("resumed_rows"):
        print(
            f"[pdfsys] resumed:  {summary['resumed_rows']} rows already on disk, "
            f"{summary['num_skipped_as_done']} inputs skipped as done"
        )
        if summary["num_skipped_as_done"] == 0:
            print(
                "[pdfsys] warning: carried rows but skipped nothing — the paths on "
                "disk do not match the paths being processed, so this leg is "
                "redoing work. Check --pdf-dir / --path-root against the earlier leg.",
                file=sys.stderr,
            )
    if summary.get("repaired_tail_bytes"):
        print(
            f"[pdfsys] repaired: dropped {summary['repaired_tail_bytes']} bytes of "
            f"an interrupted final line from results.jsonl",
            file=sys.stderr,
        )
    if summary.get("resumed_stage_mismatch"):
        print(
            f"[pdfsys] warning: resuming a run that was started with stages "
            f"{' → '.join(summary['resumed_stage_mismatch'])}, now running "
            f"{' → '.join(cfg.stages)}. Resume skips whole documents, not "
            f"stages — the {summary['resumed_rows']} rows already on disk keep "
            f"the old depth. Re-run without --resume to redo them.",
            file=sys.stderr,
        )
    print(f"[pdfsys] processed {summary['num_pdfs']} PDFs in {summary['wall_seconds']:.1f}s")
    print(f"[pdfsys] backends:  {summary['by_backend']}")
    if summary.get("by_stage_b"):
        print(f"[pdfsys] stage-b:   {summary['by_stage_b']}")
    print(
        f"[pdfsys] extracted={summary['num_extracted']} "
        f"skipped={summary['num_skipped']} "
        f"scored={summary['num_scored']} errors={summary['num_errors']}"
    )
    if summary.get("by_skip_reason"):
        print(f"[pdfsys] skipped:   {summary['by_skip_reason']}")
    # Filtering OCR work out of a mupdf lane is the normal hand-off. Filtering
    # *mupdf* work out of an OCR lane is not: it means this box decided the
    # document needs no OCR, while the box that queued it decided the opposite
    # and has already handed it away. It is then in no lane at all.
    lane = cfg.extract_backends or []
    stranded = summary.get("by_filtered_backend", {}).get("mupdf", 0)
    if stranded and lane and "mupdf" not in lane:
        print(
            f"[pdfsys] warning: {stranded} documents were routed to mupdf here but "
            f"queued for lane {', '.join(lane)} by the box that sent them — this "
            f"box's router disagrees, so they are in NO lane. Check that "
            f"--ocr-threshold matches ({cfg.router.ocr_threshold}).",
            file=sys.stderr,
        )
    if summary.get("avg_quality") is not None:
        print(f"[pdfsys] avg_quality={summary['avg_quality']:.3f}")
    print(f"[pdfsys] jsonl:     {cfg.jsonl_path}")
    print(f"[pdfsys] summary:   {summary.get('summary_path', '')}")

    # Gate on what this invocation *selected*, not on num_pdfs — that counter
    # includes rows carried in from earlier legs, so a resumed run pointed at
    # the wrong path would inherit a healthy-looking count and exit 0.
    if disc.get("selected", summary["num_pdfs"]) == 0:
        # Almost always a wrong path, or a worklist naming files this machine
        # did not mount. Exiting 0 lets a fleet script march on through every
        # bucket reporting success and produce nothing.
        print(f"[pdfsys] error: no PDFs to process from {source}", file=sys.stderr)
        return 1
    if disc.get("entries") and disc.get("missing") == disc.get("entries"):
        print(
            f"[pdfsys] error: not one of the {disc['entries']} listed paths exists "
            f"— wrong --path-root?",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    import dataclasses
    import json
    from functools import partial
    from pathlib import Path

    import pyarrow.parquet as pq

    from pdfsys_core import render_markdown

    from .dataset_build import (
        build_from_mineru_dir,
        build_from_pdf,
        iter_mineru_dirs,
        iter_pdfs,
        render_page_images,
        select_documents,
        select_pdfs,
    )
    from .dataset_writer import DatasetWriter, pairs_table

    from_pdfs = args.from_pdf_dir is not None
    src = Path(args.from_pdf_dir if from_pdfs else args.from_mineru)
    if not src.is_dir():
        print(f"Error: not a directory: {src}", file=sys.stderr)
        return 1

    # MinerU hands us crops. The mupdf lane rasterises nothing at all, so the
    # only pixels it could ever store are whole-page renders.
    images = args.images or ("pages" if from_pdfs else "crops")
    if from_pdfs and images == "crops":
        print(
            "Error: --images crops is impossible with --from-pdf-dir — the mupdf "
            "lane reads the PDF's text layer and never cuts figures out of the "
            "page. Use --images pages (whole-page rasters, figures addressed by "
            "bbox) or --images none.",
            file=sys.stderr,
        )
        return 1
    want_rasters = images == "pages"

    # Both lanes reduce to the same shape: a label to blame on failure, a
    # thunk returning (pages, blobs), and the source PDF when we already know
    # it. Everything downstream is lane-agnostic.
    work: list = []
    pdf_index: dict[str, Path] = {}

    if from_pdfs:
        selected, dropped = select_pdfs(iter_pdfs(src))
        if not selected:
            print(f"Error: no *.pdf found under {src}", file=sys.stderr)
            return 1
        work = [(str(p), partial(build_from_pdf, p), p) for _, p in selected]
    else:
        if want_rasters:
            if not args.pdf_dir:
                print(
                    "Error: --images pages requires --pdf-dir (figures are cut out of "
                    "the page raster, so the raster has to exist).",
                    file=sys.stderr,
                )
                return 1
            pdf_index = _index_pdfs_by_sha256(Path(args.pdf_dir))
            print(f"[pdfsys dataset] indexed {len(pdf_index)} source PDFs for rasterisation")

        doc_dirs = list(iter_mineru_dirs(src))
        if not doc_dirs:
            print(f"Error: no *_content_list.json found under {src}", file=sys.stderr)
            return 1
        # A run with --no-parser-images never fetched them, so the source tree
        # has no images/ at all. Building `crops` from it produces a shard whose
        # text points at figures and whose images table is empty — valid, and
        # completely unlike what was asked for.
        if (
            images == "crops"
            and not args.allow_missing_crops
            and not any((d / "images").is_dir() for d in doc_dirs)
        ):
            print(
                f"Error: --images crops but no images/ directory anywhere "
                f"under {src}. The run that produced it likely used "
                f"--no-parser-images. Use --images pages or none, or pass "
                f"--allow-missing-crops to build an image-less crops shard.",
                file=sys.stderr,
            )
            return 1

        doc_dirs, dropped = select_documents(doc_dirs)
        work = [
            (
                str(doc_dir),
                partial(
                    build_from_mineru_dir,
                    doc_dir,
                    link_figure_mentions=not args.no_mentions,
                    images=images,
                ),
                None,
            )
            for doc_dir in doc_dirs
        ]

    if dropped:
        if args.on_duplicate == "error":
            for path, doc_id, why in dropped:
                print(f"Error: {doc_id[:12]}… 有多份: {path} — {why}", file=sys.stderr)
            print(
                "Error: (doc_id, page_index) 是主键，重复会写出违约的 shard。"
                "用 --on-duplicate best 择优，或先清理输入。",
                file=sys.stderr,
            )
            return 1
        for path, doc_id, why in dropped:
            print(f"  - 跳过 {path}: {why}", file=sys.stderr)

    meta = _load_run_meta(Path(args.meta)) if args.meta else {}
    out_dir = Path(args.to_dir)
    lane = "mupdf" if from_pdfs else "mineru"

    # Two lanes write into one shard directory, so the shard name is the only
    # thing keeping them apart. pq.ParquetWriter truncates without asking, so
    # reusing a name silently replaces the other lane's work with this one's.
    #
    # A shard is up to four parquets plus a descriptor, and the media writers
    # are opened lazily — a rebuild with a different --images mode would
    # truncate pages/ and leave the previous build's images/ and page_images/
    # in place, splicing two builds into one shard. So the whole set is both
    # the collision check and what --overwrite clears.
    shard_files = [
        out_dir / sub / f"{args.shard}.parquet"
        for sub in ("pages", "images", "page_images", "pairs")
    ] + [out_dir / f"{args.shard}.meta.json"]
    clashes = [p for p in shard_files if p.exists()]
    if clashes and not args.overwrite:
        for p in clashes:
            print(f"Error: {p} 已存在", file=sys.stderr)
        print(
            f"Error: shard 名 {args.shard!r} 已被占用。换一个 --shard 名字，"
            f"或加 --overwrite 覆盖（会删掉上面这些文件）。",
            file=sys.stderr,
        )
        return 1
    for p in clashes:
        p.unlink()

    print(f"[pdfsys dataset] lane={lane} images={images} "
          f"{len(work)} documents → {out_dir}"
          + (f"（去重丢弃 {len(dropped)} 份）" if dropped else ""))

    docs: list = []
    failures = 0
    missing_pdfs = 0
    meta_hits = meta_misses = 0
    empty_builds = 0
    with DatasetWriter(
        out_dir,
        shard=args.shard,
        compression=args.compression,
        include_blocks=not args.no_blocks,
    ) as writer:
        for label, build, known_pdf in work:
            try:
                pages, blobs = build()
            except Exception as e:  # one bad document must not kill the shard
                failures += 1
                print(f"  ! {label}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            if pages and meta:
                if meta.get(pages[0].doc_id):
                    meta_hits += 1
                else:
                    meta_misses += 1
            pages = tuple(
                _apply_run_meta(p, meta.get(p.doc_id)) for p in pages
            )

            rasters: list = []
            if want_rasters:
                # The mupdf lane came from the PDF, so it already has it; the
                # MinerU lane has to find it by content hash.
                pdf_path = known_pdf or (
                    pdf_index.get(pages[0].doc_id) if pages else None
                )
                if pdf_path is None:
                    missing_pdfs += 1
                    # No raster for this document, so its bbox:// markers point
                    # into pixels the shard does not have. Re-render its text
                    # without them — the same thing --images none does, decided
                    # per document because that is where the raster is decided.
                    pages = tuple(
                        dataclasses.replace(
                            p, text=render_markdown(p.blocks, region_refs=False)
                        )
                        for p in pages
                    )
                else:
                    rasters = render_page_images(
                        pdf_path, pages, dpi=args.render_dpi
                    )
                    # render_page_images returns pages stamped with the raster
                    # id; those are the rows that must be written.
                    pages = tuple(p for p, _ in rasters)

            if not pages:
                # Built fine, produced nothing. DatasetWriter deliberately does
                # not count this as a document, so without a counter here it
                # would appear in neither `documents` nor `failed`.
                empty_builds += 1
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
        f"images={n_images} page_images={n_page_images} "
        f"empty={empty_builds} failed={failures}"
    )
    # documents + empty + failed == len(work), so a shortfall is arithmetic
    # rather than something you have to notice.
    unaccounted = len(work) - (n_docs + empty_builds + failures)
    if unaccounted:
        print(
            f"[pdfsys dataset] warning: {unaccounted} 份输入既没写进 shard，"
            f"也没记进 empty/failed —— 这是个 bug，请报告",
            file=sys.stderr,
        )
    if meta:
        # A shard whose quality columns are all null looks the same whether the
        # scorer never ran or the --meta file simply covers a different corpus.
        print(f"[pdfsys dataset] meta matched {meta_hits}/{meta_hits + meta_misses}")
        if meta_misses:
            print(
                f"[pdfsys dataset] warning: {meta_misses} 份文档在 --meta 里没有对应行，"
                f"质量分等列为空",
                file=sys.stderr,
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
                "images_mode": images,
                "render_dpi": args.render_dpi if want_rasters else None,
                "has_blocks": not args.no_blocks,
                "failed": failures,
                "empty": empty_builds,
                "missing_pdfs": missing_pdfs,
                "inputs": len(work),
                "lane": lane,
                "source": str(src),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if n_docs == 0:
        # Same reasoning as cmd_run's empty-corpus guard: a per-bucket fleet job
        # that packages nothing must not report success.
        print(
            f"[pdfsys dataset] error: 没有写出任何文档"
            f"（输入 {len(work)} 份，失败 {failures}，空 {empty_builds}）",
            file=sys.stderr,
        )
        return 1
    return 0


def _index_pdfs_by_sha256(pdf_dir) -> dict:
    """Map sha256 -> path for every PDF under ``pdf_dir``.

    Documents are identified by content hash throughout the pipeline, so this
    is the only reliable way to pair a MinerU output directory back to the PDF
    it came from; filenames are not stable across ingest.
    """
    import hashlib

    from pdfsys_core import iter_pdf_paths

    index: dict = {}
    for path in iter_pdf_paths(pdf_dir):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as e:
            print(f"Warning: cannot read {path}: {e}", file=sys.stderr)
            continue
        index.setdefault(digest, path)
    return index


def _load_run_meta(path) -> dict:
    """Index a run's results.jsonl by sha256 so dataset rows inherit its columns.

    Reports what it drops. A results.jsonl from a resumed or re-run job can
    carry the same sha256 twice (last wins), and rows that never reached a
    stage that hashes the file have no key at all — both used to be silent,
    which is how a shard ends up with null quality columns nobody can explain.
    """
    import json

    out: dict = {}
    n_lines = n_dup = n_nokey = 0
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
            n_lines += 1
            sha = row.get("sha256")
            if not sha:
                n_nokey += 1
                continue
            if sha in out:
                n_dup += 1
            out[sha] = row
    if n_dup:
        print(
            f"Warning: --meta {path.name} 有 {n_dup} 个重复 sha256（保留最后一条）",
            file=sys.stderr,
        )
    if n_nokey:
        print(
            f"Warning: --meta {path.name} 有 {n_nokey}/{n_lines} 行没有 sha256，已跳过",
            file=sys.stderr,
        )
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
        # Additive, not overriding: the builder read the extractor out of the
        # document's own middle.json, which is what actually produced these
        # pages. The run row is a second opinion and only fills a blank.
        extractor=page.extractor or (row.get("extract_backend") or ""),
        doc_n_pages=page.doc_n_pages or int(row.get("num_pages") or 0),
        doc_quality_score=row.get("quality_score"),
        quality_model=row.get("quality_model") or "",
        router_ocr_prob=row.get("ocr_prob"),
        layout_model=page.layout_model or (row.get("layout_model") or ""),
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
    elif args.command == "score":
        from .score import cmd_score
        return cmd_score(args)
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
