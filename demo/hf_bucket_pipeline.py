#!/usr/bin/env python3
"""HuggingFace Bucket → pdfsys Pipeline Demo.

Reads PDF files from a HuggingFace Storage Bucket, routes them through
the pdfsys pipeline (XGBoost router + MuPDF extraction + optional quality
scoring), and saves the extracted results locally.

Usage
-----
    # Process *all* PDFs in the bucket (uses the router's default threshold)
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc

    # Process first 50 PDFs, skip the heavy quality scorer
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --max-files 50

    # Full pipeline with quality scoring (downloads ~800 MB model on first run)
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --run-quality

    # Custom output directory
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --out ./results

    # Skip the download and process already-cached PDFs
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --use-cache

    # List files in the bucket without processing
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --list-only

    # Upload extracted results to a target bucket after processing
    python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --upload-bucket roger1024/extracted_pdf

Output layout::

    <output-dir>/
    ├── manifest.json              # batch-level summary (record per PDF)
    ├── pdfs/                      # cached PDF files (optional)
    └── extracted/
        └── <sha256[:16]>/
            ├── metadata.json      # router + quality results
            └── page_*.md          # extracted Markdown per PDF
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Lazy imports — keep the module importable even when deps are missing
# --------------------------------------------------------------------------- #

def _import_hf_hub():
    import huggingface_hub  # noqa: F401 — side-effect

    return huggingface_hub


def _import_pipeline_parts():
    """Import the pdfsys pipeline components; they are heavy (xgboost, etc.)."""
    from demo.pipeline import PipelineResult, run_pipeline

    return PipelineResult, run_pipeline


def _import_xgb_weights():
    """Trigger the one-time XGBoost weights download if needed."""
    from pdfsys_router.download_weights import target_path

    if not target_path().is_file():
        from pdfsys_router.download_weights import download

        download()


# --------------------------------------------------------------------------- #
# Bucket interaction helpers
# --------------------------------------------------------------------------- #

@dataclass
class BucketFile:
    """Represents one file entry from a HuggingFace bucket listing."""

    path: str
    size: int


def list_bucket(
    bucket_id: str,
    *,
    suffix: str = ".pdf",
    recursive: bool = True,
) -> list[BucketFile]:
    """Return all files in ``bucket_id`` that match ``suffix``.

    Raises
    ------
    RuntimeError
        If the bucket is unreachable or the user is not authenticated.
    """
    hf = _import_hf_hub()
    try:
        items = list(hf.list_bucket_tree(bucket_id, recursive=recursive))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to list bucket {bucket_id!r}. "
            f"Check your HF token: `huggingface-cli login`"
        ) from exc

    matched = []
    for it in items:
        if it.type == "file" and it.path.endswith(suffix):
            matched.append(BucketFile(path=it.path, size=it.size))
    return matched


def download_bucket_files(
    bucket_id: str,
    files: list[tuple[str, str | Path]],
) -> None:
    """Download files from a HuggingFace bucket to local paths.

    Parameters
    ----------
    bucket_id : str
        The bucket identifier, e.g. ``"roger1024/raw_doc"``.
    files : list of (remote_path, local_path)
        Pairs of remote object path and local destination path.
    """
    hf = _import_hf_hub()
    hf.download_bucket_files(bucket_id, files=files)


# --------------------------------------------------------------------------- #
# Result persistence
# --------------------------------------------------------------------------- #

@dataclass
class PipelineRecord:
    """One PDF's full pipeline result, stored as a dict for JSON serialization."""

    source_path: str
    sha256: str | None
    extracted: bool
    backend: str
    ocr_prob: float
    num_pages: int
    segments: list[dict[str, Any]]
    markdown: str
    stats: dict[str, Any]
    quality_score: float | None
    quality_num_tokens: int | None
    errors: list[str]
    wall_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "sha256": self.sha256,
            "extracted": self.extracted,
            "backend": self.backend,
            "ocr_prob": round(self.ocr_prob, 4),
            "num_pages": self.num_pages,
            "num_segments": len(self.segments),
            "markdown_chars": len(self.markdown),
            "quality_score": self.quality_score,
            "quality_num_tokens": self.quality_num_tokens,
            "errors": self.errors,
            "wall_ms": round(self.wall_ms, 1),
        }


def save_result(
    out_dir: Path,
    pdf_name: str,
    pdf_bytes: bytes,
    record: PipelineRecord,
    *,
    save_pdf: bool = False,
) -> None:
    """Persist extraction results and optionally the original PDF.

    Directory layout::

        <out_dir>/
        ├── pdfs/                  # original PDFs (if save_pdf=True)
        ├── manifest.json          # all records
        └── extracted/
            └── <sha256[:16]>/
                ├── metadata.json
                └── extracted.md
    """
    sha_prefix = (record.sha256 or pdf_name)[:16]
    extract_dir = out_dir / "extracted" / sha_prefix
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Save extracted Markdown
    md_path = extract_dir / "extracted.md"
    md_path.write_text(record.markdown, encoding="utf-8")

    # Save metadata (without the full markdown to keep it lean)
    meta = record.to_dict()
    meta["markdown"] = record.markdown  # include for convenience
    meta["segments"] = record.segments
    meta_path = extract_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Optionally save the original PDF
    if save_pdf and pdf_bytes:
        pdf_dir = out_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / f"{sha_prefix}.pdf").write_bytes(pdf_bytes)


def append_to_manifest(out_dir: Path, record: PipelineRecord) -> None:
    """Append one record to the batch manifest JSON file."""
    manifest_path = out_dir / "manifest.json"
    manifest: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    manifest.append(record.to_dict())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Core processing function
# --------------------------------------------------------------------------- #

def process_pdf_bytes(
    pdf_bytes: bytes,
    source_name: str,
    *,
    run_quality: bool = False,
    ocr_threshold: float = 0.5,
) -> PipelineRecord:
    """Process a single PDF from bytes through the full pipeline.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF content.
    source_name : str
        Original filename / path for provenance.
    run_quality : bool
        Whether to run the ModernBERT quality scorer (heavy).
    ocr_threshold : float
        Router OCR probability threshold (0.0–1.0).

    Returns
    -------
    PipelineRecord
        All results, errors, and timing.
    """
    import tempfile

    PipelineResult, run_pipeline_func = _import_pipeline_parts()

    errors: list[str] = []
    t_start = time.perf_counter()

    # Write to a temporary file — run_pipeline() expects a file path
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp_path = Path(tmp.name)

        result: PipelineResult = run_pipeline_func(
            tmp_path,
            run_quality=run_quality,
            ocr_threshold=ocr_threshold,
        )

        # Collect errors from the result object
        if result.router_error:
            errors.append(f"router: {result.router_error}")
        if result.extract_error:
            errors.append(f"extract: {result.extract_error}")
        if result.quality_error:
            errors.append(f"quality: {result.quality_error}")

        wall_ms = (
            result.wall_ms_router
            + result.wall_ms_extract
            + result.wall_ms_quality
        )

        record = PipelineRecord(
            source_path=source_name,
            sha256=result.sha256,
            extracted=(result.backend == "mupdf" and result.extract_error is None),
            backend=result.backend,
            ocr_prob=float(result.ocr_prob) if result.ocr_prob == result.ocr_prob else float("nan"),
            num_pages=result.num_pages,
            segments=result.segments,
            markdown=result.markdown,
            stats=result.extract_stats or {},
            quality_score=result.quality_score,
            quality_num_tokens=result.quality_num_tokens,
            errors=errors,
            wall_ms=wall_ms,
        )
    except Exception as exc:  # noqa: BLE001
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        record = PipelineRecord(
            source_path=source_name,
            sha256=None,
            extracted=False,
            backend="error",
            ocr_prob=float("nan"),
            num_pages=0,
            segments=[],
            markdown="",
            stats={},
            quality_score=None,
            quality_num_tokens=None,
            errors=[f"{type(exc).__name__}: {exc}"],
            wall_ms=wall_ms,
        )
    finally:
        tmp.close()
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)

    return record


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def ensure_local_path(path: str | Path) -> Path:
    """Resolve and create parent directories for a local path that may contain
    subdirectory separators."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run(
    *,
    bucket_id: str = "roger1024/raw_doc",
    output_dir: str | Path = "./hf_bucket_results",
    max_files: int | None = None,
    run_quality: bool = False,
    ocr_threshold: float = 0.5,
    list_only: bool = False,
    save_pdfs: bool = False,
    use_cache: bool = False,
    upload_bucket: str | None = None,
) -> None:
    """Main orchestration: list, download, process, save, optionally upload.

    Parameters
    ----------
    bucket_id : str
        HuggingFace bucket identifier.
    output_dir : str or Path
        Local directory for saving results.
    max_files : int or None
        Limit the number of PDFs processed (None = all).
    run_quality : bool
        Enable the ModernBERT quality scorer.
    ocr_threshold : float
        Router OCR probability threshold.
    list_only : bool
        If True, only list files and exit.
    save_pdfs : bool
        Save original PDFs alongside extracted results.
    use_cache : bool
        If True, skip downloading; process PDFs already in ``output_dir/pdfs/``.
    upload_bucket : str or None
        If set, upload the results directory to this HuggingFace bucket
        (e.g. ``"roger1024/extracted_pdf"``) after processing.
    """
    out = Path(output_dir)
    pdf_cache = out / "pdfs"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ step 1
    # Ensure the XGBoost weights are downloaded
    print("=" * 72)
    print("pdfsys · HuggingFace Bucket Pipeline")
    print(f"  bucket:    {bucket_id}")
    print(f"  output:    {out.resolve()}")
    print(f"  max files: {max_files or 'all'}")
    print(f"  quality:   {run_quality}")
    print(f"  OCR thresh:{ocr_threshold}")
    upload_str = f"  upload:    {upload_bucket}" if upload_bucket else ""
    if upload_str:
        print(upload_str)
    print("=" * 72)

    print("\n[1/4] Ensuring router weights …")
    _import_xgb_weights()

    # ------------------------------------------------------------------ step 2
    # Discover files
    print("\n[2/4] Listing bucket …")
    try:
        bucket_files = list_bucket(bucket_id, suffix=".pdf")
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        print("  Tip: run `huggingface-cli login` first.")
        return

    if not bucket_files:
        print("  No PDF files found in the bucket.")
        print("  (Is the bucket name correct? Do you have access?)")
        return

    # Sort by path for deterministic ordering
    bucket_files.sort(key=lambda f: f.path)

    print(f"  Found {len(bucket_files)} PDF(s) in bucket {bucket_id!r}.")
    if list_only:
        for bf in bucket_files:
            size_str = f"{bf.size / 1024:.1f} KB" if bf.size < 1_000_000 else f"{bf.size / 1_000_000:.1f} MB"
            print(f"    {bf.path}  ({size_str})")
        return

    # ------------------------------------------------------------------ step 3
    # Download / collect PDF bytes
    print("\n[3/4] Preparing PDFs …")

    pdfs_to_process: list[tuple[str, bytes]] = []
    skipped_count = 0

    if use_cache:
        print(f"  Using cached PDFs from {pdf_cache}/ …")
        if not pdf_cache.is_dir():
            print(f"  ERROR: cache directory {pdf_cache} does not exist.")
            return
        count = 0
        for pdf_path in sorted(pdf_cache.glob("*.pdf")):
            if max_files is not None and count >= max_files:
                break
            pdf_bytes = pdf_path.read_bytes()
            pdfs_to_process.append((pdf_path.name, pdf_bytes))
            count += 1
        print(f"  Loaded {len(pdfs_to_process)} PDF(s) from cache.")
    else:
        # Filter files to download
        files_to_download = bucket_files
        if max_files is not None and max_files < len(files_to_download):
            files_to_download = files_to_download[:max_files]
            print(f"  Limiting to first {max_files} file(s).")

        print(f"  Downloading {len(files_to_download)} PDF(s) from bucket …")
        pdf_cache.mkdir(parents=True, exist_ok=True)

        # Download each file individually for progress reporting
        for idx, bf in enumerate(files_to_download, start=1):
            local_path = pdf_cache / Path(bf.path).name
            size_str = f"{bf.size / 1024:.1f} KB" if bf.size < 1_000_000 else f"{bf.size / 1_000_000:.1f} MB"
            print(f"    [{idx}/{len(files_to_download)}] {bf.path} ({size_str})")

            try:
                download_bucket_files(
                    bucket_id,
                    files=[(bf.path, str(local_path))],
                )
                pdfs_to_process.append(
                    (Path(bf.path).name, local_path.read_bytes())
                )
            except Exception as exc:
                print(f"      ⚠ Download failed: {exc}")
                skipped_count += 1
                continue

    if not pdfs_to_process:
        print("  No PDFs to process. Exiting.")
        return

    # ------------------------------------------------------------------ step 4
    # Process each PDF
    print(f"\n[4/4] Processing {len(pdfs_to_process)} PDF(s) …")
    t_start_batch = time.perf_counter()

    processed_count = 0
    error_count = 0

    for idx, (name, pdf_bytes) in enumerate(pdfs_to_process, start=1):
        print(f"\n  --- [{idx}/{len(pdfs_to_process)}] {name} ---")

        t0 = time.perf_counter()
        record = process_pdf_bytes(
            pdf_bytes,
            source_name=name,
            run_quality=run_quality,
            ocr_threshold=ocr_threshold,
        )
        elapsed = time.perf_counter() - t0

        # Status line
        status = "✓" if record.extracted else "⚠"
        backend_str = record.backend
        quality_str = (
            f"  quality={record.quality_score:.2f}"
            if record.quality_score is not None
            else ""
        )
        errors_str = f"  errors={record.errors}" if record.errors else ""
        print(
            f"    {status} backend={backend_str}"
            f"  pages={record.num_pages}"
            f"  chars={len(record.markdown):,}"
            f"{quality_str}"
            f"{errors_str}"
            f"  {elapsed:.1f}s"
        )

        # Save results
        try:
            save_result(
                out,
                name,
                pdf_bytes,
                record,
                save_pdf=save_pdfs,
            )
            append_to_manifest(out, record)
        except Exception as exc:
            print(f"      ⚠ Failed to save result: {exc}")

        if record.errors:
            error_count += 1
        processed_count += 1

    # ------------------------------------------------------------------ summary
    t_elapsed = time.perf_counter() - t_start_batch
    print("\n" + "=" * 72)
    print(f"Done — {processed_count} processed, {error_count} with errors, {skipped_count} skipped")
    print(f"Total time: {t_elapsed:.1f}s")
    print(f"Results in: {out.resolve()}")
    print("=" * 72)

    # ------------------------------------------------------------------ step 5
    # Optionally upload results to a target bucket
    if upload_bucket:
        _upload_results(out, upload_bucket)


def _upload_results(local_dir: Path, bucket_id: str) -> None:
    """Upload the results directory to a HuggingFace bucket using sync_bucket.

    Parameters
    ----------
    local_dir : Path
        Local directory with extracted results.
    bucket_id : str
        Target bucket identifier, e.g. ``"roger1024/extracted_pdf"``.
    """
    hf = _import_hf_hub()
    bucket_path = f"hf://buckets/{bucket_id}"
    print(f"\n[5/5] Uploading results to {bucket_path} …")
    print(f"  source: {local_dir.resolve()}")
    print(f"  dest:   {bucket_path}")
    try:
        plan = hf.sync_bucket(
            source=str(local_dir),
            dest=bucket_path,
            verbose=False,
        )
        print(f"  Upload complete: {plan}")
    except Exception as exc:
        print(f"  ERROR: Upload failed: {exc}")
        print("  Tip: run `huggingface-cli login` with a token that has write access.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pdfsys pipeline on PDFs from a HuggingFace Storage Bucket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bucket",
        default="roger1024/raw_doc",
        help="HuggingFace bucket identifier (default: roger1024/raw_doc)",
    )
    parser.add_argument(
        "--out",
        default="./hf_bucket_results",
        dest="output_dir",
        help="Local output directory (default: ./hf_bucket_results)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of PDFs to process (default: all)",
    )
    parser.add_argument(
        "--run-quality",
        action="store_true",
        default=False,
        help="Enable ModernBERT quality scorer (heavy, ~800 MB download)",
    )
    parser.add_argument(
        "--ocr-threshold",
        type=float,
        default=0.5,
        help="Router OCR probability threshold 0–1 (default: 0.5)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        default=False,
        help="Only list files in the bucket, don't process",
    )
    parser.add_argument(
        "--save-pdfs",
        action="store_true",
        default=False,
        help="Save original PDF files alongside extracted results",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use PDFs from the cache directory instead of downloading",
    )
    parser.add_argument(
        "--upload-bucket",
        type=str,
        default=None,
        help="Upload results to this HuggingFace bucket after processing, "
        'e.g. "roger1024/extracted_pdf"',
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        bucket_id=args.bucket,
        output_dir=args.output_dir,
        max_files=args.max_files,
        run_quality=args.run_quality,
        ocr_threshold=args.ocr_threshold,
        list_only=args.list_only,
        save_pdfs=args.save_pdfs,
        use_cache=args.use_cache,
        upload_bucket=args.upload_bucket,
    )
