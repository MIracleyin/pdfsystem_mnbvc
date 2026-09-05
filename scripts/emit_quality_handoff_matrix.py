"""Reshape an extract_matrix JSONL (one row per (PDF, parser) pair) into
a per-file JSON for the downstream quality-score annotator.

Each file gets a single record containing ALL parser candidates side
by side, so the annotator can compare mupdf / pipeline / vlm output
for the same PDF in one read.

Input:  out/<run>/results.jsonl produced by scripts/extract_matrix.py
        + the matching markdown/ directory.
Output: out/<run>/quality_handoff_matrix.json

Schema (quality_handoff_matrix.v1):

    {
      "schema_version": "quality_handoff_matrix.v1",
      "generated_at": "<iso8601>",
      "source_run": "<basename of jsonl dir>",
      "stats": {
        "num_pdfs": int,
        "num_extractions": int,
        "by_parser": {"mupdf": int, "pipeline": int, "vlm": int},
        "num_errors": int,
        "num_with_markdown": int,
      },
      "files": [
        {
          "file_id": "<sha256>",
          "filename": "<basename>",
          "extractions": [
            {
              "parser": "mupdf|pipeline|vlm",
              "markdown": str | null,         # inlined text
              "char_count": int,
              "wall_ms": float,
              "error": str | null,
              "markdown_file": str | null,    # relative filename, informational
            },
            ...
          ],
        },
        ...
      ]
    }

Use:
  uv run python scripts/emit_quality_handoff_matrix.py \\
      out/annotation-set-demo/results.jsonl \\
      --markdown-dir out/annotation-set-demo/markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def _read_markdown(md_dir: Path | None, name: str | None) -> str | None:
    if md_dir is None or not name:
        return None
    p = md_dir / name
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        print(f"warn: cannot read {p}: {e}", file=sys.stderr)
        return None


def reshape(jsonl_path: Path, markdown_dir: Path | None) -> dict:
    grouped: dict[str, dict] = {}
    extractions_by_file: dict[str, list[dict]] = defaultdict(list)
    by_parser: Counter[str] = Counter()
    n_errors = 0
    n_with_md = 0

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fid = rec.get("file_id")
            if not fid:
                continue

            if fid not in grouped:
                grouped[fid] = {
                    "file_id": fid,
                    "filename": rec.get("filename"),
                }

            md_text = _read_markdown(markdown_dir, rec.get("markdown_file"))
            if md_text is not None:
                n_with_md += 1

            err = rec.get("error")
            if err:
                n_errors += 1

            parser = rec.get("parser") or "unknown"
            by_parser[parser] += 1

            extractions_by_file[fid].append({
                "parser": parser,
                "markdown": md_text,
                "char_count": rec.get("char_count", 0),
                "wall_ms": rec.get("wall_ms"),
                "error": err,
                "markdown_file": rec.get("markdown_file"),
            })

    files = []
    for fid, base in grouped.items():
        files.append({
            **base,
            # Stable parser order in each file's extractions list, so the
            # downstream consumer can iterate by index if it wants.
            "extractions": sorted(
                extractions_by_file[fid],
                key=lambda e: ("mupdf", "pipeline", "vlm").index(e["parser"])
                if e["parser"] in ("mupdf", "pipeline", "vlm") else 99,
            ),
        })

    return {
        "schema_version": "quality_handoff_matrix.v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": jsonl_path.parent.name,
        "stats": {
            "num_pdfs": len(files),
            "num_extractions": sum(by_parser.values()),
            "by_parser": dict(by_parser),
            "num_errors": n_errors,
            "num_with_markdown": n_with_md,
        },
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="emit_quality_handoff_matrix")
    p.add_argument("jsonl", type=Path,
                   help="extract_matrix results.jsonl")
    p.add_argument("--markdown-dir", type=Path, default=None,
                   help="Directory of <sha256>__<parser>.md files. "
                        "Defaults to <jsonl-dir>/markdown if it exists. "
                        "Pass an empty path to skip inlining.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. "
                        "Default: <jsonl-dir>/quality_handoff_matrix.json")
    args = p.parse_args(argv)

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    md_dir = args.markdown_dir
    if md_dir is None:
        default_md = args.jsonl.parent / "markdown"
        if default_md.is_dir():
            md_dir = default_md
    if md_dir and not md_dir.is_dir():
        print(f"warn: --markdown-dir {md_dir} not found; running without",
              file=sys.stderr)
        md_dir = None

    out = args.out or args.jsonl.parent / "quality_handoff_matrix.json"
    payload = reshape(args.jsonl, md_dir)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = payload["stats"]
    by = stats["by_parser"]
    parser_counts = ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
    print(
        f"wrote {out} ({stats['num_pdfs']} files × ~3 parsers = "
        f"{stats['num_extractions']} extractions; "
        f"{parser_counts}; "
        f"errors={stats['num_errors']}, "
        f"with_markdown={stats['num_with_markdown']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
