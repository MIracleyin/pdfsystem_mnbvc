"""Reshape bench JSONL into a clean per-file handoff JSON for the
downstream quality-score teammate.

Input: out/<run>/results.jsonl produced by `python -m pdfsys_bench ...`
Output: out/<run>/quality_handoff.json

Schema (quality_handoff.v1):

    {
      "schema_version": "quality_handoff.v1",
      "generated_at": "<iso8601>",
      "source_run": "<basename of jsonl dir>",
      "stats": {
        "num_pdfs": int,
        "num_published": int,
        "num_rejected": int,
        "by_parser": {"mupdf": int, "pipeline": int, "vlm": int, "deferred": int},
        "avg_quality": float | null,
      },
      "files": [
        {
          "file_id": "<sha256 hex>",
          "filename": "<basename>",
          "parser": "mupdf|pipeline|vlm|deferred",
          "cascade_decision": "publish|reject",
          "router_ocr_prob": float | null,
          "num_pages": int | null,
          "markdown_chars": int,
          "markdown_path": null,   # filled if --markdown-dir was passed to the original bench
          "quality": {
            "score": float | null,
            "num_chars": int | null,
            "num_tokens": int | null,
            "model": str | null,
          },
          "extract": {
            "stats": {...} | null,
            "error": str | null,
          },
          "cascade_final_stage": str | null,
        },
        ...
      ]
    }

Markdown text itself is NOT included; downstream consumers who want to
rescore should re-run bench with ``--markdown-dir <DIR>``, then this
script populates ``markdown_path`` from the matching ``<sha256>.md``
filename in that directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def reshape(jsonl_path: Path, markdown_dir: Path | None) -> dict:
    files = []
    by_parser: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    sum_quality = 0.0
    n_quality = 0

    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            sha = rec.get("sha256")
            md_path = None
            if markdown_dir and sha:
                cand = markdown_dir / f"{sha}.md"
                if cand.exists():
                    md_path = str(cand)

            qs = rec.get("quality_score")
            if qs is not None:
                sum_quality += qs
                n_quality += 1

            parser = rec.get("backend") or "deferred"
            by_parser[parser] += 1
            decision = rec.get("cascade_decision") or "reject"
            decisions[decision] += 1

            files.append(
                {
                    "file_id": sha,
                    "filename": Path(rec.get("pdf_path", "")).name,
                    "parser": parser,
                    "cascade_decision": decision,
                    "router_ocr_prob": rec.get("ocr_prob"),
                    "num_pages": rec.get("num_pages"),
                    "markdown_chars": rec.get("markdown_chars", 0),
                    "markdown_path": md_path,
                    "quality": {
                        "score": qs,
                        "num_chars": rec.get("quality_num_chars"),
                        "num_tokens": rec.get("quality_num_tokens"),
                        "model": rec.get("quality_model"),
                    },
                    "extract": {
                        "stats": rec.get("extract_stats") or None,
                        "error": rec.get("extract_error"),
                    },
                    "cascade_final_stage": rec.get("cascade_final_stage"),
                }
            )

    return {
        "schema_version": "quality_handoff.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": jsonl_path.parent.name,
        "stats": {
            "num_pdfs": len(files),
            "num_published": decisions.get("publish", 0),
            "num_rejected": decisions.get("reject", 0),
            "by_parser": dict(by_parser),
            "avg_quality": sum_quality / n_quality if n_quality else None,
        },
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="emit_quality_handoff")
    p.add_argument("jsonl", type=Path, help="bench results.jsonl")
    p.add_argument("--markdown-dir", type=Path, default=None,
                   help="Optional dir of <sha256>.md files produced by "
                        "`pdfsys-bench --markdown-dir`. If passed, "
                        "per-file markdown_path is filled.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Default: <jsonl dir>/quality_handoff.json")
    args = p.parse_args(argv)

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    out = args.out or args.jsonl.parent / "quality_handoff.json"
    payload = reshape(args.jsonl, args.markdown_dir)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({payload['stats']['num_pdfs']} files, "
          f"{payload['stats']['num_published']} published, "
          f"avg_quality={payload['stats']['avg_quality']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
