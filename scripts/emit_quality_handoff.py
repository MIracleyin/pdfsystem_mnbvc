"""Reshape bench JSONL into a clean per-file handoff JSON for the
downstream quality-score teammate.

Input: out/<run>/results.jsonl produced by `python -m pdfsys_bench ...`
Output: out/<run>/quality_handoff.json

Schema (quality_handoff.v2):

    {
      "schema_version": "quality_handoff.v2",
      "generated_at": "<iso8601>",
      "source_run": "<basename of jsonl dir>",
      "stats": {
        "num_pdfs": int,
        "num_published": int,
        "num_rejected": int,
        "by_parser": {"mupdf": int, "pipeline": int, "vlm": int, "deferred": int},
        "avg_quality": float | null,
        "with_markdown": int,
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
          "markdown": str | null,        # inlined text when --markdown-dir is passed
          "markdown_path": str | null,   # relative path on disk (informational)
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

When ``--markdown-dir`` is passed, each ``<sha256>.md`` file in that
directory is inlined into the matching record's ``markdown`` field.
Rejected / deferred PDFs have no markdown file and stay
``markdown: null``. Bumped to v2 from v1 (added ``markdown`` field and
``stats.with_markdown``); downstream code keyed by ``schema_version``
should pin to v2 to consume inlined text.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def reshape(jsonl_path: Path, markdown_dir: Path | None) -> dict:
    files = []
    by_parser: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    sum_quality = 0.0
    n_quality = 0
    n_with_markdown = 0

    # Relative path the consumer sees in markdown_path: relative to the
    # jsonl's parent so the JSON is portable as long as the markdown dir
    # ships alongside it.
    jsonl_dir = jsonl_path.parent.resolve()

    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            sha = rec.get("sha256")

            md_text: str | None = None
            md_rel: str | None = None
            if markdown_dir and sha:
                cand = markdown_dir / f"{sha}.md"
                if cand.exists():
                    try:
                        md_text = cand.read_text(encoding="utf-8")
                        n_with_markdown += 1
                    except OSError as e:
                        print(f"warn: cannot read {cand}: {e}", file=sys.stderr)
                    try:
                        md_rel = str(cand.resolve().relative_to(jsonl_dir))
                    except ValueError:
                        md_rel = str(cand.resolve())

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
                    "markdown": md_text,
                    "markdown_path": md_rel,
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
        "schema_version": "quality_handoff.v2",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": jsonl_path.parent.name,
        "stats": {
            "num_pdfs": len(files),
            "num_published": decisions.get("publish", 0),
            "num_rejected": decisions.get("reject", 0),
            "by_parser": dict(by_parser),
            "avg_quality": sum_quality / n_quality if n_quality else None,
            "with_markdown": n_with_markdown,
        },
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="emit_quality_handoff")
    p.add_argument("jsonl", type=Path, help="bench results.jsonl")
    p.add_argument("--markdown-dir", type=Path, default=None,
                   help="Dir of <sha256>.md files produced by "
                        "`pdfsys-bench --markdown-dir`. When passed, the "
                        "markdown text is inlined into each record (the "
                        "downstream quality-score consumer then loads a "
                        "single self-contained JSON instead of fanning "
                        "out to per-file disk reads).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Default: <jsonl dir>/quality_handoff.json")
    args = p.parse_args(argv)

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    out = args.out or args.jsonl.parent / "quality_handoff.json"
    payload = reshape(args.jsonl, args.markdown_dir)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = payload["stats"]
    aq = stats["avg_quality"]
    aq_str = f"{aq:.4f}" if aq is not None else "n/a"
    print(
        f"wrote {out} ({stats['num_pdfs']} files, "
        f"{stats['num_published']} published, "
        f"avg_quality={aq_str}, "
        f"with_markdown={stats['with_markdown']}/{stats['num_pdfs']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
