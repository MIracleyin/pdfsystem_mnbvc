"""Offline LLM external-review for release manifests.

Reads a release manifest JSONL produced by ``release_gate.py``, scores
selected rows with the LLM scorer, and writes ``quality_*_llm`` fields
back into the same file. Decision fields are NEVER modified — the LLM
is strictly an external reviewer, not part of the decision rule.

Scope:
* ``--llm-scope all``     — score every row (benchmark phase, default v1)
* ``--llm-scope review``  — score only rows with ``decision == "review"``

Resume via ``<manifest>.llm.jsonl``: an append-only checkpoint of each
``{doc_id, quality_score_llm, ...}`` record. With ``--resume``, rows
whose ``doc_id`` already appears in the checkpoint are skipped.

Hard architectural rule (enforced by convention, see the spec):
``llm_review`` MUST NOT be imported by ``loop.py``, ``cascade.py``, or
``release_gate.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .quality_llm import LlmQualityScorer

_VALID_SCOPES = ("all", "review")
_PATCH_FIELDS = (
    "quality_score_llm",
    "quality_reason_llm",
    "quality_model_llm",
    "quality_parse_error_llm",
)


def filter_manifest_rows(
    rows: Iterable[dict],
    *,
    scope: str,
) -> Iterator[dict]:
    """Yield only the rows the given scope applies to."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}; got {scope!r}")
    for row in rows:
        if scope == "all" or row.get("decision") == "review":
            yield row


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    """Write rows to ``path`` atomically via a .tmp + rename.

    On any failure (including KeyboardInterrupt) the .tmp file is
    unlinked so we never leave litter that could be mistaken for the
    real manifest by another tool.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = rec.get("doc_id")
            if sha:
                by_id[sha] = rec
    return by_id


def _append_checkpoint(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def _resolve_markdown_path(row: dict, markdown_dir: Path) -> Path | None:
    """Resolve a manifest row to its markdown file.

    Prefers an explicit ``_markdown_path`` (relative to ``markdown_dir``)
    when present; otherwise falls back to ``<doc_id>.md``.
    """
    rel = row.get("_markdown_path") or f"{row['doc_id']}.md"
    candidate = markdown_dir / rel
    return candidate if candidate.exists() else None


def _score_row(
    scorer: LlmQualityScorer,
    row: dict,
    markdown_dir: Path,
) -> dict[str, Any]:
    md_path = _resolve_markdown_path(row, markdown_dir)
    if md_path is None:
        return {
            "doc_id": row["doc_id"],
            "quality_score_llm": None,
            "quality_reason_llm": f"markdown not found for {row['doc_id']}",
            "quality_model_llm": scorer.client.config.model,
            "quality_parse_error_llm": "missing markdown",
        }
    text = md_path.read_text(encoding="utf-8", errors="replace")
    try:
        result = scorer.score(text)
    except Exception as exc:  # keep the batch going on transient errors
        return {
            "doc_id": row["doc_id"],
            "quality_score_llm": None,
            "quality_reason_llm": "",
            "quality_model_llm": scorer.client.config.model,
            "quality_parse_error_llm": f"{type(exc).__name__}: {exc}"[:300],
        }
    return {
        "doc_id": row["doc_id"],
        "quality_score_llm": result.score,
        "quality_reason_llm": result.reason,
        "quality_model_llm": result.model,
        "quality_parse_error_llm": result.parse_error,
    }


def run_review(
    *,
    manifest_path: str | Path,
    markdown_dir: str | Path,
    scope: str,
    scorer: Any = None,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Run LLM review against a manifest in place.

    ``scorer`` defaults to a fresh :class:`LlmQualityScorer` (which reads
    ``.env``). Tests pass a mock.
    """
    manifest_path = Path(manifest_path)
    markdown_dir = Path(markdown_dir)
    rows = _load_manifest(manifest_path)

    scorer = scorer if scorer is not None else LlmQualityScorer()
    ckpt_path = manifest_path.with_suffix(manifest_path.suffix + ".llm.jsonl")
    checkpoint = _load_checkpoint(ckpt_path) if resume else {}

    in_scope = list(filter_manifest_rows(rows, scope=scope))
    todo = [r for r in in_scope if r["doc_id"] not in checkpoint] if resume else in_scope

    by_id: dict[str, dict[str, Any]] = dict(checkpoint)  # start with resumed

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_score_row, scorer, r, markdown_dir): r
                for r in todo
            }
            for fut in as_completed(futures):
                rec = fut.result()
                by_id[rec["doc_id"]] = rec
                _append_checkpoint(ckpt_path, rec)

    # Patch the manifest in place.
    for row in rows:
        rec = by_id.get(row["doc_id"])
        if rec is None:
            continue
        for k in _PATCH_FIELDS:
            row[k] = rec.get(k)

    _write_manifest(manifest_path, rows)

    return {
        "num_rows": len(rows),
        "num_in_scope": len(in_scope),
        "num_resumed": len(in_scope) - len(todo) if resume else 0,
        "num_scored": len(todo),
        "checkpoint": str(ckpt_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys_bench.llm_review",
        description="Offline LLM review for release manifests (external; not in decision rule).",
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--markdown-dir", required=True, type=Path)
    p.add_argument("--llm-scope", choices=_VALID_SCOPES, default="all",
                   help="'all' (default, benchmark phase) or 'review' (production phase)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already in <manifest>.llm.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_review(
        manifest_path=args.manifest,
        markdown_dir=args.markdown_dir,
        scope=args.llm_scope,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"[llm-review] scope        = {args.llm_scope}")
    print(f"[llm-review] num_rows     = {summary['num_rows']}")
    print(f"[llm-review] num_in_scope = {summary['num_in_scope']}")
    print(f"[llm-review] num_resumed  = {summary['num_resumed']}")
    print(f"[llm-review] num_scored   = {summary['num_scored']}")
    print(f"[llm-review] checkpoint   = {summary['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
