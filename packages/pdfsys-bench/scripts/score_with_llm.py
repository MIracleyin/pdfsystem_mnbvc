"""Re-score an existing viz bundle's markdown with the LLM quality scorer.

Reads ``<src>/viz_data.json``, scores each row's markdown via
``LlmQualityScorer`` (mimo-v2.5-pro by default), and writes a new bundle
at ``<dst>/`` with quality_score_llm + quality_reason_llm patched into
every row. Static assets (index.html, assets/, markdown/, page1/,
previews.json) are hard-linked or copied so the new bundle is
self-contained.

A checkpoint file ``<dst>/.llm_scores.jsonl`` is appended after each
successful row so the job is resumable.

Usage::

    uv run python packages/pdfsys-bench/scripts/score_with_llm.py \\
        --src out/viz_final --dst out/viz_llm \\
        [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Make the script runnable directly without an install step.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "packages" / "pdfsys-bench" / "src"))

from pdfsys_bench.quality_llm import LlmQualityScorer  # noqa: E402


def _copy_bundle_skeleton(src: Path, dst: Path) -> None:
    """Mirror static assets from src → dst. Skip viz_data.json (we rewrite it)."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == "viz_data.json":
            continue
        target = dst / child.name
        if target.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, target, symlinks=False)
        else:
            shutil.copy2(child, target)


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Return {sha256: row_patch} for already-scored rows."""
    if not path.exists():
        return {}
    scored: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = rec.get("sha256")
            if sha:
                scored[sha] = rec
    return scored


def _score_one(
    scorer: LlmQualityScorer,
    row: dict[str, Any],
    bundle_root: Path,
) -> dict[str, Any]:
    sha = row["sha256"]
    md_rel = row.get("extract", {}).get("markdown_path")
    if not md_rel:
        return {
            "sha256": sha,
            "score": None,
            "reason": "no markdown_path",
            "model": scorer.client.config.model,
            "parse_error": "missing markdown",
            "wall_ms": 0.0,
        }

    md_path = bundle_root / md_rel
    if not md_path.exists():
        return {
            "sha256": sha,
            "score": None,
            "reason": f"markdown file missing: {md_rel}",
            "model": scorer.client.config.model,
            "parse_error": "missing markdown",
            "wall_ms": 0.0,
        }

    text = md_path.read_text(encoding="utf-8", errors="replace")
    t0 = time.perf_counter()
    try:
        res = scorer.score(text)
    except Exception as e:  # noqa: BLE001 — keep the batch going
        return {
            "sha256": sha,
            "score": None,
            "reason": "",
            "model": scorer.client.config.model,
            "parse_error": f"{type(e).__name__}: {e}"[:300],
            "wall_ms": (time.perf_counter() - t0) * 1000,
        }
    wall_ms = (time.perf_counter() - t0) * 1000

    return {
        "sha256": sha,
        "score": res.score,
        "reason": res.reason,
        "model": res.model,
        "parse_error": res.parse_error,
        "raw_excerpt": (res.raw_content or "")[:240],
        "wall_ms": wall_ms,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path, help="source viz bundle dir")
    p.add_argument("--dst", required=True, type=Path, help="destination viz bundle dir")
    p.add_argument("--limit", type=int, default=None, help="only score first N rows")
    p.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    p.add_argument("--resume", action="store_true", help="skip rows already in checkpoint")
    args = p.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not (src / "viz_data.json").exists():
        print(f"ERROR: {src}/viz_data.json not found", file=sys.stderr)
        return 2

    print(f"[llm-score] src = {src}")
    print(f"[llm-score] dst = {dst}")
    _copy_bundle_skeleton(src, dst)

    data = json.loads((src / "viz_data.json").read_text())
    rows: list[dict[str, Any]] = data["rows"]
    if args.limit is not None:
        rows = rows[: args.limit]
    total = len(rows)

    checkpoint_path = dst / ".llm_scores.jsonl"
    scored: dict[str, dict[str, Any]] = (
        _load_checkpoint(checkpoint_path) if args.resume else {}
    )
    if scored:
        print(f"[llm-score] resuming: {len(scored)} rows already scored")

    todo = [r for r in rows if r["sha256"] not in scored]
    print(f"[llm-score] total={total}  todo={len(todo)}  workers={args.workers}")

    scorer = LlmQualityScorer()
    print(f"[llm-score] model = {scorer.client.config.model}")

    started = time.perf_counter()
    done = len(scored)

    with checkpoint_path.open("a", encoding="utf-8") as ckpt, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score_one, scorer, r, src): r for r in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            scored[rec["sha256"]] = rec
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
            done += 1
            elapsed = time.perf_counter() - started
            score_str = f"{rec['score']:.1f}" if rec.get('score') is not None else " - "
            reason_excerpt = (rec.get("reason") or "")[:60]
            err = rec.get("parse_error") or ""
            tag = f" [parse_error: {err}]" if err else ""
            print(
                f"[{done:3d}/{total}] {rec['sha256'][:12]}  "
                f"score={score_str}  ({elapsed:.1f}s elapsed)  "
                f"{reason_excerpt}{tag}",
                flush=True,
            )

    # Patch the rows in place and write new viz_data.json.
    model_id = scorer.client.config.model
    avg_scores: list[float] = []
    for r in rows:
        sha = r["sha256"]
        rec = scored.get(sha)
        if rec is None:
            continue
        q = r.setdefault("quality", {})
        q["score_llm"] = rec.get("score")
        q["reason_llm"] = rec.get("reason")
        q["model_llm"] = rec.get("model") or model_id
        if rec.get("parse_error"):
            q["parse_error_llm"] = rec["parse_error"]
        if rec.get("score") is not None:
            avg_scores.append(float(rec["score"]))

    # Carry the patched rows back into the dataset, preserving rows that
    # were not in the --limit slice (they keep their original quality
    # block, with no LLM fields).
    if args.limit is not None:
        patched_by_sha = {r["sha256"]: r for r in rows}
        data["rows"] = [patched_by_sha.get(r["sha256"], r) for r in data["rows"]]
    else:
        data["rows"] = rows

    agg = data.setdefault("aggregate", {})
    if avg_scores:
        agg["avg_quality_llm"] = sum(avg_scores) / len(avg_scores)
        agg["count_quality_llm"] = len(avg_scores)
    meta = data.setdefault("run_meta", {})
    meta["llm_quality_model"] = model_id
    meta["llm_scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    (dst / "viz_data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"[llm-score] wrote {dst}/viz_data.json  "
          f"(scored {len(avg_scores)} rows, avg_llm={agg.get('avg_quality_llm'):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
