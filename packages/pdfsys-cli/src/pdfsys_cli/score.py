"""Quality-score a finished run's markdown, separately from producing it.

Scoring used to be reachable only as a continuation of extraction: the runner
called it with the ``ExtractedDoc`` still in memory, so the only way to get a
quality column was to re-run the whole pipeline. That is the wrong shape once
the pipeline is split across machines. The CPU box extracts most of the corpus
and the GPU box extracts the rest; the scorer is one ModernBERT on one GPU, and
both lanes need to reach it.

So this reads a run's ``results.jsonl`` plus its ``markdown/`` directory and
writes the same rows back with the quality columns filled. Nothing is
re-extracted. Only text crosses the network — clipped to what the server would
truncate to anyway — so the CPU box can score against a remote scorer without
shipping its markdown anywhere.

Crash-safety works the way the runner's does. Scores are appended to a
checkpoint as they arrive; the output file is written once, atomically, at the
end. A killed run resumes from the checkpoint rather than re-scoring.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ScoreModelMismatch", "ScoreReport", "score_run"]

#: The server truncates here too (``_quality_server._score``). Clipping on this
#: side means the difference never crosses the wire.
DEFAULT_MAX_CHARS = 40_000


class ScoreModelMismatch(RuntimeError):
    """The scorer is not serving the model this run was told to expect."""


class ScorerUnreachable(RuntimeError):
    """The quality server could not be reached, or did not answer as one."""


@dataclass
class ScoreReport:
    rows: int = 0
    scored: int = 0            # scored by this invocation
    already: int = 0           # carried from the checkpoint or already in the input
    missing_markdown: int = 0  # nothing on disk, and nothing explaining why
    handed_off: int = 0        # no markdown because another lane owns the text
    no_key: int = 0            # no sha256, so no markdown filename to look for
    failed: int = 0
    model: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Did this run do what it was asked to?

        Not merely "some row has a score". A scorer that dies partway leaves a
        column that is mostly null and looks exactly like a corpus with little
        text — so failures outweighing successes is a failed run, not a
        partial one.
        """
        if self.scored + self.already == 0:
            return False
        return self.failed <= self.scored


def _check_model(expect: str | None) -> str:
    """Ask the scorer what it is serving, and refuse a surprise.

    With ``QUALITY_URL`` set the client's ``--model`` is advisory: the server
    already loaded whatever it was started with, and ``score()`` takes the name
    from the response. Two lanes scored against two servers therefore produce
    one column holding two different scales, with nothing in the data to say
    so. One GET before any work makes that impossible.
    """
    import httpx

    external = os.environ.get("QUALITY_URL")
    if not external:
        # No server to ask yet — the client will start one with the model it
        # was given, so there is nothing to disagree with.
        return expect or ""
    try:
        resp = httpx.get(f"{external.rstrip('/')}/health", timeout=10.0)
        resp.raise_for_status()
        serving = str(resp.json().get("model") or "")
    except Exception as e:  # httpx errors, non-JSON bodies, anything
        raise ScorerUnreachable(
            f"could not ask {external} what it is serving: "
            f"{type(e).__name__}: {e}. Is the quality server up, and finished "
            f"loading its model?"
        ) from e
    if expect and serving != expect:
        # Fails closed: a server that will not say what it is serving is the
        # case the flag exists for, not one to wave through.
        raise ScoreModelMismatch(
            f"{external} is serving {serving or '(it would not say)'}, not the "
            f"expected {expect!r}. Scores from two different models are two "
            f"different scales; putting them in one column makes the column "
            f"meaningless."
        )
    return serving


def score_run(
    results: Path,
    markdown_dir: Path,
    out: Path,
    *,
    model: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    workers: int = 4,
    resume: bool = False,
    rescore: bool = False,
    progress: Any = None,
) -> ScoreReport:
    """Fill the quality columns of *results* from the markdown in *markdown_dir*.

    Every input row reaches the output, scored or not — a document with no
    markdown keeps its null quality columns rather than disappearing.
    """
    from pdfsys_bench.quality import OcrQualityScorer

    from .runner import scan_jsonl

    report = ScoreReport()
    report.model = _check_model(model)

    checkpoint = out.with_suffix(out.suffix + ".partial")
    done: dict[str, dict[str, Any]] = {}
    # --rescore means "the previous scores are wrong", which includes the
    # checkpoint's. Consulting it first would let --resume quietly win.
    if resume and not rescore and checkpoint.exists():
        def _carry(row: dict[str, Any]) -> None:
            sha = row.get("sha256")
            if sha:
                done[sha] = row

        _, good = scan_jsonl(checkpoint, _carry)
        if good < checkpoint.stat().st_size:
            # Truncate the torn tail before appending, or the next line is
            # spliced onto a half-written one and the whole checkpoint stops
            # parsing on the leg after this.
            os.truncate(checkpoint, good)
    elif checkpoint.exists():
        checkpoint.unlink()

    # Pass 1: what needs scoring. Only keys are held, never the rows — a 218k
    # results.jsonl is hundreds of megabytes and this needs a work list.
    todo: dict[str, Path] = {}
    keep: set[str] = set()  # rows whose existing score must survive the merge

    def _plan(row: dict[str, Any]) -> None:
        report.rows += 1
        sha = row.get("sha256")
        if not sha:
            report.no_key += 1
            return
        if sha in done:
            # Counted here rather than as len(done): a checkpoint entry that
            # matches no input row is not work this run accounted for, and
            # counting it would satisfy the "scored nothing" guard.
            report.already += 1
            return
        if row.get("quality_score") is not None and not rescore:
            report.already += 1
            keep.add(sha)
            return
        md = markdown_dir / f"{sha}.md"
        if not md.is_file():
            # A row this run deliberately produced no text for — lane-filtered,
            # deferred, or failed — has nothing to score, and saying so as a
            # missing file would make the CPU lane's normal output look broken.
            if row.get("skip_reason") or row.get("error_class"):
                report.handed_off += 1
            else:
                report.missing_markdown += 1
            return
        todo[sha] = md

    scan_jsonl(results, _plan)

    if todo:
        # Let OcrQualityScorer's own default stand when there is no name to
        # give it. Passing None would override the default, not fall back to it.
        name = model or report.model
        scorer = OcrQualityScorer(
            max_chars=max_chars, **({"model_name": name} if name else {})
        )
        # Resolve the server once, before the pool: _ensure_server is not
        # thread-safe, and N workers racing it would start N subprocesses.
        scorer._ensure_server()
        try:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as ckpt:
                for chunk in _chunks(sorted(todo.items()), _CHUNK):
                    # One pool per chunk: Executor.map submits every item
                    # eagerly, so handing it the whole corpus would materialise
                    # 218k futures — the allocation the streaming passes exist
                    # to avoid.
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for scored in pool.map(
                            lambda item: _score_one(scorer, item[0], item[1], max_chars),
                            chunk,
                        ):
                            if scored.get("_error"):
                                report.failed += 1
                                if len(report.failures) < 5:
                                    report.failures.append(scored["_error"])
                                continue
                            done[scored["sha256"]] = scored
                            report.scored += 1
                            ckpt.write(json.dumps(scored, ensure_ascii=False) + "\n")
                            ckpt.flush()
                            if progress is not None:
                                progress(report)
        finally:
            scorer.close()

    # Pass 2: stream the input again and write the merged output atomically, so
    # a kill here leaves the previous output intact and the checkpoint usable.
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _write(handle: Any) -> None:
        def _emit(row: dict[str, Any]) -> None:
            sha = row.get("sha256") or ""
            # `keep` holds rows pass 1 decided to leave alone. Without it, a
            # second row with the same sha256 — the same PDF twice in a corpus —
            # would get scored and its result written onto this one, replacing a
            # score the report just claimed was untouched.
            merged = None if sha in keep else done.get(sha)
            if merged:
                row = {
                    **row,
                    **{k: v for k, v in merged.items() if not k.startswith("_")},
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        scan_jsonl(results, _emit)

    # surrogatepass: a filename the OS handed us as undecodable bytes reaches
    # results.jsonl as a lone surrogate, and the default strict encoder would
    # abort pass 2 after every document had already been scored.
    with tmp.open("w", encoding="utf-8", errors="surrogatepass") as f:
        _write(f)
    os.replace(tmp, out)
    checkpoint.unlink(missing_ok=True)
    return report


#: Work per pool. Big enough to keep the server busy, small enough that the
#: futures for one chunk are a rounding error next to the corpus.
_CHUNK = 2048


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _score_one(
    scorer: Any, sha: str, md_path: Path, max_chars: int
) -> dict[str, Any]:
    """Score one document. Errors are data, so one bad row cannot end a batch."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        q = scorer.score(text[:max_chars])
    except Exception as e:
        return {"sha256": sha, "_error": f"{md_path.name}: {type(e).__name__}: {e}"[:300]}
    return {
        "sha256": sha,
        "quality_score": q.score,
        "quality_num_chars": q.num_chars,
        "quality_num_tokens": q.num_tokens,
        "quality_model": q.model,
    }


def cmd_score(args: Any) -> int:
    """CLI entry point for ``pdfsys score``."""
    from .runner import CorruptResultsError

    results = Path(args.results)
    markdown_dir = Path(args.markdown_dir)
    out = Path(args.out)

    if not results.is_file():
        print(f"Error: --results not found: {results}", file=sys.stderr)
        return 1
    if not markdown_dir.is_dir():
        print(f"Error: --markdown-dir not found: {markdown_dir}", file=sys.stderr)
        return 1
    if out.exists() and not (args.resume or args.overwrite):
        print(
            f"Error: {out} already exists. Pass --resume to continue it, or "
            f"--overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        report = score_run(
            results, markdown_dir, out,
            model=args.model,
            max_chars=args.max_chars,
            workers=args.workers,
            resume=args.resume,
            rescore=args.rescore,
        )
    except (ScoreModelMismatch, ScorerUnreachable) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except CorruptResultsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"[pdfsys score] model:   {report.model or '(local subprocess)'}")
    print(
        f"[pdfsys score] rows={report.rows} scored={report.scored} "
        f"already={report.already} no-text={report.handed_off} "
        f"no-markdown={report.missing_markdown} "
        f"no-sha256={report.no_key} failed={report.failed}"
    )
    for f in report.failures:
        print(f"  ! {f}", file=sys.stderr)
    if report.failed > len(report.failures):
        print(
            f"  ! … and {report.failed - len(report.failures)} more",
            file=sys.stderr,
        )
    if report.missing_markdown:
        # Distinct from `no-text`: these rows claim an extraction that produced
        # no file, which usually means the run had no --markdown-dir at all.
        print(
            f"[pdfsys score] warning: {report.missing_markdown} rows extracted "
            f"but have no {markdown_dir}/<sha256>.md — they keep null quality "
            f"columns. Was the run given --markdown-dir?",
            file=sys.stderr,
        )
    print(f"[pdfsys score] out:     {out}")

    if not report.ok:
        # Either nothing was scored at all — an output file identical to its
        # input — or failures outweighed successes, which leaves a column that
        # is mostly null and looks just like a corpus with little text.
        if report.scored + report.already == 0:
            print("[pdfsys score] error: not one row was scored", file=sys.stderr)
        else:
            print(
                f"[pdfsys score] error: {report.failed} documents failed against "
                f"{report.scored} scored — the scorer looks unhealthy, and the "
                f"column this wrote is mostly null.",
                file=sys.stderr,
            )
        return 1
    return 0
